"""Go execution harness: argv construction, classification, and parallel driving.

Three properties matter here, and each one has bitten real harnesses before:

**No shell.** Arguments come from a model, so they are untrusted. Every command
is built as an argument vector and executed without a shell; there is no code
path that concatenates a model string into a command line.

**Status separation.** A failing test suite is a *legitimate model-visible
result*; a crashed container is an *infrastructure failure*. Merging them makes a
broken harness look like a weak model (or vice versa), and with parallelism the
merged count scales with the fleet size.

**Resume from disk.** The driver is told which keys are already done and skips
them, so a crashed run continues instead of starting over. Progress is recorded
through the :class:`~gotooltrain.evalstore.ResultStore`, not in memory.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from math import comb
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import HarnessError, ValidationError
from .gotools import (
    GO_TOOLS as GOOLS_ALL,
)
from .gotools import (
    TOOLS_BY_NAME,
    ToolDefinition,
    assert_catalog_is_executable,
    commandless_tools,
    truncate_output,
)
from .workspace import EditOutcome, edit_file, write_file

#: Exit codes the Go toolchain uses for its own failures.
GO_TEST_FAILURE_EXIT: Final[int] = 1

#: Exit code used when the harness itself killed the command.
TIMEOUT_EXIT: Final[int] = 124


class Status(str, Enum):
    """Outcome of one tool execution."""

    #: Command ran and succeeded.
    OK = "ok"
    #: Command ran and failed in a way the model should see (e.g. tests failed).
    TOOL_ERROR = "tool_error"
    #: The model's request was invalid: unknown tool, bad arguments.
    MODEL_ERROR = "model_error"
    #: Infrastructure failure: container, timeout, missing binary.
    HARNESS_ERROR = "harness_error"


@dataclass(frozen=True, slots=True)
class ExecRequest:
    """One tool invocation to execute."""

    task_id: str
    sample_index: int
    tool_name: str
    arguments: Mapping[str, Any]
    #: Where the command runs. Each worker needs its own; the harness never
    #: shares one directory across concurrent tasks.
    workspace: Path
    timeout_s: int = 120
    #: Position of this call within its turn. Two calls to the same tool in one
    #: turn are distinct attempts and must not share a result-store key.
    call_index: int = 0
    #: Which agent turn this call belongs to. A trajectory revisits the same tool
    #: on later turns (``go_test`` before and after a fix), so the turn has to be
    #: part of the key or the second run would be mistaken for a cached first.
    turn_index: int = 0


@dataclass(frozen=True, slots=True)
class ExecResult:
    """What happened, kept separate from what the model will be shown."""

    task_id: str
    sample_index: int
    tool_name: str
    status: Status
    exit_code: int | None
    stdout: str
    duration_ms: int
    #: Populated only for HARNESS_ERROR; names the infrastructure fault.
    harness_error: str = ""
    #: Position of the originating call within its turn, so a result can be
    #: matched back to the ``tool_call`` that asked for it.
    call_index: int = 0

    @property
    def is_model_visible(self) -> bool:
        """Whether this output belongs in a ``<tool_result>`` block."""
        return self.status in (Status.OK, Status.TOOL_ERROR)

    def to_record(self) -> dict[str, Any]:
        """Serialisable form, ready for the result store."""
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "duration_ms": self.duration_ms,
            "harness_error": self.harness_error,
            "call_index": self.call_index,
        }


class Executor(Protocol):
    """Runs an argv inside an isolated workspace."""

    def run(self, argv: Sequence[str], request: ExecRequest) -> tuple[int | None, str, str]:
        """Return ``(exit_code, stdout, stderr)``; raise only for infrastructure loss."""


# --------------------------------------------------------------- argv building


def validate_arguments(tool: ToolDefinition, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Check a call against the tool's schema. Failures are the model's fault."""
    properties: Mapping[str, Any] = tool.parameters["properties"]
    required: Sequence[str] = tool.parameters.get("required", [])

    missing = [name for name in required if name not in arguments]
    if missing:
        raise ValidationError(f"{tool.name} is missing required argument(s): {missing}")
    unexpected = sorted(set(arguments) - set(properties))
    if unexpected:
        raise ValidationError(f"{tool.name} received unexpected argument(s): {unexpected}")
    for name, spec in properties.items():
        if name not in arguments:
            continue
        value = arguments[name]
        if spec.get("type") == "string" and not isinstance(value, str):
            raise ValidationError(
                f"{tool.name}.{name} must be a string, got {type(value).__name__}"
            )
    return {name: arguments[name] for name in properties if name in arguments}


def build_argv(tool: ToolDefinition, arguments: Mapping[str, Any]) -> list[str]:
    """Render a tool's command template into an argument vector.

    The template is trusted (it ships with the catalogue); the substituted values
    are not. They become individual argv entries and are never joined into a
    string, so quoting and separators cannot be used to inject a command.
    """
    checked = validate_arguments(tool, arguments)
    if not tool.command_template:
        return []
    rendered: list[str] = []
    for token in tool.command_template.split():
        if token.startswith("{") and token.endswith("}"):
            name = token[1:-1]
            rendered.append(str(checked.get(name, "")))
        else:
            rendered.append(token)
    return rendered


def plan(request: ExecRequest) -> tuple[ToolDefinition, list[str]]:
    """Resolve the tool and its argv, or raise a model-attributable error."""
    try:
        tool = TOOLS_BY_NAME[request.tool_name]
    except KeyError as exc:
        raise ValidationError(f"unknown tool {request.tool_name!r}") from exc
    return tool, build_argv(tool, request.arguments)


# ----------------------------------------------------------------- execution


class LocalExecutor:
    """Runs commands on the host with no shell.

    Only appropriate for trusted workspaces (the canary self-check, and local
    development). Untrusted model output belongs in the container pool.
    """

    def __init__(self, cwd: Path | None = None) -> None:
        """Pin the working directory; defaults to the request workspace."""
        self.cwd = cwd

    def run(self, argv: Sequence[str], request: ExecRequest) -> tuple[int | None, str, str]:
        """Run without a shell; raise ``HarnessError`` only if the environment is lost."""
        if not argv:
            return 0, "", ""
        try:
            completed = subprocess.run(  # noqa: S603 - argv list, shell=False
                list(argv),
                cwd=str(self.cwd or request.workspace),
                capture_output=True,
                text=True,
                timeout=request.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise HarnessError(f"{argv[0]} not found: {exc}") from exc
        except subprocess.TimeoutExpired:
            return TIMEOUT_EXIT, "", f"timed out after {request.timeout_s}s"
        return completed.returncode, completed.stdout, completed.stderr


def execute(request: ExecRequest, executor: Executor) -> ExecResult:
    """Run one request and classify the outcome."""
    started = time.monotonic()
    try:
        tool, argv = plan(request)
    except ValidationError as exc:
        return ExecResult(
            task_id=request.task_id,
            sample_index=request.sample_index,
            tool_name=request.tool_name,
            status=Status.MODEL_ERROR,
            exit_code=None,
            stdout="",
            duration_ms=int((time.monotonic() - started) * 1000),
            harness_error=str(exc),
            call_index=request.call_index,
        )

    if not argv:
        return _mutate_workspace(request, tool, started)

    try:
        exit_code, stdout, stderr = executor.run(argv, request)
    except HarnessError as exc:
        return ExecResult(
            task_id=request.task_id,
            sample_index=request.sample_index,
            tool_name=tool.name,
            status=Status.HARNESS_ERROR,
            exit_code=None,
            stdout="",
            duration_ms=int((time.monotonic() - started) * 1000),
            harness_error=str(exc),
            call_index=request.call_index,
        )

    output = stdout if stdout.strip() else stderr
    # Tool output goes through the same budget as the catalogue declares, so the
    # model sees exactly the shape it will see at serving time.
    output = truncate_output(output, tool.output_budget_chars)
    status = Status.OK if exit_code == 0 else Status.TOOL_ERROR
    return ExecResult(
        task_id=request.task_id,
        sample_index=request.sample_index,
        tool_name=tool.name,
        status=status,
        exit_code=exit_code,
        stdout=output,
        duration_ms=int((time.monotonic() - started) * 1000),
        call_index=request.call_index,
    )


#: Tools with no command template, and the workspace operation each one performs.
#: A new commandless tool must be added here, so it cannot silently no-op.
WORKSPACE_MUTATIONS: Final[Mapping[str, Callable[[Path, dict[str, Any]], EditOutcome]]] = {
    "write_file": write_file,
    "edit_file": edit_file,
}


def assert_harness_covers_catalog() -> None:
    """Fail at import-time-equivalent if a tool has no way to run.

    Called by the eval entry point. Without it, a tool that is neither a command
    nor a workspace mutation would report success for doing nothing.
    """
    assert_catalog_is_executable(commandless_tools())
    unhandled = sorted(set(commandless_tools()) - set(WORKSPACE_MUTATIONS))
    if unhandled:
        raise HarnessError(
            f"tools are commandless but have no workspace handler: {unhandled}. Add a handler "
            f"in WORKSPACE_MUTATIONS or give the tool a command template."
        )
    unknown = sorted(set(WORKSPACE_MUTATIONS) - {t.name for t in GOOLS_ALL})
    if unknown:
        raise HarnessError(f"workspace handlers reference unknown tools: {unknown}")


def _mutate_workspace(request: ExecRequest, tool: ToolDefinition, started: float) -> ExecResult:
    """Apply a non-command tool directly to the workspace.

    These tools edit files rather than run programs. An empty argv is *not*
    treated as "nothing to do": that would report success for an edit that never
    happened, training the model to believe its changes landed. A tool without a
    registered handler is a harness bug and raises.
    """
    handler = WORKSPACE_MUTATIONS.get(tool.name)
    if handler is None:
        raise HarnessError(
            f"tool {tool.name!r} has no command template and no workspace handler; refusing to "
            f"report a success that did not happen"
        )
    try:
        outcome = handler(request.workspace, dict(request.arguments))
    except ValidationError as exc:
        return ExecResult(
            task_id=request.task_id,
            sample_index=request.sample_index,
            tool_name=tool.name,
            status=Status.MODEL_ERROR,
            exit_code=None,
            stdout="",
            duration_ms=int((time.monotonic() - started) * 1000),
            harness_error=str(exc),
            call_index=request.call_index,
        )
    return ExecResult(
        task_id=request.task_id,
        sample_index=request.sample_index,
        tool_name=tool.name,
        status=Status.OK,
        exit_code=0,
        stdout=truncate_output(outcome.message, tool.output_budget_chars),
        duration_ms=int((time.monotonic() - started) * 1000),
        call_index=request.call_index,
    )


# ----------------------------------------------------------------- canary


@dataclass(frozen=True, slots=True)
class CanaryCase:
    """A probe with a known outcome."""

    name: str
    request: ExecRequest
    should_fail: bool


def run_canary(executor: Executor, cases: Sequence[CanaryCase]) -> None:
    """Prove the executor can tell success from failure before a fan-out.

    A harness that cannot discriminate returns success for everything, which
    would report 100% on every run and look like a great model. Refuse to start.
    """
    if not cases:
        raise HarnessError("canary set is empty; refusing to start a run that cannot be trusted")
    failures: list[str] = []
    for case in cases:
        result = execute(case.request, executor)
        observed_fail = result.status is not Status.OK
        if observed_fail != case.should_fail:
            failures.append(
                f"{case.name}: expected {'failure' if case.should_fail else 'success'}, "
                f"got {result.status.value}"
            )
    if failures:
        raise HarnessError(
            "canary failed; the executor cannot discriminate success from failure, so any "
            f"score it produces is meaningless: {failures}"
        )


# ----------------------------------------------------------------- parallel


def run_many(
    requests: Sequence[ExecRequest],
    executor: Executor,
    *,
    max_parallel: int = 4,
    should_run: Callable[[ExecRequest], bool] | None = None,
    on_result: Callable[[ExecResult], None] | None = None,
) -> list[ExecResult]:
    """Execute requests concurrently, returning results in input order.

    ``should_run`` is the resume predicate: a request it rejects is not executed.
    ``on_result`` is called as each result lands, so the caller can persist it and
    survive a crash. Ordering is restored before returning so aggregation is
    deterministic regardless of which task finished first.
    """
    if max_parallel < 1:
        raise HarnessError(f"max_parallel must be >= 1, got {max_parallel}")
    selected = [
        (index, request)
        for index, request in enumerate(requests)
        if should_run is None or should_run(request)
    ]
    if not selected:
        return []

    results: list[ExecResult | None] = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {pool.submit(execute, request, executor): index for index, request in selected}
        for future, index in futures.items():
            result = future.result()
            results[index] = result
            if on_result is not None:
                on_result(result)
    return [result for result in results if result is not None]


def tool_binaries() -> list[str]:
    """Every command the catalogue will run, deduplicated and in catalogue order.

    Derived from the templates rather than hard-coded, so a tool added later is
    covered without anyone remembering to update a list.
    """
    seen: list[str] = []
    for tool in GOOLS_ALL:
        if not tool.command_template:
            continue
        binary = tool.command_template.split()[0]
        if binary not in seen:
            seen.append(binary)
    return seen


def missing_tool_binaries() -> list[str]:
    """Catalogue commands that are not resolvable on PATH.

    Checked before a fan-out because a missing binary would otherwise surface as a
    ``HARNESS_ERROR`` on every single sample, after a full pass of wasted work, and
    read like a broken harness rather than a missing dependency. Note that a
    canary alone cannot catch this: a missing binary makes every command "fail",
    which is exactly what a must-fail probe expects.
    """
    return [binary for binary in tool_binaries() if shutil.which(binary) is None]


# --------------------------------------------------------------- aggregation


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k).

    With ``k > n`` every sample is used, so the result is 1.0 when anything
    passed and 0.0 otherwise.
    """
    if n < 0 or c < 0 or c > n:
        raise HarnessError(f"invalid pass@k inputs n={n} c={c}")
    if k < 1:
        raise HarnessError(f"k must be >= 1, got {k}")
    if k > n:
        return 1.0 if c > 0 else 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


@dataclass(frozen=True, slots=True)
class TaskSummary:
    """Aggregated outcome for one task across its samples."""

    task_id: str
    samples: int
    passed: int
    harness_errors: int
    model_errors: int
    tool_errors: int
    pass_at_1: float
    pass_at_n: float

    def to_record(self) -> dict[str, Any]:
        """Serialisable form for the result store and the final report."""
        return {
            "task_id": self.task_id,
            "samples": self.samples,
            "passed": self.passed,
            "harness_errors": self.harness_errors,
            "model_errors": self.model_errors,
            "tool_errors": self.tool_errors,
            "pass_at_1": self.pass_at_1,
            "pass_at_n": self.pass_at_n,
        }


def summarise(results: Iterable[ExecResult], *, k: int = 1) -> list[TaskSummary]:
    """Group by task and compute pass@k.

    Sorted by ``task_id`` so two runs over the same results produce byte
    identical reports; without sorting, parallel completion order leaks into the
    numbers and real movement is indistinguishable from noise.
    """
    grouped: dict[str, list[ExecResult]] = {}
    for result in results:
        grouped.setdefault(result.task_id, []).append(result)

    summaries: list[TaskSummary] = []
    for task_id in sorted(grouped):
        bucket = sorted(grouped[task_id], key=lambda r: r.sample_index)
        passed = sum(1 for r in bucket if r.status is Status.OK)
        summaries.append(
            TaskSummary(
                task_id=task_id,
                samples=len(bucket),
                passed=passed,
                harness_errors=sum(1 for r in bucket if r.status is Status.HARNESS_ERROR),
                model_errors=sum(1 for r in bucket if r.status is Status.MODEL_ERROR),
                tool_errors=sum(1 for r in bucket if r.status is Status.TOOL_ERROR),
                pass_at_1=pass_at_k(len(bucket), passed, 1),
                pass_at_n=pass_at_k(len(bucket), passed, k),
            )
        )
    return summaries
