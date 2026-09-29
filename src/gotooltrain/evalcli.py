"""Command-line entry points for the eval harness.

Two commands, matching the two halves of the session-judge flow:

``queue``   run the agent loop and write the trajectories that need grading
``judge``   read graded verdicts and produce the scored report

Both are thin. The logic lives in :mod:`gotooltrain.evalrun` and
:mod:`gotooltrain.judge`, so anything the CLI can do is testable without a
subprocess.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import HarnessError, ToolTrainError
from .evalrun import (
    RunConfig,
    canary_cases,
    ingest_judge_verdicts,
    judge_queue,
    run_evaluation,
)
from .evalstore import ResultStore, sha256_text
from .generator import HttpGenerator
from .gorun import (
    ExecRequest,
    Executor,
    LocalExecutor,
    missing_tool_binaries,
    run_canary,
)
from .judge import (
    SessionJudge,
    read_judge_queue,
    read_verdicts,
    write_judge_queue,
)
from .sandbox import (
    ContainerExecutor,
    ContainerPool,
    DockerBackend,
    SandboxSpec,
    docker_available,
)
from .schema import FORMAT_VERSION
from .template import load_template_source, template_format_version


def _load_tasks(path: str) -> list[dict[str, Any]]:
    """Read a task list from JSON or JSONL, with or without a fixture directory."""
    source = Path(path)
    if not source.is_file():
        raise ToolTrainError(f"task file not found: {source}")
    text = source.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    if source.suffix == ".jsonl":
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ToolTrainError(f"{source}:{number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ToolTrainError(f"{source}:{number} must be a JSON object")
            records.append(record)
    else:
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolTrainError(f"{source} is not valid JSON: {exc}") from exc
        if not isinstance(decoded, list) or not all(isinstance(r, dict) for r in decoded):
            raise ToolTrainError(f"{source} must hold a list of task objects")
        records = list(decoded)
    for index, record in enumerate(records):
        if "id" not in record:
            raise ToolTrainError(f"{source}: task {index} has no 'id'")
    return records


def _config(args: argparse.Namespace) -> RunConfig:
    """Build the run config from the command line.

    ``template_sha`` is taken from the shipped template rather than trusted from
    the arguments, so a run cannot be keyed by a template this build does not
    actually use.
    """
    source = load_template_source()
    declared = template_format_version(source)
    if declared != FORMAT_VERSION:
        raise ToolTrainError(
            f"the installed template declares {declared!r} but this build expects "
            f"{FORMAT_VERSION!r}. Refusing to evaluate against a token format the run "
            f"was not trained on."
        )
    return RunConfig(
        model_id=args.model,
        model_revision=args.revision,
        dataset_version=args.dataset_version,
        seed=args.seed,
        decode_params=json.loads(args.decode_params),
        n_samples=args.n_samples,
        max_turns=args.max_turns,
        template_sha=sha256_text(source),
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options every command needs, so both halves key runs identically."""
    parser.add_argument("--store", required=True, help="result store root")
    parser.add_argument("--tasks", required=True, help="task file (.json or .jsonl)")
    parser.add_argument("--model", required=True, help="model id under evaluation")
    parser.add_argument("--revision", required=True, help="model revision or checkpoint")
    parser.add_argument("--dataset-version", required=True, help="holdout set version")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument(
        "--decode-params", default='{"temperature": 0.0}', help="JSON decode parameters"
    )


def _executor(args: argparse.Namespace) -> Executor:
    """Build the executor for this run.

    ``--sandbox docker`` runs model-authored code in a pooled container; the
    default runs it on the host and says so. The default is refused for anything
    other than a local smoke test, because untrusted generated Go code on the
    host is exactly the thing the container pool exists to prevent.
    """
    fixture = Path(args.fixture) if args.fixture else None
    if fixture is not None and not fixture.is_dir():
        raise ToolTrainError(f"fixture directory not found: {fixture}")

    if args.sandbox == "local":
        if not args.allow_local_execution:
            raise ToolTrainError(
                "refusing to execute model-authored code on the host. Pass "
                "--allow-local-execution to override, or use --sandbox docker."
            )
        # The fixture is validated above for both backends so it is never silently
        # ignored; here it only seeds the per-sample workspaces, and the executor
        # runs in whatever directory the harness hands it. Pinning a separate cwd
        # would mean the model edits one repository while the tests read another.
        return LocalExecutor()

    if not docker_available():
        raise ToolTrainError(
            "Docker was requested for the sandbox but is not available on this host. "
            "Install Docker, or use --sandbox local --allow-local-execution for a "
            "trusted smoke test."
        )
    spec = SandboxSpec(image=args.image, memory_mb=args.memory_mb, cpus=args.cpus)
    pool = ContainerPool(backend=DockerBackend(), spec=spec, size=args.workers)
    return ContainerExecutor(pool, fixture=fixture)


def cmd_queue(args: argparse.Namespace) -> int:
    """Run the agent loop and write the trajectories awaiting a verdict."""
    config = _config(args)
    store = ResultStore(args.store)
    tasks = _load_tasks(args.tasks)
    generator = HttpGenerator(
        base_url=args.model_url,
        model=args.model,
        temperature=config.decode_params.get("temperature", 0.0),
        max_tokens=int(config.decode_params.get("max_tokens", 4096)),
    )
    executor = _executor(args)
    workspace = Path(args.workspace).expanduser()
    # Scratch state is created under the system temp directory unless the operator
    # says otherwise. A default of "." would drop per-sample directories into
    # whatever directory the command happened to be run from -- usually a source
    # tree -- and a stray directory there is both litter and a confusing thing to
    # debug when it shadows a real path.
    workspace.mkdir(parents=True, exist_ok=True)

    if args.sandbox == "local":
        # A canary cannot catch a missing binary: it makes every command "fail",
        # which is exactly what a must-fail probe expects. The dependency is
        # therefore checked directly, from the catalogue, before a sample is spent.
        absent = missing_tool_binaries()
        if absent:
            raise ToolTrainError(
                f"the tool catalogue needs {absent}, which is not on PATH. Every tool call "
                f"would fail as a harness error. Install it, or run inside the sandbox image."
            )
    # Probe before the fan-out. A missing RTK or a toolchain that cannot report a
    # failure must stop the run here, naming the cause, rather than surfacing as a
    # harness error on every sample after a full pass of wasted work.
    if not args.skip_canary:
        probe = workspace / "canary"
        probe.mkdir(parents=True, exist_ok=True)
        run_canary(executor, canary_cases(probe))
        print("canary passed", file=sys.stderr)

    report = run_evaluation(
        config,
        tasks,
        store,
        generator,
        executor,
        run_id=args.run_id,
        judge=None,
        workspace=workspace,
        fixture=args.fixture,
        max_parallel=args.workers,
    )
    entries = judge_queue(config, tasks, store)
    written = write_judge_queue(args.out, entries)
    print(
        f"execution: {len(report.summaries)} tasks, mean_pass@1={report.mean_pass_at_1:.3f} "
        f"({report.scoring}), truncated={report.truncated}",
        file=sys.stderr,
    )
    print(f"wrote {written} trajectory/ies awaiting judgement to {args.out}", file=sys.stderr)
    if written == 0:
        print(
            "nothing to judge: every sample is either already judged or was lost to a "
            f"harness error ({report.unjudged_tasks} task(s) unjudged). Check the report "
            "before trusting an unchanged score.",
            file=sys.stderr,
        )
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """Ingest verdicts and print the scored report.

    Nothing is executed here. The trajectories already exist in the store; a
    missing one is a run that never completed, and inventing it to satisfy the
    queue would grade a sample that never ran.
    """
    config = _config(args)
    store = ResultStore(args.store)
    tasks = _load_tasks(args.tasks)
    queue = read_judge_queue(args.queue)
    verdicts = read_verdicts(args.verdicts, queue)
    # One identity for the whole command. Ingesting under a different name than
    # the judge reports would make the store reject its own verdicts on resume.
    session = SessionJudge(verdicts)
    ingest_judge_verdicts(config, args.run_id, store, verdicts, judge_model=session.identity)
    report = run_evaluation(
        config,
        tasks,
        store,
        _StoreOnlyGenerator(),
        _NeverExecuted(),
        run_id=args.run_id,
        judge=session,
    )
    print(json.dumps(report.to_record(), indent=2, sort_keys=True))
    return 0


class _NeverExecuted:
    """An executor that cannot run anything.

    Judging reads stored results. Reaching this means a trajectory was missing, and
    that is a store problem to report, not something to work around by running the
    command the model asked for a second time.
    """

    def run(self, argv: Sequence[str], request: ExecRequest) -> tuple[int | None, str, str]:
        """Refuse every request."""
        raise ToolTrainError(
            "the judge command never executes: trajectories must already be in the "
            f"store. Missing trajectory for {request.task_id}#{request.sample_index}."
        )


class _StoreOnlyGenerator:
    """A generator that refuses to sample.

    Judging reads what the queue command already stored. Reaching this means a
    trajectory is missing, and sampling a fresh one would grade something the
    verdict file was never written about.
    """

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 - protocol shape
        """Refuse every request."""
        raise ToolTrainError(
            "the judge command never samples: every trajectory must already be in the "
            "store. Re-run the queue command for the missing tasks."
        )


def _add_sandbox(parser: argparse.ArgumentParser) -> None:
    """Execution options for the command that actually runs code."""
    parser.add_argument(
        "--sandbox",
        choices=("docker", "local"),
        default="docker",
        help="where model-authored code runs (default: docker)",
    )
    parser.add_argument(
        "--image",
        default="gotooltrain/go-sandbox:0.1.0",
        help=(
            "sandbox image (default: the one with rtk in it; a stock golang image "
            "fails every go_build and go_test call as a harness error)"
        ),
    )
    parser.add_argument("--workers", type=int, default=4, help="concurrent samples and containers")
    parser.add_argument("--memory-mb", type=int, default=3072)
    parser.add_argument("--cpus", type=float, default=2.0)
    parser.add_argument(
        "--fixture", default=None, help="repository state restored before each task"
    )
    parser.add_argument(
        "--skip-canary",
        action="store_true",
        help="run without the preflight probe (only for a harness already proven on this host)",
    )
    parser.add_argument(
        "--allow-local-execution",
        action="store_true",
        help="permit host execution of model-authored code (trusted smoke tests only)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``."""
    parser = argparse.ArgumentParser(prog="gotooltrain-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    queue = sub.add_parser("queue", help="run the agent loop and write trajectories to grade")
    _add_common(queue)
    queue.add_argument("--run-id", default=None)
    queue.add_argument("--out", required=True, help="queue file to write")
    queue.add_argument("--model-url", required=True, help="OpenAI-compatible base URL")
    queue.add_argument(
        "--workspace",
        default=str(Path(tempfile.gettempdir()) / "gotooltrain-eval"),
        help="root the per-sample workspaces live under (default: a temp directory)",
    )
    _add_sandbox(queue)
    queue.set_defaults(func=cmd_queue)

    judge = sub.add_parser("judge", help="ingest verdicts and print the scored report")
    _add_common(judge)
    judge.add_argument("--run-id", required=True)
    judge.add_argument("--queue", required=True, help="the queue these verdicts answer")
    judge.add_argument("--verdicts", required=True, help="graded verdict file")
    judge.set_defaults(func=cmd_judge)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ToolTrainError, HarnessError) as exc:
        # HarnessError is a sibling of ToolTrainError, not a subclass, so it is
        # caught explicitly: a dead container or an unreachable model is an
        # expected failure and deserves the same clean message as a bad argument,
        # not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
