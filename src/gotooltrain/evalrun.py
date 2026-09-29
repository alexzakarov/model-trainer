"""Agent-loop evaluation: generate -> execute -> feed back -> ... -> judge.

The pipeline is a trajectory, not three independent passes. A Go agent reads a
file, edits it, runs the tests, reads the failure, and edits again; evaluating
only the first turn would score the model on something no user would ask for, and
would train on trajectories that end before they are useful.

Structure that follows from that:

**Turn-indexed keys.** Every (task, sample, turn) pair gets its own fingerprint,
so a crash on turn 4 resumes at turn 4 and an unchanged turn is never re-billed.
``go_test`` on turn 0 and turn 3 are different attempts and must not collide.

**Tool results go back to the model.** The generator receives the conversation
so far, including ``<tool_result>`` blocks in the same shape the template emits.
The alternative -- generating once and executing whatever came back -- produces
single-turn data that teaches the model to guess at outcomes.

**Sequential inside a turn, parallel across samples.** Turns within one sample
are causally dependent and cannot overlap, but independent samples run
concurrently. Parallelism that cut across turns would execute edits against a
workspace state the model has not seen yet.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import EvalStoreError
from .evalstore import (
    MANIFEST,
    Fingerprint,
    ResultStore,
    canonical_json,
    sha256_text,
)
from .gorun import (
    CanaryCase,
    ExecRequest,
    ExecResult,
    Executor,
    Status,
    TaskSummary,
    assert_harness_covers_catalog,
    pass_at_k,
    run_many,
)
from .gotools import catalog_sha
from .judge import QueueEntry, SessionJudge, render_trajectory
from .schema import FORMAT_VERSION
from .template import load_template_source

HARNESS_VERSION: Final[str] = "0.2.0"
STAGE_GENERATE: Final[str] = "generate"
STAGE_EXECUTE: Final[str] = "execute"
STAGE_JUDGE: Final[str] = "judge"

#: Scoring was decided as execution *and* a strong judge. A run that skips the
#: judge measures something different, so it cannot be compared to one that has it.
SCORING_JUDGE: Final[str] = "judge"
SCORING_EXECUTION_ONLY: Final[str] = "execution_only"
#: Scored in a coding session rather than by a served endpoint. Cheapest option and
#: the least reproducible, so it gets its own label instead of borrowing "judge".
SCORING_SESSION_JUDGE: Final[str] = "session_judge"

#: Enough turns for read -> edit -> test -> read failure -> edit -> test. Beyond
#: this the agent is looping, and a looping agent is a failure to be scored, not
#: an excuse to keep spending tokens.
DEFAULT_MAX_TURNS: Final[int] = 8


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything that identifies a run, and therefore every attempt in it."""

    model_id: str
    model_revision: str
    dataset_version: str
    seed: int
    decode_params: Mapping[str, Any]
    n_samples: int = 1
    max_turns: int = DEFAULT_MAX_TURNS
    image_digest: str = "none"
    harness_version: str = HARNESS_VERSION
    format_version: str = FORMAT_VERSION
    template_sha: str = field(default_factory=lambda: sha256_text(load_template_source()))
    tool_catalog_sha: str = field(default_factory=catalog_sha)

    def __post_init__(self) -> None:
        """Reject a configuration that cannot produce a keyed attempt."""
        if self.n_samples < 1:
            raise EvalStoreError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.max_turns < 1:
            raise EvalStoreError(f"max_turns must be >= 1, got {self.max_turns}")
        if not self.model_id or not self.model_revision:
            raise EvalStoreError("model_id and model_revision are required")

    def fingerprint(
        self, task_id: str, sample_index: int, stage: str, turn_index: int = 0
    ) -> Fingerprint:
        """Content key for one attempt in one stage of one turn.

        ``turn_index`` is folded into ``stage`` rather than added as a field: it
        is part of the stage's identity ("the generation of turn 2"), and keeping
        the fingerprint schema stable means old stores stay readable.
        """
        return Fingerprint(
            task_id=task_id,
            sample_index=sample_index,
            stage=f"{stage}:t{turn_index}",
            model_id=self.model_id,
            model_revision=self.model_revision,
            format_version=self.format_version,
            template_sha=self.template_sha,
            harness_version=self.harness_version,
            dataset_version=self.dataset_version,
            seed=self.seed,
            decode_params=dict(self.decode_params),
            tool_catalog_sha=self.tool_catalog_sha,
            image_digest=self.image_digest,
        )

    def manifest(self) -> dict[str, Any]:
        """Run manifest, written once and compared on restart."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "dataset_version": self.dataset_version,
            "harness_version": self.harness_version,
            "n_samples": self.n_samples,
            "max_turns": self.max_turns,
            "seed": self.seed,
            "decode_params": dict(self.decode_params),
            "image_digest": self.image_digest,
            "format_version": self.format_version,
            "template_sha": self.template_sha,
            "tool_catalog_sha": self.tool_catalog_sha,
        }


class Generator(Protocol):
    """Produces one assistant turn, given the conversation so far."""

    def generate(
        self,
        task: Mapping[str, Any],
        sample_index: int,
        turn_index: int,
        messages: Sequence[Mapping[str, Any]],
        fingerprint: Fingerprint,
    ) -> Mapping[str, Any]:
        """Return e.g. ``{"text": ..., "tool_calls": [...]}``; empty calls ends the sample."""


class Judge(Protocol):
    """Scores a completed trajectory."""

    def judge(self, record: Mapping[str, Any], fingerprint: Fingerprint) -> Mapping[str, Any]:
        """Return e.g. ``{"score": 1.0, "reason": ...}``."""


@dataclass(slots=True)
class RunReport:
    """Deterministic output of a run.

    ``scoring`` is part of the result, not metadata. A run scored by an LLM judge
    and a run scored only by command exit codes measure different things, and the
    numbers are not comparable; recording the mode means a reader can tell which
    one they are looking at instead of assuming.
    """

    run_id: str
    summaries: list[TaskSummary]
    judged: dict[str, float]
    stage_counts: dict[str, int]
    reused: dict[str, int]
    #: Samples that hit the turn limit while still calling tools. Reported apart
    #: from pass rate so a looping agent is visible rather than averaged away.
    truncated: int = 0
    #: ``SCORING_JUDGE`` or ``SCORING_EXECUTION_ONLY``.
    scoring: str = SCORING_JUDGE
    #: Identity of the judge, recorded so two runs are never silently compared.
    judge_identity: str = ""
    #: Tasks that had no judge verdict, i.e. lost to a harness error. A judge run
    #: where this is non-zero scored fewer tasks than it ran.
    unjudged_tasks: int = 0

    @property
    def scored_by_judge(self) -> bool:
        """Whether anything other than exit codes decided the scores."""
        return self.scoring in (SCORING_JUDGE, SCORING_SESSION_JUDGE)

    @property
    def reproducible(self) -> bool:
        """Whether re-deriving the score would give the same number.

        Only the *judge* is the unreproducible part. Execution results are
        byte-reproducible and the store memoises them by key, so an
        execution-only run is reproducible; a served judge at temperature 0 is close
        to it. A session judge is the exception: the same trajectory graded in a
        later session may get a different verdict, and a report that hid that would
        invite comparing it against a served run as if they measured the same thing.
        """
        return self.scoring != SCORING_SESSION_JUDGE

    @property
    def mean_pass_at_1(self) -> float:
        """Average pass@1 across tasks; 0.0 for an empty run."""
        if not self.summaries:
            return 0.0
        return sum(s.pass_at_1 for s in self.summaries) / len(self.summaries)

    @property
    def mean_pass_at_n(self) -> float:
        """Average pass@k across tasks; 0.0 for an empty run."""
        if not self.summaries:
            return 0.0
        return sum(s.pass_at_n for s in self.summaries) / len(self.summaries)

    def to_record(self) -> dict[str, Any]:
        """Serialisable form, safe to diff between runs."""
        return {
            "scoring": self.scoring,
            "judge_identity": self.judge_identity,
            "scored_by_judge": self.scored_by_judge,
            "reproducible": self.reproducible,
            "unjudged_tasks": self.unjudged_tasks,
            "run_id": self.run_id,
            "tasks": [s.to_record() for s in self.summaries],
            "mean_pass_at_1": self.mean_pass_at_1,
            "mean_pass_at_n": self.mean_pass_at_n,
            "judged": self.judged,
            "stage_counts": self.stage_counts,
            "reused": self.reused,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class Turn:
    """One assistant turn and the tool results it caused."""

    index: int
    generated: Mapping[str, Any]
    results: tuple[ExecResult, ...] = ()

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Tool calls the model asked for in this turn."""
        return _tool_calls(self.generated)

    @property
    def has_harness_error(self) -> bool:
        """Whether infrastructure failed during this turn."""
        return any(r.status is Status.HARNESS_ERROR for r in self.results)

    def to_record(self) -> dict[str, Any]:
        """Serialisable trajectory turn for the judge."""
        return {
            "turn": self.index,
            "generated": dict(self.generated),
            "executions": [r.to_record() for r in self.results],
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The full multi-turn record of one sample."""

    task_id: str
    sample_index: int
    turns: tuple[Turn, ...]
    #: True when the turn budget ran out while the model was still calling tools.
    truncated: bool = False

    @property
    def all_results(self) -> tuple[ExecResult, ...]:
        """Every execution across every turn."""
        return tuple(result for turn in self.turns for result in turn.results)

    @property
    def has_harness_error(self) -> bool:
        """Whether any turn hit an infrastructure failure."""
        return any(turn.has_harness_error for turn in self.turns)

    @property
    def all_ok(self) -> bool:
        """Whether every command that ran succeeded, and something ran.

        "All OK" over an empty trajectory is not a pass: the model did nothing.
        """
        results = self.all_results
        return bool(results) and all(r.status is Status.OK for r in results)

    def to_record(self) -> dict[str, Any]:
        """Serialisable trajectory for the judge."""
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "turns": [turn.to_record() for turn in self.turns],
            "truncated": self.truncated,
        }


def _tool_calls(generated: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls, refusing a malformed container and dropping bad items.

    A non-dict entry is dropped rather than fatal: one malformed call should not
    discard a whole trajectory that may contain the run's only good turn. A
    non-list ``tool_calls`` is an interface error and does raise.
    """
    calls = generated.get("tool_calls") or []
    if not isinstance(calls, list):
        raise EvalStoreError(f"generated tool_calls must be a list, got {type(calls).__name__}")
    return [c for c in calls if isinstance(c, dict)]


def _observation(result: ExecResult, call: Mapping[str, Any]) -> dict[str, Any]:
    """Render one tool result as the model will see it.

    Harness failures are rendered as an explicit infrastructure fault rather than
    as empty output: showing an empty ``<tool_result>`` would teach the model
    that a crashed container means "the command printed nothing".
    """
    call_id = str(call.get("id") or f"{result.tool_name}-{result.call_index}")
    if result.status is Status.HARNESS_ERROR:
        content = f"[infrastructure error] {result.harness_error or 'execution failed'}"
    else:
        content = result.stdout
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": result.tool_name,
        "status": result.status.value,
        "content": content,
    }


def _messages(task: Mapping[str, Any], turns: Sequence[Turn]) -> list[dict[str, Any]]:
    """Rebuild the conversation in the order the model saw it.

    Tool results follow their turn's assistant message, matching the Anthropic
    block order the template emits. The generator receives exactly this, so a
    model cannot be evaluated on a conversation it would never be served.
    """
    history: list[dict[str, Any]] = [{"role": "user", "content": str(task.get("prompt", ""))}]
    for turn in turns:
        history.append(_assistant_message(turn))
        for call, result in zip(turn.tool_calls, turn.results, strict=False):
            history.append(_observation(result, call))
    return history


def _assistant_message(turn: Turn) -> dict[str, Any]:
    """Render one assistant turn, carrying tool calls only when there are any.

    A turn that called no tools has no ``tool_calls`` key at all: emitting an
    empty list would make the model expect a matching ``<tool_result>`` block that
    never arrives, and a template that renders one would train on a transcript
    that never occurs at serving time.
    """
    message: dict[str, Any] = {
        "role": "assistant",
        "content": str(turn.generated.get("text", "")),
    }
    if turn.tool_calls:
        message["tool_calls"] = turn.tool_calls
    return message


def _generate_turn(
    config: RunConfig,
    task: Mapping[str, Any],
    sample_index: int,
    turn_index: int,
    turns: Sequence[Turn],
    store: ResultStore,
    generator: Generator,
    run_id: str,
) -> tuple[Mapping[str, Any], bool]:
    """Produce one turn, reusing a stored one when present."""
    fingerprint = config.fingerprint(str(task["id"]), sample_index, STAGE_GENERATE, turn_index)
    stored = store.get(fingerprint.key)
    if stored is not None:
        return stored, True
    payload = generator.generate(
        task, sample_index, turn_index, _messages(task, turns), fingerprint
    )
    store.put(fingerprint.key, payload)
    store.append_event(
        run_id,
        {"event": "completed", "stage": STAGE_GENERATE, "key": fingerprint.key, "turn": turn_index},
    )
    return payload, False


def _prepare_workspace(
    root: str, task_id: str, sample_index: int, fixture: str | Path | None
) -> Path:
    """Return this sample's own workspace, seeded from the fixture once.

    Samples run concurrently, so they cannot share a directory: a model that writes
    a file would otherwise race another sample's ``go test`` against a repository
    neither of them was given. Each gets its own copy, created once and then reused
    on resume, where the stored execution results mean nothing is re-run anyway.

    Restoring by copy rather than by cleaning in place is deliberate: cleaning can
    leave a file a previous sample created, and two samples would quietly inherit
    each other's work.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task_id)
    target = Path(root) / f"{safe}-s{sample_index}"
    if target.is_dir():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if fixture is not None:
        shutil.copytree(fixture, target)
    else:
        target.mkdir(parents=True)
    return target


def _execute_turn(
    config: RunConfig,
    task: Mapping[str, Any],
    sample_index: int,
    turn_index: int,
    generated: Mapping[str, Any],
    store: ResultStore,
    executor: Executor,
    run_id: str,
    workspace: Path,
    max_parallel: int,
) -> tuple[tuple[ExecResult, ...], int]:
    """Run one turn's tool calls, in parallel, reusing stored results."""
    calls = _tool_calls(generated)
    if not calls:
        return (), 0

    requests = [
        ExecRequest(
            task_id=str(task["id"]),
            sample_index=sample_index,
            tool_name=str(call.get("name", "")),
            arguments=call.get("arguments") or {},
            workspace=workspace,
            call_index=call_index,
            turn_index=turn_index,
        )
        for call_index, call in enumerate(calls)
    ]

    fingerprint = config.fingerprint(str(task["id"]), sample_index, STAGE_EXECUTE, turn_index)
    results: list[ExecResult | None] = [None] * len(requests)
    pending: list[tuple[int, ExecRequest]] = []
    for index, request in enumerate(requests):
        stored = store.get(_call_key(fingerprint, request))
        if stored is None:
            pending.append((index, request))
        else:
            results[index] = _result_from_record(request, stored)

    reused = len(requests) - len(pending)
    if pending:
        fresh = run_many([r for _, r in pending], executor, max_parallel=max_parallel)
        for (index, request), result in zip(pending, fresh, strict=True):
            results[index] = result
            key = _call_key(fingerprint, request)
            store.put(key, result.to_record())
            store.append_event(
                run_id,
                {
                    "event": "completed",
                    "stage": STAGE_EXECUTE,
                    "key": key,
                    "turn": turn_index,
                },
            )
    return tuple(r for r in results if r is not None), reused


def _call_key(fingerprint: Fingerprint, request: ExecRequest) -> str:
    """Execution key for one tool call within one turn.

    The turn fingerprint is shared by every call in that turn, so the call
    position and tool name are folded in. Without this, two calls to the same
    tool in one turn would collide and the second would be refused as an
    idempotency violation.
    """
    return sha256_text(
        canonical_json(
            {
                **fingerprint.to_dict(),
                "call_index": request.call_index,
                "tool_name": request.tool_name,
            }
        )
    )


def _result_from_record(request: ExecRequest, record: Mapping[str, Any]) -> ExecResult:
    return ExecResult(
        task_id=request.task_id,
        sample_index=request.sample_index,
        tool_name=str(record.get("tool_name", request.tool_name)),
        status=Status(record.get("status", Status.OK.value)),
        exit_code=record.get("exit_code"),
        stdout=str(record.get("stdout", "")),
        duration_ms=int(record.get("duration_ms", 0)),
        harness_error=str(record.get("harness_error", "")),
        call_index=request.call_index,
    )


def _run_sample(
    config: RunConfig,
    task: Mapping[str, Any],
    sample_index: int,
    store: ResultStore,
    generator: Generator,
    executor: Executor,
    run_id: str,
    workspace: str,
    fixture: str | Path | None,
    max_parallel: int,
    on_sample: Callable[[str, int, str], None] | None,
) -> Trajectory:
    """Drive one sample's agent loop to completion.

    Stops when the model stops calling tools, or when the turn budget is
    exhausted. A sample that ends on the turn limit is marked ``truncated``: it
    is not an error, but it must not be silently scored as a finished trajectory.
    """
    turns: list[Turn] = []
    sample_workspace = _prepare_workspace(workspace, str(task["id"]), sample_index, fixture)
    for turn_index in range(config.max_turns):
        generated, _ = _generate_turn(
            config, task, sample_index, turn_index, turns, store, generator, run_id
        )
        if on_sample is not None:
            on_sample(str(task["id"]), sample_index, f"{STAGE_GENERATE}:{turn_index}")

        if not _tool_calls(generated):
            turns.append(Turn(index=turn_index, generated=generated))
            return Trajectory(str(task["id"]), sample_index, tuple(turns))

        results, _ = _execute_turn(
            config,
            task,
            sample_index,
            turn_index,
            generated,
            store,
            executor,
            run_id,
            sample_workspace,
            max_parallel,
        )
        if on_sample is not None:
            on_sample(str(task["id"]), sample_index, f"{STAGE_EXECUTE}:{turn_index}")
        turns.append(Turn(index=turn_index, generated=generated, results=results))

    return Trajectory(str(task["id"]), sample_index, tuple(turns), truncated=True)


def _judge_stage(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    trajectories: Mapping[tuple[str, int], Trajectory],
    store: ResultStore,
    judge: Judge,
    run_id: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Score each sample. Infrastructure failures are never judged."""
    scores: dict[str, float] = {}
    reused = {STAGE_JUDGE: 0}
    for task in tasks:
        task_id = str(task["id"])
        for sample_index in range(config.n_samples):
            trajectory = trajectories.get((task_id, sample_index))
            if trajectory is None or trajectory.has_harness_error:
                # A harness error means we learned nothing about the model;
                # scoring it would turn broken infrastructure into a regression.
                continue
            fingerprint = config.fingerprint(task_id, sample_index, STAGE_JUDGE)
            stored = store.get(fingerprint.key)
            if stored is not None:
                # A stored verdict is only reusable by the judge that wrote it.
                # Without this check a session grade would be handed back under a
                # served judge's name, and two different judgements would be reported
                # as one number.
                recorded_by = str(stored.get("judge_model", ""))
                expected_by = _judge_identity(judge)
                if recorded_by and expected_by and recorded_by != expected_by:
                    raise EvalStoreError(
                        f"{task_id}#{sample_index} was already judged by {recorded_by!r}, but this "
                        f"run uses {expected_by!r}. Reusing the old verdict would report one "
                        f"judge's opinion as another's. Use a fresh store, or re-run under the "
                        f"original judge."
                    )
                scores[f"{task_id}#{sample_index}"] = float(stored.get("score", 0.0))
                reused[STAGE_JUDGE] += 1
                continue
            record = {"task": dict(task), **trajectory.to_record()}
            verdict = judge.judge(record, fingerprint)
            if not isinstance(verdict, Mapping):
                raise EvalStoreError(
                    f"judge returned {type(verdict).__name__} for {task_id}#{sample_index}, "
                    "expected a mapping with a 'score'. A judge that declined to grade must "
                    "raise or say so explicitly; returning nothing would leave the sample "
                    "silently ungraded and shrink the denominator."
                )
            if "score" not in verdict:
                raise EvalStoreError(
                    f"judge verdict for {task_id}#{sample_index} has no 'score' key: "
                    f"{sorted(verdict)}"
                )
            score = verdict["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise EvalStoreError(
                    f"judge verdict for {task_id}#{sample_index} has score={score!r}, which is "
                    "not a number"
                )
            store.put(fingerprint.key, verdict)
            store.append_event(
                run_id, {"event": "completed", "stage": STAGE_JUDGE, "key": fingerprint.key}
            )
            scores[f"{task_id}#{sample_index}"] = float(score)
    return scores, reused


def run_evaluation(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    store: ResultStore,
    generator: Generator,
    executor: Executor,
    *,
    run_id: str | None = None,
    judge: Judge | None = None,
    workspace: str | Path | None = None,
    fixture: str | Path | None = None,
    max_parallel: int = 4,
    k: int = 1,
    on_sample: Callable[[str, int, str], None] | None = None,
) -> RunReport:
    """Run the agent loop over every task sample and return a deterministic report.

    ``workspace`` is a *root*: each sample gets its own directory beneath it, so
    concurrent samples cannot see each other's edits. ``fixture`` is the repository
    state each of those directories is seeded from, if any.

    ``workspace`` defaults to a directory under the system temp path. A default of
    ``"."`` would create per-sample directories in whatever directory the caller
    happened to be running from -- a source tree, in practice -- leaving litter
    behind and, when a generated name collides with a real one, a genuinely
    confusing thing to debug. Pass an explicit path to keep them somewhere known.

    Re-running is safe and cheap: every turn reuses whatever the store already
    holds, so a crashed run continues at the turn it reached. The generator and
    executor are called concurrently across samples, so both must be thread-safe.
    """
    root = str(workspace) if workspace else str(Path(tempfile.gettempdir()) / "gotooltrain-eval")
    if not tasks:
        raise EvalStoreError("no tasks to evaluate")
    for index, task in enumerate(tasks):
        if "id" not in task:
            raise EvalStoreError(f"tasks[{index}] has no 'id'")

    # Refuse to start if any tool is neither a command nor a workspace mutation:
    # it would report success without doing anything.
    assert_harness_covers_catalog()

    # The judge is part of the run's identity, not a detail of how it was scored.
    # Recording it means a resume under a different judge is refused rather than
    # silently blending verdicts from two different models. Only the judge
    # identity is recorded: the scoring mode is derived from it, so recording both
    # would only let them disagree.
    run = run_id or store.new_run_id("eval")
    manifest = _run_manifest(config, run, store, judge)
    if (store.run_dir(run) / MANIFEST).is_file():
        # _run_manifest has already refused every difference except the judge
        # upgrade, so writing here can only ever record that one field.
        store.rewrite_manifest(run, manifest)
    store.start_run(run, manifest)

    jobs = [(task, index) for task in tasks for index in range(config.n_samples)]
    trajectories: dict[tuple[str, int], Trajectory] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        futures = {
            pool.submit(
                _run_sample,
                config,
                task,
                sample_index,
                store,
                generator,
                executor,
                run,
                root,
                fixture,
                max_parallel,
                on_sample,
            ): (str(task["id"]), sample_index)
            for task, sample_index in jobs
        }
        for future, key in futures.items():
            trajectories[key] = future.result()

    scores: dict[str, float] = {}
    reused_judge: dict[str, int] = {STAGE_JUDGE: 0}
    if judge is not None:
        scores, reused_judge = _judge_stage(config, tasks, trajectories, store, judge, run)

    summaries = _summarise(
        config, tasks, [trajectories[key] for key in sorted(trajectories)], scores, k
    )
    judged_task_ids = {key.split("#", 1)[0] for key in scores}
    return RunReport(
        run_id=run,
        summaries=summaries,
        judged=scores,
        stage_counts=store.stats(run),
        reused=reused_judge,
        truncated=sum(1 for t in trajectories.values() if t.truncated),
        scoring=_scoring_mode(judge),
        judge_identity=_judge_identity(judge),
        unjudged_tasks=(
            sum(1 for t in tasks if str(t["id"]) not in judged_task_ids) if judge is not None else 0
        ),
    )


def canary_cases(workspace: Path) -> tuple[CanaryCase, ...]:
    """Probes that must pass and must fail, run before any fan-out.

    Two jobs, and the second matters more than the first. A missing binary -- RTK,
    most likely -- otherwise surfaces as a ``HARNESS_ERROR`` on *every* sample,
    which reads like a broken harness and costs a full run to discover. Probing
    first turns that into an immediate refusal naming the missing command. The
    failing probe is the one that stops a harness which cannot tell success from
    failure from reporting 100% on everything.
    """
    return (
        CanaryCase(
            "toolchain present",
            ExecRequest(
                task_id="canary",
                sample_index=0,
                tool_name="go_doc",
                arguments={"symbol": "errors.Is"},
                workspace=workspace,
                call_index=0,
            ),
            should_fail=False,
        ),
        CanaryCase(
            "failures are visible",
            ExecRequest(
                task_id="canary",
                sample_index=0,
                tool_name="go_test",
                arguments={"pkg": "./definitely-not-a-package"},
                workspace=workspace,
                call_index=0,
            ),
            should_fail=True,
        ),
    )


def _run_manifest(config: RunConfig, run: str, store: ResultStore, judge: object) -> dict[str, Any]:
    """Build this run's manifest, refusing any change to a run already started.

    A run's identity is write-once, with exactly one sanctioned progression: a run
    that started with no judge may record the judge that later graded it. Every
    other difference is refused, because a different seed, revision or catalogue
    under the same run id would blend two configurations into one number.

    Which fields differ is computed here rather than left to the store to notice,
    so the rule and its single exception live in one place.
    """
    manifest = {**config.manifest(), "judge": _judge_identity(judge)}
    directory = store.run_dir(run)
    if not (directory / MANIFEST).is_file():
        return manifest

    stored = store.load_manifest(run)
    differing = {
        field
        for field in set(stored) | set(manifest)
        if field != "store_version" and stored.get(field) != manifest.get(field)
    }
    if not differing:
        return manifest
    if differing == {"judge"} and not stored.get("judge"):
        return manifest
    raise EvalStoreError(
        f"run {run} already exists with a different manifest: {sorted(differing)} differ. "
        "Use a new run id for a different configuration or a different judge, or delete "
        "the run directory deliberately."
    )


def _scoring_mode(judge: object) -> str:
    """Which measurement produced the scores.

    Three distinct things are being reported, and collapsing any two of them would
    let a reader compare numbers that do not measure the same quantity: exit codes
    only, a served judge, or a session judge that will not reproduce.
    """
    if judge is None:
        return SCORING_EXECUTION_ONLY
    if isinstance(judge, SessionJudge):
        return SCORING_SESSION_JUDGE
    return SCORING_JUDGE


def rebuild_trajectory(
    config: RunConfig,
    task: Mapping[str, Any],
    sample_index: int,
    store: ResultStore,
) -> Trajectory | None:
    """A finished trajectory from stored artifacts, or ``None`` if it never finished.

    Public because training-data mining needs the same reconstruction the judge
    does: a trajectory that was never fully executed must not be mined any more
    than it may be graded.
    """
    return _trajectory_from_store(config, task, sample_index, store)


def trajectory_messages(task: Mapping[str, Any], trajectory: Trajectory) -> list[dict[str, Any]]:
    """The conversation as the model saw it, ready to become a training record.

    The same function the generator was driven with, so a record can never be
    built in an order the model was not actually served.
    """
    return _messages(task, trajectory.turns)


def _trajectory_from_store(
    config: RunConfig,
    task: Mapping[str, Any],
    sample_index: int,
    store: ResultStore,
) -> Trajectory | None:
    """Rebuild a finished trajectory from stored artifacts, or None if absent.

    Generation and execution are both persisted, so a trajectory can be reassembled
    without re-running anything. That is what makes an offline judge possible: the
    queue is a pure function of the store, so a crashed or resumed run still grades
    exactly the trajectories that were executed.
    """
    task_id = str(task["id"])
    turns: list[Turn] = []
    for turn_index in range(config.max_turns):
        fingerprint = config.fingerprint(task_id, sample_index, STAGE_GENERATE, turn_index)
        generated = store.get(fingerprint.key)
        if generated is None:
            # No generation here means either the sample never started, or it called
            # a tool and its follow-up turn never ran because the run died. Both mean
            # there is no finished trajectory: queueing one would have the judge
            # score a run that never completed, and an empty report entry would
            # silently shrink the denominator.
            return None
        calls = _tool_calls(generated)
        if not calls:
            turns.append(Turn(index=turn_index, generated=generated))
            return Trajectory(task_id, sample_index, tuple(turns))

        execute_fingerprint = config.fingerprint(task_id, sample_index, STAGE_EXECUTE, turn_index)
        results: list[ExecResult] = []
        for call_index, call in enumerate(calls):
            request = ExecRequest(
                task_id=task_id,
                sample_index=sample_index,
                tool_name=str(call.get("name", "")),
                arguments=call.get("arguments") or {},
                workspace=Path(),
                call_index=call_index,
                turn_index=turn_index,
            )
            stored = store.get(_call_key(execute_fingerprint, request))
            if stored is None:
                # A half-finished turn: the generation is there but the execution is
                # not, so there is no trajectory to grade. Reporting a partial one
                # would have the judge score a run that never completed.
                return None
            results.append(_result_from_record(request, stored))
        turns.append(Turn(index=turn_index, generated=generated, results=tuple(results)))
    return Trajectory(task_id, sample_index, tuple(turns), truncated=True)


def judge_queue(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    store: ResultStore,
) -> list[QueueEntry]:
    """Trajectories that are finished and not yet judged, in a stable order.

    Samples lost to a harness error are excluded: a judge cannot grade a run that
    the infrastructure took away, and asking it to would invite a verdict about
    our failure rather than the model's work.
    """
    entries: list[QueueEntry] = []
    for task in tasks:
        task_id = str(task["id"])
        for sample_index in range(config.n_samples):
            fingerprint = config.fingerprint(task_id, sample_index, STAGE_JUDGE)
            if store.get(fingerprint.key) is not None:
                continue
            trajectory = _trajectory_from_store(config, task, sample_index, store)
            if trajectory is None or trajectory.has_harness_error:
                continue
            entries.append(
                QueueEntry(
                    key=fingerprint.key,
                    task_id=task_id,
                    sample_index=sample_index,
                    trajectory=render_trajectory({"task": dict(task), **trajectory.to_record()}),
                    prompt=str(task.get("prompt", "")),
                )
            )
    return entries


def ingest_judge_verdicts(
    config: RunConfig,
    run_id: str,
    store: ResultStore,
    verdicts: Mapping[str, Mapping[str, Any]],
    *,
    judge_model: str,
) -> int:
    """Write session verdicts into the store under their queue keys; return the count.

    Each verdict is stored with ``allow_identical_rewrite=False`` so a re-ingest
    with different content is refused rather than overwriting a grade already given.
    """
    for key, verdict in sorted(verdicts.items()):
        payload = {
            "score": float(verdict["score"]),
            "reason": str(verdict.get("reason", "")),
            "judge_model": judge_model,
        }
        store.put(key, payload, allow_identical_rewrite=False)
        store.append_event(run_id, {"event": "completed", "stage": STAGE_JUDGE, "key": key})
    return len(verdicts)


def _judge_identity(judge: object) -> str:
    """A stable name for the judge, recorded so runs stay attributable.

    Two runs scored by different judges produce numbers that look comparable and
    are not. Recording the identity in the manifest means a resume under a swapped
    judge is refused, and the report says which judge produced the score.
    """
    if judge is None:
        return ""
    identity = getattr(judge, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    return f"{type(judge).__module__}.{type(judge).__qualname__}"


def _summarise(
    config: RunConfig,
    tasks: Sequence[Mapping[str, Any]],
    trajectories: Sequence[Trajectory],
    scores: Mapping[str, float],
    k: int,
) -> list[TaskSummary]:
    """Aggregate per task, driven by the task list rather than the results.

    Every task appears in the report even when the model called no tools. A task
    with no executions is scored as zero rather than dropped: omitting it would
    silently inflate every average, and "the model did nothing" is a failure, not
    a non-event.

    A task whose samples were all lost to harness errors has no verdict. It is
    reported as zero and surfaced in ``RunReport.unjudged_tasks``, because a
    passing number derived from the tasks that happened to work would overstate
    the model.

    With a judge, the judged score decides pass/fail, because "every command
    succeeded" is necessary but not sufficient. Without one, execution status is
    the only signal available.
    """
    summaries: list[TaskSummary] = []
    for task in tasks:
        task_id = str(task["id"])
        samples = [t for t in trajectories if t.task_id == task_id]
        results = [r for t in samples for r in t.all_results]

        # Samples keep their identity in both modes: a task always reports
        # n_samples, so a run where a third of the samples died to a harness error
        # cannot present itself as a smaller, cleaner run. An unjudged sample counts
        # as a non-pass, never as an absent one.
        values: list[float] = []
        for index in range(config.n_samples):
            key = f"{task_id}#{index}"
            if key in scores:
                values.append(scores[key])
            elif scores:
                # Judge mode, no verdict for this sample: either a harness error took
                # the run away, or the judge itself failed on this one. Either way
                # there is no evidence the model did the work, so it counts as a
                # non-pass. Scoring it from execution status would credit the model
                # for a sample that was never actually assessed.
                values.append(0.0)
            else:
                execution_only = next((t for t in samples if t.sample_index == index), None)
                values.append(1.0 if execution_only is not None and execution_only.all_ok else 0.0)
        passed = sum(1 for v in values if v >= 1.0)
        errors = {
            "harness_errors": sum(1 for r in results if r.status is Status.HARNESS_ERROR),
            "model_errors": sum(1 for r in results if r.status is Status.MODEL_ERROR),
            "tool_errors": sum(1 for r in results if r.status is Status.TOOL_ERROR),
        }

        total = len(values) or 1
        summaries.append(
            TaskSummary(
                task_id=task_id,
                samples=total,
                passed=passed,
                harness_errors=errors["harness_errors"],
                model_errors=errors["model_errors"],
                tool_errors=errors["tool_errors"],
                pass_at_1=passed / total,
                pass_at_n=pass_at_k(total, passed, k),
            )
        )
    return summaries
