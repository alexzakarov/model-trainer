"""The offline judge path: queue, session verdicts, and honest labelling.

The flow being defended: the eval writes finished trajectories to a queue, they are
graded elsewhere, the verdicts come back into the store, and a resumed run reports
the scores -- while admitting in the report that the grading is not reproducible.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain import (
    EvalStoreError,
    HarnessError,
    IdempotencyViolationError,
    ResultStore,
)
from gotooltrain.evalrun import (
    SCORING_EXECUTION_ONLY,
    SCORING_SESSION_JUDGE,
    STAGE_JUDGE,
    RunConfig,
    ingest_judge_verdicts,
    judge_queue,
    run_evaluation,
)
from gotooltrain.judge import (
    SESSION_JUDGE_MODEL,
    VERDICTS_VERSION,
    SessionJudge,
    read_judge_queue,
    read_verdicts,
    write_judge_queue,
    write_verdicts,
)

TASKS: list[dict[str, Any]] = [
    {"id": "go-0001", "prompt": "add a test for the parser"},
    {"id": "go-0002", "prompt": "fix the nil map panic"},
]


def sample(calls: list[dict[str, Any]] | None = None, text: str = "") -> dict[str, Any]:
    return {"tool_calls": calls or [], "text": text}


def call(name: str, pkg: str = "./parser") -> dict[str, Any]:
    return {"name": name, "arguments": {"pkg": pkg}}


class ScriptedGenerator:
    """Replays a per-task script; an exhausted script ends the sample."""

    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        """Adopt a per-task script."""
        self.script = script or {
            "go-0001": [sample([call("go_test")]), sample(text="fixed")],
            "go-0002": [sample([call("go_test")]), sample(text="fixed")],
        }
        self.calls = 0

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Return the scripted turn."""
        self.calls += 1
        turns = self.script[str(task["id"])]
        return turns[turn_index] if turn_index < len(turns) else sample(text="done")


class CountingExecutor:
    """Executor stub returning a passing result."""

    def __init__(self, exit_code: int = 0) -> None:
        """Record the canned exit code."""
        self.exit_code = exit_code
        self.calls = 0

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Pretend to execute."""
        self.calls += 1
        return self.exit_code, "ok  parser  0.4s", ""


class BrokenExecutor:
    """Executor stub that loses the container."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Raise a harness error."""
        raise HarnessError("container is not running")


def config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-4B-go",
        "model_revision": "rev-a",
        "dataset_version": "holdout-1",
        "seed": 7,
        "decode_params": {"temperature": 0.0},
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> ResultStore:
    return ResultStore(tmp_path / "store")


def executed(store: ResultStore, run: RunConfig | None = None, **kwargs: Any) -> ResultStore:
    """Run the agent loop with no judge, leaving trajectories in the store."""
    run = run or config(**kwargs)
    run_evaluation(run, TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="run-1")
    return store


# ----------------------------------------------------------------- the queue


def test_the_queue_holds_one_entry_per_finished_sample(store: ResultStore) -> None:
    executed(store)
    entries = judge_queue(config(), TASKS, store)
    assert [e.label for e in entries] == ["go-0001#0", "go-0002#0"]


def test_the_queue_carries_the_evidence_not_just_the_prompt(store: ResultStore) -> None:
    """A judge that sees only the prompt would be guessing."""
    executed(store)
    entry = judge_queue(config(), TASKS, store)[0]
    assert "add a test for the parser" in entry.prompt
    assert "ok  parser  0.4s" in entry.trajectory
    assert "go_test" in entry.trajectory


def test_the_queue_key_is_the_store_key(store: ResultStore) -> None:
    """So a verdict is filed against the trajectory it was written for."""
    run = config()
    executed(store, run)
    entry = judge_queue(run, TASKS, store)[0]
    assert entry.key == run.fingerprint("go-0001", 0, STAGE_JUDGE).key


def test_a_harness_lost_sample_is_not_queued(store: ResultStore) -> None:
    """Judging a run our infrastructure destroyed grades the wrong thing."""
    run = config()
    run_evaluation(run, TASKS, store, ScriptedGenerator(), BrokenExecutor(), run_id="run-1")
    assert judge_queue(run, TASKS, store) == []


def test_an_already_judged_sample_is_not_queued_again(store: ResultStore) -> None:
    """Resume must not re-grade work that already has a verdict."""
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    write_verdicts(str(store.root.parent / "v.jsonl"), {e.key: {"score": 1.0} for e in entries})
    ingest_judge_verdicts(
        run,
        "run-1",
        store,
        {e.key: {"score": 1.0} for e in entries},
        judge_model=SESSION_IDENTITY,
    )
    assert judge_queue(run, TASKS, store) == []


def test_the_queue_round_trips_through_a_file(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = str(tmp_path / "queue.jsonl")
    assert write_judge_queue(path, entries) == len(entries)
    assert read_judge_queue(str(path)) == entries


def judge_queue_of(tmp_path: pathlib.Path) -> list[Any]:
    """Build a small queue for the file round-trip tests."""
    store = ResultStore(tmp_path / "store")
    executed(store)
    return judge_queue(config(), TASKS, store)


def test_a_queue_from_another_build_is_refused(tmp_path: pathlib.Path) -> None:
    """A stale prompt would mean grading against a format that no longer exists."""
    path = tmp_path / "queue.jsonl"
    path.write_text(json.dumps({"version": "judge-queue-v0", "key": "k"}) + "\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="queue version"):
        read_judge_queue(str(path))


def test_a_missing_queue_file_is_reported(tmp_path: pathlib.Path) -> None:
    with pytest.raises(HarnessError, match="not found"):
        read_judge_queue(str(tmp_path / "nope.jsonl"))


def test_a_corrupt_queue_line_is_reported(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="not valid JSON"):
        read_judge_queue(str(path))


# ------------------------------------------------------------- session judge


#: The identity a SessionJudge reports; verdicts must be filed under exactly this.
SESSION_IDENTITY = SessionJudge({}).identity


def grade_all(entries: list[Any], score: float = 1.0) -> dict[str, dict[str, Any]]:
    """A verdict for every queue entry."""
    return {e.key: {"score": score, "reason": "graded in session"} for e in entries}


def test_session_verdicts_come_back_as_a_judged_run(
    store: ResultStore, tmp_path: pathlib.Path
) -> None:
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    verdicts = grade_all(entries)
    ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY)

    generator = ScriptedGenerator()
    executor = CountingExecutor()
    report = run_evaluation(
        run,
        TASKS,
        store,
        generator,
        executor,
        run_id="run-1-judged",
        judge=SessionJudge(verdicts),
    )
    assert generator.calls == 0, "the trajectories are reused, not re-run"
    assert executor.calls == 0
    assert report.judged == {"go-0001#0": 1.0, "go-0002#0": 1.0}
    assert report.mean_pass_at_1 == 1.0


def test_a_session_judged_run_says_it_is_not_reproducible(
    store: ResultStore, tmp_path: pathlib.Path
) -> None:
    """The whole trade-off, stated in the report instead of remembered."""
    run = config()
    executed(store, run)
    verdicts = grade_all(judge_queue(run, TASKS, store))
    report = run_evaluation(
        run,
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1-judged",
        judge=SessionJudge(verdicts),
    )
    assert report.scoring == SCORING_SESSION_JUDGE
    assert report.scored_by_judge
    assert report.reproducible is False
    assert report.to_record()["reproducible"] is False
    assert SESSION_JUDGE_MODEL in report.judge_identity


def test_an_execution_only_run_is_reproducible(store: ResultStore) -> None:
    """Only the judge is the unreproducible part.

    Without one, the score is a function of execution results the store memoises
    byte-exactly, so re-deriving it gives the same number.
    """
    report = run_evaluation(
        config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="run-1"
    )
    assert report.scoring == SCORING_EXECUTION_ONLY
    assert report.reproducible is True


def test_a_missing_session_verdict_is_refused(store: ResultStore) -> None:
    """A silently ungraded sample would shrink the denominator."""
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    partial = grade_all([entries[0]])
    with pytest.raises(HarnessError, match="no session verdict"):
        run_evaluation(
            run,
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-1-judged",
            judge=SessionJudge(partial),
        )


def test_the_session_judge_names_itself(store: ResultStore) -> None:
    judge = SessionJudge({})
    assert judge.identity.endswith("@session")
    assert SESSION_JUDGE_MODEL in judge.identity


# ------------------------------------------------------------------ verdicts


def test_verdicts_are_checked_against_the_queue(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = str(tmp_path / "v.jsonl")
    write_verdicts(path, grade_all(entries))
    assert read_verdicts(path, entries) == grade_all(entries)


def test_a_verdict_for_an_unknown_key_is_refused(tmp_path: pathlib.Path) -> None:
    """Queue and verdicts from different runs would grade the wrong trajectories."""
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "v.jsonl"
    path.write_text(
        json.dumps({"version": VERDICTS_VERSION, "key": "f" * 64, "score": 1.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(HarnessError, match="not in the queue"):
        read_verdicts(str(path), entries)


def test_a_duplicate_verdict_is_refused(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "v.jsonl"
    write_verdicts(str(tmp_path / "v.jsonl"), grade_all([entries[0]]))
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(HarnessError, match="twice"):
        read_verdicts(str(path), [entries[0]])


def test_a_missing_verdict_is_refused(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = str(tmp_path / "v.jsonl")
    write_verdicts(path, grade_all([entries[0]]))
    with pytest.raises(HarnessError, match="no verdict"):
        read_verdicts(path, entries)


def test_a_verdict_with_the_wrong_version_is_refused(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "v.jsonl"
    path.write_text(
        json.dumps({"version": "v0", "key": entries[0].key, "score": 1.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(HarnessError, match="declares version"):
        read_verdicts(str(path), entries)


def test_a_missing_verdict_file_is_reported(tmp_path: pathlib.Path) -> None:
    with pytest.raises(HarnessError, match="not found"):
        read_verdicts(str(tmp_path / "nope.jsonl"), [])


# ------------------------------------------------------------------- ingest


def test_ingest_refuses_to_overwrite_a_grade(store: ResultStore) -> None:
    """Re-grading with a different answer must be a deliberate act."""
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    ingest_judge_verdicts(run, "run-1", store, grade_all(entries), judge_model=SESSION_IDENTITY)
    with pytest.raises(IdempotencyViolationError, match="already holds different content"):
        ingest_judge_verdicts(
            run, "run-1", store, grade_all(entries, score=0.0), judge_model=SESSION_IDENTITY
        )


def test_re_ingesting_the_same_verdict_is_refused_too(store: ResultStore) -> None:
    """Even an identical rewrite is refused, so a re-ingest is always deliberate."""
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    verdicts = grade_all(entries)
    ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY)
    with pytest.raises(IdempotencyViolationError, match="identical content"):
        ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY)


def test_ingest_stores_the_verdict_verbatim(store: ResultStore) -> None:
    run = config()
    executed(store, run)
    entries = judge_queue(run, TASKS, store)
    verdicts = grade_all(entries)
    assert ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY) == 2
    assert store.get(entries[0].key)["score"] == 1.0


def test_ingest_records_the_stage_in_the_event_log(store: ResultStore) -> None:
    run = config()
    executed(store, run)
    verdicts = grade_all(judge_queue(run, TASKS, store))
    ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY)
    assert store.completed_keys("run-1", stage=STAGE_JUDGE)


def test_a_served_and_a_session_judge_are_not_interchangeable(store: ResultStore) -> None:
    """Two different judgements must not share a number."""
    run = config()
    executed(store, run)
    verdicts = grade_all(judge_queue(run, TASKS, store))
    ingest_judge_verdicts(run, "run-1", store, verdicts, judge_model=SESSION_IDENTITY)

    class Served:
        model = "some-hosted-model"
        base_url = "https://example.test/v1"

        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Contradict the session grade."""
            return {"score": 0.0}

    with pytest.raises(EvalStoreError):
        run_evaluation(
            run,
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-2",
            judge=Served(),
        )


def test_a_queue_with_blank_lines_is_read(tmp_path: pathlib.Path) -> None:
    """A hand-edited file with a trailing blank line is not corrupt."""
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "q.jsonl"
    write_judge_queue(str(path), entries)
    path.open("a", encoding="utf-8").write("\n\n")
    assert read_judge_queue(path) == entries


def test_a_corrupt_verdict_line_is_reported(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "v.jsonl"
    path.write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="not valid JSON"):
        read_verdicts(str(path), entries)


def test_a_verdict_file_with_blank_lines_is_read(tmp_path: pathlib.Path) -> None:
    entries = judge_queue_of(tmp_path)
    path = tmp_path / "v.jsonl"
    write_verdicts(str(path), grade_all(entries))
    path.open("a", encoding="utf-8").write("\n")
    assert read_verdicts(str(path), entries) == grade_all(entries)


class CrashingGenerator(ScriptedGenerator):
    """Calls a tool on turn 0, then dies before the follow-up turn exists."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Lose the model server on the second turn."""
        if turn_index >= 1:
            raise HarnessError("model server died")
        return sample([call("go_test")])


def test_a_crashed_sample_is_not_queued(store: ResultStore) -> None:
    """Turn 0 called a tool, so the sample was not finished.

    Queueing it would have the judge score a run that never completed; dropping it
    silently would shrink the denominator. It is left unjudged and counted.
    """
    run = config()
    with pytest.raises(HarnessError, match="model server died"):
        run_evaluation(run, TASKS, store, CrashingGenerator(), CountingExecutor(), run_id="run-1")
    assert judge_queue(run, TASKS, store) == []


def test_a_sample_that_never_started_is_not_queued(store: ResultStore) -> None:
    run = config()
    executed(store, run)
    empty_store = ResultStore(store.root.parent / "empty-store")
    assert judge_queue(run, TASKS, empty_store) == []


def test_a_looping_sample_is_queued_as_truncated(store: ResultStore) -> None:
    """The turn budget ran out mid-trajectory; the judge still needs to see it."""
    run = config(max_turns=2)
    looping = {
        "go-0001": [sample([call("go_test")])] * 4,
        "go-0002": [sample([call("go_test")])] * 4,
    }
    run_evaluation(
        run, TASKS, store, ScriptedGenerator(dict(looping)), CountingExecutor(), run_id="run-1"
    )
    entries = judge_queue(run, TASKS, store)
    assert len(entries) == 2
    assert "incomplete" in entries[0].trajectory


def test_a_turn_with_an_unexecuted_call_is_not_queued(tmp_path: pathlib.Path) -> None:
    """Generation landed but its execution did not: nothing finished to grade."""
    run = config()
    store = ResultStore(tmp_path / "half")
    store.put(run.fingerprint("go-0001", 0, "generate", 0).key, sample([call("go_test")]))
    assert judge_queue(run, TASKS, store) == []
