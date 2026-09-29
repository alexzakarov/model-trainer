"""Agent-loop orchestration: multi-turn feedback, resume, and honest reporting.

The properties defended here:

* a tool result is fed back to the model in the shape it will be served
* turns are causally ordered; samples are independent and run concurrently
* a crash resumes at the turn it reached, not at the start
* a sample that did nothing is not silently omitted from the score
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain import EvalStoreError, HarnessError, ResultStore
from gotooltrain.evalrun import (
    STAGE_EXECUTE,
    STAGE_GENERATE,
    RunConfig,
    run_evaluation,
)
from gotooltrain.gorun import Status

TASKS: list[dict[str, Any]] = [
    {"id": "go-0001", "prompt": "add a test for the parser"},
    {"id": "go-0002", "prompt": "fix the nil map panic"},
]


def sample(calls: list[dict[str, Any]] | None = None, text: str = "") -> dict[str, Any]:
    return {"tool_calls": calls or [], "text": text}


def call(name: str, pkg: str = "./parser", call_id: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "arguments": {"pkg": pkg}}
    if call_id is not None:
        entry["id"] = call_id
    return entry


class ScriptedGenerator:
    """Replays a per-task script of turns and records the conversations it saw."""

    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        """Adopt a per-task script; an exhausted script ends the sample."""
        self.script = script or {
            "go-0001": [sample([call("go_test")])],
            "go-0002": [sample([call("go_test")])],
        }
        #: Keyed by (task, sample, turn) so tests never depend on completion order.
        self.conversations: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        self.calls = 0

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Record the conversation and return the scripted turn."""
        self.calls += 1
        self.conversations[(str(task["id"]), sample_index, turn_index)] = [
            dict(m) for m in messages
        ]
        turns = self.script[str(task["id"])]
        return turns[turn_index] if turn_index < len(turns) else sample(text="done")

    def conversation(self, task_id: str, turn_index: int) -> list[dict[str, Any]]:
        """The conversation the model was shown before that turn."""
        return self.conversations[(task_id, 0, turn_index)]


class AlwaysJudge:
    """Judge stub that returns a fixed score."""

    def __init__(self, score: float = 1.0) -> None:
        """Fix the score to hand out."""
        self.score = score
        self.calls = 0
        self.records: list[dict[str, Any]] = []

    def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
        """Score without inspecting the trajectory."""
        self.calls += 1
        self.records.append(dict(record))
        return {"score": self.score, "reason": "scripted"}


class CountingExecutor:
    """Executor stub that returns a canned exit code and output."""

    def __init__(self, exit_code: int = 0, stdout: str = "ok  parser  0.4s") -> None:
        """Record the canned result and the call count."""
        self.exit_code = exit_code
        self.stdout = stdout
        self.calls = 0

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Pretend to execute, counting the calls."""
        self.calls += 1
        return self.exit_code, self.stdout, ""


class BrokenExecutor:
    """Executor stub that fails the way a dead container would."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Raise a harness error."""
        raise HarnessError("container is not running")


def config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-4B-go",
        "model_revision": "rev-a",
        "dataset_version": "go-ut-bench-holdout-1",
        "seed": 7,
        "decode_params": {"temperature": 0.0, "max_tokens": 2048},
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> ResultStore:
    return ResultStore(tmp_path / "store")


# --------------------------------------------------------------------- config


def test_config_rejects_a_zero_sample_count() -> None:
    with pytest.raises(EvalStoreError, match="n_samples must be"):
        config(n_samples=0)


def test_config_rejects_a_zero_turn_budget() -> None:
    with pytest.raises(EvalStoreError, match="max_turns must be"):
        config(max_turns=0)


def test_config_requires_model_identity() -> None:
    with pytest.raises(EvalStoreError, match="model_id and model_revision"):
        config(model_revision="")


def test_manifest_captures_every_identity_field() -> None:
    manifest = config().manifest()
    for field in (
        "model_id",
        "model_revision",
        "dataset_version",
        "harness_version",
        "n_samples",
        "max_turns",
        "seed",
        "decode_params",
        "template_sha",
        "tool_catalog_sha",
        "format_version",
    ):
        assert field in manifest


def test_turns_get_distinct_keys() -> None:
    """Turn 0 and turn 1 of one sample must not collide."""
    run = config()
    assert (
        run.fingerprint("t", 0, STAGE_GENERATE, 0).key
        != run.fingerprint("t", 0, STAGE_GENERATE, 1).key
    )


def test_turn_is_part_of_the_stage_name() -> None:
    run = config()
    assert run.fingerprint("t", 0, STAGE_GENERATE, 3).stage == "generate:t3"


# ----------------------------------------------------------------- agent loop


def test_generation_stops_when_the_model_calls_nothing(store: ResultStore) -> None:
    generator = ScriptedGenerator({"go-0001": [sample()], "go-0002": [sample()]})
    report = run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    assert generator.calls == 2, "one turn each, then both models finished"
    assert [s.task_id for s in report.summaries] == ["go-0001", "go-0002"]


def test_tool_results_are_fed_back_to_the_model(store: ResultStore) -> None:
    """The core of the agent loop: turn 2 must see turn 1's output."""
    generator = ScriptedGenerator(
        {
            "go-0001": [sample([call("go_test", call_id="c1")]), sample(text="fixed it")],
            "go-0002": [sample(text="no tools needed")],
        }
    )
    run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    second_turn = generator.conversation("go-0001", 1)
    roles = [m["role"] for m in second_turn]
    assert roles == ["user", "assistant", "tool"], roles
    assert second_turn[2]["content"] == "ok  parser  0.4s"
    assert second_turn[2]["tool_call_id"] == "c1"


def test_the_task_prompt_is_the_first_message(store: ResultStore) -> None:
    generator = ScriptedGenerator()
    run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    assert generator.conversation("go-0001", 0)[0] == {
        "role": "user",
        "content": "add a test for the parser",
    }


def test_a_failing_test_is_visible_to_the_model_on_the_next_turn(
    store: ResultStore,
) -> None:
    """A red suite is the model's own feedback loop; hiding it teaches nothing."""
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")]), sample(text="fixing")], "go-0002": [sample()]}
    )
    run_evaluation(
        config(),
        TASKS,
        store,
        generator,
        CountingExecutor(exit_code=1, stdout="FAIL: undefined: foo"),
        run_id="run-1",
    )
    assert "FAIL: undefined: foo" in generator.conversation("go-0001", 1)[2]["content"]


def test_a_harness_failure_is_labelled_not_shown_as_empty(store: ResultStore) -> None:
    """Empty output would teach the model that a dead container means 'no output'."""
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")]), sample(text="retry")], "go-0002": [sample()]}
    )
    run_evaluation(config(), TASKS, store, generator, BrokenExecutor(), run_id="run-1")
    content = generator.conversation("go-0001", 1)[2]["content"]
    assert "infrastructure error" in content
    assert "container is not running" in content


def test_turn_budget_stops_a_looping_agent(store: ResultStore) -> None:
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")])] * 10, "go-0002": [sample()]}
    )
    report = run_evaluation(
        config(max_turns=3), TASKS, store, generator, CountingExecutor(), run_id="run-1"
    )
    assert report.truncated == 1
    assert generator.calls == 3 + 1


def test_parallel_tool_calls_all_execute(store: ResultStore) -> None:
    generator = ScriptedGenerator(
        {
            "go-0001": [sample([call("go_build", "./..."), call("go_test", "./parser")], "done")],
            "go-0002": [sample()],
        }
    )
    executor = CountingExecutor()
    run_evaluation(config(), TASKS, store, generator, executor, run_id="run-1")
    assert executor.calls == 2


def test_tool_results_stay_matched_to_their_calls(store: ResultStore) -> None:
    """Ordering matters: a result shown against the wrong call teaches the wrong thing."""
    generator = ScriptedGenerator(
        {
            "go-0001": [
                sample(
                    [
                        call("go_build", "./...", call_id="b"),
                        call("go_test", "./parser", call_id="t"),
                    ],
                    "done",
                )
            ],
            "go-0002": [sample()],
        }
    )
    run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    conversation = generator.conversation("go-0001", 1)
    assert [m["name"] for m in conversation if m["role"] == "tool"] == ["go_build", "go_test"]
    assert [m["tool_call_id"] for m in conversation if m["role"] == "tool"] == ["b", "t"]


def test_the_same_tool_on_two_turns_is_two_attempts(store: ResultStore) -> None:
    """go_test before and after a fix are different runs of the world."""
    generator = ScriptedGenerator(
        {
            "go-0001": [sample([call("go_test")]), sample([call("go_test")]), sample(text="done")],
            "go-0002": [sample()],
        }
    )
    executor = CountingExecutor()
    run_evaluation(config(), TASKS, store, generator, executor, run_id="run-1")
    assert executor.calls == 2, "both go_test calls ran"


# --------------------------------------------------------------------- judging


def test_judge_sees_the_whole_trajectory(store: ResultStore) -> None:
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")]), sample(text="done")], "go-0002": [sample()]}
    )
    judge = AlwaysJudge()
    run_evaluation(
        config(), TASKS, store, generator, CountingExecutor(), run_id="run-1", judge=judge
    )
    go1 = next(r for r in judge.records if r["task_id"] == "go-0001")
    assert [t["turn"] for t in go1["turns"]] == [0, 1]
    assert go1["turns"][0]["executions"][0]["status"] == Status.OK.value


def test_judge_decides_the_score(store: ResultStore) -> None:
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(score=1.0),
    )
    assert report.judged == {"go-0001#0": 1.0, "go-0002#0": 1.0}
    assert report.mean_pass_at_1 == 1.0


def test_judge_can_fail_a_task(store: ResultStore) -> None:
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(score=0.0),
    )
    assert report.mean_pass_at_1 == 0.0


def test_harness_errors_are_never_judged(store: ResultStore) -> None:
    """Scoring an infrastructure failure would look like a model regression."""
    judge = AlwaysJudge()
    report = run_evaluation(
        config(), TASKS, store, ScriptedGenerator(), BrokenExecutor(), run_id="run-1", judge=judge
    )
    assert report.judged == {}


def test_a_trajectory_free_of_harness_errors_is_judged(store: ResultStore) -> None:
    generator = ScriptedGenerator({"go-0001": [sample()], "go-0002": [sample()]})
    judge = AlwaysJudge()
    run_evaluation(
        config(), TASKS, store, generator, CountingExecutor(), run_id="run-1", judge=judge
    )
    assert judge.calls == 2


# --------------------------------------------------------------------- resume


def test_rerun_reuses_every_turn(store: ResultStore) -> None:
    first = ScriptedGenerator()
    first_executor = CountingExecutor()
    first_judge = AlwaysJudge()
    run_evaluation(config(), TASKS, store, first, first_executor, run_id="run-1", judge=first_judge)
    # Each task calls a tool on turn 0, so it needs a turn 1 to finish.
    assert (first.calls, first_executor.calls, first_judge.calls) == (4, 2, 2)

    second = ScriptedGenerator()
    second_executor = CountingExecutor()
    second_judge = AlwaysJudge()
    report = run_evaluation(
        config(), TASKS, store, second, second_executor, run_id="run-1", judge=second_judge
    )
    assert second.calls == 0, "a resumed run must not re-generate"
    assert second_executor.calls == 0, "a resumed run must not re-execute"
    assert second_judge.calls == 0, "a resumed run must not re-judge"
    assert report.mean_pass_at_1 == 1.0


def test_resume_continues_at_the_turn_it_reached(store: ResultStore) -> None:
    """Pre-seed turn 0 for one sample; the run must start that sample at turn 1."""
    run = config()
    store.put(run.fingerprint("go-0001", 0, STAGE_GENERATE, 0).key, sample([call("go_test")]))
    store.put(
        run.fingerprint("go-0001", 0, STAGE_EXECUTE, 0).key,
        {
            "task_id": "go-0001",
            "sample_index": 0,
            "tool_name": "go_test",
            "status": "ok",
            "exit_code": 0,
            "stdout": "cached",
            "duration_ms": 1,
            "harness_error": "",
        },
    )

    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")]), sample(text="done")], "go-0002": [sample()]}
    )
    executor = CountingExecutor()
    run_evaluation(run, TASKS, store, generator, executor, run_id="run-1")
    assert executor.calls == 1, "only go-0002's tool call actually ran"


def test_a_changed_model_revision_forces_fresh_work(store: ResultStore) -> None:
    run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="a")
    executor = CountingExecutor()
    run_evaluation(
        config(model_revision="rev-b"), TASKS, store, ScriptedGenerator(), executor, run_id="b"
    )
    assert executor.calls == 2, "a new revision must re-execute"


def test_a_changed_tool_catalogue_forces_fresh_execution(store: ResultStore) -> None:
    run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="a")
    executor = CountingExecutor()
    run_evaluation(
        config(tool_catalog_sha="deadbeef"), TASKS, store, ScriptedGenerator(), executor, run_id="b"
    )
    assert executor.calls == 2


def test_a_larger_turn_budget_reaches_further(store: ResultStore) -> None:
    """A looping agent stored turns 0-1 under a budget of 2; a budget of 3 must go on.

    The stored turns are reused, not regenerated: turn 2 is the same turn 2
    whatever the budget, so only the newly reachable turn costs a generation.
    """
    looping = {"go-0001": [sample([call("go_test")])] * 5, "go-0002": [sample()]}
    run_evaluation(
        config(max_turns=2),
        TASKS,
        store,
        ScriptedGenerator(dict(looping)),
        CountingExecutor(),
        run_id="a",
    )
    generator = ScriptedGenerator(dict(looping))
    run_evaluation(config(max_turns=3), TASKS, store, generator, CountingExecutor(), run_id="b")
    assert generator.calls == 1, "only the newly reachable turn is generated"
    assert generator.conversation("go-0001", 2)[0]["role"] == "user"
    assert [m["role"] for m in generator.conversation("go-0001", 2)] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def test_rerunning_a_different_config_under_the_same_run_id_is_refused(
    store: ResultStore,
) -> None:
    run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="r")
    with pytest.raises(Exception, match="different manifest"):
        run_evaluation(
            config(seed=99), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="r"
        )


# ------------------------------------------------------------------ reporting


def test_a_model_that_does_nothing_scores_zero(store: ResultStore) -> None:
    """Dropping the task would inflate every average."""
    generator = ScriptedGenerator({"go-0001": [sample()], "go-0002": [sample()]})
    report = run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    assert len(report.summaries) == 2
    assert report.mean_pass_at_1 == 0.0
    assert report.mean_pass_at_n == 0.0


def test_a_sample_with_no_tool_calls_is_not_a_pass(store: ResultStore) -> None:
    generator = ScriptedGenerator({"go-0001": [sample()], "go-0002": [sample([call("go_test")])]})
    report = run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    assert report.summaries[0].passed == 0
    assert report.summaries[1].passed == 1


def test_a_failing_command_is_not_a_pass(store: ResultStore) -> None:
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(exit_code=1),
        run_id="run-1",
    )
    assert report.mean_pass_at_1 == 0.0
    assert report.summaries[0].tool_errors >= 1


def test_report_is_order_independent(tmp_path: pathlib.Path) -> None:
    script = {"go-0001": [sample([call("go_test")]), sample(text="done")], "go-0002": [sample()]}
    a = run_evaluation(
        config(),
        TASKS,
        ResultStore(tmp_path / "a"),
        ScriptedGenerator(dict(script)),
        CountingExecutor(),
        run_id="run-a",
        max_parallel=4,
    )
    b = run_evaluation(
        config(),
        TASKS,
        ResultStore(tmp_path / "b"),
        ScriptedGenerator(dict(script)),
        CountingExecutor(),
        run_id="run-b",
        max_parallel=1,
    )
    assert [s.to_record() for s in a.summaries] == [s.to_record() for s in b.summaries]
    assert a.mean_pass_at_n == b.mean_pass_at_n


def test_pass_at_n_over_several_samples(store: ResultStore) -> None:
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")])] * 4, "go-0002": [sample()] * 4}
    )

    class TwoOfFourPass(AlwaysJudge):
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Pass the first two samples only."""
            self.calls += 1
            return {"score": 1.0 if int(fingerprint.sample_index) < 2 else 0.0}

    report = run_evaluation(
        config(n_samples=4),
        TASKS,
        store,
        generator,
        CountingExecutor(),
        run_id="run-1",
        judge=TwoOfFourPass(),
        k=2,
    )
    assert report.summaries[0].samples == 4
    assert report.summaries[0].passed == 2
    assert report.summaries[0].pass_at_1 == 0.5
    assert report.summaries[0].pass_at_n == pytest.approx(1 - 1 / 6)


def test_report_serialises(store: ResultStore) -> None:
    record = run_evaluation(
        config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="run-1"
    ).to_record()
    assert record["run_id"] == "run-1"
    assert len(record["tasks"]) == 2
    assert record["truncated"] == 0


def test_progress_callback_fires_per_turn(store: ResultStore) -> None:
    seen: list[tuple[str, int, str]] = []
    run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1",
        on_sample=lambda t, i, s: seen.append((t, i, s)),
    )
    assert ("go-0001", 0, f"{STAGE_GENERATE}:0") in seen
    assert ("go-0001", 0, f"{STAGE_EXECUTE}:0") in seen


# ---------------------------------------------------------------- validation


def test_empty_task_list_is_refused(store: ResultStore) -> None:
    with pytest.raises(EvalStoreError, match="no tasks"):
        run_evaluation(config(), [], store, ScriptedGenerator(), CountingExecutor())


def test_a_task_without_an_id_is_refused(store: ResultStore) -> None:
    with pytest.raises(EvalStoreError, match="has no 'id'"):
        run_evaluation(config(), [{"prompt": "x"}], store, ScriptedGenerator(), CountingExecutor())


def test_non_list_tool_calls_are_refused(store: ResultStore) -> None:
    class BadGenerator:
        def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
            """Return a malformed payload."""
            return {"tool_calls": {"name": "go_test"}}

    with pytest.raises(EvalStoreError, match="tool_calls must be a list"):
        run_evaluation(
            config(), TASKS[:1], store, BadGenerator(), CountingExecutor(), run_id="run-1"
        )


def test_malformed_call_entries_are_dropped_not_fatal(store: ResultStore) -> None:
    """One bad entry must not discard an otherwise good trajectory."""
    generator = ScriptedGenerator({"go-0001": [sample(["not a dict"])], "go-0002": [sample()]})
    report = run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    assert len(report.summaries) == 2


def test_run_id_is_generated_when_absent(store: ResultStore) -> None:
    report = run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor())
    assert report.run_id.startswith("eval-")


def test_stored_execution_status_round_trips(store: ResultStore) -> None:
    from gotooltrain.evalrun import _result_from_record
    from gotooltrain.gorun import ExecRequest

    request = ExecRequest(
        task_id="go-0001",
        sample_index=0,
        tool_name="go_test",
        arguments={},
        workspace=pathlib.Path(),
        call_index=2,
        turn_index=1,
    )
    result = _result_from_record(
        request,
        {"status": Status.TOOL_ERROR.value, "stdout": "FAIL", "exit_code": 1, "duration_ms": 7},
    )
    assert result.status is Status.TOOL_ERROR
    assert result.call_index == 2
    assert result.tool_name == "go_test"


def test_an_empty_report_has_zero_averages() -> None:
    """The means must be defined before the first task lands, not a ZeroDivisionError."""
    from gotooltrain.evalrun import RunReport

    empty = RunReport(run_id="r", summaries=[], judged={}, stage_counts={}, reused={})
    assert empty.mean_pass_at_1 == 0.0
    assert empty.mean_pass_at_n == 0.0


def test_a_final_turn_without_tool_calls_omits_the_tool_calls_field(
    store: ResultStore,
) -> None:
    """Only turns that call tools carry the field; a stray empty list would be noise."""
    generator = ScriptedGenerator(
        {"go-0001": [sample([call("go_test")]), sample(text="done")], "go-0002": [sample()]}
    )
    run_evaluation(config(), TASKS, store, generator, CountingExecutor(), run_id="run-1")
    final = generator.conversation("go-0001", 1)
    assert "tool_calls" not in final[-1]
    assert generator.conversation("go-0002", 0) == [
        {"role": "user", "content": "fix the nil map panic"}
    ]


def test_a_turn_with_results_but_no_calls_renders_no_tool_calls_field() -> None:
    """Covers the assistant-message shape directly, independent of any run."""
    from gotooltrain.evalrun import Turn, _messages

    turn = Turn(index=0, generated=sample(text="just talking"), results=())
    assert _messages({"prompt": "p"}, [turn]) == [
        {"role": "user", "content": "p"},
        {"role": "assistant", "content": "just talking"},
    ]


def test_a_judge_can_declare_its_own_identity(store: ResultStore) -> None:
    """Two runs scored by different judges must not look interchangeable."""

    class Named(AlwaysJudge):
        identity = "my-judge-v3"

    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1",
        judge=Named(),
    )
    assert report.judge_identity == "my-judge-v3"
    assert report.scored_by_judge


def test_an_anonymous_judge_still_gets_an_identity(store: ResultStore) -> None:
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(),
    )
    assert "AlwaysJudge" in report.judge_identity


def test_a_run_without_a_judge_is_labelled_execution_only(store: ResultStore) -> None:
    """The gap this closes: a judge's-less number must not look like a judge's."""
    report = run_evaluation(
        config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="run-1"
    )
    assert report.scoring == "execution_only"
    assert not report.scored_by_judge
    assert report.judge_identity == ""
    record = report.to_record()
    assert record["scoring"] == "execution_only"
    assert record["scored_by_judge"] is False


def test_the_judge_is_part_of_the_run_manifest(store: ResultStore) -> None:
    run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="r")
    assert store.load_manifest("r")["judge"] == ""


def test_a_run_without_a_judge_may_gain_one(store: ResultStore) -> None:
    """Execute first, grade afterwards: the normal order, not a conflict."""
    run_evaluation(config(), TASKS, store, ScriptedGenerator(), CountingExecutor(), run_id="r")
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="r",
        judge=AlwaysJudge(),
    )
    assert report.scored_by_judge
    assert store.load_manifest("r")["judge"] == "test_evalrun.AlwaysJudge"


def test_resuming_with_a_different_judge_is_refused(store: ResultStore) -> None:
    """Otherwise the run would blend verdicts from two models into one number."""

    class JudgeA(AlwaysJudge):
        identity = "judge-a"

    class JudgeB(AlwaysJudge):
        identity = "judge-b"

    run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        CountingExecutor(),
        run_id="r",
        judge=JudgeA(),
    )
    with pytest.raises(EvalStoreError, match=r"\['judge'\] differ"):
        run_evaluation(
            config(),
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="r",
            judge=JudgeB(),
        )


def test_unjudged_tasks_are_counted_when_harness_errors_eat_them(
    store: ResultStore,
) -> None:
    """A pass rate over only the tasks that worked would overstate the model."""
    report = run_evaluation(
        config(),
        TASKS,
        store,
        ScriptedGenerator(),
        BrokenExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(),
    )
    assert report.unjudged_tasks == len(TASKS)
    assert report.mean_pass_at_1 == 0.0


def test_a_judge_that_returns_nothing_is_refused(store: ResultStore) -> None:
    """A silently ungraded sample would shrink the denominator and inflate the score."""

    class Silent(AlwaysJudge):
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Decline to grade without saying so."""
            return None

    with pytest.raises(EvalStoreError, match="expected a mapping"):
        run_evaluation(
            config(),
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-1",
            judge=Silent(),
        )


def test_a_judge_verdict_without_a_score_is_refused(store: ResultStore) -> None:
    class ReasonOnly(AlwaysJudge):
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Return prose with no verdict."""
            return {"reason": "looks fine to me"}

    with pytest.raises(EvalStoreError, match="no 'score' key"):
        run_evaluation(
            config(),
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-1",
            judge=ReasonOnly(),
        )


def test_a_non_numeric_score_is_refused(store: ResultStore) -> None:
    class VerdictString(AlwaysJudge):
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Return a score that is not a number."""
            return {"score": "pass"}

    with pytest.raises(EvalStoreError, match="not a number"):
        run_evaluation(
            config(),
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-1",
            judge=VerdictString(),
        )


def test_a_boolean_score_is_refused(store: ResultStore) -> None:
    class VerdictBool(AlwaysJudge):
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """True would be read as 1.0 without complaint."""
            return {"score": True}

    with pytest.raises(EvalStoreError, match="not a number"):
        run_evaluation(
            config(),
            TASKS,
            store,
            ScriptedGenerator(),
            CountingExecutor(),
            run_id="run-1",
            judge=VerdictBool(),
        )


def test_harness_errors_keep_the_full_denominator(store: ResultStore) -> None:
    """A pass rate over only the samples that worked would overstate the model."""
    report = run_evaluation(
        config(n_samples=2),
        TASKS,
        store,
        ScriptedGenerator(),
        BrokenExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(),
    )
    for summary in report.summaries:
        assert summary.samples == 2
        assert summary.passed == 0


class HalfBrokenExecutor:
    """Fails only for sample 0, so one task is partially judged."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Lose the container for the first sample of each task."""
        if request.sample_index == 0:
            raise HarnessError("container is not running")
        return 0, "ok  parser  0.4s", ""


def test_a_partially_judged_task_keeps_its_denominator(store: ResultStore) -> None:
    """Sample 0 died, sample 1 was judged: the task must still report 2 samples.

    Dropping the lost sample would report pass@1 of 1.0 for a task where half the
    evidence never existed.
    """
    report = run_evaluation(
        config(n_samples=2),
        TASKS,
        store,
        ScriptedGenerator(),
        HalfBrokenExecutor(),
        run_id="run-1",
        judge=AlwaysJudge(),
    )
    for summary in report.summaries:
        assert summary.samples == 2
        assert summary.passed == 1
        assert summary.harness_errors == 1
    assert report.mean_pass_at_1 == 0.5
    assert report.unjudged_tasks == 0, "the task was judged, just not on every sample"


def test_each_sample_gets_its_own_workspace(tmp_path: pathlib.Path) -> None:
    """Concurrent samples cannot share a directory; that would be a data race."""
    from gotooltrain.evalrun import _prepare_workspace

    root = tmp_path / "ws"
    first = _prepare_workspace(str(root), "go-0001", 0, None)
    second = _prepare_workspace(str(root), "go-0001", 1, None)
    assert first != second
    assert first.is_dir() and second.is_dir()


def test_a_workspace_is_seeded_from_the_fixture(tmp_path: pathlib.Path) -> None:
    from gotooltrain.evalrun import _prepare_workspace

    fixture = tmp_path / "repo"
    fixture.mkdir()
    (fixture / "go.mod").write_text("module x\n", encoding="utf-8")
    target = _prepare_workspace(str(tmp_path / "ws"), "go-0001", 0, fixture)
    assert (target / "go.mod").read_text(encoding="utf-8") == "module x\n"
    (target / "scribble").write_text("x", encoding="utf-8")
    assert not (fixture / "scribble").exists(), "the fixture is never written to"


def test_an_existing_workspace_is_reused(tmp_path: pathlib.Path) -> None:
    """Resume must not wipe the directory; nothing will be re-executed anyway."""
    from gotooltrain.evalrun import _prepare_workspace

    first = _prepare_workspace(str(tmp_path / "ws"), "go-0001", 0, None)
    (first / "state").write_text("x", encoding="utf-8")
    again = _prepare_workspace(str(tmp_path / "ws"), "go-0001", 0, None)
    assert (again / "state").exists()


def test_a_task_id_with_path_separators_cannot_escape(tmp_path: pathlib.Path) -> None:
    from gotooltrain.evalrun import _prepare_workspace

    root = tmp_path / "ws"
    target = _prepare_workspace(str(root), "../../escape", 0, None)
    assert root.resolve() in target.resolve().parents
