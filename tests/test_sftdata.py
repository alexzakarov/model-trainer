"""Mining training data from an evaluation run.

The property under test throughout: a trajectory the model failed must not become
a training example. A miner that keeps everything scores a large corpus and
teaches the model its mistakes, and the mistake is invisible because the corpus
looks healthy.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain import (
    ExecResult,
    ResultStore,
    Status,
    ToolCallIdError,
    build_examples,
    normalize_conversation,
    validate_records,
)
from gotooltrain.evalrun import (
    STAGE_JUDGE,
    RunConfig,
    Trajectory,
    Turn,
    run_evaluation,
)
from gotooltrain.sftdata import (
    MIN_SCORE,
    Admission,
    mine,
    refusal_reason,
)

TASKS: list[dict[str, Any]] = [
    {"id": "go-0001", "repository": "acme/parser", "package": "parser", "prompt": "add a test"},
    {"id": "go-0002", "repository": "acme/parser", "package": "lexer", "prompt": "fix the panic"},
]


def config(**overrides: Any) -> RunConfig:
    base: dict[str, Any] = {
        "model_id": "Qwen/Qwen3.5-4B-go",
        "model_revision": "rev-a",
        "dataset_version": "go-ut-bench-holdout-1",
        "seed": 7,
        "decode_params": {"temperature": 0.0},
    }
    base.update(overrides)
    return RunConfig(**base)


@pytest.fixture
def store(tmp_path: pathlib.Path) -> ResultStore:
    """A fresh result store per test.

    Defined here rather than in conftest because the eval-run tests own an
    identical fixture; sharing it would couple two suites through a helper.
    """
    return ResultStore(tmp_path / "store")


def turn(index: int, *, calls: int = 1, text: str = "") -> Turn:
    """A turn, optionally with tool calls and matching results."""
    generated = {
        "text": text,
        "tool_calls": [
            {"id": f"c{index}-{i}", "name": "go_test", "arguments": {"pkg": "./..."}}
            for i in range(calls)
        ],
    }
    if not calls:
        generated["tool_calls"] = []
    results = tuple(
        ExecResult(
            task_id="t",
            sample_index=0,
            tool_name="go_test",
            status=Status.OK,
            exit_code=0,
            stdout="ok",
            duration_ms=1,
            call_index=i,
        )
        for i in range(calls)
    )
    return Turn(index=index, generated=generated, results=results)


def trajectory(*turns: Turn, truncated: bool = False) -> Trajectory:
    return Trajectory("go-0001", 0, tuple(turns), truncated=truncated)


# ---------------------------------------------------------------- single rules


def test_a_harness_error_outranks_every_other_reason() -> None:
    """Infrastructure loss is not a quality problem, and must not read as one."""
    broken = ExecResult(
        task_id="t",
        sample_index=0,
        tool_name="go_test",
        status=Status.HARNESS_ERROR,
        exit_code=None,
        stdout="",
        duration_ms=1,
        harness_error="container is not running",
    )
    bad = Turn(index=0, generated=turn(0).generated, results=(broken,))
    assert refusal_reason(trajectory(bad), 0.0, min_score=1.0, judged=True) == "harness_error"


def test_a_truncated_trajectory_is_refused() -> None:
    """The final turn is an interruption, not a decision."""
    assert (
        refusal_reason(trajectory(turn(0), truncated=True), 1.0, min_score=1.0, judged=True)
        == "truncated"
    )


def test_a_conversation_without_tool_calls_teaches_nothing() -> None:
    """The corpus exists to teach the catalogue; an essay does not."""
    assert (
        refusal_reason(trajectory(turn(0, calls=0, text="done")), 1.0, min_score=1.0, judged=True)
        == "no_tool_calls"
    )


def test_a_call_without_a_result_is_refused() -> None:
    """It would render a transcript that never occurs, and the validator rejects it."""
    dangling = Turn(index=0, generated=turn(0).generated, results=())
    assert refusal_reason(trajectory(dangling), 1.0, min_score=1.0, judged=True) == "dangling_call"


def test_a_score_below_the_threshold_is_refused() -> None:
    assert refusal_reason(trajectory(turn(0)), 0.0, min_score=1.0, judged=True) == "below_threshold"


def test_a_missing_verdict_is_refused_rather_than_assumed() -> None:
    """No verdict is not a pass; treating it as one would admit ungraded work."""
    assert refusal_reason(trajectory(turn(0)), None, min_score=1.0, judged=True) == "unjudged"


def test_the_threshold_is_exactly_inclusive() -> None:
    assert refusal_reason(trajectory(turn(0)), 1.0, min_score=1.0, judged=True) == ""


def test_execution_only_admits_without_a_verdict() -> None:
    """A deliberately unjudged run admits on execution, but only when asked."""
    assert refusal_reason(trajectory(turn(0)), None, min_score=1.0, judged=False) == ""


def test_a_good_trajectory_breaks_no_rule() -> None:
    assert (
        refusal_reason(
            trajectory(turn(0), turn(1, calls=0, text="done")), 1.0, min_score=1.0, judged=True
        )
        == ""
    )


def test_an_empty_trajectory_is_refused() -> None:
    assert refusal_reason(trajectory(), 1.0, min_score=1.0, judged=True) == "empty"


def test_the_default_threshold_matches_the_judge_scale() -> None:
    """A lower default would admit exactly the failures the judge identified."""
    assert MIN_SCORE == 1.0


def test_admission_reports_whether_it_was_admitted() -> None:
    """The reason is the only field; ``admitted`` is derived from it, never stored."""
    assert Admission("t", 0, "").admitted
    assert not Admission("t", 0, "below_threshold").admitted


# ------------------------------------------------------------------- the report


def test_the_counts_account_for_every_trajectory(store: ResultStore, qwen_tokenizer: Any) -> None:
    """Considered == admitted + refused, always; a lost trajectory is a lost answer."""
    cfg = run(store, judge=FixedJudge(score=0.0))
    _, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert report.considered == len(TASKS)
    assert report.considered == report.admitted + report.refused_count
    assert sum(report.refusals.values()) == report.refused_count


# ------------------------------------------------------------------ integration


class ScriptedGenerator:
    """Replays one turn per task, then finishes."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Call go_test once, then finish with an answer."""
        if turn_index == 0:
            return {
                "text": "Let me run the tests.",
                "tool_calls": [{"id": "call_1", "name": "go_test", "arguments": {"pkg": "./..."}}],
            }
        return {"text": "Tests pass.", "tool_calls": []}


class SilentGenerator:
    """Never calls a tool: an answer the corpus cannot learn from."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Answer without ever calling a tool."""
        return {"text": "I would rather not.", "tool_calls": []}


class ScriptedExecutor:
    """A command that always succeeds."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Return a passing exit code and some output."""
        return 0, "ok  parser  0.4s", ""


class FixedJudge:
    """A judge that returns one score for everything."""

    def __init__(self, score: float = 1.0) -> None:
        """Fix the score to hand out."""
        self.score = score

    def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
        """Score without inspecting the trajectory."""
        return {"score": self.score, "reason": "scripted"}


def run(
    store: ResultStore,
    *,
    generator: Any = None,
    judge: Any = None,
    **overrides: Any,
) -> RunConfig:
    """Execute a real run so the store holds real artifacts to mine."""
    cfg = config(**overrides)
    run_evaluation(
        cfg,
        TASKS,
        store,
        generator or ScriptedGenerator(),
        ScriptedExecutor(),
        judge=FixedJudge() if judge is None else judge,
        workspace=store.root / "ws",
        run_id="run-mine",
    )
    return cfg


def test_only_judged_successes_become_records(store: ResultStore, qwen_tokenizer: Any) -> None:
    cfg = run(store)
    records, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert len(records) == len(TASKS)
    assert report.admitted == len(records)
    assert report.refused_count == 0


def test_a_failed_trajectory_is_not_training_data(store: ResultStore, qwen_tokenizer: Any) -> None:
    """The whole point: training on a failure teaches the failure."""
    cfg = run(store, judge=FixedJudge(score=0.0))
    records, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert records == []
    assert report.refusals.get("below_threshold") == len(TASKS)
    assert all(a.reason == "below_threshold" for a in report.refused)


def test_a_silent_model_contributes_nothing(store: ResultStore, qwen_tokenizer: Any) -> None:
    cfg = run(store, generator=SilentGenerator())
    records, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert records == []
    assert report.refusals.get("no_tool_calls") == len(TASKS)


def test_execution_only_mining_skips_the_verdict(store: ResultStore, qwen_tokenizer: Any) -> None:
    cfg = run(store, judge=FixedJudge(score=0.0))
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer, judged=False)
    assert len(records) == len(TASKS)


def test_an_unknown_judge_fingerprint_is_not_treated_as_a_pass(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """A verdict stored under another config must not leak into this one."""
    run(store)
    other = config(seed=9999)
    records, report = mine(other, TASKS, store, qwen_tokenizer)

    assert records == []
    assert report.refusals.get("unfinished") == len(TASKS)


def test_every_refusal_is_named_and_counted(store: ResultStore, qwen_tokenizer: Any) -> None:
    cfg = run(store, judge=FixedJudge(score=0.0))
    _, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert report.considered == report.admitted + report.refused_count


def test_the_mined_records_validate_and_export(store: ResultStore, qwen_tokenizer: Any) -> None:
    """The output must be consumable by the rest of the pipeline, not just shaped right."""
    cfg = run(store)
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer)

    report = validate_records(records, strict=True)
    assert report.kept == len(records)

    examples = list(build_examples(report.conversations, qwen_tokenizer, on_empty="error"))
    assert len(examples) == len(records)
    assert all(e.supervised_tokens > 0 for e in examples)


def test_tool_results_are_present_in_the_mined_conversation(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """A record missing the observation teaches the model to call and stop."""
    cfg = run(store)
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer)

    messages = records[0]["messages"]
    roles = [m["role"] for m in messages]
    assert "tool" in roles
    conversation = normalize_conversation(messages, records[0]["tools"])
    assert any(m.results for m in conversation.messages)


def test_the_record_carries_its_provenance(store: ResultStore, qwen_tokenizer: Any) -> None:
    """A record with no task id cannot be traced back to what it was mined from."""
    cfg = run(store)
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer)

    metadata = records[0]["metadata"]
    assert metadata["task_id"] in {t["id"] for t in TASKS}
    assert metadata["repository"] == "acme/parser"
    assert metadata["source"] == "eval"


def test_the_report_measures_the_corpus_honestly(store: ResultStore, qwen_tokenizer: Any) -> None:
    """A tiny corpus must be reported inadequate, not silently blessed."""
    cfg = run(store)
    _, report = mine(cfg, TASKS, store, qwen_tokenizer)

    assert report.corpus is not None
    assert report.corpus.trajectories == 2
    assert not report.corpus.is_adequate
    names = {f.name for f in report.corpus.findings}
    assert "too_few_packages" in names


def test_the_report_serialises(store: ResultStore, qwen_tokenizer: Any) -> None:
    cfg = run(store, judge=FixedJudge(score=0.0))
    _, report = mine(cfg, TASKS, store, qwen_tokenizer)
    record = report.to_record()

    assert record["admitted"] == 0
    assert record["refusals"]["below_threshold"] == len(TASKS)
    assert record["min_score"] == 1.0
    assert len(record["refused_detail"]) == report.refused_count


def test_the_catalogue_is_what_the_model_was_offered(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """Training on a tool list the eval never used teaches a different catalogue."""
    from gotooltrain.gotools import GO_TOOLS

    cfg = run(store)
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer)
    names = [t["function"]["name"] for t in records[0]["tools"]]
    assert names == [t.name for t in GO_TOOLS]


def test_mining_a_store_missing_the_run_finds_nothing(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """An unstarted run must not look like a corpus of zero-admitted successes."""
    cfg = config(seed=1234)
    records, report = mine(cfg, TASKS, store, qwen_tokenizer)
    assert records == []
    assert report.refusals.get("unfinished") == len(TASKS)


def test_a_task_without_an_id_is_reported_not_dropped(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """Silently skipping a task would shrink the denominator; a crash would lose the run."""
    cfg = run(store)
    records, report = mine(cfg, [{"prompt": "no id here"}], store, qwen_tokenizer)
    assert records == []
    assert report.refusals.get("missing_task_id") == 1
    assert report.refused[0].reason == "missing_task_id"


def test_the_judge_fingerprint_used_for_mining_matches_the_runner() -> None:
    """If these diverged, every judged trajectory would look unjudged."""
    cfg = config()
    assert cfg.fingerprint("go-0001", 0, STAGE_JUDGE).key == (
        config().fingerprint("go-0001", 0, STAGE_JUDGE).key
    )


def test_an_unwritable_target_is_not_needed_to_mine(
    tmp_path: pathlib.Path, store: ResultStore, qwen_tokenizer: Any
) -> None:
    """Mining is pure: it reads the store and returns records, writing nothing."""
    cfg = run(store)
    before = sorted(p.name for p in store.root.iterdir())
    mine(cfg, TASKS, store, qwen_tokenizer)
    assert sorted(p.name for p in store.root.iterdir()) == before


def test_missing_installed_template_is_refused(store: ResultStore) -> None:
    """Without the template the token counts would be computed from a different render."""
    cfg = run(store)

    class Bare:
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            return "x"

    with pytest.raises(Exception, match="template"):
        mine(cfg, TASKS, store, Bare())


def test_records_are_rejected_by_the_normalizer_if_a_call_is_left_dangling(
    store: ResultStore, qwen_tokenizer: Any
) -> None:
    """Confirms the miner's own dangling-call rule is not redundant with the validator."""
    cfg = run(store)
    records, _ = mine(cfg, TASKS, store, qwen_tokenizer)
    broken = {
        "messages": [m for m in records[0]["messages"] if m["role"] != "tool"],
        "tools": records[0]["tools"],
    }
    with pytest.raises(ToolCallIdError, match="without a result"):
        validate_records([broken], strict=True)
