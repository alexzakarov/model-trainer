"""Execution reward and preference pairs.

The reward is the part of preference learning that cannot be learned from: if it is
wrong, the model is optimised towards the wrong thing and the loss curve still looks
healthy. So each rule that decides a reward has a test that names the failure it
prevents.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain import ResultStore
from gotooltrain.evalrun import RunConfig, Trajectory, Turn, run_evaluation
from gotooltrain.gorun import ExecResult, Status
from gotooltrain.reward import (
    FAILURE,
    SUCCESS,
    PreferencePair,
    build_preferences,
    dpo_example,
    pair_from_samples,
    score_result,
    score_trajectory,
    verification_results,
)


def result(
    tool: str = "go_test",
    status: Status = Status.OK,
    *,
    index: int = 0,
    stdout: str = "ok",
) -> ExecResult:
    """One execution outcome."""
    return ExecResult(
        task_id="t",
        sample_index=0,
        tool_name=tool,
        status=status,
        exit_code=0 if status is Status.OK else 1,
        stdout=stdout,
        duration_ms=1,
        call_index=index,
        harness_error="container died" if status is Status.HARNESS_ERROR else "",
    )


def turn(index: int, *results: ExecResult, calls: int = 1, text: str = "") -> Turn:
    """A turn, with matching tool calls for the results it carries."""
    generated: dict[str, Any] = {
        "text": text,
        "tool_calls": [
            {"id": f"c{index}-{i}", "name": r.tool_name, "arguments": {"pkg": "./..."}}
            for i, r in enumerate(results)
        ][:calls]
        or [{"id": f"c{index}-0", "name": "go_test", "arguments": {}}]
        if results
        else [],
    }
    if not results:
        generated["tool_calls"] = []
    return Turn(index=index, generated=generated, results=results)


def trajectory(*turns: Turn, truncated: bool = False) -> Trajectory:
    return Trajectory("go-0001", 0, tuple(turns), truncated=truncated)


# ------------------------------------------------------------------- the reward


def test_a_successful_command_is_a_success() -> None:
    assert score_result(result(status=Status.OK)) == SUCCESS


def test_a_command_that_ran_and_failed_is_a_failure() -> None:
    """The command ran; the model owns the outcome."""
    assert score_result(result(status=Status.TOOL_ERROR)) == FAILURE


def test_bad_arguments_are_the_models_failure() -> None:
    assert score_result(result(status=Status.MODEL_ERROR)) == FAILURE


def test_a_harness_error_has_no_reward() -> None:
    """Not zero: zero is a claim about the model, and the container died."""
    assert score_result(result(status=Status.HARNESS_ERROR)) is None


def test_only_the_declared_verification_counts() -> None:
    """A model cannot be paid for a command the task never asked for."""
    results = [result("read_file"), result("go_test"), result("grep")]
    assert [
        r.tool_name for r in verification_results(trajectory(turn(0, *results)), ["go_test"])
    ] == ["go_test"]


def test_the_last_verification_call_decides() -> None:
    """A rescue is the behaviour worth reinforcing, so iteration must not be punished."""
    early_fail = result(status=Status.TOOL_ERROR, index=0)
    later_pass = result(status=Status.OK, index=1)
    sample = score_trajectory(trajectory(turn(0, early_fail), turn(1, later_pass)), ["go_test"])
    assert sample.reward == SUCCESS


def test_a_regression_is_a_failure() -> None:
    """Pass then fail is a failure: the final state is what is rewarded."""
    early_pass = result(status=Status.OK, index=0)
    later_fail = result(status=Status.TOOL_ERROR, index=1)
    sample = score_trajectory(trajectory(turn(0, early_pass), turn(1, later_fail)), ["go_test"])
    assert sample.reward == FAILURE


def test_a_harness_failure_is_not_read_as_a_failed_verification() -> None:
    """The verification never really ran, so it cannot count as one that failed."""
    real = result(status=Status.OK, index=0)
    broken = result(status=Status.HARNESS_ERROR, index=1)
    sample = score_trajectory(trajectory(turn(0, real), turn(1, broken)), ["go_test"])
    assert sample.reward == SUCCESS
    assert sample.harness_errors == 1


def test_a_trajectory_whose_only_verification_broke_has_no_reward() -> None:
    broken = result(status=Status.HARNESS_ERROR)
    sample = score_trajectory(trajectory(turn(0, broken)), ["go_test"])
    assert sample.reward is None
    assert sample.verification_ran is False


def test_a_task_without_verification_cannot_be_rewarded() -> None:
    """Falling back to "did anything succeed" would pay for unrelated commands."""
    sample = score_trajectory(trajectory(turn(0, result(status=Status.OK))), [])
    assert sample.reward is None
    assert sample.verification_ran is False


def test_a_trajectory_that_never_ran_the_verification_has_no_reward() -> None:
    sample = score_trajectory(trajectory(turn(0, result("read_file"))), ["go_test"])
    assert sample.reward is None
    assert sample.verification_ran is False


# ------------------------------------------------------------------- the pairs


def sample(reward: float | None, index: int = 0) -> tuple[Any, Trajectory]:
    from gotooltrain.reward import ScoredSample

    finished = trajectory(turn(0, result(status=Status.OK)), turn(1, calls=0, text="done"))
    return (
        ScoredSample(
            task_id="go-0001",
            sample_index=index,
            reward=reward,
            verification_ran=reward is not None,
            harness_errors=0,
        ),
        finished,
    )


def test_a_pass_and_a_fail_make_a_pair() -> None:
    pair, reason = pair_from_samples("go-0001", "fix it", [sample(1.0, 0), sample(0.0, 1)])
    assert reason == ""
    assert pair is not None
    assert pair.chosen_reward == 1.0
    assert pair.rejected_reward == 0.0
    assert pair.margin == 1.0


def test_two_passes_are_not_a_preference() -> None:
    """Inventing an order between two successes trains on noise."""
    pair, reason = pair_from_samples("go-0001", "fix it", [sample(1.0, 0), sample(1.0, 1)])
    assert pair is None
    assert reason == "no_contrast"


def test_two_failures_are_not_a_preference() -> None:
    pair, reason = pair_from_samples("go-0001", "fix it", [sample(0.0, 0), sample(0.0, 1)])
    assert pair is None
    assert reason == "no_contrast"


def test_one_gradable_sample_is_not_enough() -> None:
    pair, reason = pair_from_samples("go-0001", "fix it", [sample(1.0, 0), sample(None, 1)])
    assert pair is None
    assert reason == "too_few_gradable_samples"


def test_a_truncated_sample_is_not_compared() -> None:
    """An interruption is not a conclusion, so it cannot be the rejected side."""
    from gotooltrain.reward import ScoredSample

    truncated = Trajectory(
        "go-0001",
        1,
        (turn(0, result(status=Status.TOOL_ERROR)),),
        truncated=True,
    )
    scored = ScoredSample("go-0001", 1, 0.0, True, 0)
    pair, reason = pair_from_samples("go-0001", "fix it", [sample(1.0, 0), (scored, truncated)])
    assert pair is None
    assert reason == "too_few_gradable_samples"


def test_a_margin_can_be_required() -> None:
    """The reward is binary, so the only margins are 0 and 1.

    Requiring one therefore rejects everything, which is the point: an operator who
    wants only full contrasts gets them, rather than a graded reward appearing from
    nowhere.
    """
    pair, reason = pair_from_samples(
        "go-0001", "fix it", [sample(1.0, 0), sample(0.0, 1)], min_margin=1.0
    )
    assert pair is None
    assert reason == "no_contrast"

    # A margin below the observed one keeps the pair.
    kept, _ = pair_from_samples(
        "go-0001", "fix it", [sample(1.0, 0), sample(0.0, 1)], min_margin=0.5
    )
    assert kept is not None


def test_the_pair_carries_both_conversations() -> None:
    pair, _ = pair_from_samples("go-0001", "fix it", [sample(1.0, 0), sample(0.0, 1)])
    assert pair is not None
    assert pair.prompt == "fix it"
    assert pair.chosen[0]["role"] == "user"
    assert pair.rejected[0]["role"] == "user"


def test_the_dpo_record_keeps_the_reward_that_justified_it() -> None:
    pair = PreferencePair(
        task_id="go-0001",
        prompt="fix it",
        chosen=[{"role": "assistant", "content": "a"}],
        rejected=[{"role": "assistant", "content": "b"}],
        chosen_reward=1.0,
        rejected_reward=0.0,
        margin=1.0,
    )
    record = dpo_example(pair)
    assert record["metadata"]["chosen_reward"] == 1.0
    assert record["metadata"]["source"] == "execution"


# ------------------------------------------------------------------ the run tie-in


TASKS: list[dict[str, Any]] = [
    {
        "id": "go-0001",
        "repository": "acme/parser",
        "package": "parser",
        "prompt": "make the test pass",
        "verification": ["go_test"],
    },
    {
        "id": "go-0002",
        "repository": "acme/parser",
        "package": "lexer",
        "prompt": "make the test pass",
        "verification": ["go_test"],
    },
]


class PerSampleGenerator:
    """Calls go_test once, then answers."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """One tool call on the first turn, then a final answer."""
        if turn_index == 0:
            return {
                "text": f"sample {sample_index}",
                "tool_calls": [{"id": "c0", "name": "go_test", "arguments": {"pkg": "./..."}}],
            }
        return {"text": "done", "tool_calls": []}


class PerSampleExecutor:
    """Succeeds for sample 0 and fails for any other sample."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Return an exit code that depends on which sample called."""
        if request.sample_index == 0:
            return 0, "ok  parser  0.4s", ""
        return 1, "FAIL  parser", ""


def seed(tmp_path: pathlib.Path, n_samples: int = 2) -> tuple[ResultStore, RunConfig]:
    store = ResultStore(tmp_path / "store")
    config = RunConfig(
        model_id="Qwen/Qwen3.5-4B-go",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=7,
        decode_params={"temperature": 0.0},
        n_samples=n_samples,
    )
    run_evaluation(
        config,
        TASKS,
        store,
        PerSampleGenerator(),
        PerSampleExecutor(),
        workspace=tmp_path / "ws",
        run_id="run-reward",
    )
    return store, config


def test_a_run_supplies_pairs_per_task(tmp_path: pathlib.Path) -> None:
    """Sample 0 passes and sample 1 fails, so every task has one contrast."""
    store, config = seed(tmp_path)
    pairs, report = build_preferences(config, TASKS, store)

    assert len(pairs) == len(TASKS)
    assert report.pairs == 2
    assert report.refusals == {}
    assert all(pair.chosen_reward == 1.0 and pair.rejected_reward == 0.0 for pair in pairs)


def test_a_run_with_no_contrast_reports_why(tmp_path: pathlib.Path) -> None:
    """One sample per task cannot contrast with itself."""
    store, config = seed(tmp_path, n_samples=1)
    pairs, report = build_preferences(config, TASKS, store)

    assert pairs == []
    assert report.refusals.get("too_few_gradable_samples") == len(TASKS)


def test_the_report_records_the_rewards_it_measured(tmp_path: pathlib.Path) -> None:
    store, config = seed(tmp_path)
    _, report = build_preferences(config, TASKS, store)
    assert report.rewards["go-0001"] == [1.0, 0.0]
    record = report.to_record()
    assert record["pairs"] == 2
    assert record["rewards"]["go-0001"] == [1.0, 0.0]


def test_a_task_without_a_verification_yields_no_pair(tmp_path: pathlib.Path) -> None:
    """No declared verification means no reward, so no preference either."""
    store, config = seed(tmp_path)
    unverifiable = [{**task, "verification": []} for task in TASKS]
    pairs, report = build_preferences(config, unverifiable, store)

    assert pairs == []
    assert report.refusals.get("nothing_gradable") == len(TASKS)


def test_a_task_without_an_id_is_reported(tmp_path: pathlib.Path) -> None:
    store, config = seed(tmp_path)
    pairs, report = build_preferences(config, [{"prompt": "no id"}], store)
    assert pairs == []
    assert report.refusals.get("missing_task_id") == 1


def test_an_unstarted_run_yields_nothing(tmp_path: pathlib.Path) -> None:
    store, _ = seed(tmp_path)
    other = RunConfig(
        model_id="Qwen/Qwen3.5-4B-go",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=9999,
        decode_params={"temperature": 0.0},
        n_samples=2,
    )
    pairs, report = build_preferences(other, TASKS, store)
    assert pairs == []
    assert report.refusals.get("nothing_gradable") == len(TASKS)


def test_a_min_margin_is_honoured_end_to_end(tmp_path: pathlib.Path) -> None:
    store, config = seed(tmp_path)
    pairs, report = build_preferences(config, TASKS, store, min_margin=1.0)
    assert pairs == []
    assert report.refusals.get("no_contrast") == len(TASKS)
    assert report.min_margin == 1.0


def test_the_same_task_at_two_samples_is_preferred_over_nothing(
    tmp_path: pathlib.Path,
) -> None:
    """The chosen and rejected sides must be different samples, not the same one twice."""
    store, config = seed(tmp_path)
    pairs, _ = build_preferences(config, TASKS, store)
    for pair in pairs:
        assert pair.chosen != pair.rejected, "a pair compared a sample with itself"


@pytest.mark.parametrize("reward", [SUCCESS, FAILURE])
def test_the_reward_values_are_binary(reward: float) -> None:
    """A graded reward would be an invention: the domain's outcome is pass or fail."""
    assert reward in (0.0, 1.0)
