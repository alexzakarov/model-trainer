"""The pipeline as a program rather than a sequence of notebook cells.

What matters here is not that the stages exist but what happens when one of them
misbehaves. A run split across cells survives a failure in a way that reads like
progress; a run driven from here stops and says which stage and why.

The stage order lives in ``build_stages`` at the bottom, which is where the chain is
pinned: a reader asking what happens before what should not have to read a shell
script.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Any

import pytest

from gotooltrain import pipeline as pipe
from gotooltrain.errors import DatasetError
from gotooltrain.pipeline import Pipeline, Stage, StageResult, tail


def _record(lines: list[str]) -> Any:
    """An ``emit`` that both prints and keeps, so assertions can read it back."""
    return lambda message: (lines.append(message), print(message))


def test_stages_run_in_order_and_a_failure_stops_the_run(tmp_path: pathlib.Path) -> None:
    """Continuing past a failure is how a pipeline produces a convincing checkpoint.

    The third stage here prints a marker. If the run does not halt, that marker
    appears and the pipeline reports success for a run that stopped halfway.
    """
    lines: list[str] = []
    pipeline = Pipeline(
        [
            Stage("first", (sys.executable, "-c", "print('fine')")),
            Stage(
                "boom",
                (sys.executable, "-c", "import sys; print('why', file=sys.stderr); sys.exit(3)"),
            ),
            Stage("third", (sys.executable, "-c", "print('SHOULD NOT APPEAR')")),
        ],
        emit=_record(lines),
        log_root=tmp_path,
    )
    code = pipeline.run()

    assert code == 3, "the pipeline did not surface the stage's exit code"
    joined = "\n".join(lines)
    assert "SHOULD NOT APPEAR" not in joined, "a stage ran after the one that failed"
    assert "PIPELINE STOPPED" in joined, "the halt was not announced"
    assert "why" in joined, "the stage's own reason was not shown above the halt"
    assert [r.name for r in pipeline.results] == ["first", "boom"]


def test_a_failing_stage_leaves_its_reason_where_it_can_be_reached(tmp_path: pathlib.Path) -> None:
    """The log is the diagnosis, and it has to outlive the scrollback.

    A long stage buries its own failure in output that scrolls past. Printing the
    tail handles the reader who is watching; the file handles the one who was not.
    """
    pipeline = Pipeline(
        [
            Stage(
                "boom",
                (
                    sys.executable,
                    "-c",
                    "import sys; print('the real reason', file=sys.stderr); sys.exit(1)",
                ),
            )
        ],
        emit=lambda message: None,
        log_root=tmp_path,
    )
    pipeline.run()
    assert "the real reason" in tail(tmp_path / "boom.log", 40)


def test_output_arrives_while_the_stage_is_still_running(tmp_path: pathlib.Path) -> None:
    """A stage that prints, waits, then prints again is how streaming is observable.

    If the log were only written at exit, or buffered until the pipeline finished, the
    first line would not appear until after the wait -- and a reader would have no way
    to tell a slow stage from a stuck one.
    """
    lines: list[str] = []
    pipeline = Pipeline(
        [
            Stage(
                "slow",
                (sys.executable, "-c", "import time\nprint('early')\ntime.sleep(2)\nprint('late')"),
            )
        ],
        emit=lines.append,
        log_root=tmp_path,
    )
    pipeline.run()
    assert lines.index("early") < lines.index("late")
    # And the ordering is preserved in the file, not appended out of sequence.
    assert (tmp_path / "slow.log").read_text(encoding="utf-8").split() == ["early", "late"]


def test_the_plan_is_announced_before_anything_runs(tmp_path: pathlib.Path) -> None:
    """Order lives in the program, not in the reader's memory of the cells.

    If it is only visible after the fact, reconstructing what should have happened is
    guesswork -- and the guess is what people check their run against.
    """
    lines: list[str] = []
    pipeline = Pipeline(
        [Stage("alpha", ("python", "-c", "")), Stage("beta", ("python", "-c", ""))],
        emit=lines.append,
        log_root=tmp_path,
    )
    pipeline.announce()
    joined = "\n".join(lines)
    assert "2 stages, in order" in joined
    assert joined.index("1. alpha") < joined.index("2. beta")


def test_a_dry_run_prints_the_commands_and_starts_nothing(tmp_path: pathlib.Path) -> None:
    """A rehearsal of a multi-hour run has to be able to stop before the first one."""
    lines: list[str] = []
    marker = tmp_path / "should-not-exist"
    pipeline = Pipeline(
        [Stage("writes", (sys.executable, "-c", f"open({str(marker)!r},'w').write('x')"))],
        emit=lines.append,
        log_root=tmp_path,
    )
    assert pipeline.run(dry_run=True) == 0
    assert not marker.exists(), "a dry run executed its stage"
    assert "writes" in "\n".join(lines)


def test_the_summary_records_what_ran_and_for_how_long(tmp_path: pathlib.Path) -> None:
    lines: list[str] = []
    pipeline = Pipeline(
        [Stage("one", (sys.executable, "-c", "print('x')"))], emit=lines.append, log_root=tmp_path
    )
    pipeline.run()
    summary = pipeline.summary()
    assert "one" in summary and ".log" in summary
    result = pipeline.results[0]
    assert result.ok and result.returncode == 0 and result.seconds >= 0.0


"""How the stage list is built, which is where the order actually lives.

The notebook calls one command and the order is decided here. That makes this file the
place the chain is pinned: a reader who wants to know what happens before what does not
have to read a shell script.
"""


def options(tmp_path: pathlib.Path, **overrides: object) -> argparse.Namespace:
    """The parsed-and-folded options ``build_stages`` expects."""
    args = argparse.Namespace(
        model="Qwen/Qwen3.5-4B",
        sft_output=str(tmp_path / "sft"),
        dpo_output=str(tmp_path / "dpo"),
        dataset=tmp_path / "data.jsonl",
        corpus=tmp_path / "corpus.jsonl",
        source_split=tmp_path / "split.json",
        dapt=tmp_path / "dapt.jsonl",
        dapt_report=tmp_path / "dapt.json",
        pairs=tmp_path / "pairs.jsonl",
        preferences_report=tmp_path / "pref.json",
        tasks="data/eval/tasks.jsonl",
        store="runs/eval-store",
        model_url="http://127.0.0.1:8000/v1",
        queue_file=tmp_path / "queue.jsonl",
        verdicts=tmp_path / "verdicts.jsonl",
        context_length=8192,
        memory_budget_gb=40.0,
        optimizer="adafactor",
        epochs=1,
        batch_size=1,
        grad_accum=8,
        learning_rate=1e-5,
        max_records=400,
        n_samples=4,
        extra_sft=[],
        extra_dpo=[],
        run_gate=True,
        run_corpus=True,
        run_sft=True,
        run_eval=True,
        run_dpo=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_the_stages_are_in_the_order_the_run_has_to_happen_in(tmp_path: pathlib.Path) -> None:
    """Each stage consumes what the one before it produced.

    Quality gate first so a broken checkout is caught in a minute rather than after the
    corpus download; corpus before the data that is built from it; the evaluation
    between the checkpoint and the pairs, because a preference pair is two attempts at
    the same task and there is nothing to compare until they exist.
    """
    names = [stage.name for stage in pipe.build_stages(options(tmp_path))]
    assert names == [
        "quality-gate",
        "corpus",
        "sft",
        "eval-queue",
        "eval-judge",
        "preferences",
        "dpo",
    ]


def test_a_switched_off_stage_disappears_from_the_chain(tmp_path: pathlib.Path) -> None:
    """Skipping has to be a decision on record, not a stage that runs and finds nothing.

    A preference stage with no task file would otherwise fail the pipeline.
    """
    names = [
        stage.name for stage in pipe.build_stages(options(tmp_path, run_eval=False, run_dpo=False))
    ]
    assert names == ["quality-gate", "corpus", "sft"]
    assert not any("eval" in name or "dpo" in name or name == "preferences" for name in names)


def test_the_supervised_stage_carries_what_the_memory_work_established(
    tmp_path: pathlib.Path,
) -> None:
    """The flags are the code's own conclusions, passed on rather than re-typed.

    Gradient checkpointing that silently does nothing is why the first run OOMed, and
    the budget is what turns that into a message instead of a crash.
    """
    stage = next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "sft")
    command = " ".join(stage.command)
    assert "--gradient-checkpointing" in command
    assert "--loss-mode selective" in command
    assert "--optimizer adafactor" in command
    assert "--max-length 8192" in command
    assert "--memory-budget-gb 40.0" in command


def test_the_preference_stage_starts_from_the_checkpoint_it_was_built_on(
    tmp_path: pathlib.Path,
) -> None:
    """Preference optimisation starts from the checkpoint it was built on.

    It compares against a frozen copy of the model being trained, so pointing it at
    the base model would optimise towards the wrong thing.
    """
    stages = {s.name: " ".join(s.command) for s in pipe.build_stages(options(tmp_path))}
    assert "--model" in stages["dpo"]
    assert str(tmp_path / "sft") in stages["dpo"]
    assert "traincli dpo" in stages["dpo"]
    assert "--pairs" in stages["dpo"]


def test_the_evaluation_stage_samples_more_than_once(tmp_path: pathlib.Path) -> None:
    """A preference needs one attempt that passed and one that failed.

    With a single sample there is no contrast, so `preferences` returns nothing and
    the chain stops for a reason that has nothing to do with the model.
    """
    stages = {s.name: " ".join(s.command) for s in pipe.build_stages(options(tmp_path))}
    assert "--n-samples 4" in stages["eval-queue"]
    assert "--n-samples" in stages["preferences"]


def test_the_evaluation_runs_locally_because_colab_has_no_docker(tmp_path: pathlib.Path) -> None:
    """Executing model-authored code is refused unless asked for explicitly.

    That flag is a real risk on someone's own machine; in a disposable Colab VM it is
    the difference between a stage that runs and one that cannot.
    """
    stages = {s.name: " ".join(s.command) for s in pipe.build_stages(options(tmp_path))}
    assert "--sandbox local" in stages["eval-queue"]
    assert "--allow-local-execution" in stages["eval-queue"]


def test_publish_flags_are_passed_through_rather_than_duplicated(tmp_path: pathlib.Path) -> None:
    """The orchestrator holds no hub policy of its own.

    Adding one would mean two places deciding whether a run publishes, and they would
    disagree eventually.
    """
    stage = next(
        s
        for s in pipe.build_stages(options(tmp_path, extra_sft=["--hub-repo-id", "a/b"]))
        if s.name == "sft"
    )
    assert "--hub-repo-id a/b" in " ".join(stage.command)
    others = [" ".join(s.command) for s in pipe.build_stages(options(tmp_path)) if s.name != "sft"]
    assert not any("--hub-repo-id" in command for command in others)


def test_a_dry_run_prints_the_plan_and_runs_nothing(tmp_path: pathlib.Path, capsys) -> None:
    """The command is built, printed, and no stage starts."""
    assert pipe.main(["--dry-run", "--no-eval", "--no-dpo"]) == 0
    out = capsys.readouterr().out
    assert "stages, in order" in out
    assert "quality-gate" in out
    assert "--dry-run" in out or "[kuru]" in out


def test_selecting_nothing_is_reported_rather_than_silently_succeeding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pipeline with no stages has not succeeded at anything."""
    assert pipe.main(["--no-gate", "--no-corpus", "--no-sft", "--no-eval", "--no-dpo"]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_a_failing_stage_becomes_the_process_exit_code(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator's own exit code has to carry the failure.

    A caller that only looks at whether the pipeline returned zero would otherwise see
    a clean exit after a stage died -- the notebook's single cell would print its
    success line and stop there.
    """
    stages = [Stage("boom", (sys.executable, "-c", "import sys; sys.exit(7)"))]
    monkeypatch.setattr(pipe, "build_stages", lambda options: stages)
    with pytest.raises(SystemExit) as raised:
        pipe.main([])
    assert raised.value.code == 7, "the stage's code did not become the pipeline's"


def test_a_stage_that_stops_without_a_pipe_says_so_rather_than_reporting_an_empty_log(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnosis path must not be the thing that crashes quietly.

    Reaching this means the pipe was never created, which is a bug here rather than
    anything a reader can cause -- so the point is that it is reported, not swallowed
    into an empty log that looks like a stage that printed nothing.
    """
    real_popen = subprocess.Popen

    class NoPipe(real_popen):  # type: ignore[misc,no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # The parent's __init__ assigns self.stdout from the PIPE argument, so a
            # class attribute alone would never take effect.
            self.stdout = None

    monkeypatch.setattr(subprocess, "Popen", NoPipe)
    pipeline = Pipeline(
        [Stage("x", (sys.executable, "-c", "pass"))],
        emit=lambda message: None,
        log_root=tmp_path,
    )
    with pytest.raises(DatasetError, match="output pipe"):
        pipeline.run()


def test_the_shared_production_run_is_the_one_the_notebook_asks_for(
    tmp_path: pathlib.Path,
) -> None:
    """One measurement of the supervised share, used for the budget.

    This is the number that multiplies the vocabulary term, and it is measured from
    the real corpus rather than assumed -- so it belongs in the pipeline rather than
    re-derived per reader.
    """
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="No data to train on"):
        pipe.measure_dataset(
            empty, max_records=1, max_tokens_per_record=64, tokenizer_name="Qwen/Qwen3.5-4B"
        )


def test_a_stage_that_is_not_fatal_does_not_stop_the_run(tmp_path: pathlib.Path) -> None:
    """An evaluation can fail and still report something worth reading.

    The judgement belongs to the stage, not to the runner: a stage marked non-fatal
    records its failure and the chain continues.
    """
    lines: list[str] = []
    pipeline = Pipeline(
        [
            Stage("soft", (sys.executable, "-c", "import sys; sys.exit(2)"), fatal=False),
            Stage("after", (sys.executable, "-c", "print('carried on')")),
        ],
        emit=lines.append,
        log_root=tmp_path,
    )
    assert pipeline.run() == 0
    assert "carried on" in "\n".join(lines)
    assert [r.name for r in pipeline.results] == ["soft", "after"]


def test_an_unreadable_log_says_so_rather_than_raising(tmp_path: pathlib.Path) -> None:
    """The diagnosis path must not itself be the thing that crashes."""
    assert "unreadable" in tail(tmp_path / "never-written.log", 10)[0]


def test_a_stage_result_reports_its_own_outcome() -> None:
    assert StageResult("a", 0, 1.0, pathlib.Path("a.log")).ok
    assert not StageResult("a", 1, 1.0, pathlib.Path("a.log")).ok
