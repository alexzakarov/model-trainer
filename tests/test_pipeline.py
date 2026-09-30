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
import importlib.util
import json
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
        split_dir=tmp_path / "splits",
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
        training_mode="full",
        epochs=1,
        batch_size=1,
        grad_accum=8,
        learning_rate=1e-5,
        max_records=400,
        max_tokens_per_record=8192,
        n_samples=4,
        extra_sft=[],
        extra_dpo=[],
        run_gate=True,
        run_device_check=True,
        run_token_check=True,
        run_repo=True,
        run_tasks=True,
        mutate=True,
        mutations=1,
        log_root=tmp_path / "logs",
        model_name="policy",
        revision="sft-8192",
        dataset_version="go-ut-bench-val",
        repo_url="https://example.invalid/repo.git",
        repo_name="example/repo",
        repo_dir=tmp_path / "repo",
        task_limit=8,
        vllm_port=8000,
        serve_timeout=5,
        serve=True,
        serve_command=["python", "-m", "vllm.entrypoints.openai.api_server"],
        run_corpus=True,
        run_data=True,
        run_sft=True,
        run_eval=True,
        run_dpo=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_an_in_process_stage_runs_and_stops_the_chain_when_it_refuses(
    tmp_path: pathlib.Path,
) -> None:
    """Work in this interpreter is still a stage, and still has to stop the chain.

    Preparing the corpus happens here rather than in a subprocess because it is cheap.
    That must not cost it its place in the plan, its log entry, or the guarantee that
    an empty corpus ends the run with an explanation instead of a trainer error later.
    """
    from gotooltrain.errors import DatasetError as Refusal

    def refuses(emit: Any) -> int:
        emit("measuring")
        raise Refusal("none of them usable")

    lines: list[str] = []
    pipeline = Pipeline(
        [
            Stage("sft-data", ("in-process", "label"), run=refuses),
            Stage("after", (sys.executable, "-c", "print('SHOULD NOT APPEAR')")),
        ],
        emit=lines.append,
        log_root=tmp_path,
    )
    assert pipeline.run() == 2
    joined = "\n".join(lines)
    assert "SHOULD NOT APPEAR" not in joined, "the chain continued past a refused stage"
    assert "none of them usable" in joined
    assert json.loads((tmp_path / "last_failure.json").read_text(encoding="utf-8"))["stage"] == (
        "sft-data"
    )


def test_an_in_process_stage_without_work_says_so_rather_than_reporting_success(
    tmp_path: pathlib.Path,
) -> None:
    """A stage that is declared but not wired must fail loudly on the first run."""
    pipeline = Pipeline(
        [Stage("empty", ("in-process", "label"))],
        emit=lambda message: None,
        log_root=tmp_path,
    )
    with pytest.raises(DatasetError, match="carries no work"):
        pipeline._run_in_process(pipeline.stages[0])  # the branch under test


def test_the_card_is_checked_before_anything_is_downloaded(tmp_path: pathlib.Path) -> None:
    """A full fine-tune of a 4B model has a 17.34 GB floor and needs ~29 GB at 8192.

    A wrong accelerator found after the corpus download costs ten minutes; found as a
    CUDA OOM forty minutes into training it costs an hour and leaves a run that looked
    like it was working. The check is a stage, and it is first -- so it is in the plan,
    it can be seen in a dry run, and nothing upstream of it has run yet.
    """
    names = [stage.name for stage in pipe.build_stages(options(tmp_path))]
    assert names[0] == "device", f"the card check is not first: {names}"
    assert names.index("device") < names.index("corpus"), "and precedes the download"

    stage = next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "device")
    assert stage.run is not None, "the stage has no work attached"
    lines: list[str] = []
    assert stage.run(lines.append) == 0
    # No CUDA here, and that is reported rather than treated as a refusal: the plan is
    # still worth rehearsing on a machine that cannot run it.
    assert any("device" in line for line in lines), lines


def test_a_card_smaller_than_the_budget_is_called_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate below is for a different machine, and saying so is the point.

    Reported rather than fatal, because the number to change is the budget -- and a
    refusal that does not say which knob to turn is a refusal people work around.
    """
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda index: types.SimpleNamespace(
            name="Fake T4", total_memory=15 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    lines: list[str] = []
    assert pipe._check_device(options(tmp_path, memory_budget_gb=40.0), lines.append) == 0

    joined = "\n".join(lines)
    assert "Fake T4" in joined
    assert "smaller than the 40 GB" in joined
    assert "--memory-budget-gb" in joined, "the refusal names the knob to turn"


def test_a_card_that_cannot_hold_the_run_is_not_a_silent_continue(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No CUDA has to be visible in the log, or a CPU rehearsal reads as a real run."""
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake)

    lines: list[str] = []
    assert pipe._check_device(options(tmp_path), lines.append) == 0
    assert any("no CUDA" in line for line in lines), lines


def test_a_card_that_fits_is_reported_without_a_warning(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A card that can hold the run should not be told it cannot.

    A warning printed unconditionally is worse than none: it trains the reader to skip
    the line, which is the line that matters when the card is wrong.
    """
    import types

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda index: types.SimpleNamespace(
            name="A100-SXM4-40GB", total_memory=40 * 1024**3
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    lines: list[str] = []
    assert pipe._check_device(options(tmp_path, memory_budget_gb=40.0), lines.append) == 0
    assert any("A100" in line for line in lines)
    assert not any("smaller than" in line for line in lines), lines


def test_a_host_without_torch_skips_the_card_check_rather_than_failing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline is a Python program before it is a GPU program.

    Preparing a corpus, checking a plan and rehearsing the stage order are all things
    someone may want on a machine that has never had torch installed -- and failing
    here would take all of that away for a check that only matters to the last stage.
    """
    import builtins

    real_import = builtins.__import__

    def without_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    lines: list[str] = []
    assert pipe._check_device(options(tmp_path), lines.append) == 0
    assert any("torch not installed" in line for line in lines), lines


def test_a_publishing_run_without_a_token_is_refused_before_the_download(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token is knowable in the first second; the corpus is not.

    The trainer already refuses on a missing token before it loads the model, but by
    then the corpus has been fetched and measured. Measured for real: a run with the
    secret present in the Colab panel still stopped at the training stage, because
    nothing had put it in the process environment -- and the check that would have
    said so in the first second did not exist.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    lines: list[str] = []
    with pytest.raises(DatasetError, match="HF_TOKEN is not set"):
        pipe._check_token(options(tmp_path, extra_sft=["--hub-repo-id", "a/b"]), lines.append)

    names = [stage.name for stage in pipe.build_stages(options(tmp_path))]
    assert names.index("token") < names.index("corpus"), names


def test_a_run_that_does_not_publish_needs_no_token(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a credential a run never uses is how people learn to ignore checks."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    lines: list[str] = []
    assert pipe._check_token(options(tmp_path), lines.append) == 0
    assert any("none is needed" in line for line in lines), lines


def test_a_rehearsal_needs_no_token_even_with_a_repo_named(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rehearsal uploads nothing, so it must not demand the credential uploading needs."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    lines: list[str] = []
    code = pipe._check_token(
        options(tmp_path, extra_sft=["--hub-repo-id", "a/b", "--hub-dry-run"]), lines.append
    )
    assert code == 0
    assert any("rehearsed" in line for line in lines), lines


def test_a_token_that_is_present_is_reported_not_echoed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that prints the secret it found has moved it into a notebook output."""
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_not_be_printed")
    lines: list[str] = []
    assert (
        pipe._check_token(options(tmp_path, extra_sft=["--hub-repo-id", "a/b"]), lines.append) == 0
    )
    assert not any("hf_this_must_not_be_printed" in line for line in lines), lines


def test_cloning_reuses_a_checkout_that_is_already_there(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running the chain is normal, and re-cloning is not.

    The task builder reads whatever is in that directory, so an existing *checkout* is
    reused and anything else is refused. A silent reuse of a directory that is not a
    checkout would build tasks against whatever happened to be there.
    """
    directory = tmp_path / "repo"
    (directory / ".git").mkdir(parents=True)
    called: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        pipe, "stream_command", lambda command, log, emit: called.append(tuple(command)) or 0
    )
    lines: list[str] = []
    assert pipe._clone_repository(options(tmp_path, repo_dir=directory), lines.append) == 0
    assert called == [], "it cloned over a checkout that was already there"
    assert any("already cloned" in line for line in lines), lines


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path: pathlib.Path) -> None:
    """Otherwise the tasks are built from whatever is in that directory."""
    directory = tmp_path / "repo"
    directory.mkdir()
    (directory / "some-file.txt").write_text("someone's work", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a git checkout"):
        pipe._clone_repository(options(tmp_path, repo_dir=directory), lambda m: None)


def test_a_fresh_clone_is_shallow_and_names_the_repository(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The task builder reads the working tree, not the history.

    Fetching a decade of commits would cost minutes and buy nothing.
    """
    calls: list[tuple[str, ...]] = []

    def record(command: Any, log: Any, emit: Any) -> int:
        calls.append(tuple(str(part) for part in command))
        return 0

    monkeypatch.setattr(pipe, "stream_command", record)
    assert pipe._clone_repository(options(tmp_path), lambda m: None) == 0
    assert calls[0][:2] == ("git", "clone")
    assert "--depth" in calls[0], "a shallow clone, because only the tree is read"
    assert "https://example.invalid/repo.git" in calls[0]


def test_the_server_is_stopped_even_when_sampling_fails(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server left running holds the card the preference run needs next.

    That run fits two whole models on it, and the failure mode -- a successful
    evaluation followed by an out-of-memory error that has nothing to do with memory --
    is one nobody would attribute to a leaked process.
    """
    terminated: list[bool] = []

    class FakeServer:
        """A server that stays up until it is terminated.

        ``poll`` returning a number means the process has *exited*; ``None`` means it is
        still running. Getting that backwards sends the readiness loop down the
        died-early branch on a server that is fine.
        """

        def poll(self) -> int | None:
            return 0 if terminated else None

        def terminate(self) -> None:
            terminated.append(True)

        def wait(self, timeout: int = 0) -> int:
            return 0

    monkeypatch.setattr(pipe.subprocess, "Popen", lambda *a, **k: FakeServer())
    monkeypatch.setattr(pipe, "stream_command", lambda *a, **k: 7)
    # Ready immediately, so the readiness loop breaks and sampling runs (and fails).
    # Patched on the real module, because the function imports it where it uses it.
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Ready())

    code = pipe._serve_and_generate(options(tmp_path), lambda m: None)
    assert code == 7, "the sampling exit code was not passed on"
    assert terminated, "the server was left running"


class _Ready:
    """A minimal stand-in for the readiness response."""

    status = 200

    def read(self) -> bytes:
        return b"{}"

    def __enter__(self) -> _Ready:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_a_server_that_dies_prints_its_own_reason(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first attempt reported "the server exited 1" and nothing else.

    Its output had been sent to DEVNULL, so the reason was destroyed before anyone
    could read it. The tail has to be in the failure, because there is nowhere else the
    reason exists.
    """
    log_root = tmp_path / "logs"
    log_root.mkdir(parents=True)

    class DeadServer:
        """A server that writes a reason and then exits, as a real one does."""

        returncode = 1

        def __init__(self, argv: Any, stdout: Any = None, stderr: Any = None) -> None:
            # The stage opens the log before this is constructed, so the reason has to
            # arrive through the handle it was given rather than being pre-written.
            if stdout is not None:
                stdout.write("CUDA out of memory\narchitectures not supported\n")
                stdout.flush()

        def poll(self) -> int:
            return 1

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int = 0) -> int:
            return 1

    monkeypatch.setattr(pipe.subprocess, "Popen", DeadServer)
    lines: list[str] = []
    with pytest.raises(DatasetError, match="exited 1 before it was ready"):
        pipe._serve_and_generate(options(tmp_path, log_root=log_root), lines.append)
    joined = "\n".join(lines)
    assert "not supported" in joined, "the server's own reason was not shown"


def test_an_externally_served_model_is_used_without_starting_anything(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-serve is for a host where the server already runs.

    Starting another would fail on the port, and a port conflict reads like a model
    problem to whoever is looking at it.
    """
    started: list[bool] = []
    monkeypatch.setattr(pipe.subprocess, "Popen", lambda *a, **k: started.append(True))
    monkeypatch.setattr(pipe, "stream_command", lambda *a, **k: 0)
    lines: list[str] = []
    code = pipe._serve_and_generate(options(tmp_path, serve=False), lines.append)
    assert code == 0
    assert started == [], "a server was started despite --no-serve"
    assert any("already" in line for line in lines), lines


def test_a_server_that_never_becomes_ready_gives_up_and_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Waiting forever is worse than failing: the reader has no idea whether to wait.

    Loading a 4B checkpoint is silent for minutes, so the loop reports what the server
    last said while it tries -- otherwise "still loading" and "wedged" look identical,
    and the next move would be killing a process that was working.
    """
    import urllib.request

    terminated: list[bool] = []

    class LoadingServer:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            terminated.append(True)

        def wait(self, timeout: int = 0) -> int:
            return 0

    monkeypatch.setattr(pipe.subprocess, "Popen", lambda *a, **k: LoadingServer())

    def never_ready(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", never_ready)

    # Time is the thing under test, so it is driven rather than waited on. It starts
    # past the reporting interval so the first pass has something to say.
    clock = {"now": 100.0}
    monkeypatch.setattr(pipe.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(
        pipe.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds)
    )

    lines: list[str] = []
    with pytest.raises(DatasetError, match="was not ready in"):
        pipe._serve_and_generate(options(tmp_path, serve_timeout=20), lines.append)
    assert terminated, "the server was left running after the wait gave up"
    assert any("loading" in line for line in lines), lines


def test_a_named_revision_is_used_as_given(tmp_path: pathlib.Path) -> None:
    """The revision is what stops a resumed run from reusing stale samples.

    It keys the store, so a run that changed checkpoint but kept the key would grade
    itself against results from a model that is not the one being trained. Left unset
    it is derived from the context length; given, it is left alone, because a name
    someone chose is not a value to second-guess.
    """
    assert (
        pipe.main(
            [
                "--dry-run",
                "--no-device",
                "--no-token",
                "--no-gate",
                "--no-corpus",
                "--no-data",
                "--no-sft",
                "--no-eval",
                "--no-dpo",
                "--revision",
                "my-checkpoint",
            ]
        )
        == 0
    )


def test_the_served_module_is_read_off_the_command() -> None:
    """The check has to know what to look for without being told twice."""
    assert pipe._served_module(["python", "-m", "vllm.entrypoints.openai.api_server"]) == (
        "vllm.entrypoints.openai.api_server"
    )
    assert pipe._served_module(["/usr/bin/serve", "--port", "8000"]) is None
    # A dangling -m names nothing, and guessing would check the wrong module.
    assert pipe._served_module(["python", "-m"]) is None


def test_suppressing_the_mutation_keeps_the_task_stage(tmp_path: pathlib.Path) -> None:
    """--no-mutate is for a checkout whose tests already fail.

    The task builder still has to run: it is what turns those failures into the records
    the evaluation reads.
    """
    names = [stage.name for stage in pipe.build_stages(options(tmp_path, mutate=False))]
    assert "mutate" not in names
    assert "tasks" in names, "suppressing the mutation must not suppress the tasks"


def test_a_server_that_cannot_be_imported_is_refused_before_it_is_started(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking after starting it means the reason is a dead process, not a missing one.

    The install is deliberately not done from here: `pip install vllm` can move torch,
    and a pipeline that rearranges the environment it runs in is one whose failures have
    two possible causes. The message names the command instead.
    """
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(DatasetError, match="which is not installed"):
        pipe._serve_and_generate(
            options(
                tmp_path,
                serve_command=["python", "-m", "definitely_not_installed_anywhere"],
            ),
            lambda m: None,
        )


def test_breaking_nothing_is_refused_rather_than_passed_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuing with a healthy checkout stops two stages later, about the wrong thing.

    The task builder would write an empty file, the evaluation would sample nothing, and
    the failure would arrive as "no preference pairs" -- pointing at the pairs rather
    than at the reason there are none.
    """
    from gotooltrain import tasks as task_module

    monkeypatch.setattr(task_module, "harvest_packages", lambda root, limit: ["pkg"])
    monkeypatch.setattr(task_module, "introduce_failures", lambda root, pkgs, limit: [])
    with pytest.raises(DatasetError, match="could be made to fail"):
        pipe._introduce_failure(options(tmp_path), lambda m: None)


def test_a_checkout_with_no_packages_says_so(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to look at is a different problem from nothing breakable."""
    from gotooltrain import tasks as task_module

    monkeypatch.setattr(task_module, "harvest_packages", lambda root, limit: [])
    with pytest.raises(DatasetError, match="no Go packages"):
        pipe._introduce_failure(options(tmp_path), lambda m: None)


def test_the_packages_it_broke_are_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What was broken belongs in the run's record.

    The fixture the model is given is no longer the repository that was cloned, and a
    reader comparing the two needs to know that.
    """
    from gotooltrain import tasks as task_module

    monkeypatch.setattr(task_module, "harvest_packages", lambda root, limit: ["a", "b"])
    monkeypatch.setattr(task_module, "introduce_failures", lambda root, pkgs, limit: ["b"])
    lines: list[str] = []
    assert pipe._introduce_failure(options(tmp_path), lines.append) == 0
    assert any("b" in line for line in lines), lines


def test_the_stages_are_in_the_order_the_run_has_to_happen_in(tmp_path: pathlib.Path) -> None:
    """Each stage consumes what the one before it produced.

    Quality gate first so a broken checkout is caught in a minute rather than after the
    corpus download; the corpus before the data built from it, and that data before the
    stage that trains on it; the evaluation between the checkpoint and the pairs,
    because a preference pair is two attempts at the same task and there is nothing to
    compare until they exist.
    """
    names = [stage.name for stage in pipe.build_stages(options(tmp_path))]
    assert names == [
        "device",
        "token",
        "quality-gate",
        "corpus",
        "sft-data",
        "sft",
        "repo",
        "mutate",
        "tasks",
        "eval",
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
    assert names == ["device", "token", "quality-gate", "corpus", "sft-data", "sft"]
    assert not any(name in {"repo", "tasks", "eval", "preferences", "dpo"} for name in names)


def test_the_supervised_stage_carries_what_the_memory_work_established(
    tmp_path: pathlib.Path,
) -> None:
    """The flags are the code's own conclusions, passed on rather than re-typed.

    Gradient checkpointing that silently does nothing is why the first run OOMed, and
    the budget is what turns that into a message instead of a crash.
    """
    stage = next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "sft")
    command = " ".join(str(part) for part in stage.command)
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
    stages = {
        s.name: " ".join(str(part) for part in s.command)
        for s in pipe.build_stages(options(tmp_path))
    }
    assert "--model" in stages["dpo"]
    assert str(tmp_path / "sft") in stages["dpo"]
    assert "traincli dpo" in stages["dpo"]
    assert "--pairs" in stages["dpo"]


def test_the_evaluation_stage_samples_more_than_once(tmp_path: pathlib.Path) -> None:
    """A preference needs one attempt that passed and one that failed.

    With a single sample there is no contrast, so `preferences` returns nothing and
    the chain stops for a reason that has nothing to do with the model. The sampling
    command is built inside the evaluation stage rather than in the plan, because the
    stage has to run it while the server it started is still alive -- so the assertion
    is on that function's own argv.
    """
    import inspect

    source = inspect.getsource(pipe._serve_and_generate)
    assert '"--n-samples"' in source and "options.n_samples" in source
    stages = {
        s.name: " ".join(str(part) for part in s.command)
        for s in pipe.build_stages(options(tmp_path))
    }
    assert "--n-samples" in stages["preferences"]


def test_the_evaluation_runs_locally_because_colab_has_no_docker(tmp_path: pathlib.Path) -> None:
    """Executing model-authored code is refused unless asked for explicitly.

    That flag is a real risk on someone's own machine; in a disposable Colab VM it is
    the difference between a stage that runs and one that cannot.
    """
    import inspect

    source = inspect.getsource(pipe._serve_and_generate)
    assert '"--sandbox"' in source and '"local"' in source
    assert '"--allow-local-execution"' in source, "the sandbox choice is explicit"
    # And the model served is the checkpoint this run produced, not the base.
    assert (
        '"--model",\n                options.sft_output' in source or "options.sft_output" in source
    )


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
    assert "--hub-repo-id a/b" in " ".join(str(part) for part in stage.command)
    others = [
        " ".join(str(part) for part in s.command)
        for s in pipe.build_stages(options(tmp_path))
        if s.name != "sft"
    ]
    assert not any("--hub-repo-id" in command for command in others)


def test_a_dry_run_prints_the_plan_and_runs_nothing(tmp_path: pathlib.Path, capsys) -> None:
    """The command is built, printed, and no stage starts."""
    assert pipe.main(["--dry-run", "--no-eval", "--no-dpo"]) == 0
    out = capsys.readouterr().out
    assert "stages, in order" in out
    assert "quality-gate" in out
    assert "[dry]" in out, f"the stages were not marked as rehearsals: {out}"


def test_the_corpus_can_be_re_measured_without_re_running_the_chain(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A separate entry point, because re-measuring is cheap and the pipeline is not.

    Re-running the gate and the corpus download to change one filter would cost minutes
    and change nothing else, so the measurement is also reachable on its own.
    """
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="No data to train on"):
        pipe.main(
            [
                "prepare-data",
                "--corpus",
                str(empty),
                "--dataset",
                str(tmp_path / "sft.jsonl"),
            ]
        )


def _tiny_corpus(path: pathlib.Path) -> pathlib.Path:
    """One usable record, so the preparation path can actually succeed."""
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "make the test pass"},
                    {"role": "assistant", "content": "Tests pass."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_the_data_stage_writes_the_corpus_the_next_stage_reads(tmp_path: pathlib.Path) -> None:
    """The happy path, which is the one a run depends on.

    The refusals are tested elsewhere; what matters here is that the stage returns
    success and leaves the file behind, because the trainer's input is that file and
    nothing else in the chain writes it.
    """
    source = _tiny_corpus(tmp_path / "corpus.jsonl")
    destination = tmp_path / "sft.jsonl"
    lines: list[str] = []

    assert (
        pipe.main(
            [
                "prepare-data",
                "--corpus",
                str(source),
                "--dataset",
                str(destination),
                "--max-records",
                "10",
            ]
        )
        == 0
    )
    assert destination.is_file() and destination.read_text(encoding="utf-8").strip()

    # And the stage's own callable reaches the same file from the same options.
    stage = next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "sft-data")
    assert stage.run is not None
    options(tmp_path).corpus = source
    assert stage.run(lines.append) == 0
    assert any("wrote" in line for line in lines)


def test_the_data_stage_reads_the_same_switches_as_the_rest_of_the_chain(
    tmp_path: pathlib.Path,
) -> None:
    """One place decides where the corpus is and how long a record may be.

    A second hard-coded copy of those paths would mean a reader who changes
    ``--corpus`` watches the run prepare one file and train on another. Proved by
    running the stage's own callable against the file the options name: if it were
    reading a path of its own, this refusal would not name that one.
    """
    stage = next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "sft-data")
    assert stage.run is not None
    # Nothing at `options.corpus`, so the refusal names the path it looked at.
    with pytest.raises(DatasetError, match=r"corpus\.jsonl"):
        stage.run(lambda message: None)


def test_the_corpus_stage_produces_the_input_it_reads(tmp_path: pathlib.Path) -> None:
    """A stage must not depend on a file some other cell happened to leave behind.

    This failed for real: the corpus stage named a split file, the cell that fetched
    it had been folded away when the notebook collapsed, and the run stopped after the
    gate had already passed with "Go-UT-Bench split not found". The gate being green
    first is what made it look like something else.
    """
    command = " ".join(
        next(s for s in pipe.build_stages(options(tmp_path)) if s.name == "corpus").command
    )
    assert "--download-to" in command, "the stage fetches its own input"
    assert "--source" not in command, "and does not depend on a path nothing produces"


def test_every_module_the_pipeline_names_actually_exists(tmp_path: pathlib.Path) -> None:
    """A stage that names a module that is not installed fails after the gate passes.

        This is not hypothetical: the corpus stage named ``gotooltrain.data`` while the
        module is ``gotooltrain.datacli``. Every other test here asserted on the command
        *strings*, and a string can be perfectly well formed and still name nothing. The
        gate ran, passed, and the run then stopped two stages in with
        "No module named gotooltrain.data" -- the reader's first thought would have been
    that the install was broken.
    """
    import importlib.util

    for stage in pipe.build_stages(options(tmp_path)):
        if "-m" not in stage.command:
            continue  # an in-process stage runs here; it has no command line to check
        for index, token in enumerate(stage.command):
            if token == "-m":
                name = stage.command[index + 1]
                assert importlib.util.find_spec(name) is not None, (
                    f"stage {stage.name} runs `python -m {name}` and that module does not exist"
                )


def test_every_flag_the_pipeline_passes_is_one_the_cli_defines(tmp_path: pathlib.Path) -> None:
    """A flag the CLI does not define aborts argparse with exit code 2.

    Checked against each CLI's own help text, so it is the real surface the pipeline
    depends on rather than a private helper. The alternative -- finding out by
    watching a multi-hour run stop at the last stage -- is not a test. A stage's
    non-flag arguments are skipped: they are paths and numbers, not switches.
    """
    import subprocess

    def help_for(command: tuple[str, ...]) -> str:
        """The help of the sub-command this stage actually invokes.

        Flags live on the sub-command, not on the top-level parser: ``--help`` at the
        top lists only the sub-commands, which is how a stage can pass a switch that
        does not exist and still look fine to a reader.
        """
        index = command.index("-m")
        name = command[index + 1]
        rest = command[index + 2 :]
        subcommand = rest[0] if rest and not rest[0].startswith("-") else None
        argv = [sys.executable, "-m", name] + ([subcommand] if subcommand else []) + ["--help"]
        # The module and sub-command come from this module's own stage table, never from a
        # dataset or a downloaded file.
        result = subprocess.run(  # noqa: S603 - argv built from the stage table above
            argv, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{' '.join(argv[2:])} failed: {result.stderr[:300]}"
        return result.stdout + result.stderr

    stages = pipe.build_stages(options(tmp_path))
    for stage in stages:
        if "-m" not in stage.command:
            continue  # an in-process stage runs here; it has no command line to check
        index = stage.command.index("-m")
        module_name = str(stage.command[index + 1])
        # str() because a stage may carry a Path for a repository or a task file, and
        # the question here is about flags, not about what a value is made of.
        rest = [str(part) for part in stage.command[index + 2 :]]
        subcommand = rest[0] if rest and not rest[0].startswith("-") else ""
        for position, token in enumerate(rest):
            if not token.startswith("--"):
                continue
            nxt = rest[position + 1] if position + 1 < len(rest) else ""
            if nxt and not nxt.startswith("-"):
                continue  # the token after this one is its value, not another switch
            assert token in help_for(stage.command), (
                f"stage {stage.name} passes {token}, which "
                f"`{module_name} {subcommand}`.strip() does not define"
            )


def test_a_failed_stage_is_recorded_where_a_caller_can_name_it(tmp_path: pathlib.Path) -> None:
    """The reader should not have to scroll back through thousands of lines.

    The notebook's single cell runs the pipeline as a subprocess and reads this file
    to say which stage stopped, so the name survives the scrollback that the log
    itself does not.
    """
    pipeline = Pipeline(
        [Stage("boom", (sys.executable, "-c", "import sys; sys.exit(4)"))],
        emit=lambda message: None,
        log_root=tmp_path,
    )
    pipeline.run()
    record = json.loads((tmp_path / "last_failure.json").read_text(encoding="utf-8"))
    assert record == {"stage": "boom", "exit_code": 4}


def test_auto_continues_from_what_is_already_published() -> None:
    """Asking the Hub what is there turns a data-loss default into a safe one.

    Every push replaces the repository, so starting from the base model when a
    trained checkpoint exists destroys it at the first push.
    """
    assert pipe.resolve_resume(
        ["--resume-from", "auto", "--loss-mode", "selective"],
        repo_id="a/b",
        fetch=lambda _: {"model.safetensors", "config.json"},
    ) == ["--resume-from", "a/b", "--loss-mode", "selective"]


def test_auto_starts_from_the_base_model_when_nothing_is_published() -> None:
    """An empty repository is a first run, not a failure."""
    assert pipe.resolve_resume(
        ["--resume-from", "auto"], repo_id="a/b", fetch=lambda _: {"README.md"}
    ) == ["--resume-from", ""]


def test_a_query_that_fails_is_not_reported_as_an_empty_repository() -> None:
    """A failed query is not an empty repository.

    Both look identical from the outside and only one of them is true. Downgrading a
    failed query to a fresh run would start from the base model and overwrite whatever
    is published -- the exact loss the check exists to prevent. So it refuses, and says
    which way to go if the refusal is wrong.
    """

    def broken(_: str) -> set[str]:
        raise RuntimeError("401 unauthorized")

    with pytest.raises(DatasetError, match="unknown"):
        pipe.resolve_resume(["--resume-from", "auto"], repo_id="a/b", fetch=broken)


def test_a_concrete_resume_is_never_second_guessed() -> None:
    """An explicit repository id or revision is a decision, not a question."""

    def broken(_: str) -> set[str]:
        raise AssertionError("the Hub must not be consulted for a concrete value")

    assert pipe.resolve_resume(["--resume-from", "other/repo:v2"], repo_id="a/b", fetch=broken) == [
        "--resume-from",
        "other/repo:v2",
    ]
    # And an explicit empty string means "start over on purpose".
    assert pipe.resolve_resume(["--resume-from", ""], repo_id="a/b", fetch=broken) == [
        "--resume-from",
        "",
    ]


def test_a_resume_switch_with_no_value_says_what_was_expected() -> None:
    with pytest.raises(DatasetError, match="without a value"):
        pipe.resolve_resume(["--resume-from"], repo_id="a/b")


def test_nothing_to_pass_through_is_returned_unchanged() -> None:
    assert pipe.resolve_resume(["--loss-mode", "selective"], repo_id="a/b") == [
        "--loss-mode",
        "selective",
    ]


def test_a_failure_record_that_cannot_be_written_is_reported_not_swallowed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure is already on screen; a second failure must not replace it.

    Writing the record is a convenience for the caller, so if the disk refuses, the
    run's own error has to stay the thing that is reported.
    """
    lines: list[str] = []
    pipeline = Pipeline(
        [Stage("boom", (sys.executable, "-c", "import sys; sys.exit(2)"))],
        emit=lines.append,
        log_root=tmp_path,
    )
    # Only the failure record is unwritable; the stage's own log uses `open` and must
    # still be written, or this test would prove nothing about the record.
    monkeypatch.setattr(
        pathlib.Path,
        "write_text",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("read-only")),
        raising=False,
    )
    pipeline.run()
    assert any("could not record" in line for line in lines)
    assert (tmp_path / "boom.log").is_file(), "the stage's own log still has to exist"


def test_the_default_listing_asks_the_hub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam is optional: with no ``fetch`` the real Hub is consulted.

    Covered separately because every other test passes one, and a default that quietly
    did nothing would leave ``--resume-from auto`` silently starting from the base
    model -- the exact loss this function exists to prevent.
    """
    hub = pytest.importorskip("huggingface_hub")
    asked: list[str] = []

    class Stub:
        def list_repo_files(self, repo_id: str) -> list[str]:
            asked.append(repo_id)
            return ["model.safetensors"]

    monkeypatch.setattr(hub, "HfApi", Stub)
    assert pipe.resolve_resume(["--resume-from", "auto"], repo_id="a/b") == [
        "--resume-from",
        "a/b",
    ]
    assert asked == ["a/b"], "the Hub was not asked, so nothing was resolved"


def test_selecting_nothing_is_reported_rather_than_silently_succeeding(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A pipeline with no stages has not succeeded at anything."""
    assert (
        pipe.main(
            [
                "--no-device",
                "--no-token",
                "--no-gate",
                "--no-corpus",
                "--no-data",
                "--no-sft",
                "--no-eval",
                "--no-dpo",
            ]
        )
        == 0
    )
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
