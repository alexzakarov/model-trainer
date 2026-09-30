"""The whole run as one command, so a notebook does not have to be a program.

A notebook is a bad place to hold a pipeline. It is a sequence of cells with hidden
state between them: if one dies, the ones after it still run against whatever the
previous one left behind, and the reader is left guessing which stage actually
happened. Splitting a run across eleven cells means the *order* is carried in a
person's memory.

So the pipeline lives here, where it can be tested, and the notebook's job shrinks
to starting it. Three properties this file exists to guarantee:

* **Order is data, not memory.** The stages are a list, printed before anything
  runs, so the reader sees the plan rather than reconstructing it afterwards.
* **A failed stage stops the run.** The log tail is printed and the process exits
  non-zero. Continuing past a failure is how a pipeline produces a checkpoint that
  looks complete.
* **Output streams and is kept.** Each stage's lines are printed as they arrive and
  appended to ``runs/pipeline/<stage>.log``, so a long stage is watchable and its
  diagnosis survives the scrollback.

The stages are separate processes on purpose. A CUDA OOM or a torch crash in one
stage must not leave the orchestrator with a poisoned runtime, and a subprocess
cannot poison anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from . import build_notebook
from .errors import DatasetError

#: The repository a run publishes to, and the one ``--resume-from auto`` asks about.
#: Imported rather than repeated: a second copy would be a second place where the
#: answer to "where did this checkpoint go" lives, and a rename would leave the two
#: silently disagreeing -- the run trains, the checkpoint uploads, and it lands
#: somewhere nobody is looking.
DEFAULT_HUB_REPO: Final[str] = build_notebook.DEFAULT_HF_REPO_ID

#: Where per-stage logs land. Under ``runs/`` so it is ignored like the rest.
LOG_ROOT = pathlib.Path("runs") / "pipeline"


@dataclass(frozen=True, slots=True)
class Stage:
    """One step: what it is called, and how to start it."""

    name: str
    command: tuple[str, ...]
    #: Whether a failure here should stop the pipeline. A dry run failing is a
    #: rehearsal failing; an evaluation failing may be reportable and not fatal,
    #: because it produces a report either way.
    fatal: bool = True
    #: Work done in this interpreter instead of another command. Only set for a stage
    #: whose ``command`` is the placeholder ``("in-process", <label>)`` -- the label is
    #: there so a dry run can still print something, not to be dispatched on.
    run: Callable[[Callable[[str], None]], int] | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one stage did, for the record and for the final summary."""

    name: str
    returncode: int
    seconds: float
    log: pathlib.Path

    @property
    def ok(self) -> bool:
        """Whether the stage succeeded."""
        return self.returncode == 0


@dataclass
class Pipeline:
    """An ordered run of stages, each isolated and each logged.

    ``emit`` is the one place output goes, so the same object runs under a test with
    a collector and on Colab with ``print`` -- the orchestration is not retested
    through the notebook that happens to call it.
    """

    stages: Sequence[Stage]
    emit: Callable[[str], None] = field(default=lambda message: print(message))
    log_root: pathlib.Path = LOG_ROOT
    results: list[StageResult] = field(default_factory=list)

    def announce(self) -> None:
        """Say what is about to happen, before any of it has."""
        self.emit(f"{len(self.stages)} stages, in order:")
        for index, stage in enumerate(self.stages, start=1):
            self.emit(f"  {index:2}. {stage.name}")
        self.emit("")

    def run(self, *, dry_run: bool = False) -> int:
        """Run every stage in order; return the first fatal failure's exit code."""
        self.log_root.mkdir(parents=True, exist_ok=True)
        for stage in self.stages:
            if dry_run:
                self.emit(f"[dry] {stage.name}: {' '.join(stage.command)}")
                continue
            code = self._run_in_process(stage) if stage.run is not None else self._one(stage)
            if code != 0 and stage.fatal:
                self.emit(
                    f"\nPIPELINE STOPPED at {stage.name} ({code}). "
                    "Its log and last lines are above."
                )
                # Written so a caller can say which stage failed without scrolling
                # back through output that may be thousands of lines long.
                self._record_failure(stage, code)
                return code
        return 0

    def _run_in_process(self, stage: Stage) -> int:
        """A stage that is work in this interpreter rather than another command.

        Only one: preparing the corpus. It is cheap, and a second interpreter start
        would reload the tokenizer for no reason. It is still a stage, because a stage
        is what appears in the plan, stops the chain when it fails, and gets a log.
        """
        self.emit(f"\n=== {stage.name} ===")
        if stage.run is None:
            raise DatasetError(f"stage {stage.name} is in-process but carries no work to do")
        try:
            return int(stage.run(self.emit))
        except DatasetError as exc:
            # A refusal, not a crash: the chain stops and says why, which is the whole
            # reason an empty corpus is caught here rather than by the trainer.
            self.emit(f"{stage.name} refused: {exc}")
            return 2

    def _record_failure(self, stage: Stage, code: int) -> None:
        """Name the failed stage where a caller can read it."""
        try:
            (self.log_root / "last_failure.json").write_text(
                json.dumps({"stage": stage.name, "exit_code": code}, indent=2), encoding="utf-8"
            )
        except OSError:
            # The failure is already on screen; failing to also write it to disk must
            # not replace one clear error with a second confusing one.
            self.emit("(could not record which stage failed to disk)")

    def _one(self, stage: Stage) -> int:
        """Run one stage, streaming its output and keeping it."""
        log_path = self.log_root / f"{stage.name}.log"
        started = time.monotonic()
        self.emit(f"\n=== {stage.name} ===")
        self.emit(f"$ {' '.join(stage.command)}")
        # Unbuffered, so a stage that prints before its first step still says so.
        command = (
            (sys.executable, "-u", *stage.command[1:])
            if stage.command[0] == "python"
            else stage.command
        )
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            # Every element comes from this module's own stage table or from the
            # operator's command line, never from a dataset or a downloaded file --
            # which is the distinction ruff's S603 is asking about. A task file's
            # contents are arguments to `gotooltrain`, not to this.
            process = subprocess.Popen(  # noqa: S603 - argv is constructed here, not parsed from data
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            stdout = process.stdout
            if stdout is None:
                # Only reachable if the pipe above were removed; failing here beats
                # iterating None and reporting it as a silent empty log.
                raise DatasetError("the stage's output pipe was not created")
            for line in stdout:
                self.emit(line.rstrip("\n"))
                log.write(line)
            code = process.wait()
        seconds = time.monotonic() - started
        self.results.append(StageResult(stage.name, code, seconds, log_path))
        if code != 0:
            self.emit(f"--- {log_path} (last 30 lines) ---")
            for line in tail(log_path, 30):
                self.emit(f"  {line}")
        else:
            self.emit(f"--- {stage.name} bitti ({seconds:.1f}s) ---")
        return code

    def summary(self) -> str:
        """A compact record of what ran, in the order it ran."""
        lines = ["", "stage summary:"]
        for result in self.results:
            mark = "ok" if result.ok else f"{result.returncode}"
            lines.append(f"  {mark:>4}  {result.seconds:8.1f}s  {result.name}  -> {result.log}")
        return "\n".join(lines)


def resolve_resume(
    extra: Sequence[str],
    *,
    repo_id: str,
    fetch: Callable[[str], Iterable[str]] | None = None,
) -> list[str]:
    """Turn ``--resume-from auto`` into a decision, before the first stage runs.

    ``auto`` means "continue from whatever this project already published, if anything".
    The alternative default -- start from the base model -- destroys a trained
    checkpoint at the first push, because every push replaces the repository. Asking
    the Hub what is there costs one API call and makes the safe choice the default.

    "Bilmiyorum" is not "yok". A query that fails, a token that is missing and a
    network that is down all all mean the repository's contents are unknown, and the
    three of them look identical from the outside. Only one of them means there is
    nothing to resume from, so a failed query is refused rather than quietly
    downgraded to a fresh run that would overwrite whatever is there.

    Pass-through when the value is already concrete: an explicit repository id, a
    revision, or an explicit empty string meaning "start over on purpose".

    ``fetch`` is the seam the tests use. The default asks the Hub, so no caller has to
    know it exists.
    """
    values = list(extra)
    try:
        index = values.index("--resume-from")
    except ValueError:
        return values
    if index + 1 >= len(values):
        raise DatasetError(
            "--resume-from was given without a value. Pass a repository id to continue "
            "from one, 'auto' to continue from whatever is already published, or an "
            "empty string to start over on purpose."
        )

    if values[index + 1] != "auto":
        return values

    try:
        files = set((fetch or _hub_listing)(repo_id))
    except Exception as exc:  # every failure means "unknown", not "empty"
        raise DatasetError(
            f"--resume-from auto could not read {repo_id} ({type(exc).__name__}). "
            "Whether a checkpoint is published there is unknown, and starting from the "
            "base model would overwrite it at the first push. Set --resume-from "
            f'{repo_id} to continue from it deliberately, or --resume-from "" to '
            "start over on purpose."
        ) from exc

    weights = {"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"}
    values[index + 1] = repo_id if weights & files else ""
    return values


def _hub_listing(repo_id: str) -> Iterable[str]:
    """What the Hub says a repository contains."""
    from huggingface_hub import HfApi

    files: Iterable[str] = HfApi().list_repo_files(repo_id)
    return files


def tail(path: pathlib.Path, lines: int) -> list[str]:
    """The last few lines of a log, or a note if it cannot be read."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return [f"(log okunamadi: {path})"]


def measure_dataset(
    source: pathlib.Path,
    *,
    max_records: int,
    max_tokens_per_record: int,
    tokenizer_name: str,
) -> dict[str, Any]:
    """Read, render and filter the corpus before anything expensive happens.

    This lives here rather than in the notebook so it is tested. Its decisions are not
    obvious: a record that *reaches* the context limit has been truncated, and
    truncation preserves the leading prompt tokens, so a record whose assistant turn
    began before the limit still has supervised tokens left. Measuring "is there
    supervision" alone therefore approves records that the trainer will later refuse
    as over-length. The check is the length, and it is here.

    Measured on the first 400 records of this corpus: 74 of them were truncated in
    exactly this way, with real lengths of 8.5K to 17.6K.
    """
    from transformers import AutoTokenizer

    from . import (
        catalog,
        install_template,
        load_template_source,
        normalize_conversation,
        read_jsonl,
        render_example,
    )

    tools = catalog()
    tokenizer = install_template(
        AutoTokenizer.from_pretrained(tokenizer_name), load_template_source()
    )

    records = list(read_jsonl(source))[:max_records]
    kept: list[dict[str, Any]] = []
    dropped_invalid = dropped_truncated = 0
    total_tokens = supervised_tokens = longest = 0
    first_rendered: Any = None

    for record in records:
        try:
            conversation = normalize_conversation(record["messages"], tools)
        except Exception:
            dropped_invalid += 1
            continue
        example = render_example(tokenizer, conversation, max_length=max_tokens_per_record)

        # Sitting exactly on the ceiling means truncated. See the docstring.
        if len(example.input_ids) >= max_tokens_per_record:
            dropped_truncated += 1
            continue
        # There is deliberately no "supervised nothing" check here. It was one, and it
        # was unreachable: `normalize_conversation` refuses an empty assistant turn and
        # refuses a tool call with no result, both before rendering, so a record that
        # reaches this line always supervises something. Verified with a whitespace
        # answer and with a bare tool call. A guard no input can reach is a claim
        # nothing can test, and if a future template ever renders zero supervision the
        # trainer's own guard says so -- at the point where it would matter.
        kept.append({"messages": record["messages"], "tools": tools})
        total_tokens += len(example.input_ids)
        supervised_tokens += example.supervised_tokens
        longest = max(longest, len(example.input_ids))
        if first_rendered is None:
            first_rendered = example

    if not kept:
        raise DatasetError(
            f"{len(records)} records, none of them usable "
            f"({dropped_truncated} truncated, {dropped_invalid} invalid). No data to "
            "train on; starting this run would "
            "waste the GPU and produce a checkpoint identical to the one it started from."
        )

    return {
        "records_read": len(records),
        "kept": len(kept),
        "dropped_truncated": dropped_truncated,
        "dropped_invalid": dropped_invalid,
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_share": supervised_tokens / max(1, total_tokens),
        "longest": longest,
        "average": total_tokens // len(kept),
        "examples": kept,
        "first_rendered": first_rendered,
    }


def _check_device(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Fail in seconds if the card cannot hold the run.

    The estimate the training stage prints is the same arithmetic; getting it before
    the download means a wrong accelerator costs ten seconds instead of ten minutes,
    and -- more importantly -- it is the difference between "this card is too small"
    and a CUDA OOM that arrives after the run has already been going long enough to
    look like it was working.

    No CUDA is not a refusal. The pipeline runs under test on a CPU-only machine, and
    a plan is still worth rehearsing there. It is reported, because a run that will
    fall over at the first forward pass should say so rather than appear healthy.
    """
    try:
        import torch
    except ImportError:
        emit("device: torch not installed; skipping the card check")
        return 0

    if not torch.cuda.is_available():
        emit(
            "device: no CUDA device visible. This run will not train here; the check "
            "is reported rather than fatal so the plan can still be rehearsed."
        )
        return 0

    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / (1024**3)
    emit(f"device: {properties.name}, {total_gb:.1f} GB")
    if total_gb + 1e-6 < options.memory_budget_gb:
        emit(
            f"device: the card is smaller than the {options.memory_budget_gb:.0f} GB this "
            "run is fitted against, so the estimate below is for a different machine. "
            "Lower --memory-budget-gb to this card's size to get a refusal you can act "
            "on, or point the run at the card it was fitted for."
        )
    return 0


def _write_sft_data(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Prepare the training corpus from the options this run was given.

    A closure rather than fixed paths: the stage reads the same ``--corpus`` and
    ``--model`` the rest of the chain does, so a reader changing one switch does not
    have to find a second hard-coded copy of it.
    """
    prepare_data(
        pathlib.Path(options.corpus),
        pathlib.Path(options.dataset),
        max_records=options.max_records,
        max_tokens_per_record=options.max_tokens_per_record,
        tokenizer_name=options.model,
        emit=emit,
    )
    return 0


def build_stages(options: argparse.Namespace) -> list[Stage]:
    """The chain, in the order it has to happen, from the options given."""
    sft = options.sft_output
    stages: list[Stage] = []

    # First, before anything is downloaded: is this card even in the conversation?
    # A full fine-tune of a 4B model has a 17.34 GB resident floor and needs about
    # 29 GB at 8192 tokens. Discovering that after the corpus download wastes minutes,
    # and discovering it as a CUDA OOM forty minutes into training wastes an hour and
    # leaves a run that looks like it was working.
    if options.run_device_check:
        stages.append(
            Stage(
                "device",
                ("in-process", "is there a card that can hold this run"),
                run=lambda emit: _check_device(options, emit),
            )
        )

    if options.run_gate:
        stages.append(Stage("quality-gate", ("python", "-m", "pytest", "-q", "-x", "-rs")))
    if options.run_corpus:
        stages.append(
            Stage(
                "corpus",
                (
                    "python",
                    "-m",
                    "gotooltrain.datacli",
                    "go-pairs",
                    "--download-to",
                    str(options.split_dir),
                    "--out",
                    str(options.dapt),
                    "--report",
                    str(options.dapt_report),
                    "--unit-tests-out",
                    str(options.corpus),
                ),
            )
        )
    if options.run_data:
        # Runs in this process, not a subprocess: the measurement is cheap, the
        # tokenizer is already in memory for nothing else, and an extra interpreter
        # start here would be pure overhead. It is a stage so that it appears in the
        # plan, stops the chain on an empty corpus, and has its own log entry.
        stages.append(
            Stage(
                "sft-data",
                ("in-process", "measure, filter and write data/sft.jsonl"),
                run=lambda emit: _write_sft_data(options, emit),
            )
        )

    if options.run_sft:
        stages.append(
            Stage(
                "sft",
                (
                    "python",
                    "-m",
                    "gotooltrain.traincli",
                    "sft",
                    "--model",
                    options.model,
                    "--output",
                    sft,
                    "--tokenizer",
                    options.model,
                    "--dataset",
                    str(options.dataset),
                    "--dtype",
                    "bfloat16",
                    "--epochs",
                    str(options.epochs),
                    "--batch-size",
                    str(options.batch_size),
                    "--grad-accum",
                    str(options.grad_accum),
                    "--lr",
                    str(options.learning_rate),
                    "--max-length",
                    str(options.context_length),
                    "--memory-budget-gb",
                    str(options.memory_budget_gb),
                    "--loss-mode",
                    "selective",
                    "--optimizer",
                    options.optimizer,
                    "--training-mode",
                    options.training_mode,
                    "--gradient-checkpointing",
                    *options.extra_sft,
                ),
            )
        )
    if options.run_eval:
        # Two commands, not one: the agent loop writes trajectories, and the judge
        # ingests verdicts into the store. `preferences` reads the store, so both
        # have to have run before a pair can exist.
        stages.append(
            Stage(
                "eval-queue",
                (
                    "python",
                    "-m",
                    "gotooltrain.evalcli",
                    "queue",
                    "--store",
                    options.store,
                    "--tasks",
                    options.tasks,
                    "--model",
                    options.model,
                    "--revision",
                    f"sft-{options.context_length}",
                    "--dataset-version",
                    "go-ut-bench-val",
                    "--n-samples",
                    str(options.n_samples),
                    "--model-url",
                    options.model_url,
                    "--out",
                    str(options.queue_file),
                    "--sandbox",
                    "local",
                    "--allow-local-execution",
                ),
            )
        )
        stages.append(
            Stage(
                "eval-judge",
                (
                    "python",
                    "-m",
                    "gotooltrain.evalcli",
                    "judge",
                    "--store",
                    options.store,
                    "--tasks",
                    options.tasks,
                    "--model",
                    options.model,
                    "--revision",
                    f"sft-{options.context_length}",
                    "--dataset-version",
                    "go-ut-bench-val",
                    "--queue",
                    str(options.queue_file),
                    "--verdicts",
                    str(options.verdicts),
                ),
            )
        )
        stages.append(
            Stage(
                "preferences",
                (
                    "python",
                    "-m",
                    "gotooltrain.datacli",
                    "preferences",
                    "--store",
                    options.store,
                    "--tasks",
                    options.tasks,
                    "--model",
                    options.model,
                    "--revision",
                    f"sft-{options.context_length}",
                    "--dataset-version",
                    "go-ut-bench-val",
                    "--n-samples",
                    str(options.n_samples),
                    "--out",
                    str(options.pairs),
                    "--report",
                    str(options.preferences_report),
                ),
            )
        )
    if options.run_dpo:
        stages.append(
            Stage(
                "dpo",
                (
                    "python",
                    "-m",
                    "gotooltrain.traincli",
                    "dpo",
                    "--model",
                    sft,
                    "--output",
                    options.dpo_output,
                    "--tokenizer",
                    sft,
                    "--pairs",
                    str(options.pairs),
                    "--dtype",
                    "bfloat16",
                    "--epochs",
                    str(options.epochs),
                    "--batch-size",
                    str(options.batch_size),
                    "--grad-accum",
                    str(options.grad_accum),
                    "--lr",
                    "5e-6",
                    "--max-length",
                    str(options.context_length),
                    "--memory-budget-gb",
                    str(options.memory_budget_gb),
                    "--loss-mode",
                    "selective",
                    "--optimizer",
                    options.optimizer,
                    "--training-mode",
                    options.training_mode,
                    "--gradient-checkpointing",
                    *options.extra_dpo,
                ),
            )
        )
    return stages


def prepare_data(
    source: pathlib.Path,
    destination: pathlib.Path,
    *,
    max_records: int,
    max_tokens_per_record: int,
    tokenizer_name: str,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure, filter and write the corpus the training stage will read.

    Writing belongs here rather than in a notebook cell because the file is an input
    to the next stage: a stage that depends on a file some other cell left behind is
    how the run stopped with "split not found" and would have stopped again with
    "no such dataset" one stage later. The counts are printed because a reader who
    cannot see how many records survived the filter has no idea whether the run is
    training on the corpus they think it is.
    """
    report = measure_dataset(
        source,
        max_records=max_records,
        max_tokens_per_record=max_tokens_per_record,
        tokenizer_name=tokenizer_name,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in report["examples"]:
            handle.write(json.dumps(record) + "\n")

    emit(
        f"corpus: {report['kept']}/{report['records_read']} kept, "
        f"{report['dropped_truncated']} truncated, {report['dropped_invalid']} invalid"
    )
    emit(
        f"tokens: {report['total_tokens']} total, {report['supervised_tokens']} supervised "
        f"({report['supervised_share']:.1%}); longest {report['longest']}"
    )
    emit(f"wrote {destination}")
    return report


def add_common(parser: argparse.ArgumentParser) -> None:
    """Options that describe the run rather than one stage."""
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--sft-output", default="runs/colab-sft")
    parser.add_argument("--dpo-output", default="runs/colab-dpo")
    parser.add_argument("--dataset", default="data/sft.jsonl", type=pathlib.Path)
    parser.add_argument("--corpus", default="data/go-unit-tests.jsonl", type=pathlib.Path)
    parser.add_argument(
        "--split-dir",
        default="data/go-ut-bench",
        type=pathlib.Path,
        help="where the published splits are fetched; the stage produces its own input",
    )
    parser.add_argument("--dapt", default="data/go-dapt.jsonl", type=pathlib.Path)
    parser.add_argument("--dapt-report", default="data/go-dapt-report.json", type=pathlib.Path)
    parser.add_argument("--pairs", default="data/preferences.jsonl", type=pathlib.Path)
    parser.add_argument(
        "--preferences-report", default="runs/preferences-report.json", type=pathlib.Path
    )
    parser.add_argument("--tasks", default="data/eval/tasks.jsonl")
    parser.add_argument("--store", default="runs/eval-store")
    parser.add_argument("--model-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--queue-file", default="runs/eval-queue.jsonl", type=pathlib.Path)
    parser.add_argument("--verdicts", default="runs/eval-verdicts.jsonl", type=pathlib.Path)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--memory-budget-gb", type=float, default=40.0)
    parser.add_argument("--optimizer", default="adafactor")
    parser.add_argument(
        "--training-mode",
        default="full",
        choices=("full", "qlora"),
        help=(
            "full updates every weight; qlora keeps the base frozen in 4 bits and trains "
            "adapters, which is what fits a 4B model on a small card"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-records", type=int, default=400)
    parser.add_argument(
        "--max-tokens-per-record",
        type=int,
        default=8192,
        help="a record that reaches this ceiling has been truncated, and is dropped",
    )
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument(
        "--hub-repo", default=DEFAULT_HUB_REPO, help="the repository auto-resume consults"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline, or prepare the corpus on its own."""
    parser = argparse.ArgumentParser(prog="gotooltrain-pipeline", description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=("run", "prepare-data"),
        help="run the chain, or only measure and write the training corpus",
    )
    add_common(parser)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-corpus", action="store_true")
    parser.add_argument("--no-device", action="store_true")
    parser.add_argument("--no-data", action="store_true")
    parser.add_argument("--no-sft", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--no-dpo", action="store_true")
    parser.add_argument(
        "--extra-sft", nargs=argparse.REMAINDER, default=[], help="passed through to the sft stage"
    )
    parser.add_argument(
        "--extra-dpo", nargs=argparse.REMAINDER, default=[], help="passed through to the dpo stage"
    )
    args = parser.parse_args(argv)

    # The parsed arguments, with the stage switches and the pass-through lists
    # folded in under the names the stage builder reads. Copied rather than
    # splatted-and-overridden, which collides on every duplicated key.
    values = dict(vars(args))
    values.update(
        run_gate=not args.no_gate,
        run_corpus=not args.no_corpus,
        run_device_check=not args.no_device,
        run_data=not args.no_data,
        run_sft=not args.no_sft,
        run_eval=not args.no_eval,
        run_dpo=not args.no_dpo,
        extra_sft=list(args.extra_sft),
        extra_dpo=list(args.extra_dpo),
    )
    options = argparse.Namespace(**values)

    # `auto` is resolved here, in code that is tested, rather than in the
    # notebook cell that used to hold it. Every caller gets the same decision.
    options.extra_sft = resolve_resume(options.extra_sft, repo_id=options.hub_repo)
    options.extra_dpo = resolve_resume(options.extra_dpo, repo_id=options.hub_repo)

    if args.command == "prepare-data":
        # Its own entry point so the corpus can be re-measured without re-running the
        # gate and the download. The measurement is cheap; the pipeline is not.
        prepare_data(
            options.corpus,
            options.dataset,
            max_records=options.max_records,
            max_tokens_per_record=options.max_tokens_per_record,
            tokenizer_name=options.model,
        )
        return 0

    stages = build_stages(options)
    if not stages:
        print("no stages selected; nothing to do")
        return 0

    pipeline = Pipeline(stages)
    pipeline.announce()
    code = pipeline.run(dry_run=args.dry_run)
    print(pipeline.summary())
    if code:
        raise SystemExit(code)
    return 0


if __name__ == "__main__":  # pragma: no cover - the console script is the entry point
    raise SystemExit(main())
