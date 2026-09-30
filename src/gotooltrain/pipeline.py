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
import pathlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import DatasetError

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
                self.emit(f"[kuru] {stage.name}: {' '.join(stage.command)}")
                continue
            code = self._one(stage)
            if code != 0 and stage.fatal:
                self.emit(
                    f"\nPIPELINE STOPPED at {stage.name} ({code}). "
                    "Its log and last lines are above."
                )
                return code
        return 0

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


def build_stages(options: argparse.Namespace) -> list[Stage]:
    """The chain, in the order it has to happen, from the options given."""
    sft = options.sft_output
    stages: list[Stage] = []
    if options.run_gate:
        stages.append(Stage("quality-gate", ("python", "-m", "pytest", "-q", "-x", "-rs")))
    if options.run_corpus:
        stages.append(
            Stage(
                "corpus",
                (
                    "python",
                    "-m",
                    "gotooltrain.data",
                    "go-pairs",
                    "--source",
                    str(options.source_split),
                    "--out",
                    str(options.dapt),
                    "--report",
                    str(options.dapt_report),
                    "--unit-tests-out",
                    str(options.corpus),
                ),
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
                    "gotooltrain.data",
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
                    "--gradient-checkpointing",
                    *options.extra_dpo,
                ),
            )
        )
    return stages


def add_common(parser: argparse.ArgumentParser) -> None:
    """Options that describe the run rather than one stage."""
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--sft-output", default="runs/colab-sft")
    parser.add_argument("--dpo-output", default="runs/colab-dpo")
    parser.add_argument("--dataset", default="data/sft.jsonl", type=pathlib.Path)
    parser.add_argument("--corpus", default="data/go-unit-tests.jsonl", type=pathlib.Path)
    parser.add_argument(
        "--source-split", default="data/go-ut-bench/train_data.json", type=pathlib.Path
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-records", type=int, default=400)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(prog="gotooltrain-pipeline", description=__doc__)
    add_common(parser)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--no-corpus", action="store_true")
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
        run_sft=not args.no_sft,
        run_eval=not args.no_eval,
        run_dpo=not args.no_dpo,
        extra_sft=list(args.extra_sft),
        extra_dpo=list(args.extra_dpo),
    )
    options = argparse.Namespace(**values)

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
