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
import importlib.util
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
                self.emit(f"[dry] {stage.name}: {' '.join(str(part) for part in stage.command)}")
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
        # str() because a stage may carry a Path; the command is printed for a reader,
        # and Popen does not need the conversion itself.
        self.emit(f"$ {' '.join(str(part) for part in stage.command)}")
        code = self.stream(stage.command, log_path)
        seconds = time.monotonic() - started
        self.results.append(StageResult(stage.name, code, seconds, log_path))
        if code != 0:
            self.emit(f"--- {log_path} (last 30 lines) ---")
            for line in tail(log_path, 30):
                self.emit(f"  {line}")
        else:
            self.emit(f"--- {stage.name} bitti ({seconds:.1f}s) ---")
        return code

    def stream(self, command: Sequence[str], log_path: pathlib.Path) -> int:
        """Run a command, printing each line as it arrives and keeping them all."""
        return stream_command(command, log_path, self.emit)

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


def stream_command(
    command: Sequence[str], log_path: pathlib.Path, emit: Callable[[str], None]
) -> int:
    """Run a command, printing each line as it arrives and keeping them all.

    Module level rather than only a method because the evaluation stage runs commands
    inside itself -- the model server it starts has to be alive while its samples are
    generated -- and a second copy of this loop would be a second place for the
    buffering, the log and the exit code to be got wrong.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered, so a command that prints before its first step still says so.
    argv = (sys.executable, "-u", *command[1:]) if command[0] == "python" else tuple(command)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        # Every element comes from this module's own stage table or from the operator's
        # command line, never from a dataset or a downloaded file -- which is the
        # distinction ruff's S603 is asking about.
        process = subprocess.Popen(  # noqa: S603 - argv is constructed here, not parsed from data
            argv,
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
            emit(line.rstrip("\n"))
            log.write(line)
        return process.wait()


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


def _check_token(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Refuse a publishing run without a token, before anything is downloaded.

    The trainer already refuses on a missing token before it loads the model, which is
    the expensive part -- but by then the corpus has been fetched and measured. A
    token is knowable in the first second, so it is checked in the first second.

    Only a run that publishes needs one: with no ``--hub-repo-id``, or with
    ``--hub-dry-run``, the environment is never asked for it.
    """
    import os

    extras = [*options.extra_sft, *options.extra_dpo]
    if "--hub-repo-id" not in extras:
        emit("token: no Hub publication configured, so none is needed")
        return 0
    if "--hub-dry-run" in extras:
        emit("token: --hub-dry-run is set, so the upload is rehearsed and no token is needed")
        return 0
    if not os.environ.get("HF_TOKEN"):
        raise DatasetError(
            "HF_TOKEN is not set, and this run publishes to the Hugging Face Hub. Set "
            "it before the run starts -- as a Colab secret, or in the environment -- or "
            "pass --hub-dry-run to rehearse the schedule without uploading. Finding "
            "this out after the corpus download is minutes spent on a run that could "
            "not have finished."
        )
    emit("token: HF_TOKEN is set")
    return 0


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


def _served_module(command: Sequence[str]) -> str | None:
    """The module a ``python -m <module>`` serve command names, if it names one."""
    parts = [str(part) for part in command]
    if "-m" in parts:
        index = parts.index("-m")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _introduce_failure(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Break a passing package so the evaluation has something to ask for.

    Refuses when it breaks nothing. The alternative -- continuing to the task builder
    with a healthy checkout -- produces an empty task file and stops two stages later
    with a message about the corpus rather than about the cause.
    """
    from .tasks import harvest_packages, introduce_failures

    packages = harvest_packages(options.repo_dir, limit=options.task_limit or 10_000)
    if not packages:
        raise DatasetError(
            f"no Go packages under {options.repo_dir}, so there is nothing to build a task from."
        )
    emit(f"mutate: {len(packages)} package(s) to consider, breaking up to {options.mutations}")
    broken = introduce_failures(options.repo_dir, packages, limit=options.mutations)
    if not broken:
        raise DatasetError(
            f"none of the {len(packages)} package(s) under {options.repo_dir} could be made "
            "to fail. A task is a package whose tests fail, so there is nothing to build "
            "one from: the evaluation would produce no samples, and no samples means no "
            "preference pair and nothing for preference optimisation to learn from."
        )
    emit(f"mutate: broke {broken}")
    return 0


def _clone_repository(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Fetch the repository the evaluation tasks are built from.

    A task needs a repository state to be worked in and a command that decides whether
    the work succeeded; ``gotooltrain-data tasks`` produces both from a checkout. So the
    checkout is the first thing the preference chain needs and the first thing that can
    be missing.

    Shallow: the task builder reads the working tree, not the history, so fetching a
    decade of commits would cost minutes and buy nothing. Already cloned is not an
    error -- re-running the chain is normal -- but a directory that exists and is *not*
    a checkout is, because the task builder would then read whatever is there.
    """
    directory = pathlib.Path(options.repo_dir)
    if directory.is_dir():
        if (directory / ".git").exists():
            emit(f"repository: {directory} already cloned, reusing it")
            return 0
        raise DatasetError(
            f"{directory} exists and is not a git checkout. Point --repo-dir somewhere "
            "else, or remove it: the task builder reads this directory, and reading "
            "whatever happens to be there would build tasks against the wrong source."
        )

    emit(f"repository: cloning {options.repo_url} (shallow) into {directory}")
    return stream_command(
        ("git", "clone", "--depth", "1", options.repo_url, str(directory)),
        pathlib.Path(options.log_root) / "repo.log",
        emit,
    )


def _serve_and_generate(options: argparse.Namespace, emit: Callable[[str], None]) -> int:
    """Serve the trained checkpoint, sample from it, then stop the server.

    The evaluation is the only stage that needs two processes at once: the agent loop
    asks a model server for completions and runs the Go it produces. The server is
    started here, waited for, and stopped in a ``finally`` -- a server left running
    would hold the card for the preference-optimisation stage that follows, which fits
    two whole models on it.

    The server's output goes to its own file, and the tail is printed when it dies.
    That is not a detail: the first attempt at this sent the server's output to
    ``DEVNULL`` and then reported "the server exited 1", which is the entire content of
    the failure and none of the reason.
    """
    import urllib.error
    import urllib.request

    server_log = pathlib.Path(options.log_root) / "server.log"
    # Created here, not assumed: on a fresh run nothing has made the log directory yet,
    # and the failure for that would be a FileNotFoundError inside the branch that was
    # supposed to start a server.
    server_log.parent.mkdir(parents=True, exist_ok=True)

    if options.serve:
        # Checked before the server is started, and before the sampling that depends on
        # it. The install is *not* done here: `pip install vllm` can move torch, and a
        # pipeline that rearranges the environment it is running in is a pipeline whose
        # failures have two possible causes. The message names the command instead.
        module = _served_module(options.serve_command)
        if module and importlib.util.find_spec(module) is None:
            raise DatasetError(
                f"the sampling stage needs `{module}`, which is not installed. Install "
                f"it ({sys.executable} -m pip install {module.split('.')[0]}) or run with "
                "--no-serve against a server you started yourself. Without samples there "
                "are no preference pairs, and preference optimisation would have nothing "
                "to learn from."
            )
    server: Any = None
    try:
        if options.serve:
            # The model is the checkpoint this run just produced, not the base: a
            # preference pair has to come from the policy being improved, or it ranks
            # attempts the run never made.
            serve_argv = [
                *options.serve_command,
                "--model",
                options.sft_output,
                "--port",
                str(options.vllm_port),
                "--served-model-name",
                options.model_name,
                "--max-model-len",
                str(options.context_length),
                "--gpu-memory-utilization",
                "0.85",
            ]
            emit(f"server: starting on port {options.vllm_port}")
            server = subprocess.Popen(  # noqa: S603 - argv comes from this module's options
                serve_argv,
                stdout=server_log.open("w", encoding="utf-8", buffering=1),
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + options.serve_timeout
            last_report = 0.0
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    emit(f"--- {server_log} (last 40 lines) ---")
                    for line in tail(server_log, 40):
                        emit(f"  {line}")
                    raise DatasetError(
                        f"the model server exited {server.returncode} before it was ready. "
                        "Its reason is above; without it there is nothing to sample from, "
                        "and an evaluation with no samples produces no preference pair."
                    )
                try:
                    # The URL is the operator's own --model-url, defaulting to
                    # localhost. It is not built from a dataset.
                    with urllib.request.urlopen(  # noqa: S310 - the operator's own endpoint
                        f"{options.model_url}/models", timeout=2
                    ) as response:
                        emit(f"server: ready ({response.status})")
                        break
                except (urllib.error.URLError, TimeoutError, OSError):
                    # Loading a 4B checkpoint is silent for minutes; silence reads as a
                    # hang, and the next move would be killing a process that was working.
                    if time.monotonic() - last_report > 30:
                        last_report = time.monotonic()
                        # The server may not have printed anything yet, so the last line
                        # is optional. Indexing it blindly would raise from inside the
                        # wait -- turning "still starting" into a crash.
                        said = tail(server_log, 1)
                        emit(f"  ...loading (server log: {said[-1][:110] if said else 'empty'})")
                    time.sleep(5)
            else:
                raise DatasetError(
                    f"the model server was not ready in {options.serve_timeout}s; {server_log} "
                    "has what it said while it tried."
                )
        else:
            emit(f"server: --no-serve, assuming one is already at {options.model_url}")

        code = stream_command(
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
                options.model_name,
                "--revision",
                options.revision,
                "--dataset-version",
                options.dataset_version,
                "--n-samples",
                str(options.n_samples),
                "--model-url",
                options.model_url,
                "--out",
                str(options.queue_file),
                "--fixture",
                options.repo_dir,
                "--workspace",
                str(pathlib.Path(options.log_root) / "workspaces"),
                "--sandbox",
                "local",
                "--allow-local-execution",
            ),
            pathlib.Path(options.log_root) / "eval-queue.log",
            emit,
        )
        if code != 0:
            emit("--- the sampling stage did not finish; no pairs can come from it ---")
        return code
    finally:
        if server is not None and server.poll() is None:
            emit("server: stopping it now, so the next stage has the card to itself")
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - a hung server on a real host
                server.kill()


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

    # A token is knowable in the first second; the download is not. Checking it next
    # means an unset HF_TOKEN costs seconds rather than a corpus fetch.
    if options.run_token_check:
        stages.append(
            Stage(
                "token",
                ("in-process", "is there a token, if this run publishes"),
                run=lambda emit: _check_token(options, emit),
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
    # The task file the evaluation reads has to be built, and it is built from a
    # checkout: a task is a repository state plus the command that decides whether the
    # work succeeded. Nothing produced either before this, so `--tasks` named a file
    # that did not exist and the evaluation could not have started.
    # Cloning and task-building exist only to feed the evaluation, so they follow it:
    # switching the evaluation off is also a decision not to fetch a repository
    # nothing will read.
    if options.run_eval and options.run_repo:
        stages.append(
            Stage(
                "repo",
                ("in-process", f"clone {options.repo_name} for the evaluation tasks"),
                run=lambda emit: _clone_repository(options, emit),
            )
        )
    if options.run_eval and options.run_tasks:
        # The task builder keeps only packages whose tests *fail*, because a package the
        # model already passes measures nothing. A healthy checkout therefore yields no
        # tasks at all -- measured: gin's first three packages all passed, and the
        # builder wrote an empty file, so the chain stopped one stage before the
        # evaluation that needs it.
        #
        # So a failure is introduced first. Only packages that pass are touched, and the
        # change is kept only if it really makes them fail.
        if options.mutate:
            stages.append(
                Stage(
                    "mutate",
                    ("in-process", "introduce a fixable failure so there is a task"),
                    run=lambda emit: _introduce_failure(options, emit),
                )
            )
        stages.append(
            Stage(
                "tasks",
                (
                    "python",
                    "-m",
                    "gotooltrain.datacli",
                    "tasks",
                    "--repository",
                    options.repo_dir,
                    "--repository-name",
                    options.repo_name,
                    "--out",
                    options.tasks,
                    "--limit",
                    str(options.task_limit),
                ),
            )
        )

    if options.run_eval:
        # One stage, because it is the only thing here that needs two processes at
        # once: the agent loop asks a model server for completions and runs the Go it
        # produces. The server is started and stopped inside the stage, so it cannot
        # leak into the preference run that follows and needs the whole card.
        #
        # There is deliberately no `eval-judge` stage. The execution reward reads the
        # tool outcomes out of the store that `queue` writes; `judge` ingests
        # LLM-judged verdicts from a file, and nothing in this chain produces one. A
        # stage that asks for a file nothing writes is a stage that cannot run.
        stages.append(
            Stage(
                "eval",
                ("in-process", "serve the checkpoint, then sample with the agent loop"),
                run=lambda emit: _serve_and_generate(options, emit),
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
                    options.model_name,
                    "--revision",
                    options.revision,
                    "--dataset-version",
                    options.dataset_version,
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
    parser.add_argument("--log-root", default=str(LOG_ROOT), type=pathlib.Path)
    # The evaluation samples from a served checkpoint. The revision keys the store, so
    # a resumed run reuses exactly the samples it already paid for.
    parser.add_argument("--model-name", default="policy", help="the name the server serves under")
    parser.add_argument(
        "--revision", default=None, help="the checkpoint revision keyed in the store"
    )
    parser.add_argument("--dataset-version", default="go-ut-bench-val")
    parser.add_argument("--repo-url", default="https://github.com/gin-gonic/gin.git")
    parser.add_argument("--repo-name", default="gin-gonic/gin")
    parser.add_argument("--repo-dir", default="runs/eval-repo", type=pathlib.Path)
    parser.add_argument("--task-limit", type=int, default=8, help="cap packages, 0 for all")
    parser.add_argument(
        "--mutations",
        type=int,
        default=1,
        help="how many passing packages to break, so the evaluation has tasks to ask for",
    )
    parser.add_argument(
        "--no-mutate",
        action="store_true",
        help="take the repository as it is; only useful for a checkout whose tests already fail",
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--serve-timeout", type=int, default=1800)
    parser.add_argument("--no-serve", action="store_true", help="assume a server is already up")
    parser.add_argument(
        "--serve-command",
        nargs=argparse.REMAINDER,
        default=["python", "-m", "vllm.entrypoints.openai.api_server"],
        help="the server to sample from; its remaining flags are built from the other options",
    )
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
    parser.add_argument("--no-token", action="store_true")
    parser.add_argument("--no-repo", action="store_true")
    parser.add_argument("--no-tasks", action="store_true")
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
        run_token_check=not args.no_token,
        run_repo=not args.no_repo,
        run_tasks=not args.no_tasks,
        mutate=not args.no_mutate,
        run_data=not args.no_data,
        run_sft=not args.no_sft,
        run_eval=not args.no_eval,
        run_dpo=not args.no_dpo,
        extra_sft=list(args.extra_sft),
        extra_dpo=list(args.extra_dpo),
    )
    # The revision names the checkpoint in the store's content key. Left unset it is
    # derived from the context length, so two runs at different lengths never share
    # stored samples -- which would silently reuse results from a model that is not the
    # one being graded.
    if values["revision"] is None:
        values["revision"] = f"sft-{values['context_length']}"
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
