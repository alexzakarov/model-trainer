"""Command-line entry point for turning an evaluation into training data.

``mine``     judged trajectories -> supervised records plus a corpus report
``measure``  a corpus file -> what it can actually teach, and what it cannot
``tasks``    a Go repository -> task records, with the already-passing ones removed

Thin by design: the logic lives in :mod:`gotooltrain.sftdata`,
:mod:`gotooltrain.corpus` and :mod:`gotooltrain.tasks`, so everything here is
testable without a subprocess.

Two of these refuse to pretend. ``measure`` exits non-zero when the corpus is
inadequate, because a build step that reports a number and returns success is a
build step whose number will be ignored. ``tasks`` drops packages the base model
already passes, because a task that is already solved measures nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .corpus import read_trajectories, summarise_targets
from .errors import ToolTrainError
from .evalrun import RunConfig
from .evalstore import ResultStore, sha256_text
from .gocorpus import (
    admit,
    dapt_records,
    deduplicate,
    download_splits,
    measure_dapt,
    read_splits,
    unit_test_messages,
)
from .reward import build_preferences, dpo_example
from .sftdata import MIN_SCORE, mine
from .tasks import (
    GoTask,
    build_task_from_package,
    harvest_packages,
    task_works_now,
    validate_task,
    write_tasks,
)
from .template import install_template, load_template_source

#: The base model whose tokenizer defines the token counts. Pinned rather than
#: taken from the run, because the corpus is measured in the tokens the trainer
#: will produce, and a different tokenizer measures a different corpus.
DEFAULT_TOKENIZER: str = "Qwen/Qwen3.5-4B"


def _tokenizer(name: str) -> Any:  # noqa: ANN401 - transformers is an optional dependency
    """Load the base tokenizer with this repository's chat template installed.

    Imported lazily so the ``measure`` and ``tasks`` commands do not need
    transformers installed: measuring a corpus file is a pure-data operation.
    """
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ToolTrainError(
            "the 'mine' command needs transformers to tokenize; install the 'train' extra"
        ) from exc
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    return install_template(tokenizer, load_template_source())


def _config(args: argparse.Namespace) -> RunConfig:
    """The run identity the mined records must be keyed to."""
    source = load_template_source()
    return RunConfig(
        model_id=args.model,
        model_revision=args.revision,
        dataset_version=args.dataset_version,
        seed=args.seed,
        decode_params={"temperature": 0.0},
        n_samples=args.n_samples,
        max_turns=args.max_turns,
        template_sha=sha256_text(source),
    )


def _load_tasks_file(path: str) -> list[dict[str, Any]]:
    """Read the task list that was evaluated, refusing a file without ids."""
    source = Path(path)
    if not source.is_file():
        raise ToolTrainError(f"task file not found: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        decoded = json.loads(text)
        if not isinstance(decoded, list):
            raise ToolTrainError(f"{source} must hold a list of task objects")
        records = list(decoded)
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "id" not in record:
            raise ToolTrainError(f"{source}: task {index} has no 'id'")
    return records


def cmd_mine(args: argparse.Namespace) -> int:
    """Write the supervised records a judged run supports, and the report.

    The report is written whatever the outcome, including when nothing was
    admitted: an empty result with a reason is a finding, and a command that
    silently wrote an empty file would hide it.
    """
    tokenizer = _tokenizer(args.tokenizer)
    config = _config(args)
    store = ResultStore(args.store)
    tasks = _load_tasks_file(args.tasks)

    records, report = mine(
        config,
        tasks,
        store,
        tokenizer,
        min_score=args.min_score,
        judged=not args.execution_only,
    )

    out = Path(args.out)
    write_jsonl_records(out, records)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_record(), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"mined {report.admitted} of {report.considered} trajectory/ies from "
        f"{report.tasks} task(s) -> {out}",
        file=sys.stderr,
    )
    if report.refusals:
        for reason, count in sorted(report.refusals.items()):
            print(f"  refused: {reason} = {count}", file=sys.stderr)

    if report.admitted == 0:
        # Not an error -- a run where the model solved nothing is a real result --
        # but returning success would let a pipeline train on an empty file.
        print(
            "no trajectory met the admission rules, so no training data was written. "
            "See the report for which rule each one broke.",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    """Report what a corpus can teach, and fail when it cannot teach the catalogue."""
    from .corpus import measure

    trajectories = read_trajectories(args.corpus)
    report = measure(trajectories)
    print(json.dumps(report.to_record(), indent=2, sort_keys=True))
    if not report.is_adequate:
        print("corpus is not adequate:", file=sys.stderr)
        for finding in report.findings:
            print(f"  {finding.name}: {finding.detail}", file=sys.stderr)
        targets = summarise_targets(report)
        if targets:
            print("next data targets:", file=sys.stderr)
            for target in targets:
                print(f"  {json.dumps(target, sort_keys=True)}", file=sys.stderr)
        return 1
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    """Build task records from a Go repository, dropping the already-passing ones.

    A package whose tests already pass is not a task: the base model solves it
    without help, so it measures nothing and would inflate every score. Checking
    costs a ``go test`` per package, which is the price of a corpus that means
    something.
    """
    repository = Path(args.repository)
    if not repository.is_dir():
        raise ToolTrainError(f"repository not found: {repository}")
    packages = harvest_packages(repository, limit=args.limit)
    if not packages:
        raise ToolTrainError(f"no Go packages found under {repository}")

    tasks: list[GoTask] = []
    already_passing = 0
    for package in packages:
        built = build_task_from_package(
            repository,
            repository=args.repository_name,
            package=package,
            prompt=args.prompt.format(package=package),
        )
        # The fixture is resolved against the repository root, exactly as the
        # evaluator will resolve it; validating here means a corpus cannot contain a
        # task whose fixture is missing.
        validate_task(built, repository)
        if args.require_failing and task_works_now(repository, built):
            already_passing += 1
            continue
        tasks.append(built)

    written = write_tasks(args.out, tasks) if tasks else 0
    print(
        f"{written} task(s) from {len(packages)} package(s); "
        f"{already_passing} dropped as already passing",
        file=sys.stderr,
    )
    if not tasks:
        print(
            "every package was already passing, so no task was written. A task needs a "
            "failing test to be worth asking for; author one, or pass --require-failing "
            "off deliberately to keep them anyway.",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_go_pairs(args: argparse.Namespace) -> int:
    """Import the Go corpus: write DAPT records, optional test prompts, and a report.

    Licence screening happens here rather than in a note: the two repositories whose
    licences need a decision are refused unless ``--include-review`` says otherwise,
    and any repository outside the documented ten is refused outright.
    """
    if args.download_to:
        sources = download_splits(args.download_to)
        print(f"downloaded {len(sources)} split(s) to {args.download_to}", file=sys.stderr)
    elif args.source:
        sources = [Path(p) for p in args.source]
    else:
        # No input at all is a usage mistake, not an empty corpus: reporting it as
        # "0 files" would read as a finding about the data.
        raise ToolTrainError(
            "no corpus source given; pass --source <split.json> (repeatable) or "
            "--download-to <dir> to fetch the published splits"
        )

    pairs = read_splits(sources)
    kept, refused = admit(pairs, include_review=args.include_review)
    kept, duplicates_removed = deduplicate(kept)
    report = measure_dapt(kept, refused=refused, duplicates_removed=duplicates_removed)

    write_jsonl_records(Path(args.out), dapt_records(kept))
    if args.unit_tests_out:
        write_jsonl_records(
            Path(args.unit_tests_out),
            [
                {
                    "messages": unit_test_messages(p),
                    "metadata": {
                        "repository": p.repository,
                        "path": p.code_path,
                        "commit": p.code_commit,
                        "sha256": p.sha256,
                    },
                }
                for p in kept
            ],
        )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report.to_record(), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"{report.files} source file(s) from {report.repositories} repository/ies -> {args.out}",
        file=sys.stderr,
    )
    for refusal in refused:
        print(f"  refused {refusal.repository}: {refusal.reason}", file=sys.stderr)
    if not report.is_adequate:
        for finding in report.findings:
            print(
                f"  inadequate: {finding.name} ({finding.measured}, needs {finding.required})",
                file=sys.stderr,
            )
        return 1
    return 0


def write_jsonl_records(path: Path, records: Sequence[dict[str, Any]]) -> int:
    """Write records as JSONL, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def cmd_preferences(args: argparse.Namespace) -> int:
    """Turn a run's pass@N samples into preference pairs, with a report.

    Execution decides the reward, so no judge is consulted: the same run that
    graded the model also ranks its own attempts, and a pair only exists when one
    sample's declared verification passed and another's failed. Nothing is
    tokenized here, so this needs neither the tokenizer nor a GPU.
    """
    config = _config(args)
    store = ResultStore(args.store)
    tasks = _load_tasks_file(args.tasks)

    pairs, report = build_preferences(config, tasks, store, min_margin=args.min_margin)
    write_jsonl_records(Path(args.out), [dpo_example(pair) for pair in pairs])
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report.to_record(), indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"{report.pairs} preference pair(s) from {report.tasks} task(s) -> {args.out}",
        file=sys.stderr,
    )
    for reason, count in sorted(report.refusals.items()):
        print(f"  refused: {reason} = {count}", file=sys.stderr)
    if not pairs:
        print(
            "no task produced a contrast. A preference needs one sample whose "
            "verification passed and another whose failed; raise --n-samples above 1, "
            "and check that each task declares a 'verification' tool.",
            file=sys.stderr,
        )
        return 2
    return 0


def _add_run_identity(parser: argparse.ArgumentParser) -> None:
    """The arguments that key a run, so records mined from it are traceable.

    Mirrors evalcli's common options rather than importing it: this is a different
    command with a different job, and the alternative is sharing a private helper
    across two CLIs.
    """
    parser.add_argument("--store", required=True, help="result store root")
    parser.add_argument("--tasks", required=True, help="the task file that was evaluated")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``."""
    parser = argparse.ArgumentParser(prog="gotooltrain-data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    mine_parser = sub.add_parser("mine", help="judged trajectories -> supervised records")
    mine_parser.add_argument("--store", required=True, help="result store root")
    mine_parser.add_argument("--tasks", required=True, help="the task file that was evaluated")
    mine_parser.add_argument("--out", required=True, help="dataset JSONL to write")
    mine_parser.add_argument("--report", required=True, help="report JSON to write")
    mine_parser.add_argument("--model", required=True)
    mine_parser.add_argument("--revision", required=True)
    mine_parser.add_argument("--dataset-version", required=True)
    mine_parser.add_argument("--seed", type=int, default=0)
    mine_parser.add_argument("--n-samples", type=int, default=1)
    mine_parser.add_argument("--max-turns", type=int, default=8)
    mine_parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    mine_parser.add_argument(
        "--min-score",
        type=float,
        default=MIN_SCORE,
        help="lowest judge score admitted as training data (default: 1.0)",
    )
    mine_parser.add_argument(
        "--execution-only",
        action="store_true",
        help="admit on exit codes alone, for a run that deliberately had no judge",
    )
    mine_parser.set_defaults(func=cmd_mine)

    measure_parser = sub.add_parser("measure", help="report what a corpus can teach")
    measure_parser.add_argument("--corpus", required=True, help="corpus JSONL to measure")
    measure_parser.set_defaults(func=cmd_measure)

    tasks_parser = sub.add_parser("tasks", help="build task records from a Go repository")
    tasks_parser.add_argument("--repository", required=True, help="repository checkout")
    tasks_parser.add_argument("--repository-name", required=True, help="name in the records")
    tasks_parser.add_argument("--out", required=True, help="task JSONL to write")
    tasks_parser.add_argument("--limit", type=int, default=0, help="cap packages, 0 for all")
    tasks_parser.add_argument(
        "--prompt",
        default="Make the tests in {package} pass.",
        help="prompt template; {package} is substituted",
    )
    tasks_parser.add_argument(
        "--allow-passing",
        dest="require_failing",
        action="store_false",
        help="keep already-passing packages anyway (they measure nothing)",
    )
    # Already-passing packages are dropped by default: a task the base model solves
    # measures nothing. Keeping them is a deliberate, named choice, not a default.
    tasks_parser.set_defaults(func=cmd_tasks, require_failing=True)

    go_parser = sub.add_parser("go-pairs", help="import and screen the Go DAPT corpus")
    go_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="a Go-UT-Bench split file (repeatable); ignored with --download-to",
    )
    go_parser.add_argument(
        "--download-to",
        default=None,
        help="fetch every published split into this directory first",
    )
    go_parser.add_argument("--out", required=True, help="DAPT JSONL to write")
    go_parser.add_argument("--report", required=True, help="report JSON to write")
    go_parser.add_argument(
        "--unit-tests-out",
        default=None,
        help="optional JSONL of unit-test-generation prompt records",
    )
    go_parser.add_argument(
        "--include-review",
        action="store_true",
        help="also admit the repositories whose licence needs a decision (BUSL-1.1, LGPL-3.0)",
    )
    go_parser.set_defaults(func=cmd_go_pairs)

    pref_parser = sub.add_parser(
        "preferences", help="turn a run's samples into execution-rewarded preference pairs"
    )
    _add_run_identity(pref_parser)
    pref_parser.add_argument("--out", required=True, help="preference JSONL to write")
    pref_parser.add_argument("--report", required=True, help="report JSON to write")
    pref_parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="smallest reward gap accepted as a preference (the reward is binary)",
    )
    pref_parser.set_defaults(func=cmd_preferences)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ToolTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
