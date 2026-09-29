"""The Go corpus for domain adaptation, imported and measured.

This is stage 4b: the continued-pretraining half of the pipeline, as opposed to
the agent-trajectory half that :mod:`gotooltrain.sftdata` mines. The two are
measured by different standards and must not be mixed, because they teach
different things.

The source is Go-UT-Bench (arXiv 2511.10868), 5,264 ``{code, unit test}`` pairs
from ten repositories, published with commit hashes so an import can be tied back
to a revision. Two things about it are treated as claims to verify rather than
facts to trust:

**The licence claim.** The dataset is described as drawn from permissively
licensed repositories. Two of the ten are not permissive in the ordinary sense --
Terraform moved to BUSL-1.1 and go-ethereum is LGPL-3.0 -- so the repository list
is carried here with each licence named, and the two that need a decision are
refused unless that decision is made explicitly. An unknown repository is also
refused, so refreshing the dataset cannot quietly add an eleventh one.

**The coverage claim.** "5,264 pairs" says nothing about whether the corpus can
teach Go, which is why :func:`measure_dapt` reports distinct repositories,
packages and duplicate share, and raises findings rather than a total.

Parsing is deliberately strict: a record missing a field, or carrying empty code
or an empty test, is an error rather than a dropped row. A silently dropped row
is a corpus whose size is unknown.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .corpus import Finding
from .errors import DatasetError

#: The repositories Go-UT-Bench draws from, with the licence each is under.
#: Verified against the paper's Table 1 (counts) and each project's own licence
#: file. Used to audit the dataset's "permissively licensed" claim instead of
#: repeating it.
LICENCES: Mapping[str, str] = {
    "gin-gonic/gin": "MIT",
    "gohugoio/hugo": "Apache-2.0",
    "golang/go": "BSD-3-Clause",
    "hashicorp/terraform": "BUSL-1.1",
    "kubernetes/kubernetes": "Apache-2.0",
    "moby/moby": "Apache-2.0",
    "pingcap/tidb": "Apache-2.0",
    "prometheus/prometheus": "Apache-2.0",
    "kserve/kserve": "Apache-2.0",
    "ethereum/go-ethereum": "LGPL-3.0",
}

#: Licences that are source-available or copyleft rather than permissive. Training
#: on them is an operator decision, not an assumption this module makes silently.
REVIEW_REQUIRED: Mapping[str, str] = {
    "hashicorp/terraform": (
        "BUSL-1.1 is source-available, not open source; Terraform moved to it in 2023 "
        "and the dataset's commits may fall after the change"
    ),
    "ethereum/go-ethereum": (
        "LGPL-3.0 is copyleft; whether model training counts as a derivative work is a "
        "legal question, not a technical one"
    ),
}

#: Below this many files a domain-adaptation corpus is too small to move the model;
#: the literature's working band for this kind of adaptation is 1k-10k examples.
MIN_DAPT_FILES: Final[int] = 1000

#: A single repository dominating teaches that repository's idioms.
MAX_REPOSITORY_SHARE: Final[float] = 0.60

#: Distinct repositories. Ten are available and two need a licence decision, so
#: eight is the expected figure; requiring five catches a corpus that has quietly
#: collapsed to one or two projects.
MIN_DAPT_REPOSITORIES: Final[int] = 5

#: Distinct Go packages, so the model sees varied APIs rather than one codebase.
MIN_DAPT_PACKAGES: Final[int] = 200

#: Duplicates are not harmful to a language model the way they are to an eval set,
#: but a high share means the corpus is smaller than it looks.
MAX_DUPLICATE_SHARE: Final[float] = 0.05

_PACKAGE_RE: Final[re.Pattern[str]] = re.compile(r"^\s*package\s+([A-Za-z_]\w*)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class GoPair:
    """One ``{code, unit test}`` pair, with the provenance needed to re-fetch it."""

    sha256: str
    repository: str
    code_path: str
    code: str
    code_commit: str
    test_path: str
    test: str
    test_commit: str

    @property
    def package(self) -> str:
        """The Go package the source declares, or an empty string if it declares none.

        Read from the file rather than from the path: `go` requires the clause, and a
        path convention is a convention, not a guarantee.
        """
        match = _PACKAGE_RE.search(self.code)
        return match.group(1) if match else ""

    @property
    def directory(self) -> str:
        """The directory the file lives in, within its repository."""
        parent = Path(self.code_path).parent.as_posix()
        return parent if parent != "." else "."


@dataclass(frozen=True, slots=True)
class Refusal:
    """A repository excluded from the corpus, and why."""

    repository: str
    reason: str

    def to_record(self) -> dict[str, str]:
        """Serialisable form."""
        return {"repository": self.repository, "reason": self.reason}


@dataclass(slots=True)
class DaptReport:
    """What a Go corpus contains, and whether it can teach the language."""

    files: int = 0
    repositories: int = 0
    packages: int = 0
    bytes: int = 0
    duplicate_share: float = 0.0
    duplicates_removed: int = 0
    largest_repository_share: float = 0.0
    missing_package_clause: int = 0
    repository_counts: dict[str, int] = field(default_factory=dict)
    refused: list[Refusal] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_adequate(self) -> bool:
        """Whether the corpus clears every threshold."""
        return not self.findings

    def to_record(self) -> dict[str, Any]:
        """Serialisable report for a build log or a decision record."""
        return {
            "files": self.files,
            "repositories": self.repositories,
            "packages": self.packages,
            "bytes": self.bytes,
            "duplicate_share": round(self.duplicate_share, 4),
            "duplicates_removed": self.duplicates_removed,
            "largest_repository_share": round(self.largest_repository_share, 4),
            "missing_package_clause": self.missing_package_clause,
            "repository_counts": dict(sorted(self.repository_counts.items())),
            "refused": [r.to_record() for r in self.refused],
            "adequate": self.is_adequate,
            "findings": [f.to_record() for f in self.findings],
        }


DATASET_ID: Final[str] = "Nutanix/GO-UNITTEST-BENCH"

#: The dataset's published splits. Named individually because a corpus built from
#: one split is a corpus built from part of the data, and the report should say
#: which parts it saw.
SPLITS: Final[tuple[str, ...]] = ("train_data.json", "val_data.json", "test_data.json")


def download_splits(destination: str | Path) -> list[Path]:
    """Fetch every published split into ``destination`` and return the local paths.

    Pinned to the dataset id rather than a free URL, and the hub's own cache is
    used, so a second call costs nothing. A missing ``huggingface_hub`` is a setup
    error naming the extra to install, not an empty corpus.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise DatasetError(
            "downloading the Go corpus needs huggingface_hub; install the 'train' extra"
        ) from exc

    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split in SPLITS:
        fetched = hf_hub_download(
            DATASET_ID,
            split,
            repo_type="dataset",
            local_dir=str(target),
        )
        paths.append(Path(fetched))
    return paths


def read_splits(paths: Iterable[str | Path]) -> list[GoPair]:
    """Read and concatenate several splits, keeping their order."""
    pairs: list[GoPair] = []
    for path in paths:
        pairs.extend(read_pairs(path))
    return pairs


def read_pairs(path: str | Path) -> list[GoPair]:
    """Read a Go-UT-Bench split, refusing anything malformed.

    The field names are the dataset's own, including the punctuation and the
    parenthetical, because guessing a tidier name would silently produce an empty
    column instead of an error.
    """
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"Go-UT-Bench split not found: {source}")
    try:
        decoded = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, list):
        raise DatasetError(f"{source} must hold a list of records")

    required = (
        "SHA256",
        "Repository",
        "File path in Repository",
        "Code",
        "Code Commit hash",
        "File Path for Unit Test",
        "Unit Test - (Ground Truth)",
        "Unit Test Commit hash",
    )
    pairs: list[GoPair] = []
    for index, record in enumerate(decoded):
        if not isinstance(record, dict):
            raise DatasetError(f"{source}: record {index} is not an object")
        missing = [key for key in required if key not in record]
        if missing:
            raise DatasetError(f"{source}: record {index} is missing {missing}")
        code = str(record["Code"])
        test = str(record["Unit Test - (Ground Truth)"])
        if not code.strip():
            raise DatasetError(f"{source}: record {index} has empty code")
        if not test.strip():
            raise DatasetError(f"{source}: record {index} has an empty unit test")
        pairs.append(
            GoPair(
                sha256=str(record["SHA256"]),
                repository=str(record["Repository"]),
                code_path=str(record["File path in Repository"]),
                code=code,
                code_commit=str(record["Code Commit hash"]),
                test_path=str(record["File Path for Unit Test"]),
                test=test,
                test_commit=str(record["Unit Test Commit hash"]),
            )
        )
    return pairs


def admit(
    pairs: Iterable[GoPair], *, include_review: bool = False
) -> tuple[list[GoPair], list[Refusal]]:
    """Split pairs by whether their repository's licence permits use.

    ``include_review=False`` (the default) keeps only the repositories whose
    licence is unambiguously permissive. The two that need a decision are refused
    by name, and an unknown repository is refused too -- a refreshed dataset that
    adds an eleventh repository must not be accepted by default.

    The whole list is judged before anything is filtered, so the refusals describe
    the corpus that was offered rather than the part examined before the first
    rejection.
    """
    offered = list(pairs)
    refusals: list[Refusal] = []
    refused: set[str] = set()
    for repository in sorted({pair.repository for pair in offered}):
        if repository not in LICENCES:
            reason = (
                "not one of the ten repositories this corpus is documented to use; "
                "verify its licence and add it deliberately"
            )
        elif repository in REVIEW_REQUIRED and not include_review:
            reason = REVIEW_REQUIRED[repository]
        else:
            continue
        refusals.append(Refusal(repository, reason))
        refused.add(repository)
    return [pair for pair in offered if pair.repository not in refused], refusals


def deduplicate(pairs: Iterable[GoPair]) -> tuple[list[GoPair], int]:
    """Keep the first occurrence of each distinct source file.

    The published splits overlap: the ten repositories' files appear under more
    than one split, so reading all three yields thousands of repeated sources. For
    continued pretraining a repeated file is not a worse file, it is the same file
    counted twice, which spends compute twice and inflates the corpus size.

    First occurrence wins, so the result depends on the input order and not on
    iteration order, and the caller can report how many were removed rather than
    having them disappear.
    """
    seen: set[str] = set()
    unique: list[GoPair] = []
    removed = 0
    for pair in pairs:
        if pair.sha256 in seen:
            removed += 1
            continue
        seen.add(pair.sha256)
        unique.append(pair)
    return unique, removed


def measure_dapt(
    pairs: Sequence[GoPair],
    *,
    refused: Sequence[Refusal] = (),
    duplicates_removed: int = 0,
) -> DaptReport:
    """Measure a Go corpus and state what it cannot teach.

    Every threshold is a constant rather than a parameter, for the same reason the
    agent-corpus thresholds are: a configurable target gets set to whatever the
    corpus happens to contain.
    """
    report = DaptReport(refused=list(refused), duplicates_removed=duplicates_removed)
    if not pairs:
        report.findings = [_empty_finding()]
        return report

    hashes = Counter(p.sha256 for p in pairs)
    repositories = Counter(p.repository for p in pairs)
    directories = {f"{p.repository}/{p.directory}" for p in pairs}
    report.files = len(pairs)
    report.repositories = len(repositories)
    report.packages = len(directories)
    report.bytes = sum(len(p.code.encode("utf-8")) for p in pairs)
    report.duplicate_share = sum(count - 1 for count in hashes.values()) / len(pairs)
    report.largest_repository_share = max(repositories.values()) / len(pairs)
    report.repository_counts = dict(repositories)
    report.missing_package_clause = sum(1 for p in pairs if not p.package)
    report.findings = _dapt_findings(report)
    return report


def _empty_finding() -> Finding:
    return Finding(
        name="no_go_sources",
        detail=(
            "the corpus is empty. An empty corpus is not a small corpus: adaptation on "
            "nothing produces a model that has learned nothing."
        ),
        measured="0 files",
        required=f">= {MIN_DAPT_FILES}",
    )


def _dapt_findings(report: DaptReport) -> list[Finding]:
    findings: list[Finding] = []
    if report.files < MIN_DAPT_FILES:
        findings.append(
            Finding(
                name="too_few_files",
                detail=(
                    "domain adaptation needs enough source to move the model. Below the "
                    "band the literature uses for this kind of adaptation, the change "
                    "cannot be told from noise."
                ),
                measured=f"{report.files} file(s)",
                required=f">= {MIN_DAPT_FILES}",
            )
        )
    if report.repositories < MIN_DAPT_REPOSITORIES:
        findings.append(
            Finding(
                name="too_few_repositories",
                detail=(
                    "one repository teaches one codebase's idioms. Go's variety is the "
                    "point of the domain, not a single project's style."
                ),
                measured=f"{report.repositories} repository/ies",
                required=f">= {MIN_DAPT_REPOSITORIES}",
            )
        )
    if report.largest_repository_share > MAX_REPOSITORY_SHARE:
        findings.append(
            Finding(
                name="one_repository_dominates",
                detail=(
                    "a corpus dominated by one repository teaches that repository. The "
                    "model memorises its APIs instead of learning the language."
                ),
                measured=f"largest repository is {report.largest_repository_share:.1%}",
                required=f"<= {MAX_REPOSITORY_SHARE:.0%}",
            )
        )
    if report.packages < MIN_DAPT_PACKAGES:
        findings.append(
            Finding(
                name="too_few_packages",
                detail=(
                    "package variety is what forces the model to generalise across APIs "
                    "rather than recall one layout."
                ),
                measured=f"{report.packages} distinct package(s)",
                required=f">= {MIN_DAPT_PACKAGES}",
            )
        )
    if report.duplicate_share > MAX_DUPLICATE_SHARE:
        findings.append(
            Finding(
                name="too_many_duplicates",
                detail=(
                    "duplicate source inflates the file count without adding language. The "
                    "corpus is smaller than its size suggests."
                ),
                measured=f"{report.duplicate_share:.1%} duplicates",
                required=f"<= {MAX_DUPLICATE_SHARE:.0%}",
            )
        )
    return findings


def dapt_records(pairs: Sequence[GoPair]) -> list[dict[str, Any]]:
    """One record per source file, ready to be tokenized for continued pretraining.

    Keys are ``text`` and ``metadata``, matching what the training loader expects,
    with the provenance a source file must carry to be traceable to a commit.
    """
    return [
        {
            "text": pair.code,
            "metadata": {
                "repository": pair.repository,
                "path": pair.code_path,
                "commit": pair.code_commit,
                "sha256": pair.sha256,
            },
        }
        for pair in pairs
    ]


def unit_test_messages(pair: GoPair) -> list[dict[str, str]]:
    """A single-turn instruction record asking for the file's unit tests.

    This is the task the source dataset is built for, expressed as a conversation
    the chat template can render. It is deliberately tool-free: the model is asked
    to write a test file, not to run one, so there is no tool result to feed back.
    """
    return [
        {
            "role": "user",
            "content": (
                f"Write Go unit tests for the file below. It is {pair.code_path} from "
                f"{pair.repository}. Reply with the test file only.\n\n"
                f"```go\n{pair.code}\n```"
            ),
        },
        {"role": "assistant", "content": pair.test},
    ]
