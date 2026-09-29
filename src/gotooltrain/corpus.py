"""Measure a corpus, then decide the training target from the number.

The rule this module exists to enforce: **no arbitrary example count**. "Train on
50k examples" is not a plan; it is a number someone liked. What actually matters is
coverage of the behaviour being taught, so the measurement is of *that*:

* how many distinct Go packages and repositories are represented, because a
  thousand examples drawn from one repository teach one codebase;
* which tools appear and how often, because a catalogue member with no training
  signal is a tool the model will never call;
* how many multi-turn trajectories there are, because single-turn data teaches
  single-turn behaviour and the whole premise is an agent loop;
* the supervised-token share, because completion-only loss on a mostly-prompt
  example spends most of the compute on learning to predict text it is given.

Every one of these has a failure mode that is invisible in a total. A corpus can
hit any target while being useless on all four counts, which is why the report
raises rather than prints a number.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .errors import DatasetError
from .gotools import GO_TOOLS

#: A tool must appear in at least this fraction of trajectories to be considered
#: learned. Below it the model has no signal for when to reach for the tool, and
#: an unused tool is worse than an absent one: it is advertised and never chosen.
MIN_TOOL_COVERAGE: Final[float] = 0.02

#: A tool must appear at least this many times in total, for the same reason.
MIN_TOOL_EXAMPLES: Final[int] = 50

#: Trajectories with more than one tool call are the ones that teach the loop.
MIN_MULTI_TURN_FRACTION: Final[float] = 0.30

#: Below this share of supervised tokens, most of the compute is spent predicting
#: a prompt the model was going to be given anyway.
MIN_SUPERVISED_FRACTION: Final[float] = 0.10

#: A single repository dominating the corpus teaches that repository.
MIN_REPO_FRACTION: Final[float] = 0.05

#: Distinct packages, so the model sees varied APIs rather than one codebase.
MIN_DISTINCT_PACKAGES: Final[int] = 200


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One measured training trajectory.

    ``tools`` is the ordered list of tool names called, so a turn count can be
    derived rather than trusted from a metadata field.
    """

    trajectory_id: str
    repository: str
    package: str
    #: Tools called, in order; repeats are kept because a second call is a decision.
    tools: tuple[str, ...] = ()
    #: Tokens the model is asked to predict.
    supervised_tokens: int = 0
    #: Tokens in the whole rendered example.
    total_tokens: int = 0
    #: Reasoning spans present, i.e. whether thinking is trained on this record.
    has_thinking: bool = False
    has_images: bool = False

    @property
    def turns(self) -> int:
        """How many times the model produced output."""
        return max(1, len(self.tools))

    @property
    def is_multi_turn(self) -> bool:
        """Whether the agent loop is exercised at all."""
        return len(self.tools) >= 2

    @property
    def supervised_fraction(self) -> float:
        """Share of the example that carries loss."""
        if self.total_tokens <= 0:
            return 0.0
        return self.supervised_tokens / self.total_tokens


@dataclass(frozen=True, slots=True)
class Finding:
    """One measured weakness, with the number that produced it."""

    name: str
    detail: str
    measured: str
    required: str

    def to_record(self) -> dict[str, str]:
        """Serialisable form."""
        return {
            "name": self.name,
            "detail": self.detail,
            "measured": self.measured,
            "required": self.required,
        }


@dataclass(slots=True)
class CorpusReport:
    """What a corpus actually contains, and whether it can teach the behaviour."""

    trajectories: int
    repositories: int
    packages: int
    tool_counts: dict[str, int] = field(default_factory=dict)
    tool_trajectory_coverage: dict[str, float] = field(default_factory=dict)
    multi_turn_fraction: float = 0.0
    supervised_fraction: float = 0.0
    largest_repository_fraction: float = 0.0
    thinking_fraction: float = 0.0
    image_fraction: float = 0.0
    total_tokens: int = 0
    supervised_tokens: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def is_adequate(self) -> bool:
        """Whether the corpus clears every threshold."""
        return not self.findings

    def to_record(self) -> dict[str, Any]:
        """Serialisable report for a build log or a decision record."""
        return {
            "trajectories": self.trajectories,
            "repositories": self.repositories,
            "packages": self.packages,
            "total_tokens": self.total_tokens,
            "supervised_tokens": self.supervised_tokens,
            "supervised_fraction": round(self.supervised_fraction, 4),
            "multi_turn_fraction": round(self.multi_turn_fraction, 4),
            "thinking_fraction": round(self.thinking_fraction, 4),
            "image_fraction": round(self.image_fraction, 4),
            "largest_repository_fraction": round(self.largest_repository_fraction, 4),
            "tool_counts": dict(sorted(self.tool_counts.items())),
            "tool_trajectory_coverage": {
                name: round(value, 4)
                for name, value in sorted(self.tool_trajectory_coverage.items())
            },
            "adequate": self.is_adequate,
            "findings": [f.to_record() for f in self.findings],
        }


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def measure(trajectories: Iterable[Trajectory]) -> CorpusReport:
    """Measure a corpus and state what is missing.

    The thresholds are constants, not parameters. Making them configurable would
    mean the target gets set to whatever the corpus happens to contain, which is
    the thing this module exists to prevent.
    """
    items = list(trajectories)
    total = len(items)
    tool_calls: Counter[str] = Counter()
    tool_trajectories: Counter[str] = Counter()
    repositories: Counter[str] = Counter()

    for item in items:
        repositories[item.repository] += 1
        # Counting per trajectory, not per call, is what makes this a coverage
        # measure: a model that calls grep 50 times in one trajectory has seen it
        # once, not 50 times.
        for name in set(item.tools):
            tool_trajectories[name] += 1
        tool_calls.update(item.tools)

    report = CorpusReport(
        trajectories=total,
        repositories=len(repositories),
        packages=len({f"{i.repository}/{i.package}" for i in items}),
        tool_counts=dict(tool_calls),
        tool_trajectory_coverage={
            name: _share(tool_trajectories[name], total) for name in sorted(tool_calls)
        },
        multi_turn_fraction=_share(sum(1 for i in items if i.is_multi_turn), total),
        supervised_fraction=_mean([i.supervised_fraction for i in items]),
        largest_repository_fraction=(max(repositories.values()) / total if repositories else 0.0),
        thinking_fraction=_share(sum(1 for i in items if i.has_thinking), total),
        image_fraction=_share(sum(1 for i in items if i.has_images), total),
        total_tokens=sum(i.total_tokens for i in items),
        supervised_tokens=sum(i.supervised_tokens for i in items),
    )
    report.findings = _findings(report)
    return report


def _findings(report: CorpusReport) -> list[Finding]:
    """Everything that would make this corpus insufficient, named precisely.

    Includes tools that were never used at all, which is the failure this whole
    exercise was built to catch: a catalogue the model never saw exercised.
    """
    findings: list[Finding] = []
    catalogue = sorted(t.name for t in GO_TOOLS)

    for name in catalogue:
        calls = report.tool_counts.get(name, 0)
        coverage = report.tool_trajectory_coverage.get(name, 0.0)
        if calls < MIN_TOOL_EXAMPLES or coverage < MIN_TOOL_COVERAGE:
            findings.append(
                Finding(
                    name=f"tool_undertrained:{name}",
                    detail=(
                        f"{name} appears in {calls} call(s) across "
                        f"{coverage:.1%} of trajectories. A tool advertised but never "
                        "demonstrated is one the model will not reach for."
                    ),
                    measured=f"{calls} calls, {coverage:.1%} of trajectories",
                    required=(
                        f">= {MIN_TOOL_EXAMPLES} calls, >= {MIN_TOOL_COVERAGE:.0%} of trajectories"
                    ),
                )
            )

    unknown = sorted(set(report.tool_counts) - set(catalogue))
    if unknown:
        findings.append(
            Finding(
                name="tools_outside_catalogue",
                detail=(
                    f"trajectories call {unknown}, which are not in the catalogue. Training on "
                    "a tool the model will never be offered teaches it to expect affordances "
                    "it does not have."
                ),
                measured=f"{unknown}",
                required="every called tool is in the catalogue",
            )
        )

    if report.multi_turn_fraction < MIN_MULTI_TURN_FRACTION:
        findings.append(
            Finding(
                name="too_few_multi_turn",
                detail=(
                    "Single-turn examples teach a single turn. The model is being trained for "
                    "an agent loop that reads a failure and acts on it."
                ),
                measured=f"{report.multi_turn_fraction:.1%} multi-turn",
                required=f">= {MIN_MULTI_TURN_FRACTION:.0%}",
            )
        )

    if report.supervised_fraction < MIN_SUPERVISED_FRACTION:
        findings.append(
            Finding(
                name="too_few_supervised_tokens",
                detail=(
                    "Most of the tokens are prompt. With completion-only loss the rest of the "
                    "compute is spent predicting text the model was handed anyway."
                ),
                measured=f"{report.supervised_fraction:.1%} supervised",
                required=f">= {MIN_SUPERVISED_FRACTION:.0%}",
            )
        )

    if report.largest_repository_fraction > 1 - MIN_REPO_FRACTION * 4:
        findings.append(
            Finding(
                name="one_repository_dominates",
                detail=(
                    "A corpus dominated by one repository teaches that codebase. The model "
                    "memorises its APIs instead of learning to read a new one."
                ),
                measured=f"largest repository is {report.largest_repository_fraction:.1%}",
                required=f"<= {1 - MIN_REPO_FRACTION * 4:.0%}",
            )
        )

    if report.packages < MIN_DISTINCT_PACKAGES:
        findings.append(
            Finding(
                name="too_few_packages",
                detail=(
                    "Package variety is what forces the model to read documentation and code "
                    "rather than recall a fixed API."
                ),
                measured=f"{report.packages} distinct package(s)",
                required=f">= {MIN_DISTINCT_PACKAGES}",
            )
        )

    return findings


def summarise_targets(report: CorpusReport) -> list[dict[str, Any]]:
    """The concrete next-data targets implied by what is missing.

    Deliberately counts, not fractions: "put edit_file into 160 more trajectories"
    is actionable, "improve balance" is not.

    The two thresholds fail differently and need different remedies, so the target
    says which one is binding:

    * **too few calls** -- the tool is rare everywhere. Add records that use it.
    * **too little coverage** -- the tool appears in only a sliver of the corpus.
      Spread it into the records that already exist; adding more records without
      spreading it would not move the fraction.
    """
    targets: list[dict[str, Any]] = []
    have = report.trajectories
    for name in sorted(t.name for t in GO_TOOLS):
        calls = report.tool_counts.get(name, 0)
        coverage = report.tool_trajectory_coverage.get(name, 0.0)
        calls_short = calls < MIN_TOOL_EXAMPLES
        coverage_short = coverage < MIN_TOOL_COVERAGE
        if not calls_short and not coverage_short:
            continue

        if calls_short and not coverage_short:
            # Enough records use it, just not often enough: more calls, no new data.
            targets.append(
                {
                    "tool": name,
                    "blocked_by": "calls",
                    "missing_calls": MIN_TOOL_EXAMPLES - calls,
                    "current_coverage": round(coverage, 4),
                    "trajectories_needed": 0,
                    "records_to_extend": 0,
                }
            )
        elif coverage_short and not calls_short:
            # Calls are fine; the tool is confined to too few records. Spreading it
            # into the existing corpus is what moves this, so the count is of
            # records to *change*, not to add.
            want = int(-(-have * MIN_TOOL_COVERAGE))
            currently = round(coverage * have)
            targets.append(
                {
                    "tool": name,
                    "blocked_by": "coverage",
                    "missing_calls": 0,
                    "current_coverage": round(coverage, 4),
                    "trajectories_needed": 0,
                    "records_to_extend": max(0, want - currently),
                }
            )
        else:
            # Neither threshold is met: new records that use the tool, at the rate
            # it is already used at, or at the floor if it is not used at all.
            calls_needed = MIN_TOOL_EXAMPLES - calls
            per_trajectory = max(MIN_TOOL_COVERAGE, coverage)
            targets.append(
                {
                    "tool": name,
                    "blocked_by": "calls_and_coverage",
                    "missing_calls": calls_needed,
                    "current_coverage": round(coverage, 4),
                    "trajectories_needed": int(-(-calls_needed // per_trajectory)),
                    "records_to_extend": 0,
                }
            )

    if report.multi_turn_fraction < MIN_MULTI_TURN_FRACTION:
        have_multi = round(report.multi_turn_fraction * report.trajectories)
        want_multi = int(MIN_MULTI_TURN_FRACTION * report.trajectories)
        targets.append(
            {
                "tool": None,
                "blocked_by": "multi_turn",
                "missing_calls": 0,
                "current_coverage": round(report.multi_turn_fraction, 4),
                "trajectories_needed": max(0, want_multi - have_multi),
                "records_to_extend": 0,
            }
        )

    return sorted(targets, key=lambda t: (-t["trajectories_needed"], t["tool"] or ""))


def read_trajectories(path: str | Path) -> list[Trajectory]:
    """Read measured trajectories from JSONL, validating the fields we rely on."""
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"corpus file not found: {source}")
    out: list[Trajectory] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source}:{number} is not valid JSON: {exc}") from exc
        try:
            out.append(
                Trajectory(
                    trajectory_id=str(record["id"]),
                    repository=str(record["repository"]),
                    package=str(record["package"]),
                    tools=tuple(str(t) for t in record.get("tools", [])),
                    supervised_tokens=int(record.get("supervised_tokens", 0)),
                    total_tokens=int(record.get("total_tokens", 0)),
                    has_thinking=bool(record.get("has_thinking", False)),
                    has_images=bool(record.get("has_images", False)),
                )
            )
        except KeyError as exc:
            raise DatasetError(f"{source}:{number} is missing field {exc}") from exc
    return out
