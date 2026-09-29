"""The Go tool catalogue: schemas, execution policy, and output budgets.

This module is the single definition of what the model can call. Two things
live here because getting them wrong is expensive:

**Execution policy.** Tool results must look identical in training and at
serving. The agent runs behind RTK, which compresses command output, so a model
trained on raw ``go test`` output meets a format it has never seen. Each tool
therefore declares how it is executed, and the mapping is data rather than a
convention someone remembers.

The dangerous case is spelled out as a rule: ``rtk smart`` reduces a file to a
two-line signature summary. Using it for ``read_file`` would train the model to
edit code it has only seen signatures of. It is forbidden, and
:func:`assert_no_forbidden_smart` enforces it.

**Output budgets.** A failing test suite can emit thousands of lines. Every tool
declares a character budget, and :func:`truncate_output` cuts over-budget output
with a visible marker so the model learns that content was elided rather than
that the run was short.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import accumulate
from typing import Any, Final

from .evalstore import canonical_json, sha256_text

CATALOG_VERSION: Final[str] = "go-tools-v2"

#: Marker inserted where output was removed. Visible on purpose: a silent cut
#: teaches the model that a short result means a short run.
TRUNCATION_MARKER: Final[str] = "\n... (+{count} more lines truncated)\n"

#: Characters held back for the marker. Any realistic line count fits, so the
#: truncated result is provably within budget without a final slicing pass.
_MARKER_RESERVE: Final[int] = 64


class Execution(str, Enum):
    """How a tool's result is produced."""

    #: Run through RTK, which strips boilerplate and keeps failures.
    RTK = "rtk"
    #: Passed through unchanged (no RTK rule covers it).
    RAW = "raw"
    #: File read; RTK's smart summary is explicitly rejected.
    RTK_READ = "rtk_read"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One callable tool, as advertised to the model and executed by the harness."""

    name: str
    description: str
    parameters: dict[str, Any]
    execution: Execution
    #: Command template used when execution is RTK-backed (``{path}``-style
    #: placeholders are filled by the harness, never by the model).
    command_template: str
    #: Character budget for the result before truncation.
    output_budget_chars: int

    def to_openai(self) -> dict[str, Any]:
        """Shape accepted by ``normalize_tools`` / the chat template."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def forbidden_smart(self) -> bool:
        """True for tools that read code, where RTK's smart summary is banned.

        ``rtk smart`` keeps only signatures. Wiring it to a tool the model reads
        code with would train the model to edit files it has never seen.
        """
        return self.execution is Execution.RTK_READ


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_STRING = {"type": "string"}


GO_TOOLS: Final[tuple[ToolDefinition, ...]] = (
    ToolDefinition(
        name="read_file",
        description=(
            "Read a Go source file. Returns the file contents verbatim, including "
            "package clause, imports and comments. Use this before editing a file."
        ),
        parameters=_schema(
            {"path": {**_STRING, "description": "Repo-relative path to a Go file"}},
            ["path"],
        ),
        execution=Execution.RTK_READ,
        command_template="rtk read {path}",
        output_budget_chars=20_000,
    ),
    ToolDefinition(
        name="write_file",
        description=(
            "Create or overwrite a file with the given contents. Use for new files "
            "only; prefer edit_file for changes to existing files."
        ),
        parameters=_schema(
            {
                "path": {**_STRING, "description": "Repo-relative path to write"},
                "content": {**_STRING, "description": "Full file contents"},
            },
            ["path", "content"],
        ),
        execution=Execution.RAW,
        command_template="",
        output_budget_chars=20_000,
    ),
    ToolDefinition(
        name="edit_file",
        description=(
            "Replace an exact, unique string in a file. The old_string must appear "
            "exactly once; include enough surrounding context to make it unique."
        ),
        parameters=_schema(
            {
                "path": {**_STRING, "description": "Repo-relative path to edit"},
                "old_string": {
                    **_STRING,
                    "description": "Exact text to replace, including indentation",
                },
                "new_string": {**_STRING, "description": "Replacement text"},
            },
            ["path", "old_string", "new_string"],
        ),
        execution=Execution.RAW,
        command_template="",
        output_budget_chars=4_000,
    ),
    ToolDefinition(
        name="grep",
        description=(
            "Search the repository with a regular expression. Returns matching lines "
            "grouped by file, truncated for length. Searches recursively, so a "
            "directory finds matches in everything beneath it."
        ),
        parameters=_schema(
            {
                "pattern": {**_STRING, "description": "Go regular expression"},
                "path": {
                    **_STRING,
                    "description": "Repo-relative directory to search (default: whole repo)",
                },
            },
            ["pattern"],
        ),
        execution=Execution.RTK,
        # -r is required. Without it rtk passes the path to grep, which refuses a
        # directory outright ("grep: .: Is a directory", exit 2) -- so the tool
        # would have failed on every directory, which is every realistic call.
        command_template="rtk grep -r {pattern} {path}",
        output_budget_chars=8_000,
    ),
    ToolDefinition(
        name="go_build",
        description=(
            "Compile Go packages. Returns compiler errors and warnings only; a "
            "successful build produces no output."
        ),
        parameters=_schema(
            {"pkg": {**_STRING, "description": "Package pattern, e.g. ./... or ./parser"}},
            ["pkg"],
        ),
        execution=Execution.RTK,
        command_template="rtk go build {pkg}",
        output_budget_chars=8_000,
    ),
    ToolDefinition(
        name="go_test",
        description=(
            "Run Go tests for a package pattern. Returns failures and their output; "
            "a passing package is summarised rather than listed line by line."
        ),
        parameters=_schema(
            {"pkg": {**_STRING, "description": "Package pattern, e.g. ./... or ./parser"}},
            ["pkg"],
        ),
        execution=Execution.RTK,
        command_template="rtk go test {pkg}",
        output_budget_chars=12_000,
    ),
    ToolDefinition(
        name="rtk_recall",
        description=(
            "Retrieve the full output that a compressed command elided. When a "
            "tool result ends with '[full output: rtk recall <hash>]', pass that "
            "hash here to see what was cut. A unique prefix is enough. Only useful "
            "immediately after reading a compressed result: the stored output is "
            "overwritten as commands run."
        ),
        parameters=_schema(
            {"hash": {**_STRING, "description": "Hash from a '[full output: ...]' hint"}},
            ["hash"],
        ),
        execution=Execution.RAW,
        command_template="rtk recall --full {hash}",
        output_budget_chars=20_000,
    ),
    ToolDefinition(
        name="go_doc",
        description=(
            "Show documentation for a Go symbol: signature, doc comment, and "
            "available methods. Use before calling an unfamiliar standard library "
            "function."
        ),
        parameters=_schema(
            {"symbol": {**_STRING, "description": "Symbol, e.g. errors.Is or sync.Mutex"}},
            ["symbol"],
        ),
        execution=Execution.RAW,
        command_template="go doc {symbol}",
        output_budget_chars=6_000,
    ),
    ToolDefinition(
        name="go_mod_tidy",
        description=(
            "Run go mod tidy to add missing or remove unused module requirements. "
            "Takes no arguments."
        ),
        parameters=_schema({}, []),
        execution=Execution.RAW,
        command_template="go mod tidy",
        output_budget_chars=4_000,
    ),
)

TOOLS_BY_NAME: Final[dict[str, ToolDefinition]] = {t.name: t for t in GO_TOOLS}


def catalog() -> list[dict[str, Any]]:
    """The catalogue in the shape the data layer and the template consume."""
    return [t.to_openai() for t in GO_TOOLS]


def catalog_sha() -> str:
    """Stable identity of the catalogue, used in the eval fingerprint.

    Changing a tool's description or schema changes tool selection behaviour, so
    results produced under different catalogues must not share a key.
    """
    return sha256_text(
        canonical_json(
            {
                "catalog_version": CATALOG_VERSION,
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "execution": t.execution.value,
                        "output_budget_chars": t.output_budget_chars,
                    }
                    for t in GO_TOOLS
                ],
            }
        )
    )


def get(name: str) -> ToolDefinition:
    """Look up a tool, failing loudly on an unknown name."""
    try:
        return TOOLS_BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown tool {name!r}; known tools: {sorted(TOOLS_BY_NAME)}") from exc


def assert_no_forbidden_smart() -> None:
    """Guard the rule that ``rtk smart`` must never back a tool.

    RTK's smart mode reduces a file to signatures. If it ever reached a tool the
    model reads code with, the model would learn to edit files it has not seen.
    """
    offenders = sorted(t.name for t in GO_TOOLS if "rtk smart" in t.command_template)
    if offenders:
        raise AssertionError(f"tools must not use 'rtk smart': {offenders}")


def commandless_tools() -> list[str]:
    """Tools with no command, i.e. the ones the harness applies to the workspace.

    These are the only tools allowed to omit a ``command_template``. A tool that
    is neither commandless nor has a template is a catalogue bug: the harness
    would have to invent a command, so :func:`assert_catalog_is_executable`
    rejects it.
    """
    return [t.name for t in GO_TOOLS if not t.command_template]


def assert_catalog_is_executable(commandless: Sequence[str]) -> None:
    """Every tool is either a runnable command or a declared workspace mutation.

    Checks the catalogue against the harness's handler table so a new tool cannot
    be added in a state where calling it silently does nothing.
    """
    declared = set(commandless)
    if len(declared) != len(commandless):
        raise AssertionError("duplicate tool names in the catalogue")
    missing = sorted(t.name for t in GO_TOOLS if not t.command_template and t.name not in declared)
    if missing:
        raise AssertionError(
            f"tools have no command template and are not declared commandless: {missing}"
        )


def _affordable_count(costs: Iterable[int], budget: int) -> int:
    """How many leading entries fit in ``budget`` (each cost includes a newline).

    Uses cumulative sums and a binary search, so there is no early-exit branch:
    an empty input naturally yields 0.
    """
    return bisect_right(list(accumulate(costs)), budget)


def truncate_output(text: str, budget_chars: int) -> str:
    """Cut over-budget output, leaving a visible marker.

    Keeps head *and* tail: compiler and test failures sit at the end, so a
    head-only cut would drop exactly the part the model needs. The marker states
    how many lines went missing, so the model learns the output was elided
    rather than that the run was short.
    """
    if budget_chars < 1:
        raise ValueError(f"budget_chars must be >= 1, got {budget_chars}")
    if len(text) <= budget_chars:
        return text
    # A string longer than the budget always has at least one line, so there is
    # no empty-splitlines case to guard.
    lines = text.splitlines()
    # The marker is paid for out of the budget before anything else. Sizing it
    # generously (and bounding the line count) keeps the result provably within
    # budget, so no post-hoc slicing can cut the marker itself away.
    available = budget_chars - _MARKER_RESERVE
    if available <= 0:
        return TRUNCATION_MARKER.format(count=len(lines))[:budget_chars]
    costs = [len(line) + 1 for line in lines]
    head_count = _affordable_count(costs, available // 2)
    tail_budget = available - available // 2
    tail_count = _affordable_count(reversed(costs[head_count:]), tail_budget)
    head = lines[:head_count]
    tail = lines[len(lines) - tail_count :] if tail_count else []
    removed = len(lines) - head_count - tail_count
    result = "".join(f"{line}\n" for line in head)
    result += TRUNCATION_MARKER.format(count=removed)
    result += "".join(f"{line}\n" for line in tail)
    return result
