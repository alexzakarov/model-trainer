"""Dataset construction: raw JSONL records in, tokenized examples out.

Kept separate from :mod:`gotooltrain.normalize` so that validation can be run
over a whole corpus (cheap, CPU-only, catches bad data before any GPU time is
committed) independently of tokenization (which needs the tokenizer).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import DatasetError, ValidationError
from .normalize import normalize_conversation
from .schema import Conversation
from .template import (
    RenderedExample,
    SupportsChatTemplate,
    assert_mask_sane,
    build_jsonl_record,
    render_example,
)


@dataclass(slots=True)
class SkippedRecord:
    """A record that failed validation, retained for the error report."""

    index: int
    reason: str


@dataclass(slots=True)
class NormalizationReport:
    """Outcome of a validation pass over a corpus."""

    conversations: list[Conversation] = field(default_factory=list)
    skipped: list[SkippedRecord] = field(default_factory=list)

    @property
    def kept(self) -> int:
        """Number of records that passed validation."""
        return len(self.conversations)

    @property
    def dropped(self) -> int:
        """Number of records rejected, available in ``skipped`` with reasons."""
        return len(self.skipped)

    def summary(self) -> str:
        """One-line human summary for build logs."""
        return f"kept {self.kept}, dropped {self.dropped}"


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSONL file, reporting the offending line on error."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DatasetError(f"dataset not found: {file_path}")
    with file_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{file_path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise DatasetError(f"{file_path}:{line_number} is not a JSON object")
            yield record


def write_jsonl(path: str | Path, rows: Iterable[str]) -> int:
    """Write pre-serialised JSONL lines; returns the number of rows written."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with file_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row + "\n")
            count += 1
    return count


def validate_records(
    records: Iterable[dict[str, Any]],
    *,
    sanitize_reserved_tags: bool = False,
    strict: bool = False,
) -> NormalizationReport:
    """Validate a corpus without tokenizing.

    Run this as a pre-flight gate: it is orders of magnitude cheaper than
    tokenization and surfaces systematic data problems (a broken schema, a
    dangling tool id) that would otherwise appear as a bad training run.
    """
    report = NormalizationReport()
    for index, record in enumerate(records):
        try:
            report.conversations.append(
                normalize_conversation(
                    record.get("messages", []),
                    record.get("tools"),
                    sanitize_reserved_tags=sanitize_reserved_tags,
                )
            )
        except ValidationError as exc:
            if strict:
                raise
            report.skipped.append(SkippedRecord(index=index, reason=str(exc)))
    return report


def build_examples(
    conversations: Iterable[Conversation],
    tokenizer: SupportsChatTemplate,
    *,
    max_length: int | None = None,
    assert_sane: bool = True,
    on_empty: str = "error",
) -> Iterator[RenderedExample]:
    """Tokenize validated conversations into masked training examples.

    ``max_length`` truncates from the right, which is what a left-to-right causal
    model expects but which can cut every assistant turn off the end of a long
    record. Such an example supervises nothing, so it must not reach the trainer
    as silent noise: ``on_empty='error'`` (the default) fails loudly, and
    ``on_empty='skip'`` drops it so a large corpus is not derailed by a few long
    records.
    """
    if on_empty not in ("error", "skip"):
        raise DatasetError(f"on_empty must be 'error' or 'skip', got {on_empty!r}")
    for conversation in conversations:
        example = render_example(tokenizer, conversation, max_length=max_length)
        if assert_sane and example.supervised_tokens == 0:
            if on_empty == "skip":
                continue
            raise DatasetError(
                "example has no supervised tokens after rendering; it is either "
                "assistant-free or was truncated before any assistant turn. Raise "
                "max_length, or pass on_empty='skip' to drop such records."
            )
        if assert_sane:
            assert_mask_sane(example)
        yield example


def export_jsonl(
    path: str | Path,
    conversations: Iterable[Conversation],
    tokenizer: SupportsChatTemplate,
    *,
    max_length: int | None = None,
    on_empty: str = "error",
) -> int:
    """Tokenize and write a JSONL dataset of token ids plus labels.

    Token ids are stored rather than text so training consumes exactly the
    tokens that were validated, and so a resumed run cannot silently re-render
    with a changed template.
    """
    return write_jsonl(
        path,
        (
            build_jsonl_record(e)
            for e in build_examples(
                conversations, tokenizer, max_length=max_length, on_empty=on_empty
            )
        ),
    )
