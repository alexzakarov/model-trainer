"""Dataset layer: JSONL IO, corpus validation, export."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain import (
    DatasetError,
    ValidationError,
    export_jsonl,
    normalize_conversation,
    read_jsonl,
    validate_records,
    write_jsonl,
)


@pytest.fixture
def record(sample_record: dict[str, Any]) -> dict[str, Any]:
    return sample_record


def test_read_jsonl_round_trip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [json.dumps({"a": 1}), json.dumps({"a": 2})])
    assert [r["a"] for r in read_jsonl(path)] == [1, 2]


def test_read_jsonl_skips_blank_lines(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n\n\n{"a":2}\n', encoding="utf-8")
    assert len(list(read_jsonl(path))) == 2


def test_read_jsonl_reports_the_offending_line(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a":1}\nnot json\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"bad\.jsonl:2"):
        list(read_jsonl(path))


def test_read_jsonl_rejects_non_objects(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("[1,2,3]\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a JSON object"):
        list(read_jsonl(path))


def test_read_jsonl_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="dataset not found"):
        list(read_jsonl(tmp_path / "nope.jsonl"))


def test_validate_records_keeps_and_drops(record) -> None:  # type: ignore[no-untyped-def]
    records = [
        record,
        {"messages": [{"role": "user", "content": "no assistant turn"}], "tools": record["tools"]},
        record,
    ]
    report = validate_records(records)
    assert report.kept == 2
    assert report.dropped == 1
    assert report.skipped[0].index == 1
    assert "dropped 1" in report.summary()


def test_validate_records_strict_raises() -> None:
    records = [{"messages": [{"role": "user", "content": "x"}], "tools": []}]
    with pytest.raises(ValidationError):
        validate_records(records, strict=True)


def test_export_writes_token_ids_and_labels(tmp_path: pathlib.Path, qwen_tokenizer, record) -> None:  # type: ignore[no-untyped-def]
    conv = normalize_conversation(record["messages"], record["tools"])
    out = tmp_path / "train.jsonl"
    count = export_jsonl(out, [conv], qwen_tokenizer)
    assert count == 1

    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["format_version"] == "anthropic-tools-v1"
    assert len(row["input_ids"]) == len(row["labels"]) == len(row["assistant_mask"])
    assert any(label != -100 for label in row["labels"])
    # Token ids, not text: a resumed run must not re-render with a changed template.
    assert isinstance(row["input_ids"], list)


def test_export_truncation_can_drop_all_supervision(
    tmp_path: pathlib.Path, qwen_tokenizer, record
) -> None:  # type: ignore[no-untyped-def]
    """Truncating a long record can remove every assistant turn.

    That example supervises nothing. It must fail loudly by default rather than
    enter the training set as silent noise.
    """
    conv = normalize_conversation(record["messages"], record["tools"])
    full = tmp_path / "full.jsonl"
    cut = tmp_path / "cut.jsonl"
    export_jsonl(full, [conv], qwen_tokenizer)
    with pytest.raises(DatasetError, match="no supervised tokens"):
        export_jsonl(cut, [conv], qwen_tokenizer, max_length=32)
    assert json.loads(full.read_text(encoding="utf-8").strip())["input_ids"]


def test_export_can_skip_unsupervised_examples(
    tmp_path: pathlib.Path, qwen_tokenizer, record
) -> None:  # type: ignore[no-untyped-def]
    conv = normalize_conversation(record["messages"], record["tools"])
    out = tmp_path / "cut.jsonl"
    assert export_jsonl(out, [conv], qwen_tokenizer, max_length=32, on_empty="skip") == 0
    assert not out.exists() or out.read_text(encoding="utf-8").strip() == ""


def test_export_rejects_unknown_on_empty(
    tmp_path: pathlib.Path, qwen_tokenizer, conversation
) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(DatasetError, match="on_empty"):
        export_jsonl(tmp_path / "x.jsonl", [conversation], qwen_tokenizer, on_empty="nope")
