"""Golden snapshot: the exact bytes of the rendered format.

Any change to the token format is a training-data change, so it must be a
deliberate, reviewable diff rather than a side effect. Regenerate with:

    python -m gotooltrain.regenerate_golden
"""

from __future__ import annotations

import pathlib

from gotooltrain import render_text

GOLDEN = pathlib.Path(__file__).parent / "golden" / "anthropic-tools-v1.txt"


def test_golden_file_exists() -> None:
    assert GOLDEN.is_file(), f"missing golden file {GOLDEN}; regenerate it explicitly"


def test_render_matches_golden(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    actual = render_text(qwen_tokenizer, conversation)
    expected = GOLDEN.read_text(encoding="utf-8")
    if actual != expected:
        import difflib

        diff = difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="golden",
            tofile="rendered",
        )
        raise AssertionError("rendered format drifted from the golden file:\n" + "".join(diff))


def test_golden_encodes_the_documented_record(conversation) -> None:  # type: ignore[no-untyped-def]
    """The golden file is only meaningful if it renders the documented record."""
    roles = [m.role for m in conversation.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
