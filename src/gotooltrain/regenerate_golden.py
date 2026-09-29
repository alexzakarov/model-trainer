"""Regenerate the golden render for the tool-use format.

Run explicitly whenever the format is intentionally changed:

    python -m gotooltrain.regenerate_golden

The resulting diff is the review artefact: a change to the token format is a
change to the training data and must be visible in version control.
"""

from __future__ import annotations

import pathlib
from typing import Any

from transformers import AutoTokenizer

from . import install_template, load_template_source, normalize_conversation, render_text

BASE_MODEL = "Qwen/Qwen3.5-4B"


def _project_root() -> pathlib.Path:
    """Walk up from this module to the directory that contains ``tests/``."""
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / "tests" / "golden").is_dir():
            return candidate
    raise RuntimeError(
        "cannot locate the project root: no ancestor directory contains tests/golden"
    )


GOLDEN = _project_root() / "tests" / "golden" / "anthropic-tools-v1.txt"

_MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a Go engineer."},
    {"role": "user", "content": "Add a test for the parser."},
    {
        "role": "assistant",
        "content": "Let me read the package first.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"parser/parser.go"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "package parser"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "go_test", "arguments": '{"pkg":"./parser"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "ok  parser  0.4s"},
    {"role": "assistant", "content": "Tests pass."},
]

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "go_test",
            "description": "Run go test for a package.",
            "parameters": {
                "type": "object",
                "properties": {"pkg": {"type": "string", "description": "package path"}},
                "required": ["pkg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def main() -> None:  # pragma: no cover - maintenance entry point
    """Re-render the sample conversation and overwrite the golden file."""
    tokenizer = install_template(AutoTokenizer.from_pretrained(BASE_MODEL), load_template_source())
    conversation = normalize_conversation(_MESSAGES, _TOOLS)
    rendered = render_text(tokenizer, conversation)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(rendered, encoding="utf-8")
    print(f"wrote {GOLDEN} ({len(rendered)} chars)")


if __name__ == "__main__":  # pragma: no cover
    main()
