"""Assistant-mask tests against the real tokenizer.

The mask decides what the model learns. If the tool catalogue or a
``<tool_result>`` block leaks into the loss, the model is being trained to
fabricate tool output — the failure mode this whole format exists to prevent.
These tests assert the boundary explicitly rather than counting tokens.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gotooltrain import (
    assert_mask_sane,
    normalize_conversation,
    render_example,
    render_text,
)

#: Must never be a training target: context, not assistant output.
FORBIDDEN_IN_LOSS = (
    "<available_tools>",
    "You are a Go engineer.",
    "Add a test for the parser.",
    "package parser",
    "ok  parser  0.4s",
    "<|im_start|>",
    "<|im_end|>",
)

#: Must be a training target, otherwise the example teaches nothing.
REQUIRED_IN_LOSS = (
    "Let me read the package first.",
    "Tests pass.",
    '"name": "read_file"',
    '"name": "go_test"',
)


def supervised_text(tokenizer, example):  # type: ignore[no-untyped-def]
    """Decode exactly the tokens the loss applies to."""
    ids = [t for t, label in zip(example.input_ids, example.labels) if label != -100]
    return tokenizer.decode(ids)


def _tools(*names: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"the {name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


@pytest.fixture
def example(qwen_tokenizer, conversation):  # type: ignore[no-untyped-def]
    return render_example(qwen_tokenizer, conversation)


def test_example_has_supervised_tokens(example) -> None:  # type: ignore[no-untyped-def]
    assert example.supervised_tokens > 0


def test_context_never_receives_loss(qwen_tokenizer, example) -> None:  # type: ignore[no-untyped-def]
    supervised = supervised_text(qwen_tokenizer, example)
    for probe in FORBIDDEN_IN_LOSS:
        assert probe not in supervised, f"leaked into the loss: {probe!r}"


def test_assistant_content_receives_loss(qwen_tokenizer, example) -> None:  # type: ignore[no-untyped-def]
    supervised = supervised_text(qwen_tokenizer, example)
    for probe in REQUIRED_IN_LOSS:
        assert probe in supervised, f"missing from the loss: {probe!r}"


def test_labels_agree_with_mask(example) -> None:  # type: ignore[no-untyped-def]
    for token, label, mask in zip(example.input_ids, example.labels, example.assistant_mask):
        assert (label != -100) == bool(mask)


def test_field_lengths_match(example) -> None:  # type: ignore[no-untyped-def]
    lengths = {
        len(example.input_ids),
        len(example.attention_mask),
        len(example.labels),
        len(example.assistant_mask),
    }
    assert len(lengths) == 1


def test_assert_mask_sane_accepts_a_good_example(example) -> None:  # type: ignore[no-untyped-def]
    assert_mask_sane(example)


def test_tool_result_turn_is_never_a_target(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    """A conversation ending on a tool result must still mask that result out."""
    conv = normalize_conversation(
        [
            {"role": "user", "content": "build it"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "go_test", "arguments": json.dumps({"pkg": "./..."})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "FAIL: undefined: foo"},
            {"role": "assistant", "content": "The build failed because foo is undefined."},
        ],
        _tools("go_test"),
    )
    supervised = supervised_text(qwen_tokenizer, render_example(qwen_tokenizer, conv))
    assert "FAIL: undefined: foo" not in supervised
    assert "because foo is undefined" in supervised


def test_parallel_tool_calls_all_are_targets(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    conv = normalize_conversation(
        [
            {"role": "user", "content": "both"},
            {
                "role": "assistant",
                "content": "Running both.",
                "tool_calls": [
                    {"id": "a", "function": {"name": "go_build", "arguments": "{}"}},
                    {"id": "b", "function": {"name": "go_test", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "build ok"},
            {"role": "tool", "tool_call_id": "b", "content": "tests ok"},
            {"role": "assistant", "content": "Both clean."},
        ],
        _tools("go_build", "go_test"),
    )
    supervised = supervised_text(qwen_tokenizer, render_example(qwen_tokenizer, conv))
    assert supervised.count("<tool_use>") == 2
    assert "build ok" not in supervised
    assert "tests ok" not in supervised


def test_tool_catalogue_is_rendered_even_without_a_system_prompt(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    """Regression: the catalogue used to vanish when no system message existed.

    The model would then see `tool_use` blocks for tools it was never told about.
    """
    conv = normalize_conversation(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
        ],
        _tools("go_test", "go_build", "read_file"),
    )
    text = render_text(qwen_tokenizer, conv)
    assert text.count("<available_tools>") == 1
    assert text.index('"go_build"') < text.index('"go_test"') < text.index('"read_file"')
    assert text.count("\n") >= 6, "one JSON definition per line"
    assert text.startswith("<|im_start|>system\n<available_tools>")


def test_no_system_turn_at_all_when_there_are_no_tools(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    conv = normalize_conversation(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    )
    text = render_text(qwen_tokenizer, conv)
    assert not text.startswith("<|im_start|>system")
    assert text.startswith("<|im_start|>user\n")


def test_generation_prompt_opens_an_assistant_turn(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    prompt = render_text(qwen_tokenizer, conversation, add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n")


def test_generation_prompt_without_thinking(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    prompt = qwen_tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        conversation.to_messages(),
        tools=conversation.to_tools() or None,
        chat_template=qwen_tokenizer.chat_template,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert prompt.endswith("<think>\n\n</think>\n\n")


# ---------------------------------------------------------- mask sanity guards


def _example(**overrides):  # type: ignore[no-untyped-def]
    from gotooltrain import RenderedExample

    base = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [-100, 2, 3],
        "assistant_mask": [0, 1, 1],
        "text": "abc",
    }
    base.update(overrides)
    return RenderedExample(**base)  # type: ignore[arg-type]


def test_ragged_example_is_rejected() -> None:
    from gotooltrain import TemplateError

    with pytest.raises(TemplateError, match="ragged example"):
        assert_mask_sane(_example(labels=[-100, 2]))


def test_example_without_supervision_is_rejected() -> None:
    from gotooltrain import TemplateError

    with pytest.raises(TemplateError, match="no supervised tokens"):
        assert_mask_sane(_example(labels=[-100, -100, -100], assistant_mask=[0, 0, 0]))


def test_label_mask_disagreement_is_rejected() -> None:
    from gotooltrain import TemplateError

    with pytest.raises(TemplateError, match="disagree"):
        assert_mask_sane(_example(labels=[1, 2, 3], assistant_mask=[0, 1, 1]))


def test_build_examples_can_skip_the_sanity_check(qwen_tokenizer, conversation) -> None:  # type: ignore[no-untyped-def]
    """assert_sane=False is an explicit opt-out, not a default."""
    from gotooltrain import build_examples

    examples = list(build_examples([conversation], qwen_tokenizer, assert_sane=False, max_length=4))
    assert len(examples) == 1
