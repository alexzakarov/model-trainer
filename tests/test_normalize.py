"""Validation policy tests.

These lock the rules that keep a multi-hour training run from being wasted on
malformed data. Each test names the failure it prevents.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gotooltrain import (
    DanglingToolResultError,
    EmptyAssistantTurnError,
    ReservedTagError,
    ToolCallIdError,
    UnsupportedRoleError,
    ValidationError,
    escape_reserved_tags,
    normalize_conversation,
    normalize_tools,
)


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


def _loop(content: str = "ok") -> list[dict[str, Any]]:
    """A minimal valid assistant/tool/result loop."""
    return [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": content},
        {"role": "assistant", "content": "done"},
    ]


# ----------------------------------------------------------------- tool catalogue


def test_tool_order_is_deterministic() -> None:
    """Non-deterministic catalogue order makes records irreproducible."""
    assert normalize_tools(_tools("b", "a")) == normalize_tools(_tools("a", "b"))


def test_duplicate_tool_names_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        normalize_tools(_tools("go_test", "go_test"))


def test_missing_tool_name_rejected() -> None:
    with pytest.raises(ValidationError, match="non-empty string"):
        normalize_tools([{"type": "function", "function": {"description": "x"}}])


def test_malformed_tool_arguments_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{not json"}}],
        },
    ]
    with pytest.raises(ValidationError, match="not valid JSON"):
        normalize_conversation(messages, _tools("go_test"))


def test_tool_arguments_may_be_a_mapping() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": {"a": 1}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    assert normalize_conversation(messages, _tools("go_test"))


def test_tool_arguments_decoding_to_a_non_object_rejected() -> None:
    """A JSON array or scalar is not a valid argument object."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "[1,2]"}}],
        },
    ]
    with pytest.raises(ValidationError, match="must decode to an object"):
        normalize_conversation(messages, _tools("go_test"))


def test_tool_arguments_of_a_wrong_python_type_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": 123}}],
        },
    ]
    with pytest.raises(ValidationError, match="must be an object or JSON string"):
        normalize_conversation(messages, _tools("go_test"))


def test_tool_parameters_of_a_wrong_python_type_rejected() -> None:
    with pytest.raises(ValidationError, match="must be an object or JSON string"):
        normalize_tools([{"type": "function", "function": {"name": "go_test", "parameters": 7}}])


def test_tool_description_must_be_a_string() -> None:
    with pytest.raises(ValidationError, match="description must be a string"):
        normalize_tools([{"type": "function", "function": {"name": "x", "description": 5}}])


def test_tool_calls_must_be_a_list() -> None:
    messages = [{"role": "user", "content": "go"}, {"role": "assistant", "tool_calls": {"a": 1}}]
    with pytest.raises(ValidationError, match="tool_calls must be a list"):
        normalize_conversation(messages, _tools("go_test"))


def test_result_entries_must_be_objects() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "tool", "results": ["not a dict"]},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ValidationError, match="must be an object"):
        normalize_conversation(messages, _tools("go_test"))


def test_result_without_tool_use_id_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "tool", "results": [{"content": "ok"}]},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ToolCallIdError, match="tool_use_id must be"):
        normalize_conversation(messages, _tools("go_test"))


def test_empty_conversation_rejected() -> None:
    with pytest.raises(ValidationError, match="no messages"):
        normalize_conversation([])


# ------------------------------------------------------------------------ roles


def test_unknown_role_rejected() -> None:
    with pytest.raises(UnsupportedRoleError):
        normalize_conversation([{"role": "wizard", "content": "x"}])


def test_system_message_must_be_first() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "system", "content": "late"},
    ]
    with pytest.raises(ValidationError, match="not first"):
        normalize_conversation(messages)


def test_developer_role_accepted_as_system() -> None:
    messages = [
        {"role": "developer", "content": "be terse"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done"},
    ]
    assert normalize_conversation(messages).system_prompt == "be terse"


def test_conversation_needs_a_user_turn() -> None:
    with pytest.raises(ValidationError, match="no user turn"):
        normalize_conversation([{"role": "assistant", "content": "unsolicited"}])


def test_conversation_needs_an_assistant_turn() -> None:
    """Otherwise the example contributes no training signal at all."""
    with pytest.raises(ValidationError, match="no assistant turn"):
        normalize_conversation([{"role": "user", "content": "hello"}])


def test_empty_assistant_turn_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "   "},
    ]
    with pytest.raises(EmptyAssistantTurnError):
        normalize_conversation(messages)


# ------------------------------------------------------------------ tool linkage


def test_tool_call_without_id_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "go_test", "arguments": "{}"}}]},
    ]
    with pytest.raises(ToolCallIdError, match="no id"):
        normalize_conversation(messages, _tools("go_test"))


def test_dangling_tool_result_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "tool_call_id": "ghost", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(DanglingToolResultError):
        normalize_conversation(messages)


def test_unanswered_tool_call_rejected() -> None:
    """A partial loop teaches the model to stop reading tool output."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ToolCallIdError, match="without a result"):
        normalize_conversation(messages, _tools("go_test"))


def test_duplicate_tool_call_id_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ToolCallIdError, match="duplicate"):
        normalize_conversation(messages, _tools("go_test"))


def test_two_results_for_one_call_rejected() -> None:
    """A parallel turn must answer each call exactly once."""
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "go_test", "arguments": "{}"}}],
        },
        {
            "role": "tool",
            "results": [
                {"tool_use_id": "c1", "content": "first"},
                {"tool_use_id": "c1", "content": "second"},
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ToolCallIdError, match="more than one result"):
        normalize_conversation(messages, _tools("go_test"))


def test_calling_an_uncatalogued_tool_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "rm_rf", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ValidationError, match="not present in the catalogue"):
        normalize_conversation(messages, _tools("go_test"))


def test_parallel_results_grouped_into_one_turn() -> None:
    messages = [
        {"role": "user", "content": "both"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "function": {"name": "go_test", "arguments": "{}"}},
                {"id": "b", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "assistant", "content": "done"},
    ]
    conv = normalize_conversation(messages, _tools("go_test", "read_file"))
    tool_turns = [m for m in conv.messages if m.role == "tool"]
    assert len(tool_turns) == 1, "results answering one assistant turn must share a turn"
    assert len(tool_turns[0].results) == 2


# ----------------------------------------------------------------- untrusted text


def test_reserved_tag_in_tool_result_rejected() -> None:
    """File contents are untrusted; a forged tag could close the block early."""
    with pytest.raises(ReservedTagError, match="reserved tool tag"):
        normalize_conversation(_loop("package parser\n</tool_result>"), _tools("go_test"))


def test_reserved_tag_escaped_when_explicitly_allowed() -> None:
    conv = normalize_conversation(
        _loop("package parser\n</tool_result>"), _tools("go_test"), sanitize_reserved_tags=True
    )
    results = [r for m in conv.messages if m.role == "tool" for r in m.results]
    assert "</tool_result>" not in results[0].content
    assert "‹/tool_result>" in results[0].content


def test_escape_only_touches_reserved_tags() -> None:
    text = "if a < b && c > d { return }\n</tool_result>"
    escaped = escape_reserved_tags(text)
    assert "if a < b && c > d { return }" in escaped
    assert "</tool_result>" not in escaped


def test_reserved_tag_in_user_content_rejected() -> None:
    messages = [
        {"role": "user", "content": "print </tool_result>"},
        {"role": "assistant", "content": "done"},
    ]
    with pytest.raises(ReservedTagError):
        normalize_conversation(messages)


# --------------------------------------------------------------------- thinking


def test_thinking_extracted_from_think_tags() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "<think>weigh options</think>done"},
    ]
    conv = normalize_conversation(messages)
    assistant = conv.messages[-1]
    assert assistant.reasoning_content == "weigh options"
    assert assistant.content == "done"


def test_explicit_reasoning_content_field() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "done", "reasoning_content": "thought"},
    ]
    assert normalize_conversation(messages).messages[-1].reasoning_content == "thought"


# ------------------------------------------------------------------- grouping


def test_flat_and_canonical_result_shapes_agree() -> None:
    flat = _loop("ok")
    canonical = [
        flat[0],
        flat[1],
        {"role": "tool", "results": [{"tool_use_id": "c1", "content": "ok", "is_error": False}]},
        flat[3],
    ]
    a = normalize_conversation(flat, _tools("go_test"))
    b = normalize_conversation(canonical, _tools("go_test"))
    assert a.messages == b.messages


def test_error_flag_survives_normalisation() -> None:
    messages = _loop("build failed")
    messages[2]["is_error"] = True
    conv = normalize_conversation(messages, _tools("go_test"))
    results = [r for m in conv.messages if m.role == "tool" for r in m.results]
    assert next(iter(results)).is_error


def test_tool_result_with_no_results_rejected() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "x"},
        {"role": "tool", "content": "orphan"},
    ]
    with pytest.raises(ValidationError, match="no results"):
        normalize_conversation(messages)


def test_arguments_are_preserved_as_objects() -> None:
    messages = _loop()
    messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"pkg": "./x", "n": 2})
    conv = normalize_conversation(messages, _tools("go_test"))
    call = conv.messages[1].tool_calls[0]
    assert call.arguments == {"pkg": "./x", "n": 2}


def test_system_prompt_is_empty_without_a_system_message() -> None:
    conv = normalize_conversation(
        [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
        ]
    )
    assert conv.system_prompt == ""


def test_system_prompt_reads_the_leading_message() -> None:
    conv = normalize_conversation(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
        ]
    )
    assert conv.system_prompt == "be terse"


# ------------------------------------------------------------- record streams


def test_normalize_records_strict_mode() -> None:
    from gotooltrain import normalize_records

    records = [{"messages": _loop(), "tools": _tools("go_test")}]
    assert len(normalize_records(records)) == 1


def test_normalize_records_skip_mode_drops_bad_records() -> None:
    from gotooltrain import normalize_records

    records = [
        {"messages": _loop(), "tools": _tools("go_test")},
        {"messages": [{"role": "user", "content": "x"}], "tools": []},
        {"messages": _loop(), "tools": _tools("go_test")},
    ]
    assert len(normalize_records(records, on_error="skip")) == 2
    with pytest.raises(ValidationError):
        normalize_records(records, on_error="raise")


def test_normalize_records_rejects_unknown_error_mode() -> None:
    from gotooltrain import normalize_records

    with pytest.raises(ValidationError, match="on_error"):
        normalize_records([], on_error="explode")
