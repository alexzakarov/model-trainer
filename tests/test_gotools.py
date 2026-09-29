"""The Go tool catalogue: schemas, execution policy, and truncation.

These tests treat the catalogue as a contract with the model. A weak description
or a tool wired to the wrong RTK mode does not crash anything -- it just quietly
degrades tool selection, which is the failure mode this module exists to prevent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gotooltrain import normalize_conversation, normalize_tools
from gotooltrain.gotools import (
    CATALOG_VERSION,
    GO_TOOLS,
    TRUNCATION_MARKER,
    Execution,
    ToolDefinition,
    assert_no_forbidden_smart,
    catalog,
    catalog_sha,
    get,
    truncate_output,
)

EXPECTED = {
    "read_file",
    "write_file",
    "edit_file",
    "grep",
    "go_build",
    "go_test",
    "go_doc",
    "go_mod_tidy",
    # rtk prints "[full output: rtk recall <hash>]" whenever a filter elides
    # something. Without a tool to act on that hint the model is told where the
    # rest of the output is and given no way to go there.
    "rtk_recall",
}


# ------------------------------------------------------------------- inventory


def test_the_catalogue_is_exactly_what_was_agreed() -> None:
    assert {t.name for t in GO_TOOLS} == EXPECTED
    assert len(GO_TOOLS) == len(EXPECTED)


def test_tool_names_are_unique() -> None:
    names = [t.name for t in GO_TOOLS]
    assert len(names) == len(set(names))


def test_every_tool_has_a_substantive_description() -> None:
    """A one-word description makes tool selection measurably worse."""
    for tool in GO_TOOLS:
        assert len(tool.description) >= 40, f"{tool.name} description is too thin"
        assert tool.description.endswith("."), f"{tool.name} description is not a sentence"


def test_every_schema_is_a_closed_object() -> None:
    for tool in GO_TOOLS:
        params = tool.parameters
        assert params["type"] == "object"
        assert params["additionalProperties"] is False, f"{tool.name} accepts unknown keys"
        assert set(params["required"]) <= set(params["properties"])


def test_every_property_is_described() -> None:
    for tool in GO_TOOLS:
        for name, spec in tool.parameters["properties"].items():
            assert "description" in spec, f"{tool.name}.{name} has no description"


def test_parameterless_tool_has_no_properties() -> None:
    tidy = get("go_mod_tidy")
    assert tidy.parameters["properties"] == {}
    assert tidy.parameters["required"] == []


def test_required_paths_are_covered() -> None:
    assert get("read_file").parameters["required"] == ["path"]
    assert get("go_test").parameters["required"] == ["pkg"]
    assert get("edit_file").parameters["required"] == ["path", "old_string", "new_string"]
    assert get("grep").parameters["required"] == ["pattern"]


def test_get_rejects_an_unknown_tool() -> None:
    with pytest.raises(KeyError, match="unknown tool"):
        get("rm_rf")


def test_output_budgets_are_positive_and_ordered() -> None:
    budgets = {t.name: t.output_budget_chars for t in GO_TOOLS}
    assert all(b > 0 for b in budgets.values())
    # A file read should be able to show more than a tidy log.
    assert budgets["read_file"] > budgets["go_mod_tidy"]
    # go_test needs room for several failures.
    assert budgets["go_test"] > budgets["go_build"]


# ------------------------------------------------------------ execution policy


def test_rtk_backed_tools_use_rtk() -> None:
    for name in ("grep", "go_build", "go_test"):
        tool = get(name)
        assert tool.execution is Execution.RTK
        assert tool.command_template.startswith("rtk ")


def test_rtk_less_tools_still_run_real_commands() -> None:
    """RAW means "no RTK rule covers it", not "no command".

    go_doc and go_mod_tidy once shipped with an empty template, which made them
    report success without doing anything.
    """
    for name in ("write_file", "edit_file", "go_doc", "go_mod_tidy"):
        assert get(name).execution is Execution.RAW
    assert get("go_doc").command_template == "go doc {symbol}"
    assert get("go_mod_tidy").command_template == "go mod tidy"
    assert get("write_file").command_template == ""
    assert get("edit_file").command_template == ""


def test_read_file_uses_rtk_read_never_rtk_smart() -> None:
    """``rtk smart`` reduces a file to signatures; that must never back a read."""
    read = get("read_file")
    assert read.execution is Execution.RTK_READ
    assert read.command_template == "rtk read {path}"
    assert read.forbidden_smart is True, "read_file is exactly where the ban applies"
    assert "smart" not in read.command_template


def test_no_tool_uses_rtk_smart() -> None:
    assert_no_forbidden_smart()


def test_forbidden_smart_guard_actually_fires(monkeypatch: Any) -> None:
    """The guard is worthless if it cannot fail; prove it detects a violation."""
    import gotooltrain.gotools as gotools

    offender = ToolDefinition(
        name="read_file",
        description="x" * 50 + ".",
        parameters={"type": "object", "properties": {}},
        execution=Execution.RAW,
        command_template="rtk smart {path}",
        output_budget_chars=100,
    )
    monkeypatch.setattr(gotools, "GO_TOOLS", (offender,))
    with pytest.raises(AssertionError, match="rtk smart"):
        gotools.assert_no_forbidden_smart()


def test_forbidden_smart_property_flags_read_backed_tools() -> None:
    assert get("read_file").forbidden_smart is True
    assert get("go_test").forbidden_smart is False


def test_command_templates_reference_only_declared_placeholders() -> None:
    for tool in GO_TOOLS:
        if not tool.command_template:
            continue
        declared = set(tool.parameters["properties"])
        for chunk in tool.command_template.split():
            if chunk.startswith("{") and chunk.endswith("}"):
                assert chunk[1:-1] in declared, f"{tool.name} uses undeclared {chunk}"


# ------------------------------------------------------------------ identity


def test_catalog_sha_is_stable() -> None:
    assert catalog_sha() == catalog_sha()
    assert len(catalog_sha()) == 64


def test_catalog_sha_changes_when_a_description_changes(monkeypatch: Any) -> None:
    """Descriptions drive tool selection, so they must be part of identity."""
    import gotooltrain.gotools as gotools

    before = gotools.catalog_sha()
    mutated = ToolDefinition(
        **{
            **{
                f: getattr(get("go_test"), f)
                for f in (
                    "name",
                    "parameters",
                    "execution",
                    "command_template",
                    "output_budget_chars",
                )
            },
            "description": "Completely different behaviour.",
        }
    )
    monkeypatch.setattr(gotools, "GO_TOOLS", (mutated,))
    assert gotools.catalog_sha() != before


def test_catalog_sha_changes_when_a_budget_changes(monkeypatch: Any) -> None:
    import gotooltrain.gotools as gotools

    before = gotools.catalog_sha()
    base = get("go_test")
    mutated = ToolDefinition(
        name=base.name,
        description=base.description,
        parameters=base.parameters,
        execution=base.execution,
        command_template=base.command_template,
        output_budget_chars=base.output_budget_chars + 1,
    )
    monkeypatch.setattr(gotools, "GO_TOOLS", (mutated,))
    assert gotools.catalog_sha() != before


def test_catalog_version_is_declared() -> None:
    assert CATALOG_VERSION == "go-tools-v2"


def test_catalog_payload_is_json_serialisable() -> None:
    payload = catalog()
    assert len(payload) == len(EXPECTED)
    json.dumps(payload)


# ---------------------------------------------------------- data-layer coupling


def test_catalogue_is_accepted_by_the_data_layer() -> None:
    """The catalogue must flow through the real validation path unchanged."""
    specs = normalize_tools(catalog())
    assert [s.name for s in specs] == sorted(EXPECTED)


def test_catalogue_renders_in_a_conversation(qwen_tokenizer) -> None:  # type: ignore[no-untyped-def]
    conv = normalize_conversation(
        [
            {"role": "user", "content": "run the parser tests"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "go_test", "arguments": '{"pkg":"./parser"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok parser 0.4s"},
            {"role": "assistant", "content": "They pass."},
        ],
        catalog(),
    )
    from gotooltrain import render_text

    text = render_text(qwen_tokenizer, conv)
    assert text.count("<available_tools>") == 1
    for name in EXPECTED:
        assert f'"name": "{name}"' in text
    # The *transport* is internal: the prompt must not tell the model that its
    # output is filtered, because at serving time that is not something it can
    # change and mentioning it invites reasoning about a system it cannot see
    # through. The `rtk_recall` tool name is different -- it is an affordance the
    # model is meant to call, so it belongs in the prompt.
    for leaked in ("rtk read", "rtk grep", "rtk go test", "command_template"):
        assert leaked not in text


# --------------------------------------------------------------- truncation


def test_short_output_is_untouched() -> None:
    assert truncate_output("ok  parser  0.4s", 1000) == "ok  parser  0.4s"


def test_truncation_keeps_the_failure_at_the_end() -> None:
    lines = [f"line {i}" for i in range(500)] + ["FAIL: undefined: foo"]
    out = truncate_output("\n".join(lines), 400)
    assert "FAIL: undefined: foo" in out, "the tail carries the failure"
    assert "line 0" in out
    assert "more lines truncated" in out


def test_truncation_respects_the_budget() -> None:
    text = "\n".join(f"a fairly long output line number {i}" for i in range(2000))
    for budget in (50, 120, 500, 4000):
        out = truncate_output(text, budget)
        assert len(out) <= budget, f"budget {budget} exceeded: {len(out)}"


def test_truncation_marker_counts_removed_lines() -> None:
    text = "\n".join(f"line {i}" for i in range(1000))
    out = truncate_output(text, 200)
    marker_line = next(line for line in out.splitlines() if "more lines truncated" in line)
    assert marker_line.startswith("...")


def test_truncation_of_a_single_huge_line() -> None:
    out = truncate_output("x" * 10_000, 100)
    assert len(out) <= 100


def test_truncation_rejects_a_nonsense_budget() -> None:
    with pytest.raises(ValueError, match="budget_chars must be"):
        truncate_output("x", 0)


def test_truncation_of_empty_text() -> None:
    assert truncate_output("", 100) == ""


def test_truncation_when_everything_fits_in_the_head() -> None:
    """Many short lines: the head takes them all and the tail slice is empty."""
    text = "\n".join("ab" for _ in range(400))
    out = truncate_output(text, 4000)
    assert "ab" in out
    assert len(out) <= 4000


def test_truncation_of_one_line_under_a_tiny_budget() -> None:
    """The budget is smaller than the marker reserve; the marker still fits."""
    out = truncate_output("x" * 500, 10)
    assert len(out) <= 10
    assert out


def test_truncation_marker_template_has_a_count_slot() -> None:
    assert "{count}" in TRUNCATION_MARKER
    assert "(+" in TRUNCATION_MARKER
