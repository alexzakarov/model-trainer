"""Workspace mutations: path containment, exact editing, and no silent no-ops.

The failure this guards against is specific: a tool that "succeeds" without
changing anything teaches the model that its edits landed. Every test here is
about a model believing something that is not true.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain import HarnessError, ValidationError
from gotooltrain.gorun import (
    ExecRequest,
    LocalExecutor,
    Status,
    assert_harness_covers_catalog,
    execute,
)
from gotooltrain.gotools import assert_catalog_is_executable, commandless_tools
from gotooltrain.workspace import edit_file, resolve_in_workspace, write_file


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "ws"
    (root / "parser").mkdir(parents=True)
    (root / "parser" / "parser.go").write_text(
        "package parser\n\nfunc Parse(s string) int {\n\treturn len(s)\n}\n", encoding="utf-8"
    )
    return root


def request_for(workspace: pathlib.Path, tool: str, arguments: dict[str, Any]) -> ExecRequest:
    return ExecRequest(
        task_id="t", sample_index=0, tool_name=tool, arguments=arguments, workspace=workspace
    )


# ---------------------------------------------------------------- containment


def test_relative_path_resolves_inside_the_workspace(workspace: pathlib.Path) -> None:
    assert resolve_in_workspace(workspace, "parser/parser.go") == workspace / "parser" / "parser.go"


@pytest.mark.parametrize(
    "attempt",
    ["../outside.txt", "parser/../../escape.txt", "/etc/passwd", "C:/Windows/system32"],
)
def test_paths_that_escape_the_workspace_are_refused(workspace: pathlib.Path, attempt: str) -> None:
    """Containment is checked on the resolved path, not on the string."""
    with pytest.raises(ValidationError):
        resolve_in_workspace(workspace, attempt)


def test_a_symlink_pointing_outside_is_refused(
    workspace: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A symlink inside the workspace must not become an escape hatch."""
    secret = tmp_path / "secret.txt"
    secret.write_text("token", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks unavailable on this filesystem")
    with pytest.raises(ValidationError, match="escapes the workspace"):
        write_file(workspace, {"path": "link.txt", "content": "overwritten"})


def test_an_empty_path_is_refused(workspace: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        write_file(workspace, {"path": "  ", "content": "x"})


# ------------------------------------------------------------------- write_file


def test_write_file_creates_a_new_file_with_parents(workspace: pathlib.Path) -> None:
    outcome = write_file(workspace, {"path": "internal/util/x.go", "content": "package util\n"})
    assert (workspace / "internal" / "util" / "x.go").read_text(
        encoding="utf-8"
    ) == "package util\n"
    assert "created" in outcome.message
    assert outcome.applied


def test_write_file_overwrites_and_says_so(workspace: pathlib.Path) -> None:
    write_file(workspace, {"path": "parser/parser.go", "content": "package parser // new\n"})
    assert (
        "updated"
        in write_file(
            workspace, {"path": "parser/parser.go", "content": "package parser // new\n"}
        ).message
    )
    assert (
        "updated"
        in write_file(
            workspace, {"path": "parser/parser.go", "content": "package parser // new\n"}
        ).message
    )


def test_write_file_uses_unix_newlines(workspace: pathlib.Path) -> None:
    """CRLF would produce a file that fails gofmt."""
    write_file(workspace, {"path": "a.go", "content": "package a\r\n"})
    assert (workspace / "a.go").read_bytes() == b"package a\n"


def test_write_file_requires_string_content(workspace: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="content must be a string"):
        write_file(workspace, {"path": "a.go", "content": 42})


# ------------------------------------------------------------------- edit_file


def test_edit_file_replaces_one_occurrence(workspace: pathlib.Path) -> None:
    outcome = edit_file(
        workspace,
        {"path": "parser/parser.go", "old_string": "return len(s)", "new_string": "return 0"},
    )
    assert "return 0" in (workspace / "parser" / "parser.go").read_text(encoding="utf-8")
    assert "edited" in outcome.message


def test_edit_file_rejects_an_ambiguous_match(workspace: pathlib.Path) -> None:
    """Editing the first of several matches silently breaks the wrong code."""
    (workspace / "dup.go").write_text("x := 1\nx := 2\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="appears 2 times"):
        edit_file(workspace, {"path": "dup.go", "old_string": "x :=", "new_string": "y :="})


def test_edit_file_rejects_a_stale_old_string(workspace: pathlib.Path) -> None:
    """The most common agent mistake: editing text it read one turn ago."""
    with pytest.raises(ValidationError, match="not found"):
        edit_file(
            workspace,
            {"path": "parser/parser.go", "old_string": "return total", "new_string": "return 0"},
        )


def test_edit_file_refuses_a_missing_file(workspace: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="missing file"):
        edit_file(workspace, {"path": "nope.go", "old_string": "a", "new_string": "b"})


def test_edit_file_refuses_a_no_op_edit(workspace: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="identical"):
        edit_file(
            workspace,
            {"path": "parser/parser.go", "old_string": "len(s)", "new_string": "len(s)"},
        )


def test_edit_file_refuses_a_non_string_argument(workspace: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="string old_string"):
        edit_file(workspace, {"path": "parser/parser.go", "old_string": 1, "new_string": "b"})


def test_edit_file_reports_the_line_delta(workspace: pathlib.Path) -> None:
    outcome = edit_file(
        workspace,
        {
            "path": "parser/parser.go",
            "old_string": "\treturn len(s)",
            "new_string": '\tif s == "" {\n\t\treturn 0\n\t}\n\treturn len(s)',
        },
    )
    assert "+3 lines" in outcome.message


# --------------------------------------------------------- through the harness


def test_execute_applies_a_write_and_reports_success(workspace: pathlib.Path) -> None:
    result = execute(
        request_for(workspace, "write_file", {"path": "new.go", "content": "package new\n"}),
        LocalExecutor(),
    )
    assert result.status is Status.OK
    assert (workspace / "new.go").is_file()
    assert "created" in result.stdout


def test_execute_marks_a_bad_edit_as_a_model_error(workspace: pathlib.Path) -> None:
    result = execute(
        request_for(
            workspace,
            "edit_file",
            {"path": "parser/parser.go", "old_string": "nope", "new_string": "yes"},
        ),
        LocalExecutor(),
    )
    assert result.status is Status.MODEL_ERROR
    assert "not found" in result.harness_error


def test_execute_builds_argv_for_go_doc(workspace: pathlib.Path) -> None:
    """go_doc is a plain command: argv is built and `go doc` really runs.

    Previously this tool had no command template, so it reported success without
    documenting anything. Asserting on the argv keeps that from regressing
    without depending on which Go toolchain is installed.
    """
    from gotooltrain.gorun import plan

    tool, argv = plan(request_for(workspace, "go_doc", {"symbol": "errors.Is"}))
    assert tool.name == "go_doc"
    assert argv == ["go", "doc", "errors.Is"]


def test_execute_builds_argv_for_go_mod_tidy(workspace: pathlib.Path) -> None:
    from gotooltrain.gorun import plan

    _, argv = plan(request_for(workspace, "go_mod_tidy", {}))
    assert argv == ["go", "mod", "tidy"]


def test_go_mod_tidy_without_a_module_is_a_tool_error(workspace: pathlib.Path) -> None:
    """It runs, and fails, rather than silently succeeding on nothing."""
    result = execute(request_for(workspace, "go_mod_tidy", {}), LocalExecutor())
    assert result.status is Status.TOOL_ERROR


def test_execute_refuses_an_unknown_tool(workspace: pathlib.Path) -> None:
    result = execute(request_for(workspace, "rm_rf", {"path": "/"}), LocalExecutor())
    assert result.status is Status.MODEL_ERROR
    assert "unknown tool" in result.harness_error


def test_a_commandless_tool_without_a_handler_raises(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that makes the whole scheme honest."""
    from gotooltrain import gorun

    monkeypatch.delitem(gorun.WORKSPACE_MUTATIONS, "write_file")
    with pytest.raises(HarnessError, match="no workspace handler"):
        gorun.assert_harness_covers_catalog()
    with pytest.raises(HarnessError, match="no workspace handler"):
        execute(
            request_for(workspace, "write_file", {"path": "a.go", "content": ""}), LocalExecutor()
        )


# ----------------------------------------------------------------- consistency


def test_every_commandless_tool_has_a_handler() -> None:
    assert_harness_covers_catalog()


def test_commandless_tools_are_exactly_the_mutating_ones() -> None:
    assert sorted(commandless_tools()) == ["edit_file", "write_file"]
    assert_catalog_is_executable(commandless_tools())


def test_duplicate_commandless_entries_are_rejected() -> None:
    with pytest.raises(AssertionError, match="duplicate"):
        assert_catalog_is_executable(["write_file", "write_file"])


def test_a_commandless_tool_missing_from_the_declaration_is_rejected() -> None:
    with pytest.raises(AssertionError, match="not declared commandless"):
        assert_catalog_is_executable(["edit_file"])


def test_editing_an_oversized_file_is_refused(
    workspace: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model asking to edit a 4 GB file made a mistake; honouring it stalls a worker."""
    from gotooltrain import workspace as ws

    monkeypatch.setattr(ws, "MAX_FILE_BYTES", 8)
    with pytest.raises(ValidationError, match="too large"):
        edit_file(
            workspace,
            {"path": "parser/parser.go", "old_string": "len", "new_string": "size"},
        )


def test_a_handler_for_an_unknown_tool_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A handler table entry with no tool behind it is dead weight that hides a typo."""
    from gotooltrain import gorun

    monkeypatch.setitem(gorun.WORKSPACE_MUTATIONS, "typo_tool", write_file)
    with pytest.raises(HarnessError, match="unknown tools"):
        gorun.assert_harness_covers_catalog()
