"""Every catalogue command, actually executed against a real Go module.

The unit tests prove the *shape* of each command: the argv is built from the
template, arguments are substituted, and no shell is involved. None of them can
prove the command works. That gap is not theoretical -- ``rtk grep`` shipped
without ``-r``, so every call against a directory failed with
``grep: .: Is a directory`` while passing all 500 unit tests.

So this file runs each one for real. Skipped loudly when the toolchain is absent;
never silently downgraded to a stub.
"""

from __future__ import annotations

import pathlib
import shutil
import sys
from typing import Any

import pytest

from gotooltrain.gorun import ExecRequest, LocalExecutor, Status, execute, plan
from gotooltrain.gotools import GO_TOOLS


def have(program: str) -> bool:
    """True when the program resolves on PATH."""
    return shutil.which(program) is not None


needs_go = pytest.mark.skipif(not have("go"), reason="the Go toolchain is not installed")
needs_rtk = pytest.mark.skipif(not have("rtk"), reason="rtk is not installed")

MODULE = "example.com/parser"

SOURCE = """package parser

// Count returns the sum of every value in the table.
func Count(table map[string]int) int {
	total := 0
	for _, n := range table {
		total += n
	}
	return total
}
"""

BROKEN_SOURCE = """package parser

// Count returns the number of entries in the table.
func Count(table map[string]int) int {
	return len(table)
}
"""

TEST = """package parser

import "testing"

func TestCount(t *testing.T) {
	if got := Count(map[string]int{"a": 1, "b": 2}); got != 3 {
		t.Fatalf("Count = %d, want 3", got)
	}
}
"""


@pytest.fixture
def module(tmp_path: pathlib.Path) -> pathlib.Path:
    """A small, passing Go module with one source and one test file."""
    root = tmp_path / "repo"
    (root / "parser").mkdir(parents=True)
    (root / "go.mod").write_text(f"module {MODULE}\n\ngo 1.23\n", encoding="utf-8")
    (root / "parser" / "parser.go").write_text(SOURCE, encoding="utf-8")
    (root / "parser" / "parser_test.go").write_text(TEST, encoding="utf-8")
    return root


def rtk_binary() -> str:
    """Absolute path to rtk.

    Resolved through which() rather than handed to subprocess as a bare name: the
    catalogue runs on PATH by design, but a test that shells out directly should
    not depend on the ambient PATH of whatever runner started pytest.
    """
    found = shutil.which("rtk")
    if found is None:
        pytest.skip("rtk is not on PATH")
    return found


def call(tool: str, module: pathlib.Path, **arguments: Any):  # type: ignore[no-untyped-def]
    """Run one catalogue tool against ``module`` and return the ExecResult."""
    request = ExecRequest(
        task_id="t",
        sample_index=0,
        tool_name=tool,
        arguments=arguments,
        workspace=module,
        timeout_s=300,
    )
    return execute(request, LocalExecutor(cwd=module))


# ------------------------------------------------------------------ raw tools


@needs_go
def test_go_doc_reports_a_symbol(module: pathlib.Path) -> None:
    result = call("go_doc", module, symbol="errors.Is")
    assert result.status is Status.OK
    assert "func Is" in result.stdout


@needs_go
def test_go_mod_tidy_runs(module: pathlib.Path) -> None:
    result = call("go_mod_tidy", module)
    assert result.status is Status.OK, result.stdout


# ------------------------------------------------------------------ rtk tools


@needs_rtk
@needs_go
def test_go_test_passes_on_a_good_module(module: pathlib.Path) -> None:
    result = call("go_test", module, pkg="./...")
    assert result.status is Status.OK, result.stdout


@needs_rtk
@needs_go
def test_go_test_reports_a_real_failure(module: pathlib.Path) -> None:
    """A red suite must reach the model as a failure, not as an empty result."""
    (module / "parser" / "parser.go").write_text(BROKEN_SOURCE, encoding="utf-8")
    result = call("go_test", module, pkg="./...")
    assert result.status is Status.TOOL_ERROR
    assert "TestCount" in result.stdout
    assert "want 3" in result.stdout


@needs_rtk
@needs_go
def test_go_build_is_silent_on_success(module: pathlib.Path) -> None:
    result = call("go_build", module, pkg="./...")
    assert result.status is Status.OK
    assert result.stdout.strip() == ""


@needs_rtk
def test_read_file_returns_the_source(module: pathlib.Path) -> None:
    result = call("read_file", module, path="parser/parser.go")
    assert result.status is Status.OK
    assert "func Count" in result.stdout


@needs_rtk
def test_read_file_is_verbatim_not_a_summary(module: pathlib.Path) -> None:
    """The whole reason `rtk smart` is banned: bodies must survive `rtk read`.

    The catalogue promises verbatim contents. If a reducer ever started stripping
    them, the model would be trained to edit code it has only seen signatures of --
    which is the exact failure the ban exists to prevent.
    """
    long_source = "package parser\n\nfunc Big(rows [][]int) int {\n\ttotal := 0\n"
    long_source += "".join(
        f"\tfor i := {i}; i < 10; i++ {{\n\t\ttotal += i\n\t}}\n" for i in range(40)
    )
    long_source += "\treturn total\n}\n"
    (module / "parser" / "big.go").write_text(long_source, encoding="utf-8")

    result = call("read_file", module, path="parser/big.go")
    assert result.status is Status.OK
    assert result.stdout.count("\n") == long_source.count("\n"), "lines were dropped"
    assert "total += i" in result.stdout, "function bodies were stripped"


@needs_rtk
def test_rtk_smart_would_destroy_the_source(module: pathlib.Path) -> None:
    """Proves the ban is load-bearing rather than superstition."""
    import subprocess

    source = module / "parser" / "parser.go"
    collapsed = subprocess.run(  # noqa: S603 - absolute path from which()
        [rtk_binary(), "smart", str(source)], capture_output=True, text=True, check=False
    )
    if collapsed.returncode != 0:
        pytest.skip("rtk smart is unavailable in this build")
    assert collapsed.stdout.count("\n") < 5
    assert "total += n" not in collapsed.stdout, "smart kept a body after all"
    # read_file is bound to `rtk read`, which must not be `rtk smart`.
    _, argv = plan(
        ExecRequest(
            task_id="t",
            sample_index=0,
            tool_name="read_file",
            arguments={"path": "parser/parser.go"},
            workspace=module,
        )
    )
    # The catalogue renders the tool's *name*, not its resolved path: a prompt
    # built here must not embed the build machine's directory layout.
    assert argv[:2] == ["rtk", "read"]


@needs_rtk
def test_recall_returns_the_output_a_filter_elided(module: pathlib.Path) -> None:
    """The hint rtk prints is only useful if the model can act on it."""
    import re

    (module / "parser" / "parser.go").write_text(BROKEN_SOURCE, encoding="utf-8")
    result = call("go_test", module, pkg="./...")
    assert result.status is Status.TOOL_ERROR
    hint = re.search(r"rtk recall ([0-9a-f]+)", result.stdout)
    if hint is None:
        pytest.skip("this rtk build did not elide the failure, so there is nothing to recall")

    recalled = call("rtk_recall", module, **{"hash": hint.group(1)})
    assert recalled.status is Status.OK, recalled.stdout
    # The point of the tool: recall must return *more* than the compact summary
    # that pointed at it. Returning the same bytes would make it pointless.
    assert len(recalled.stdout) > len(result.stdout)


@needs_rtk
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="rtk grep shells out to busybox on Windows, which is absent there; the suite is "
    "meant to run under WSL for this tool",
)
def test_grep_finds_a_symbol_in_a_subdirectory(module: pathlib.Path) -> None:
    """The case that shipped broken: without -r, grep refuses a directory."""
    result = call("grep", module, pattern="func Count", path=".")
    assert result.status is Status.OK, result.stdout
    assert "func Count" in result.stdout


@needs_rtk
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="rtk grep shells out to busybox on Windows, which is absent there",
)
def test_grep_reports_no_match_as_a_failure(module: pathlib.Path) -> None:
    result = call("grep", module, pattern="zzz-not-present", path=".")
    assert result.status is Status.TOOL_ERROR


# --------------------------------------------------------- workspace tools


@needs_go
def test_write_file_then_build_then_test(module: pathlib.Path) -> None:
    """The whole repair loop, in the order a model would perform it."""
    (module / "parser" / "parser.go").write_text(BROKEN_SOURCE, encoding="utf-8")
    assert call("go_test", module, pkg="./...").status is Status.TOOL_ERROR

    write = call("write_file", module, path="parser/parser.go", content=SOURCE)
    assert write.status is Status.OK
    assert "created" in write.stdout or "updated" in write.stdout

    assert call("go_build", module, pkg="./...").status is Status.OK
    assert call("go_test", module, pkg="./...").status is Status.OK


@needs_go
def test_edit_file_changes_the_source(module: pathlib.Path) -> None:
    result = call(
        "edit_file",
        module,
        path="parser/parser.go",
        old_string="total += n",
        new_string="total = total + n",
    )
    assert result.status is Status.OK
    assert "total = total + n" in (module / "parser" / "parser.go").read_text(encoding="utf-8")


# ------------------------------------------------------------- the whole set


@needs_rtk
@needs_go
def test_every_command_tool_is_runnable(module: pathlib.Path) -> None:
    """No catalogue tool may resolve to a command that cannot execute.

    One representative argument set per command-backed tool. A tool whose binary
    is missing fails as a harness error, which is the point: the catalogue must
    not ship a command the environment cannot run.
    """
    samples: dict[str, dict[str, Any]] = {
        "read_file": {"path": "parser/parser.go"},
        "grep": {"pattern": "func Count", "path": "."},
        "go_build": {"pkg": "./..."},
        "go_test": {"pkg": "./..."},
        "go_doc": {"symbol": "errors.Is"},
        "go_mod_tidy": {},
        "rtk_recall": {"hash": "deadbeef"},
        "write_file": {"path": "new.go", "content": "package parser\n"},
        "edit_file": {
            "path": "parser/parser.go",
            "old_string": "func Count",
            "new_string": "func Count",
        },
    }
    missing = [t.name for t in GO_TOOLS if t.name not in samples]
    assert not missing, f"no runnable sample for: {missing}"

    for name, arguments in samples.items():
        tool, argv = plan(
            ExecRequest(
                task_id="t",
                sample_index=0,
                tool_name=name,
                arguments=arguments,
                workspace=module,
            )
        )
        if tool.command_template:
            assert argv, f"{name} has a command template but builds no argv"
        else:
            assert not argv, f"{name} is commandless yet builds argv"
            continue
        result = call(name, module, **arguments)
        assert result.status is not Status.HARNESS_ERROR, (
            f"{name} ({' '.join(argv)}) failed as infrastructure: {result.harness_error}"
        )
        assert tool.name == name


@needs_rtk
@needs_go
def test_the_binaries_the_catalogue_needs_are_resolvable() -> None:
    from gotooltrain.gorun import missing_tool_binaries

    assert missing_tool_binaries() == []
