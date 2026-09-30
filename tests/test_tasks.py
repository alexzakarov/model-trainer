"""Task construction: every task must be measurable and reproducible.

The property under test throughout is that a task cannot be stored unless its
verification can actually run. A corpus of ungradable tasks is worse than an
empty one, because the count looks healthy.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain.errors import DatasetError
from gotooltrain.tasks import (
    VERIFIERS,
    GoTask,
    build_task_from_package,
    flip_equality,
    go_version,
    harvest_packages,
    has_go,
    introduce_failures,
    read_tasks,
    run_verifier,
    task_works_now,
    validate_task,
    write_tasks,
)


def module(root: pathlib.Path, *, passing: bool = False) -> pathlib.Path:
    """Create a small Go module whose test passes or fails as asked."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "go.mod").write_text("module example.com/x\n\ngo 1.23\n", encoding="utf-8")
    body = (
        "package x\n\n"
        "func Count(m map[string]int) int {\n\ttotal := 0\n"
        " \tfor _, n := range m {\n\t\ttotal += n\n\t}\n\treturn total\n}\n"
        if passing
        else "package x\n\nfunc Count(m map[string]int) int { return len(m) }\n"
    )
    (root / "x.go").write_text(body, encoding="utf-8")
    # Both states need a test file: without one `go test` reports "no test files"
    # and exits 0, so a broken implementation would still look done.
    (root / "x_test.go").write_text(
        "package x\n\n"
        'import "testing"\n\n'
        "func TestCount(t *testing.T) {\n"
        '\tif got := Count(map[string]int{"a": 1, "b": 2}); got != 3 {\n'
        '\t\tt.Fatalf("Count = %d, want 3", got)\n'
        "\t}\n"
        "}\n",
        encoding="utf-8",
    )
    return root


def task(**overrides: Any) -> GoTask:
    base: dict[str, Any] = {
        "task_id": "t1",
        "repository": "repo",
        "package": "pkg",
        "prompt": "make the test pass",
        "verification": ("go_test",),
        "fixture": "pkg",
    }
    base.update(overrides)
    return GoTask(**base)


# ------------------------------------------------------------- gradability


def test_a_task_without_verification_is_refused() -> None:
    """A model that fails every task and one that solves them must not look alike."""
    with pytest.raises(DatasetError, match="no verification command"):
        task(verification=())


def test_an_unknown_verifier_is_refused() -> None:
    """Asserting a command the harness will not run makes the task ungradable."""
    with pytest.raises(DatasetError, match="not among"):
        task(verification=("go_test", "cargo_test"))


def test_a_task_needs_an_identity() -> None:
    with pytest.raises(DatasetError, match="needs an id, a repository and a package"):
        task(task_id="")


def test_a_go_test_task_counts_as_multi_turn() -> None:
    """Solving it means reacting to a result, which is the whole point."""
    assert task().is_multi_turn
    assert not task(verification=("go_build",)).is_multi_turn


def test_a_task_serialises_with_its_verification() -> None:
    record = task().to_record()
    assert record["verification"] == ["go_test"]
    assert record["fixture"] == "pkg"
    assert record["metadata"] == {}


def test_the_verifier_list_is_small_and_fixed() -> None:
    """Near-equivalent verifiers are near-equivalent graders."""
    assert VERIFIERS == ("go_test", "go_build")


# ------------------------------------------------------------------ fixtures


def test_a_fixture_with_a_go_module_is_recognised(tmp_path: pathlib.Path) -> None:
    assert has_go(module(tmp_path / "repo"))


def test_a_directory_of_go_files_counts_without_a_module(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "loose"
    root.mkdir()
    (root / "a.go").write_text("package a\n", encoding="utf-8")
    assert has_go(root)


def test_a_missing_directory_is_not_a_go_repository(tmp_path: pathlib.Path) -> None:
    assert not has_go(tmp_path / "nope")


def test_a_task_pointing_at_a_missing_fixture_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="missing fixture"):
        validate_task(task(), tmp_path)


def test_a_fixture_without_go_is_refused(tmp_path: pathlib.Path) -> None:
    (tmp_path / "pkg").mkdir(parents=True)
    (tmp_path / "pkg" / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(DatasetError, match="holds no Go module"):
        validate_task(task(), tmp_path)


def test_a_valid_task_passes_validation(tmp_path: pathlib.Path) -> None:
    module(tmp_path / "repo" / "pkg")
    assert validate_task(task(fixture="repo/pkg"), tmp_path).task_id == "t1"


# ------------------------------------------------------------------- building


def test_a_task_is_built_from_a_real_package(tmp_path: pathlib.Path) -> None:
    module(tmp_path / "repo" / "internal" / "util")
    built = build_task_from_package(
        tmp_path / "repo",
        repository="repo",
        package="internal/util",
        prompt="add a test",
    )
    assert built.task_id == "repo:internal/util"
    assert built.fixture == "internal/util"


def test_a_missing_package_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match=r"does not exist"):
        build_task_from_package(tmp_path, repository="r", package="nope", prompt="p")


def test_a_package_without_go_files_is_refused(tmp_path: pathlib.Path) -> None:
    (tmp_path / "empty").mkdir(parents=True)
    with pytest.raises(DatasetError, match=r"no \.go files"):
        build_task_from_package(tmp_path, repository="r", package="empty", prompt="p")


# ------------------------------------------------------------------ verifying


def test_a_failing_test_verifies_as_not_done(tmp_path: pathlib.Path) -> None:
    if go_version() is None:
        pytest.skip("the Go toolchain is not installed")
    module(tmp_path / "repo" / "pkg")
    assert not task_works_now(tmp_path, task(fixture="repo/pkg"))


def test_a_passing_test_verifies_as_already_done(tmp_path: pathlib.Path) -> None:
    """A task the base model already solves is not evidence of capability."""
    if go_version() is None:
        pytest.skip("the Go toolchain is not installed")
    module(tmp_path / "repo" / "pkg", passing=True)
    assert task_works_now(tmp_path, task(fixture="repo/pkg"))


def test_an_unknown_verifier_is_refused_at_run_time(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="unknown verifier"):
        run_verifier(tmp_path, "cargo_test")


def test_a_missing_toolchain_is_a_setup_error_not_a_failing_task(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reported as absent, so it is never mistaken for a failure of the work."""
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "which", lambda name: None)
    with pytest.raises(DatasetError, match="Go toolchain is required"):
        run_verifier(tmp_path, "go_test")


def test_a_go_binary_that_cannot_be_spawned_reports_no_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A toolchain on PATH that will not spawn must not look like a working one."""
    import shutil as shutil_module
    import subprocess as subprocess_module

    monkeypatch.setattr(shutil_module, "which", lambda name: "/usr/bin/go")

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        """Refuse to execute at all, as a missing loader would."""
        raise OSError("cannot execute binary file")

    monkeypatch.setattr(subprocess_module, "run", boom)
    assert go_version() is None


def test_the_go_version_is_reported_when_present() -> None:
    version = go_version()
    if version is None:
        pytest.skip("the Go toolchain is not installed")
    assert "go version" in version


# ----------------------------------------------------------------- harvesting


def test_packages_are_harvested_in_a_stable_order(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    for name in ("zeta", "alpha", "mid"):
        module(repo / name)
    first = harvest_packages(repo)
    assert first == harvest_packages(repo), "a reshuffling harvest makes a diff unreadable"
    assert first == ["alpha", "mid", "zeta"]


def test_hidden_directories_are_skipped(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    module(repo / "real")
    module(repo / ".cache" / "vendored")
    assert harvest_packages(repo) == ["real"]


def test_harvesting_a_missing_repository_is_refused(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="repository not found"):
        harvest_packages(tmp_path / "nope")


def test_the_harvest_is_bounded(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "repo"
    for index in range(10):
        module(repo / f"p{index}")
    assert len(harvest_packages(repo, limit=3)) == 3


# ------------------------------------------------------------------------ io


def test_tasks_round_trip_through_a_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tasks.jsonl"
    assert write_tasks(path, [task(), task(task_id="t2")]) == 2
    assert [t.task_id for t in read_tasks(path)] == ["t1", "t2"]


def test_a_task_file_with_a_bad_verifier_is_refused_on_read(tmp_path: pathlib.Path) -> None:
    """Validation is not bypassable by writing the file directly."""
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id":"t","repository":"r","package":"p","prompt":"x",'
        '"verification":["cargo_test"],"fixture":"p"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="not among"):
        read_tasks(path)


def test_a_task_file_missing_a_field_is_named(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"id":"t"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="missing field"):
        read_tasks(path)


def test_a_corrupt_task_line_is_named(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        read_tasks(path)


def test_a_missing_task_file_is_reported(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="task file not found"):
        read_tasks(tmp_path / "nope.jsonl")


def test_a_missing_go_binary_reports_no_version(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "which", lambda name: None)
    assert go_version() is None


def test_a_go_binary_that_cannot_run_reports_no_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A toolchain on PATH that does not work must not look like a working one."""
    import shutil as shutil_module

    monkeypatch.setattr(shutil_module, "which", lambda name: "/usr/bin/go")

    class Failing:
        returncode = 1
        stdout = ""

    import subprocess as subprocess_module

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        """Report a toolchain that is present but broken."""
        return Failing()

    monkeypatch.setattr(subprocess_module, "run", boom)
    assert go_version() is None


def test_a_verifier_that_hangs_is_reported_as_a_timeout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hang must not look like a passing task."""
    import shutil as shutil_module
    import subprocess as subprocess_module

    monkeypatch.setattr(shutil_module, "which", lambda name: "/usr/bin/go")

    def hang(argv, **kwargs):  # type: ignore[no-untyped-def]
        """Never return, so the timeout path is taken."""
        raise subprocess_module.TimeoutExpired(argv, 1)

    monkeypatch.setattr(subprocess_module, "run", hang)
    code, out, err = run_verifier(tmp_path, "go_test", timeout_s=1)
    assert code == 124
    assert out == ""
    assert "timed out" in err


def test_a_task_file_with_blank_lines_is_read(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    assert read_tasks(path) == []


# ------------------------------------------------- making something to ask for


def test_an_equality_comparison_is_flipped():
    """A comparison compiles, changes behaviour, and fails an existing assertion.

    That is the loop being trained: read the failure, find the code, change it, run the
    tests again. Mutations that only break the build are easier to write and easier to
    game, because the compiler names the fix.
    """
    assert flip_equality("if a == b {") == "if a != b {"
    assert flip_equality("x := f() == g()") == "x := f() != g()"


def test_only_the_first_comparison_is_touched():
    """One change per attempt: two changes and a failure no longer says which one did it."""
    assert flip_equality("a == b && c == d") == "a != b && c == d"


def test_nothing_to_flip_is_reported_rather_than_rewritten():
    """A rewrite that changes nothing would be stored as a task that is already solved."""
    assert flip_equality("if a != b {") is None
    assert flip_equality("package p") is None


def test_a_passing_package_is_broken_so_there_is_a_task(tmp_path):
    """A healthy checkout yields no tasks, because the builder keeps only failing ones.

    So the chain stops one stage before the evaluation that needs a task, and one has
    to be introduced.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "p.go").write_text("package p\n\nfunc f(a, b int) bool { return a == b }\n")

    seen = []

    def verifier(root, verifier_name, **kwargs):
        text = (package / "p.go").read_text()
        seen.append("!=" in text)
        return (1 if "!=" in text else 0, "", "")

    assert introduce_failures(tmp_path, ["pkg"], verifier=verifier) == ["pkg"]
    assert (package / "p.go").read_text().count("!=") == 1, "the mutation was not left in place"


def test_a_mutation_that_changes_nothing_is_reverted(tmp_path):
    """Otherwise a package whose tests still pass is stored as a task already solved."""
    package = tmp_path / "pkg"
    package.mkdir()
    original = "package p\n\nfunc f(a, b int) bool { return a == b }\n"
    (package / "p.go").write_text(original)

    def verifier(root, verifier_name, **kwargs):
        return (0, "", "")  # the mutation never breaks anything

    assert introduce_failures(tmp_path, ["pkg"], verifier=verifier) == []
    assert (package / "p.go").read_text() == original, "a no-op mutation was left behind"


def test_a_package_that_already_fails_is_left_alone(tmp_path):
    """Its failure is not ours, and storing it would credit this step with work it did not do."""
    package = tmp_path / "pkg"
    package.mkdir()
    original = "package p\n\nfunc f(a, b int) bool { return a == b }\n"
    (package / "p.go").write_text(original)

    def verifier(root, verifier_name, **kwargs):
        return (1, "", "")  # already failing before anything was touched

    assert introduce_failures(tmp_path, ["pkg"], verifier=verifier) == []
    assert (package / "p.go").read_text() == original


def test_only_as_many_packages_are_broken_as_were_asked_for(tmp_path):
    """Each mutation costs a `go test`, so the count is the caller's to choose."""
    for name in ("a", "b", "c"):
        package = tmp_path / name
        package.mkdir()
        (package / "p.go").write_text("package p\n\nfunc f() bool { return 1 == 1 }\n")

    calls = []

    def verifier(root, verifier_name, **kwargs):
        mutated = "!=" in (root / "p.go").read_text()
        calls.append(mutated)
        return (1 if mutated else 0, "", "")

    assert introduce_failures(tmp_path, ["a", "b", "c"], limit=2, verifier=verifier) == ["a", "b"]
    assert len(calls) == 4, f"it kept going after the limit: {calls}"


def test_test_files_are_never_mutated(tmp_path):
    """Rewriting a test to make itself fail is not a task, it is a broken corpus.

    The model would be asked to repair a test that was deliberately sabotaged, and the
    failing assertion would point at the test rather than at the code.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "p.go").write_text("package p\n")
    (package / "p_test.go").write_text("package p\n\nfunc TestX(t *T) { _ = 1 == 1 }\n")
    before = (package / "p_test.go").read_text()

    def verifier(root, verifier_name, **kwargs):
        return (0, "", "")

    assert introduce_failures(tmp_path, ["pkg"], verifier=verifier) == []
    assert (package / "p_test.go").read_text() == before


def test_a_file_with_nothing_to_flip_is_skipped(tmp_path):
    """Not every file contains a comparison, and the next one might."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "a.go").write_text("package p\n\nvar x = 1\n")
    (package / "b.go").write_text("package p\n\nfunc f(a, b int) bool { return a == b }\n")

    def verifier(root, verifier_name, **kwargs):
        return (1 if "!=" in (package / "b.go").read_text() else 0, "", "")

    assert introduce_failures(tmp_path, ["pkg"], verifier=verifier) == ["pkg"]
    assert (package / "a.go").read_text() == "package p\n\nvar x = 1\n", "an unrelated file changed"
