"""Turn a real Go repository into a training task, and check it is measurable.

A task is only useful if the agent can be *scored* on it. "Read this file and
refactor it" has no right answer; "this test fails, make it pass" does. So the
task builder starts from a repository in a known state and records what must be
true at the end, which is the same contract the judge grades against.

The failure this guards against is a corpus of tasks nobody can grade: a model
that fails every one of them and a model that solves them look identical, and the
measured capability is zero either way. Every task therefore carries its
verification command, and a task whose verification cannot run is rejected rather
than stored.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import DatasetError

#: Commands that can decide whether a task is done. Kept small deliberately: a
#: verification that cannot be run produces an ungradable task, and a list of many
#: near-equivalents is a list of ways to grade inconsistently.
VERIFIERS: Final[tuple[str, ...]] = ("go_test", "go_build")


@dataclass(frozen=True, slots=True)
class GoTask:
    """One measurable unit of Go work, tied to a repository state."""

    task_id: str
    repository: str
    package: str
    prompt: str
    #: What must be true when the work is done.
    verification: tuple[str, ...]
    #: Where the repository snapshot lives, relative to the fixture root.
    fixture: str
    has_images: bool = False
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Refuse a task that cannot be graded or cannot be reproduced."""
        if not self.task_id or not self.repository or not self.package:
            raise DatasetError("a task needs an id, a repository and a package")
        if not self.verification:
            raise DatasetError(
                f"task {self.task_id} has no verification command. An ungradable task makes a "
                "failing model and a succeeding one look identical, so the measured "
                "capability is zero either way."
            )
        unknown = sorted(set(self.verification) - set(VERIFIERS))
        if unknown:
            raise DatasetError(
                f"task {self.task_id} verifies with {unknown}, which are not among "
                f"{list(VERIFIERS)}. Add the verifier to the catalogue rather than asserting "
                "a command the harness will not run."
            )

    @property
    def is_multi_turn(self) -> bool:
        """Whether solving it requires acting on a result, not one shot."""
        return "go_test" in self.verification

    def to_record(self) -> dict[str, Any]:
        """Serialisable task, as the eval harness and the corpus tool read it."""
        return {
            "id": self.task_id,
            "repository": self.repository,
            "package": self.package,
            "prompt": self.prompt,
            "verification": list(self.verification),
            "fixture": self.fixture,
            "has_images": self.has_images,
            "metadata": dict(self.metadata or {}),
        }


def has_go(root: str | Path) -> bool:
    """True when the directory holds a Go module or package."""
    base = Path(root)
    if not base.is_dir():
        return False
    if any(base.glob("go.mod")) or any(base.glob("go.work")):
        return True
    return any(base.rglob("*.go"))


def go_version() -> str | None:
    """The installed Go toolchain version, or None when there is none."""
    found = shutil.which("go")
    if found is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path from which()
            [found, "version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_verifier(root: str | Path, verifier: str, *, timeout_s: int = 300) -> tuple[int, str, str]:
    """Run one verification command; returns ``(exit_code, stdout, stderr)``.

    A missing Go toolchain is a setup error, not a passing task, and is reported
    as such rather than as a non-zero exit the caller might mistake for a failure
    of the work.
    """
    if verifier not in VERIFIERS:
        raise DatasetError(f"unknown verifier {verifier!r}; expected one of {list(VERIFIERS)}")
    found = shutil.which("go")
    if found is None:
        raise DatasetError("the Go toolchain is required to verify a task but is not installed")
    argv = [found, "test", "./..."] if verifier == "go_test" else [found, "build", "./..."]
    try:
        completed = subprocess.run(  # noqa: S603 - absolute path from which()
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"{verifier} timed out after {timeout_s}s"
    return completed.returncode, completed.stdout, completed.stderr


def task_works_now(root: str | Path, task: GoTask) -> bool:
    """Whether the task is already satisfied before the model touches it.

    The task's ``fixture`` is resolved against ``root`` exactly as
    :func:`validate_task` resolves it. Reading the two differently would let a
    task be validated against one directory and executed against another, and the
    mismatch shows up as a task that can never be satisfied.

    A task the base model already passes contributes nothing to a *capability*
    measurement -- it may still be useful as a no-op rehearsal, but it cannot be
    counted as evidence, and a corpus built entirely of them measures nothing.
    """
    base = Path(root) / task.fixture
    for verifier in task.verification:
        code, _, _ = run_verifier(base, verifier)
        if code != 0:
            return False
    return True


def validate_task(task: GoTask, fixture_root: str | Path) -> GoTask:
    """Check a task's fixture exists and its verification is runnable."""
    root = Path(fixture_root) / task.fixture
    if not root.is_dir():
        raise DatasetError(f"task {task.task_id} points at a missing fixture: {root}")
    if not has_go(root):
        raise DatasetError(
            f"task {task.task_id} fixture {root} holds no Go module, so its verifiers cannot "
            "run and the task is ungradable."
        )
    return task


def flip_equality(text: str) -> str | None:
    """Rewrite the first equality comparison into its opposite, or ``None``.

    A comparison compiles, changes behaviour, and surfaces as a failing assertion whose
    diagnostic names the value and the expectation -- which is the loop the model is
    being trained on: read the failure, find the code, change it, run the tests again.
    The mutations that only *break the build* are easier to write and easier to game,
    because the compiler names the fix.

    ``None`` when there is no equality to flip, rather than a rewrite that changes
    nothing: a mutation that changes nothing would be stored as a task that is already
    solved.
    """
    index = text.find("==")
    if index < 0:
        return None
    return text[:index] + "!=" + text[index + 2 :]


def introduce_failures(
    root: str | Path,
    packages: Sequence[str],
    *,
    limit: int = 1,
    timeout_s: int = 300,
    verifier: Callable[[Path, str], tuple[int, str, str]] | None = None,
) -> list[str]:
    """Make some passing packages fail, so there is something to ask for.

    The task builder keeps only packages whose verification *fails*, because a package
    the base model already passes measures nothing. A healthy repository therefore
    yields no tasks at all -- measured: gin's first three packages all passed, and the
    builder wrote an empty file.

    The change is reverted unless it actually makes the tests fail, and packages whose
    tests already fail are left alone: those failures are not ours, and storing one as
    a task would credit this step with work it did not do.

    ``verifier`` is the seam the tests use. The default runs the real toolchain, so no
    caller has to know it exists.

    Returns the packages it broke, so the caller can say what it did.
    """
    check = verifier or (lambda base, name: run_verifier(base, name, timeout_s=timeout_s))
    broken: list[str] = []
    for package in packages:
        if len(broken) >= limit:
            break
        base = Path(root) / package
        if check(base, "go_test")[0] != 0:
            continue  # already failing, so not something this step caused
        for path in sorted(base.glob("*.go")):
            if path.name.endswith("_test.go"):
                continue
            original = path.read_text(encoding="utf-8")
            mutated = flip_equality(original)
            if mutated is None:
                continue
            path.write_text(mutated, encoding="utf-8")
            if check(base, "go_test")[0] != 0:
                broken.append(package)
                break
            path.write_text(original, encoding="utf-8")
    return broken


def build_task_from_package(
    root: str | Path,
    *,
    repository: str,
    package: str,
    prompt: str,
    task_id: str | None = None,
    verification: Sequence[str] = ("go_test",),
) -> GoTask:
    """Build a task from a package directory that exists and is Go.

    Refuses a package that is not there. A task pointing at a missing fixture
    would be stored, counted toward the corpus, and only discovered to be
    unrunnable when the eval tried to execute it.
    """
    base = Path(root) / package
    if not base.is_dir():
        raise DatasetError(f"package {package} does not exist under {root}")
    if not any(base.glob("*.go")):
        raise DatasetError(f"package {package} contains no .go files")
    return GoTask(
        task_id=task_id or f"{repository}:{package}",
        repository=repository,
        package=package,
        prompt=prompt,
        verification=tuple(verification),
        fixture=package,
    )


def write_tasks(path: str | Path, tasks: Sequence[GoTask]) -> int:
    """Write tasks as JSONL for the eval harness; returns how many."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(task.to_record(), ensure_ascii=False, sort_keys=True) + "\n" for task in tasks
    )
    target.write_text(body, encoding="utf-8", newline="\n")
    return len(tasks)


def read_tasks(path: str | Path) -> list[GoTask]:
    """Read a task file, failing loudly on anything malformed."""
    source = Path(path)
    if not source.is_file():
        raise DatasetError(f"task file not found: {source}")
    tasks: list[GoTask] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{source}:{number} is not valid JSON: {exc}") from exc
        try:
            tasks.append(
                GoTask(
                    task_id=str(record["id"]),
                    repository=str(record["repository"]),
                    package=str(record["package"]),
                    prompt=str(record["prompt"]),
                    verification=tuple(str(v) for v in record["verification"]),
                    fixture=str(record["fixture"]),
                    has_images=bool(record.get("has_images", False)),
                    metadata=record.get("metadata") or {},
                )
            )
        except KeyError as exc:
            raise DatasetError(f"{source}:{number} is missing field {exc}") from exc
    return tasks


def harvest_packages(root: str | Path, *, limit: int = 10_000) -> list[str]:
    """List import-path-shaped directories under a repository, deterministically.

    Sorted so two runs over the same tree produce the same task list in the same
    order; a task set that reshuffles between runs makes a corpus diff unreadable.
    """
    base = Path(root)
    if not base.is_dir():
        raise DatasetError(f"repository not found: {base}")
    found: set[str] = set()
    for go_file in sorted(base.rglob("*.go")):
        directory = go_file.parent
        if directory == base or any(part.startswith(".") for part in directory.parts):
            continue
        found.add(directory.relative_to(base).as_posix())
        if len(found) >= limit:
            break
    return sorted(found)
