"""The sandbox against a real Docker daemon and a real Go toolchain.

Every other sandbox test fakes the backend, which proves the pool's logic but not
that the isolation holds. The properties only a live daemon can confirm are the
ones that matter: the worker really has no network, the root really is read-only,
the workspace really is mounted writable, a restarted container really cannot
report a stale result, and -- the reason this image exists -- the catalog's rtk
tools really run inside it.

Skipped loudly when Docker or the image is absent, never downgraded to a fake: a
test that passes without Docker has tested none of the above, and would report
success for a harness that cannot start a container.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator

import pytest

from gotooltrain.errors import HarnessError, ValidationError
from gotooltrain.evalrun import canary_cases
from gotooltrain.gorun import (
    ExecRequest,
    LocalExecutor,
    Status,
    execute,
    run_canary,
)
from gotooltrain.sandbox import (
    ContainerExecutor,
    ContainerPool,
    DockerBackend,
    SandboxSpec,
    container_summary,
    docker_available,
)
from gotooltrain.workspace import write_file

#: The image carries a pinned rtk, because go_build, go_test, grep and read_file
#: all dispatch through it and a stock golang image has none.
IMAGE = "gotooltrain/go-sandbox:0.1.0"


def image_present(name: str) -> bool:
    """True when the image is already on this host, so no build or network is needed."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            ["docker", "image", "inspect", name],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


pytestmark = [
    pytest.mark.skipif(not docker_available(), reason="no usable Docker daemon on this host"),
    pytest.mark.skipif(
        not image_present(IMAGE),
        reason=f"{IMAGE} is absent; build it with docker/go-sandbox/Dockerfile",
    ),
]


def spec(**overrides: object) -> SandboxSpec:
    base: dict[str, object] = {"image": IMAGE, "memory_mb": 2048, "cpus": 2.0}
    base.update(overrides)
    return SandboxSpec(**base)  # type: ignore[arg-type]


def mountable_root() -> pathlib.Path:
    """A workspace root the Docker daemon can bind-mount, and that survives a run.

    Two constraints fight here. The daemon must be able to *see* the directory:
    under WSL the daemon is Docker Desktop on Windows, so it can only mount paths
    reached through ``/mnt``, and the default ``/tmp`` is invisible to it. The
    directory must also not be the repository, because mounting a directory
    beneath a shared filesystem and then deleting it invalidates the working
    directory of every process rooted there -- after a run, ``os.getcwd()`` in a
    fresh child fails with ENOENT even though the repository is intact, and the
    tools that run next (coverage, mypy, ruff) all abort on it.

    The Windows system drive is the one location that satisfies both: reachable
    from WSL through ``/mnt``, and a local volume for the daemon rather than a
    share the repo happens to live on. The name is unique per process so a killed
    run cannot collide with the next one.
    """
    try:
        wsl = "microsoft" in pathlib.Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        wsl = False
    base = wsl_workspace_root() if wsl else pathlib.Path(tempfile.gettempdir())
    root = base / f"gotooltrain-sandbox-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def wsl_workspace_root() -> pathlib.Path:
    """Where to put worker workspaces when the daemon is Docker Desktop on WSL.

    Asking Windows for ``%TEMP%`` would be tidier, but it needs the WSL interop
    and a PATH that contains it, and this suite has already run once with a
    minimal PATH. Guessing a profile name is worse: it is wrong for any other
    user. So the location is derived from the one path that is already known to
    work -- the repository's own mount point -- and placed *beside* the repository
    rather than inside it.
    """
    repo = pathlib.Path(__file__).resolve().parents[1]
    mount = pathlib.Path("/mnt")
    parts = repo.parts
    if len(parts) > 3 and parts[1] == "mnt":
        # /mnt/<drive>/... -- keep the drive, drop the rest.
        return mount / parts[2] / "gotooltrain-sandbox-root"
    return repo.parent / "gotooltrain-sandbox-root"


@pytest.fixture
def pool() -> Iterator[ContainerPool]:
    """A two-container pool on the real image, torn down even when a test fails."""
    root = mountable_root()
    created = ContainerPool(backend=DockerBackend(), spec=spec(), size=2, root=root)
    try:
        yield created
    finally:
        created.stop()
        shutil.rmtree(root, ignore_errors=True)


def request(tool: str, workspace: pathlib.Path, **arguments: object) -> ExecRequest:
    return ExecRequest(
        task_id="t",
        sample_index=0,
        tool_name=tool,
        arguments=dict(arguments),
        workspace=workspace,
    )


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny Go module with one deliberately failing test, for real runs."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "go.mod").write_text("module example.com/x\n\ngo 1.23\n", encoding="utf-8")
    (root / "x.go").write_text(
        "package x\n\nfunc Count(m map[string]int) int { return len(m) }\n", encoding="utf-8"
    )
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


# ------------------------------------------------------------- the toolchain


def test_the_image_reports_its_own_toolchain() -> None:
    """The image refuses to be built without these; assert the built artefact."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
        ["docker", "run", "--rm", "--network", "none", IMAGE, "selfcheck"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "selfcheck ok" in completed.stdout


def test_a_container_runs_a_real_go_command(pool: ContainerPool) -> None:
    result = execute(request("go_doc", pathlib.Path(), symbol="errors.Is"), ContainerExecutor(pool))
    assert result.status is Status.OK, result.stdout + result.harness_error
    assert "func Is" in result.stdout


def test_the_rtk_tools_run_inside_the_container(pool: ContainerPool, repo: pathlib.Path) -> None:
    """The reason this image exists: the stock golang image fails these.

    Without rtk in the worker, go_test and go_build return a harness error on
    every sample, which reads as a broken model rather than a broken image.
    """
    executor = ContainerExecutor(pool, fixture=repo)
    built = execute(request("go_build", repo, pkg="./..."), executor)
    assert built.status is Status.OK, f"go_build via rtk: {built.stdout}{built.harness_error}"
    tested = execute(request("go_test", repo, pkg="./..."), executor)
    assert tested.status is not Status.HARNESS_ERROR, tested.harness_error
    assert tested.exit_code != 0, "a failing test must not be reported as a harness error"
    assert "want 3" in tested.stdout


def test_the_recall_path_works_inside_the_container(
    pool: ContainerPool, repo: pathlib.Path
) -> None:
    """Rtk elides long output and leaves a hash; the model must be able to read it back."""
    executor = ContainerExecutor(pool, fixture=repo)
    result = execute(request("go_test", repo, pkg="./..."), executor)
    if "[full output: rtk recall" not in result.stdout:
        pytest.skip("rtk did not elide this output, so there is nothing to recall")
    digest = result.stdout.split("rtk recall ")[1].split("]")[0].strip()
    recalled = execute(request("rtk_recall", repo, hash=digest), executor)
    assert recalled.status is Status.OK, recalled.harness_error
    assert "want 3" in recalled.stdout


# ---------------------------------------------------------------- the isolation


def test_a_worker_has_no_network(pool: ContainerPool) -> None:
    """A model that can reach the internet can exfiltrate the repository it is given."""
    executor = ContainerExecutor(pool)
    code, _, _ = executor.run(
        ["sh", "-c", "getent hosts example.com"],
        request("go_doc", pathlib.Path(), symbol="errors.Is"),
    )
    assert code != 0, "DNS resolved from inside a worker that must be offline"


def test_the_root_filesystem_is_read_only(pool: ContainerPool) -> None:
    executor = ContainerExecutor(pool)
    code, _, stderr = executor.run(
        ["sh", "-c", "touch /canary"], request("go_doc", pathlib.Path(), symbol="errors.Is")
    )
    assert code != 0, f"the root was writable: {stderr}"


def test_the_workspace_is_writable(pool: ContainerPool) -> None:
    """A read-only root with no writable workspace would make every task impossible."""
    executor = ContainerExecutor(pool)
    code, _, stderr = executor.run(
        ["sh", "-c", "touch canary.go"], request("go_doc", pathlib.Path(), symbol="errors.Is")
    )
    assert code == 0, f"the workspace volume is not writable: {stderr}"


def test_the_go_caches_are_writable(pool: ContainerPool) -> None:
    """Read-only root plus an unwritable GOCACHE fails every go build."""
    executor = ContainerExecutor(pool)
    code, _, stderr = executor.run(
        ["sh", "-c", 'go env GOCACHE >/dev/null && mkdir -p "$(go env GOCACHE)"'],
        request("go_doc", pathlib.Path(), symbol="errors.Is"),
    )
    assert code == 0, f"GOCACHE is not writable: {stderr}"


def test_a_fixture_reaches_the_worker(pool: ContainerPool, repo: pathlib.Path) -> None:
    executor = ContainerExecutor(pool, fixture=repo)
    code, _, _ = executor.run(
        ["sh", "-c", "test -f x_test.go"], request("go_doc", repo, symbol="errors.Is")
    )
    assert code == 0, "the fixture was not restored into the worker"


def test_a_previous_task_cannot_leak_into_the_next(pool: ContainerPool, repo: pathlib.Path) -> None:
    """The contamination bug: two tasks sharing a directory silently change results."""
    executor = ContainerExecutor(pool, fixture=repo)
    executor.run(
        ["sh", "-c", "echo leaked > leak.txt"], request("go_doc", repo, symbol="errors.Is")
    )
    code, _, _ = executor.run(
        ["sh", "-c", "test -e leak.txt"], request("go_doc", repo, symbol="errors.Is")
    )
    assert code != 0, "a file written by one task survived into the next"


def test_the_manifest_records_the_isolation(pool: ContainerPool) -> None:
    summary = container_summary(pool)
    assert summary["image"] == IMAGE
    assert summary["network"] is False
    assert summary["read_only_rootfs"] is True
    assert summary["pool_size"] == 2


def test_a_networked_worker_is_refused() -> None:
    """Reviewed, not merely defaulted: a networked worker voids every claim above."""
    with pytest.raises(HarnessError, match="network access is not permitted"):
        SandboxSpec(image=IMAGE, network=True)


# -------------------------------------------------------------- dead workers


def test_a_stopped_container_is_recycled_not_trusted(
    pool: ContainerPool, repo: pathlib.Path
) -> None:
    """A restarted container must never return the previous task's result.

    The worker is killed behind the pool's back, then the pool must notice and
    report a harness error for that call rather than reusing the dead id.
    """
    worker = pool.checkout()
    pool.backend.stop(worker.container_id)
    pool.checkin(worker)

    executor = ContainerExecutor(pool, fixture=repo)
    result = execute(request("go_test", repo, pkg="./..."), executor)
    assert result.status is Status.HARNESS_ERROR, result.stdout
    assert result.exit_code is None


def test_the_pool_recovers_after_a_dead_worker(pool: ContainerPool, repo: pathlib.Path) -> None:
    """One dead worker must cost one sample, not the run."""
    worker = pool.checkout()
    pool.backend.stop(worker.container_id)
    pool.checkin(worker)

    executor = ContainerExecutor(pool, fixture=repo)
    first = execute(request("go_test", repo, pkg="./..."), executor)
    assert first.status is Status.HARNESS_ERROR
    second = execute(request("go_test", repo, pkg="./..."), executor)
    assert second.status is not Status.HARNESS_ERROR, second.harness_error
    assert pool.live_workers == 2


def test_a_recycled_worker_gets_a_new_container_id(pool: ContainerPool) -> None:
    worker = pool.checkout()
    before = worker.container_id
    replacement = pool.recycle(worker)
    assert replacement.container_id != before
    assert pool.live_workers == 2


# ------------------------------------------------------------- the canary


def test_the_canary_passes_against_a_real_worker(pool: ContainerPool, repo: pathlib.Path) -> None:
    """A harness that cannot tell pass from fail would score every model 100%."""
    run_canary(ContainerExecutor(pool, fixture=repo), canary_cases(repo))


def test_the_canary_fails_when_the_toolchain_is_missing(
    pool: ContainerPool, repo: pathlib.Path
) -> None:
    """The refusal must be real, or it is decoration."""
    executor = ContainerExecutor(pool, fixture=repo)
    cases = canary_cases(repo)
    for case in cases:
        if case.request.tool_name != "go_doc":
            continue
        broken = type(case)(case.name, case.request, should_fail=True)
        with pytest.raises(HarnessError, match="canary failed"):
            run_canary(executor, (broken,))


def test_the_local_executor_still_works(pool: ContainerPool) -> None:
    """The container is the default, not the only option."""
    result = execute(request("go_doc", pathlib.Path(), symbol="errors.Is"), LocalExecutor())
    assert result.status is Status.OK, result.stdout


def test_a_missing_docker_binary_is_a_setup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PATH without docker must be reported, not surfaced as model failures."""
    import shutil as shutil_module

    from gotooltrain.sandbox import docker_available as available

    monkeypatch.setattr(shutil_module, "which", lambda name: None)
    assert available() is False


def test_a_workspace_write_cannot_escape_the_root(tmp_path: pathlib.Path) -> None:
    """The host-side guard the container's read-only root cannot provide."""
    with pytest.raises(ValidationError, match="escapes the workspace"):
        write_file(tmp_path, {"path": "../outside.go", "content": "package x\n"})
    with pytest.raises(ValidationError, match="absolute"):
        write_file(tmp_path, {"path": "C:/elsewhere.go", "content": "package x\n"})
