"""Container isolation: guarantees, recycling, workspace isolation.

The Docker specifics sit behind a protocol, so the guarantees that matter --
no network, read-only root, per-task workspace, recycling a dead worker -- are
verified without a Docker daemon.
"""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Sequence
from typing import Any

import pytest

from gotooltrain import HarnessError
from gotooltrain.gorun import ExecRequest
from gotooltrain.sandbox import (
    ContainerExecutor,
    ContainerPool,
    SandboxSpec,
    container_summary,
    docker_available,
)

IMAGE = "golang:1.23-bookworm"


class FakeBackend:
    """Records container lifecycle and hands out argv results."""

    def __init__(self, fail_exec_for: set[str] | None = None) -> None:
        """Scripted container backend that records every lifecycle call."""
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.execs: list[tuple[str, list[str]]] = []
        self.fail_exec_for = fail_exec_for or set()
        self.fail_next_start = 0
        self._counter = 0

    def start(self, name: str, workspace: pathlib.Path, spec: SandboxSpec) -> str:
        """Pretend to start a container, optionally failing on demand."""
        if self.fail_next_start > 0:
            self.fail_next_start -= 1
            raise HarnessError("image not found")
        self._counter += 1
        container_id = f"cid-{self._counter}"
        self.started.append(name)
        (workspace / "keep.txt").write_text("workspace", encoding="utf-8")
        return container_id

    def exec_in(
        self, container_id: str, argv: Sequence[str], timeout_s: int
    ) -> tuple[int | None, str, str]:
        """Pretend to exec, raising for containers marked as dead."""
        self.execs.append((container_id, list(argv)))
        if container_id in self.fail_exec_for:
            raise HarnessError("container is not running")
        return 0, "ok", ""

    def stop(self, container_id: str) -> None:
        """Record the destroyed container."""
        self.stopped.append(container_id)


@pytest.fixture
def spec() -> SandboxSpec:
    return SandboxSpec(image=IMAGE)


def make_pool(
    backend: FakeBackend, spec: SandboxSpec, tmp_path: pathlib.Path, size: int = 2
) -> ContainerPool:
    return ContainerPool(backend=backend, spec=spec, size=size, root=tmp_path / "ws")


# ------------------------------------------------------------------ spec rules


def test_network_is_refused_at_construction() -> None:
    """A networked worker invalidates every isolation claim in the module."""
    with pytest.raises(HarnessError, match="network access is not permitted"):
        SandboxSpec(image=IMAGE, network=True)


def test_memory_and_cpu_floors() -> None:
    with pytest.raises(HarnessError, match="memory_mb must be"):
        SandboxSpec(image=IMAGE, memory_mb=64)
    with pytest.raises(HarnessError, match="cpus must be"):
        SandboxSpec(image=IMAGE, cpus=0)


def test_docker_run_argv_encodes_the_isolation() -> None:
    argv = SandboxSpec(image=IMAGE, memory_mb=3072, cpus=2).docker_run_argv("w1", pathlib.Path())
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv
    assert argv[argv.index("--memory") + 1] == "3072m"
    assert argv[argv.index("--cpus") + 1] == "2"
    assert "no-new-privileges" in argv
    assert argv[-1] == "infinity"


def test_writable_tmpfs_covers_the_go_caches() -> None:
    spec = SandboxSpec(image=IMAGE)
    argv = spec.docker_run_argv("w1", pathlib.Path())
    tmpfs = {argv[i + 1].split(":")[0] for i, a in enumerate(argv) if a == "--tmpfs"}
    assert {"/tmp", "/go/pkg/mod", "/root/.cache/go-build"} <= tmpfs
    assert all(
        argv[i + 1].endswith("nosuid,nodev,exec") for i, a in enumerate(argv) if a == "--tmpfs"
    )


def test_the_writable_mounts_stay_executable() -> None:
    """Docker mounts every --tmpfs noexec unless told otherwise.

    `go test` compiles each test binary into the build cache and then execs it, so
    a noexec cache turns every go_test call into "permission denied" -- a
    failure that looks like a failing task rather than an unusable worker. Found
    by running a real container; the fake-backend suite could not see it.
    """
    argv = SandboxSpec(image=IMAGE).docker_run_argv("w1", pathlib.Path())
    for index, arg in enumerate(argv):
        if arg == "--tmpfs":
            assert "noexec" not in argv[index + 1]
            assert "exec" in argv[index + 1]


def test_read_only_root_can_be_disabled_explicitly() -> None:
    argv = SandboxSpec(image=IMAGE, read_only_rootfs=False).docker_run_argv("w1", pathlib.Path())
    assert "--read-only" not in argv


# ------------------------------------------------------------------ lifecycle


def test_the_workspace_directory_survives_a_reset(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    """A reset must empty the directory, never replace it.

    The workspace is the target of a Docker bind mount, and a bind mount is bound
    to the directory's inode. Deleting and recreating it detaches the mount: under
    a Linux daemon the container's /workspace then points at a directory the
    runtime refuses to exec in, and the mount keeps showing the pre-delete
    contents. Windows hides this, because its mounts resolve by path on every
    access, so the fake-backend suite passed there and a real Linux worker failed.
    """
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "keep.txt").write_text("keep\n", encoding="utf-8")

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    worker = pool.checkout()
    worker.workspace.mkdir(parents=True, exist_ok=True)
    (worker.workspace / "stale.txt").write_text("stale\n", encoding="utf-8")
    inode_before = worker.workspace.stat().st_ino

    pool.restore_workspace(worker, fixture)

    assert worker.workspace.stat().st_ino == inode_before, "the workspace was replaced, not emptied"
    assert (worker.workspace / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert not (worker.workspace / "stale.txt").exists()


def test_a_reset_removes_nested_stale_content(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    """Contamination is recursive: a stale package directory counts just as much."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "fresh.txt").write_text("fresh\n", encoding="utf-8")

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    worker = pool.checkout()
    stale = worker.workspace / "old" / "deep"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / "leftover.go").write_text("package old\n", encoding="utf-8")

    pool.restore_workspace(worker, fixture)

    assert not (worker.workspace / "old").exists()
    assert (worker.workspace / "fresh.txt").exists()


def test_a_reset_keeps_the_fixture_it_reads_from(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    """Copying a fixture into itself would delete the very files being copied."""
    fixture = tmp_path / "fixture"
    (fixture / "pkg").mkdir(parents=True)
    (fixture / "pkg" / "a.go").write_text("package a\n", encoding="utf-8")

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    worker = pool.checkout()
    worker.workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture, worker.workspace, dirs_exist_ok=True)

    pool.restore_workspace(worker, fixture)

    assert (worker.workspace / "pkg" / "a.go").read_text(encoding="utf-8") == "package a\n"
    assert (fixture / "pkg" / "a.go").exists()


def test_pool_warms_to_its_full_size(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=3)
    pool.start()
    assert backend.started and len(backend.started) == 3
    assert pool.live_workers == 3


def test_start_is_idempotent(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path)
    pool.start()
    pool.start()
    assert len(backend.started) == pool.size


def test_pool_refuses_to_start_when_a_worker_fails(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    backend = FakeBackend()
    backend.fail_next_start = 1
    pool = make_pool(backend, spec, tmp_path, size=2)
    with pytest.raises(HarnessError, match="image not found"):
        pool.start()


def test_pool_rejects_a_zero_size(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    with pytest.raises(HarnessError, match="pool size must be"):
        make_pool(FakeBackend(), spec, tmp_path, size=0).start()


def test_checkout_starts_the_pool_lazily(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path)
    worker = pool.checkout()
    assert worker.container_id
    assert backend.started


def test_checkin_makes_a_worker_available_again(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path)
    pool.start()
    worker = pool.checkout()
    pool.checkin(worker)
    assert pool.checkout().name == worker.name


def test_checkout_reports_an_exhausted_pool(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    pool.checkout()
    with pytest.raises(HarnessError, match="no idle worker"):
        pool.checkout()


def test_generation_increments_so_stale_results_are_detectable(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    first = pool.checkout()
    before = first.generation
    pool.checkin(first)
    second = pool.checkout()
    assert first is second, "size 1 must hand back the same worker"
    assert second.generation == before + 1


def test_checkin_ignores_a_worker_that_is_no_longer_live(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    """Checking in a discarded worker must not resurrect it into the idle set."""
    from gotooltrain.sandbox import Worker

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    ghost = Worker(container_id="cid-ghost", workspace=pool.root / "ghost", name="ghost")
    pool.checkin(ghost)
    real = pool.checkout()
    assert real.name != "ghost", "a dead worker must never be handed out"


def test_stop_destroys_every_worker(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=2)
    pool.start()
    pool.stop()
    assert len(backend.stopped) == 2
    assert pool.live_workers == 0


def test_stop_is_safe_to_call_twice(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    pool.stop()
    pool.stop()
    assert pool.live_workers == 0


def test_stop_after_stop_allows_a_fresh_start(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=1)
    pool.start()
    pool.stop()
    pool.start()
    assert len(backend.started) == 2 * pool.size


# ------------------------------------------------------------------ recycling


def test_recycle_replaces_a_dead_worker(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=1)
    pool.start()
    dead = pool.checkout()
    replacement = pool.recycle(dead)
    assert replacement.container_id != dead.container_id
    assert dead.container_id in backend.stopped
    assert pool.live_workers == 1


# ------------------------------------------------------------------ workspace


def test_workspace_is_restored_from_the_fixture(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    fixture = tmp_path / "fixture"
    (fixture / "parser").mkdir(parents=True)
    (fixture / "go.mod").write_text("module example\n", encoding="utf-8")

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    worker = pool.checkout()
    (worker.workspace / "stale.go").write_text("leftover", encoding="utf-8")

    pool.restore_workspace(worker, fixture)
    assert (worker.workspace / "go.mod").is_file()
    assert not (worker.workspace / "stale.go").exists(), "a reset must not keep task leftovers"
    assert (worker.workspace / ".spargia-fixture").read_text(encoding="utf-8") == worker.name


def test_workspace_restore_requires_a_real_fixture(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    pool.start()
    worker = pool.checkout()
    with pytest.raises(HarnessError, match="fixture directory not found"):
        pool.restore_workspace(worker, tmp_path / "missing")


# ----------------------------------------------------------------- executor


def _request() -> ExecRequest:
    return ExecRequest(
        task_id="t1",
        sample_index=0,
        tool_name="go_test",
        arguments={"pkg": "./..."},
        workspace=pathlib.Path(),
    )


def test_executor_runs_inside_a_container(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=1)
    executor = ContainerExecutor(pool)
    assert executor.run(["rtk", "go", "test", "./..."], _request()) == (0, "ok", "")
    assert backend.execs[0][1] == ["rtk", "go", "test", "./..."]


def test_executor_resets_the_workspace_per_task(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "go.mod").write_text("module example\n", encoding="utf-8")

    pool = make_pool(FakeBackend(), spec, tmp_path, size=1)
    executor = ContainerExecutor(pool, fixture=fixture)
    executor.run(["rtk", "go", "test", "./..."], _request())

    # The single worker is idle again; its workspace must hold the fixture and
    # nothing the previous task left behind.
    worker = pool.checkout()
    assert (worker.workspace / "go.mod").is_file()
    assert (worker.workspace / ".spargia-fixture").is_file()
    assert not (worker.workspace / "keep.txt").exists(), "task leftovers must be gone"
    pool.checkin(worker)


def test_executor_short_circuits_a_commandless_request(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    backend = FakeBackend()
    executor = ContainerExecutor(make_pool(backend, spec, tmp_path))
    assert executor.run([], _request()) == (0, "", "")
    assert backend.execs == []


def test_executor_recycles_and_reports_a_dead_container(
    tmp_path: pathlib.Path, spec: SandboxSpec
) -> None:
    backend = FakeBackend()
    pool = make_pool(backend, spec, tmp_path, size=1)
    pool.start()
    worker = pool.checkout()
    backend.fail_exec_for.add(worker.container_id)
    pool.checkin(worker)

    executor = ContainerExecutor(pool)
    with pytest.raises(HarnessError, match="not running"):
        executor.run(["rtk", "go", "test", "./..."], _request())

    assert worker.container_id in backend.stopped, "the dead container must be destroyed"
    assert pool.live_workers == 1, "the pool must keep its size after recycling"


def test_summary_records_the_isolation_settings(tmp_path: pathlib.Path, spec: SandboxSpec) -> None:
    pool = make_pool(FakeBackend(), spec, tmp_path, size=4)
    summary: dict[str, Any] = container_summary(pool)
    assert summary == {
        "image": IMAGE,
        "memory_mb": 2048,
        "cpus": 1.0,
        "read_only_rootfs": True,
        "network": False,
        "pool_size": 4,
    }


def test_docker_available_returns_a_bool() -> None:
    assert isinstance(docker_available(), bool)


# ------------------------------------------------------------- DockerBackend
#
# The backend's job is building the argv and turning Docker's exit codes into
# typed outcomes. That logic is worth testing without a daemon, so subprocess is
# stubbed; the daemon itself is covered by the canary in a real run.


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):  # type: ignore[no-untyped-def]
    import subprocess

    return subprocess.CompletedProcess(
        args=["docker"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_docker_calls_do_not_inherit_the_process_working_directory(
    monkeypatch: Any, tmp_path: pathlib.Path
) -> None:
    """The harness must not depend on a working directory it cannot control.

    Starting a worker bind-mounts a directory, and under WSL that alone
    invalidates getcwd() for every process whose working directory is on the same
    drive -- reproducible with `docker run -v` and nothing else. A harness long
    enough to run an eval would then fail its very next subprocess with ENOENT,
    and the failure would land on the model. The docker CLI resolves nothing
    relative here, so pinning the directory costs nothing.
    """
    import gotooltrain.sandbox as sb

    seen: list[object] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        """Record the working directory each docker call was given."""
        seen.append(kwargs.get("cwd"))
        return _completed(stdout="cid\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    backend = sb.DockerBackend()
    backend.start("w1", tmp_path / "w1", SandboxSpec(image=IMAGE))
    backend.is_running("cid")

    assert seen, "no docker call was made"
    for cwd in seen:
        assert cwd is not None, "a docker call inherited the process working directory"
        assert pathlib.Path(str(cwd)).is_dir()


def test_the_pinned_directory_is_evaluated_at_call_time(
    monkeypatch: Any, tmp_path: pathlib.Path
) -> None:
    """The pin must be the system temp, not whatever the caller had in mind.

    Pinning the workspace itself would be circular: that directory is the one
    being mounted, so it is exactly the one whose working directory can vanish.
    """
    import gotooltrain.sandbox as sb

    seen: list[object] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs.get("cwd"))
        return _completed(stdout="cid\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    sb.DockerBackend().is_running("cid")

    assert seen == [sb.tempfile.gettempdir()]


def test_backend_start_returns_the_container_id(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        return _completed(stdout="cid-42\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    assert sb.DockerBackend().start("w1", pathlib.Path(), SandboxSpec(image=IMAGE)) == "cid-42"
    # The mount probe runs first, so look for the detached worker among the calls
    # rather than assuming it is the only one.
    assert any(argv[:3] == ["docker", "run", "--detach"] for argv in seen)


def test_a_worker_is_only_probed_once_per_workspace_root(
    monkeypatch: Any, tmp_path: pathlib.Path
) -> None:
    """The probe proves a property of the root, so re-asking per worker is waste.

    A pool of eight workers would otherwise pay eight short-lived containers
    before any real work starts.
    """
    import gotooltrain.sandbox as sb

    probes: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        if "--rm" in argv:
            probes.append(list(argv))
        return _completed(stdout="cid\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    backend = sb.DockerBackend()
    spec = SandboxSpec(image=IMAGE)
    root = tmp_path / "root"
    for index in range(4):
        backend.start(f"w{index}", root / f"w{index}", spec)

    assert len(probes) == 1, f"probed {len(probes)} times for one root"


def test_an_unmountable_workspace_root_is_refused_before_any_worker(
    monkeypatch: Any, tmp_path: pathlib.Path
) -> None:
    """A daemon that cannot see the root must be named, not left to fail per sample.

    Under Docker Desktop's WSL integration the daemon runs on Windows and cannot
    bind-mount /tmp. The container still starts, with an empty /workspace, and the
    first exec dies with a breakout warning that reads like a model error.
    """
    import gotooltrain.sandbox as sb

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        if "--rm" in argv:
            return _completed(1, "", "invalid mount config for type bind")
        return _completed(stdout="cid\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    with pytest.raises(HarnessError, match="cannot mount the pool workspace root"):
        sb.DockerBackend().start("w1", tmp_path / "w1", SandboxSpec(image=IMAGE))


def test_a_hanging_mount_probe_is_reported_as_a_setup_problem(
    monkeypatch: Any, tmp_path: pathlib.Path
) -> None:
    """A daemon that never returns must not hang a run forever."""
    import subprocess

    import gotooltrain.sandbox as sb

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(argv, 120)

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    with pytest.raises(HarnessError, match="did not return for a mount probe"):
        sb.DockerBackend().start("w1", tmp_path / "w1", SandboxSpec(image=IMAGE))


def test_a_failed_probe_leaves_no_marker_behind(monkeypatch: Any, tmp_path: pathlib.Path) -> None:
    """The probe's own file must not be mistaken for fixture content."""
    import gotooltrain.sandbox as sb

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        return _completed(1, "", "no such volume")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    workspace = tmp_path / "w1"
    with pytest.raises(HarnessError):
        sb.DockerBackend().start("w1", workspace, SandboxSpec(image=IMAGE))
    assert not (workspace / ".spargia-mount-probe").exists()


def test_backend_start_wraps_a_failure(monkeypatch: Any) -> None:
    """The worker run itself can fail after the mount probe has already passed."""
    import gotooltrain.sandbox as sb

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        """Let the probe through, then refuse to start the worker."""
        if "--rm" in argv:
            return _completed(0, "", "")
        return _completed(125, "", "no such image")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    with pytest.raises(HarnessError, match="no such image"):
        sb.DockerBackend().start("w1", pathlib.Path(), SandboxSpec(image=IMAGE))


def test_backend_start_reports_a_missing_docker(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("docker")

    monkeypatch.setattr(sb.subprocess, "run", boom)
    with pytest.raises(HarnessError, match="is not installed"):
        sb.DockerBackend().start("w1", pathlib.Path(), SandboxSpec(image=IMAGE))


def test_backend_exec_builds_the_docker_exec_argv(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        return _completed(stdout="ok  parser\n")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    code, out, err = sb.DockerBackend().exec_in("cid-1", ["rtk", "go", "test", "./..."], 30)
    assert (code, out, err) == (0, "ok  parser\n", "")
    assert seen[0] == ["docker", "exec", "-w", "/workspace", "cid-1", "rtk", "go", "test", "./..."]


def test_backend_exec_returns_the_real_exit_code(monkeypatch: Any) -> None:
    """A command that fails on its own merits stays a tool error.

    The container is up in this case, so the exit code and output belong to the
    command; only a *dead* container may be reclassified as infrastructure loss.
    """
    import gotooltrain.sandbox as sb

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        """Fail the exec, and report the container as still running."""
        if "inspect" in argv:
            return _completed(0, "true\n", "")
        return _completed(1, "FAIL", "build failed")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    code, out, err = sb.DockerBackend().exec_in("cid-1", ["x"], 30)
    assert code == 1
    assert "FAIL" in out and "build failed" in err


def test_backend_exec_reports_a_dead_container_as_a_harness_error(monkeypatch: Any) -> None:
    """`docker exec` on a removed container exits 1, exactly like a failed build.

    Without the state check the pool would score a dead worker as a model
    failure, and a run whose workers died would report a confident wrong number
    instead of an infrastructure error. Found by killing a real container.
    """
    import gotooltrain.sandbox as sb

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        """Fail the exec, and report the container as gone."""
        if "inspect" in argv:
            return _completed(1, "", "No such container: cid-1")
        return _completed(1, "", "Error response from daemon: No such container: cid-1")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    with pytest.raises(sb.HarnessError, match="is not running"):
        sb.DockerBackend().exec_in("cid-1", ["x"], 30)


def test_the_running_probe_asks_docker_rather_than_guessing(monkeypatch: Any) -> None:
    """Matching stderr text would break the moment the daemon rewords it."""
    import gotooltrain.sandbox as sb

    seen: list[list[str]] = []

    def fake_run(argv, **kw):  # type: ignore[no-untyped-def]
        seen.append(list(argv))
        return _completed(0, "false\n", "")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    assert sb.DockerBackend().is_running("cid-1") is False
    assert seen[0][:3] == ["docker", "inspect", "--format"]


def test_backend_exec_converts_a_timeout(monkeypatch: Any) -> None:
    import subprocess

    import gotooltrain.sandbox as sb

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(sb.subprocess, "run", boom)
    assert sb.DockerBackend().exec_in("cid-1", ["x"], 5) == (124, "", "timed out after 5s")


def test_backend_stop_is_forceful_and_best_effort(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    seen: list[list[str]] = []
    monkeypatch.setattr(
        sb.subprocess, "run", lambda argv, **kw: seen.append(list(argv)) or _completed(1)
    )
    sb.DockerBackend().stop("cid-9")
    assert seen[0] == ["docker", "rm", "--force", "cid-9"]


def test_docker_available_true_when_the_daemon_answers(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(sb.subprocess, "run", lambda argv, **kw: _completed(0, "27.0.0"))
    assert sb.docker_available() is True


def test_docker_available_false_when_the_daemon_is_down(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _n: "/usr/bin/docker")
    monkeypatch.setattr(sb.subprocess, "run", lambda argv, **kw: _completed(1, "", "no daemon"))
    assert sb.docker_available() is False


def test_docker_available_false_without_the_binary(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _n: None)
    assert sb.docker_available() is False


def test_docker_available_false_when_the_probe_hangs(monkeypatch: Any) -> None:
    import subprocess

    import gotooltrain.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _n: "/usr/bin/docker")

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(sb.subprocess, "run", boom)
    assert sb.docker_available() is False


def test_docker_available_false_when_the_probe_cannot_run(monkeypatch: Any) -> None:
    import gotooltrain.sandbox as sb

    monkeypatch.setattr(sb.shutil, "which", lambda _n: "/usr/bin/docker")

    def boom(argv, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("docker")

    monkeypatch.setattr(sb.subprocess, "run", boom)
    assert sb.docker_available() is False
