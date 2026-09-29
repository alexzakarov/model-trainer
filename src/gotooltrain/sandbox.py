"""Container isolation for Go execution.

Untrusted model output is code, and code runs commands. Running it on the host
would put the eval machine inside the blast radius; running it in a fresh
container per task would cost more throughput than the parallelism is worth. So:
a **warm pool** of containers, reused across tasks, with the guarantees that
matter:

* **No network.** Dependencies are vendored in the fixture; nothing needs to
  fetch anything at run time.
* **Read-only root filesystem.** Only the task workspace and the Go caches are
  writable, mounted from outside.
* **Memory and CPU caps** per worker, so one runaway test cannot starve the host.
* **A fresh workspace per task**, restored from a read-only fixture. Two tasks
  sharing a directory is the contamination bug that silently changes results.
* **Automatic recycling**: a worker whose container died is destroyed and
  replaced, and the failure is reported as a harness error rather than retried.

The Docker specifics live behind :class:`ContainerBackend`, so the pool logic
(reservation, recycling, workspace resets, error classification) is testable
without a Docker daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, LifoQueue
from typing import Any, Final, Protocol

from .errors import HarnessError
from .gorun import ExecRequest

#: Marker file proving a workspace was restored from the fixture.
_FIXTURE_MARKER: Final[str] = ".spargia-fixture"

#: Marker file proving the daemon can see the pool's workspace root.
_MOUNT_MARKER: Final[str] = ".spargia-mount-probe"

#: Writable paths inside the read-only container.
_GO_TMP: Final[str] = "/tmp"


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """How a worker container is created.

    ``network`` exists only so its value is visible in a review. It is rejected at
    construction time rather than silently honoured, because a networked worker
    invalidates every isolation claim this module makes.
    """

    image: str
    memory_mb: int = 2048
    cpus: float = 1.0
    read_only_rootfs: bool = True
    network: bool = False
    #: Extra directories that must be writable for the Go toolchain.
    writable_tmpfs: tuple[str, ...] = ("/tmp", "/root/.cache/go-build", "/go/pkg/mod")

    def __post_init__(self) -> None:
        """Reject configurations that would void the isolation guarantees."""
        if self.network:
            raise HarnessError(
                "sandbox network access is not permitted: dependencies must be vendored so a "
                "worker cannot fetch code at run time"
            )
        if self.memory_mb < 256:
            raise HarnessError(f"memory_mb must be >= 256, got {self.memory_mb}")
        if self.cpus <= 0:
            raise HarnessError(f"cpus must be > 0, got {self.cpus}")

    def docker_run_argv(self, name: str, workspace: Path) -> list[str]:
        """Build the ``docker run`` argv for one warm worker."""
        argv: list[str] = [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            f"{self.memory_mb}m",
            "--cpus",
            str(self.cpus),
            "--workdir",
            "/workspace",
            "--volume",
            f"{workspace.resolve()}:/workspace",
        ]
        if self.read_only_rootfs:
            argv.append("--read-only")
        for path in self.writable_tmpfs:
            # `exec` is required and not optional: Docker mounts every --tmpfs
            # noexec by default, and `go test` compiles each test binary into
            # GOCACHE and then execs it. Without exec the mount turns every
            # go_test call into "fork/exec ...: permission denied", which reads
            # as a failing task rather than as a worker that cannot run tests.
            argv.extend(["--tmpfs", f"{path}:rw,nosuid,nodev,exec"])
        argv.extend(["--security-opt", "no-new-privileges", self.image, "sleep", "infinity"])
        return argv


class ContainerBackend(Protocol):
    """The Docker operations the pool needs."""

    def start(self, name: str, workspace: Path, spec: SandboxSpec) -> str:
        """Start a container and return its id."""

    def exec_in(
        self, container_id: str, argv: Sequence[str], timeout_s: int
    ) -> tuple[int | None, str, str]:
        """Run an argv in the container; return ``(exit_code, stdout, stderr)``."""

    def stop(self, container_id: str) -> None:
        """Destroy the container, tolerating one that is already gone."""


class DockerBackend:
    """Real backend. Every call is an argv list; no shell is ever used."""

    def __init__(self) -> None:
        """Track which host directories the daemon has already been shown to mount."""
        self._verified: dict[str, bool] = {}

    def start(self, name: str, workspace: Path, spec: SandboxSpec) -> str:
        """Start a detached worker container and return its id."""
        self.assert_mountable(workspace, spec)
        completed = self._run(["docker", *spec.docker_run_argv(name, workspace)[1:]], timeout_s=120)
        if completed.returncode != 0:
            raise HarnessError(
                f"could not start worker {name!r} from image {spec.image!r}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.stdout.strip()

    def assert_mountable(self, workspace: Path, spec: SandboxSpec) -> None:
        """Fail now if the daemon cannot actually mount the pool's workspace root.

        A directory the daemon cannot resolve does not stop the container from
        starting. The container comes up with an empty or unrelated ``/workspace``
        and the first ``docker exec`` dies with "current working directory is
        outside of container mount namespace root -- possible container breakout
        detected". Left alone that surfaces as a non-zero exit on every sample of
        a run, attributed to the model, and the real cause -- a host path the
        daemon does not share -- is never named.

        The common cause is Docker Desktop's WSL integration: the daemon runs on
        Windows, so it can only bind-mount paths reached through ``/mnt``, while
        ``tempfile.gettempdir()`` inside WSL is ``/tmp``, which it cannot see. The
        remedy is to point the pool's ``root`` at a shared directory. The check
        costs one short-lived container per distinct root and then stops asking.
        """
        key = str(workspace.parent)
        cached = self._verified.get(key)
        if cached is True:
            return
        workspace.mkdir(parents=True, exist_ok=True)
        marker = workspace / _MOUNT_MARKER
        marker.write_text(workspace.name, encoding="utf-8")
        try:
            completed = self._run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--volume",
                    f"{workspace.resolve()}:/probe",
                    spec.image,
                    "sh",
                    "-c",
                    f"test -f /probe/{_MOUNT_MARKER}",
                ],
                timeout_s=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise HarnessError(
                f"the Docker daemon did not return for a mount probe of {key}. Docker Desktop's "
                f"WSL integration cannot bind-mount WSL-internal paths such as /tmp; set the "
                f"pool's root to a directory under /mnt that the daemon shares"
            ) from exc
        finally:
            marker.unlink(missing_ok=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise HarnessError(
                f"the Docker daemon cannot mount the pool workspace root {key}, so a worker "
                f"would see an empty /workspace and every sample would fail as a model error. "
                f"Set the pool's root to a directory the daemon shares (under /mnt when the "
                f"daemon is Docker Desktop on WSL). The daemon said: {detail}"
            )
        self._verified[key] = True

    def exec_in(
        self, container_id: str, argv: Sequence[str], timeout_s: int
    ) -> tuple[int | None, str, str]:
        """Run an argv in a running container via ``docker exec``.

        A container that has died raises :class:`HarnessError` rather than
        returning a non-zero exit code. This distinction is the whole reason the
        pool can tell infrastructure loss from a failing task: ``docker exec``
        against a removed container exits 1 with "No such container" on stderr,
        which is indistinguishable from a command that failed on its own merits.
        Classified as a tool error it would be scored as a model failure, and a
        run whose workers died would report a confidently wrong number. So the
        container's real state is checked rather than inferred from the message.
        """
        command = ["docker", "exec", "-w", "/workspace", container_id, *argv]
        try:
            completed = self._run(command, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout_s}s"
        if completed.returncode != 0 and not self.is_running(container_id):
            raise HarnessError(
                f"worker container {container_id[:12]} is not running; "
                f"docker exec reported: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        return completed.returncode, completed.stdout, completed.stderr

    def is_running(self, container_id: str) -> bool:
        """Ask Docker whether the container still exists and is up.

        The probe is a separate call rather than a match on ``exec``'s stderr, so
        it stays correct if the daemon rewords that message.
        """
        completed = self._run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_id], timeout_s=60
        )
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def stop(self, container_id: str) -> None:
        """Force-remove a container, best effort."""
        # Best effort by design: a container that is already gone is the goal
        # state, and a stop failure must not mask the original error.
        subprocess.run(  # noqa: S603 - argv list, shell=False
            ["docker", "rm", "--force", container_id],
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _run(argv: Sequence[str], *, timeout_s: int) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603 - argv list, shell=False
                list(argv),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                # An explicit working directory, because inheriting the process's
                # own is not safe. Starting a worker bind-mounts a directory, and
                # under WSL that alone invalidates getcwd() for every process
                # whose working directory is on the same drive as the mount --
                # a DrvFs effect, reproducible with `docker run -v` alone. A
                # harness long enough to run an eval would then fail its very
                # next subprocess with ENOENT, blamed on the model. The docker
                # CLI resolves no relative paths here, so pinning the directory
                # to the system temp is safe and keeps the failure out.
                cwd=tempfile.gettempdir(),
            )
        except FileNotFoundError as exc:
            raise HarnessError(f"{argv[0]} is not installed: {exc}") from exc


@dataclass(slots=True)
class Worker:
    """One warm container plus the workspace it owns."""

    container_id: str
    workspace: Path
    name: str
    #: Bumped on every checkout so a stale result can never be attributed to a
    #: recycled container.
    generation: int = 0


@dataclass
class ContainerPool:
    """A warm pool of workers, reused across tasks and recycled when they die."""

    backend: ContainerBackend
    spec: SandboxSpec
    size: int = 2
    #: Where per-worker workspaces are created. Defaults under the system temp
    #: directory rather than to a relative path: a relative default writes into
    #: whatever directory the process happens to have been started from, which in
    #: practice means scattering scratch state through a source tree.
    root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "gotooltrain-sandbox")
    _idle: LifoQueue[Worker] = field(default_factory=LifoQueue, init=False, repr=False)
    _live: dict[str, Worker] = field(default_factory=dict, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        """Warm the pool. Fails loudly if any worker cannot start."""
        if self.size < 1:
            raise HarnessError(f"pool size must be >= 1, got {self.size}")
        if self._started:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(self.size):
            self._spawn()
        self._started = True

    def _spawn(self) -> Worker:
        name = f"gotooltrain-{uuid.uuid4().hex[:12]}"
        workspace = self.root / name
        workspace.mkdir(parents=True, exist_ok=True)
        container_id = self.backend.start(name, workspace, self.spec)
        worker = Worker(container_id=container_id, workspace=workspace, name=name)
        self._live[name] = worker
        self._idle.put(worker)
        return worker

    def _discard(self, worker: Worker) -> None:
        """Destroy a worker and drop it from the pool's bookkeeping."""
        self._live.pop(worker.name, None)
        try:
            self.backend.stop(worker.container_id)
        finally:
            shutil.rmtree(worker.workspace, ignore_errors=True)

    def checkout(self) -> Worker:
        """Reserve a worker, starting the pool on first use."""
        if not self._started:
            self.start()
        try:
            worker = self._idle.get_nowait()
        except Empty as exc:
            raise HarnessError("no idle worker available") from exc
        worker.generation += 1
        return worker

    def checkin(self, worker: Worker) -> None:
        """Return a healthy worker to the idle set."""
        if worker.name in self._live:
            self._idle.put(worker)

    def recycle(self, worker: Worker) -> Worker:
        """Destroy a dead worker and start a replacement."""
        self._discard(worker)
        return self._spawn()

    def stop(self) -> None:
        """Destroy every worker. Safe to call more than once."""
        while not self._idle.empty():
            try:
                worker = self._idle.get_nowait()
            except Empty:  # pragma: no cover - defensive drain
                break
            self._discard(worker)
        self._live.clear()
        self._started = False

    def restore_workspace(self, worker: Worker, fixture: Path) -> None:
        """Reset a worker's workspace from a read-only fixture.

        A task must never see files another task wrote, and the reset must
        achieve that *without* replacing the workspace directory itself.

        The directory is emptied in place and the fixture copied into it, rather
        than removed and recreated. The workspace is the target of a Docker bind
        mount, and a bind mount is bound to the directory's inode: deleting that
        inode leaves the container's ``/workspace`` pointing at a detached
        directory. Linux then refuses to exec in it -- "current working directory
        is outside of container mount namespace root" -- and the mount keeps
        showing the pre-delete contents. Windows hides this, because its mounts
        are resolved by path on every access, so the suite passed there and failed
        only under a Linux daemon. Recreating the directory would also leave the
        restored fixture invisible inside the running worker.

        Entries are removed one at a time so nothing survives the reset, which is
        the property the previous remove-and-recreate approach was reaching for.
        """
        if not fixture.is_dir():
            raise HarnessError(f"fixture directory not found: {fixture}")
        root = worker.workspace
        root.mkdir(parents=True, exist_ok=True)
        for entry in root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        for entry in fixture.iterdir():
            target = root / entry.name
            if entry.is_dir() and not entry.is_symlink():
                shutil.copytree(entry, target, symlinks=True)
            else:
                shutil.copy2(entry, target, follow_symlinks=False)
        (root / _FIXTURE_MARKER).write_text(worker.name, encoding="utf-8")

    @property
    def live_workers(self) -> int:
        """How many containers the pool currently owns."""
        return len(self._live)


class ContainerExecutor:
    """Adapts the pool to the harness ``Executor`` interface."""

    def __init__(self, pool: ContainerPool, fixture: Path | None = None) -> None:
        """Wrap a pool; ``fixture`` is restored before every task."""
        self.pool = pool
        self.fixture = fixture

    def run(self, argv: Sequence[str], request: ExecRequest) -> tuple[int | None, str, str]:
        """Run one argv in a pooled container, recycling it if the container died."""
        if not argv:
            return 0, "", ""
        worker = self.pool.checkout()
        try:
            if self.fixture is not None:
                self.pool.restore_workspace(worker, self.fixture)
            try:
                return self.pool.backend.exec_in(worker.container_id, argv, request.timeout_s)
            except HarnessError:
                # The container is gone or unusable: replace it so the pool keeps
                # its size, and report the infrastructure loss to the caller.
                self.pool.recycle(worker)
                raise
            finally:
                if worker.name in self.pool._live:
                    self.pool.checkin(worker)
        except HarnessError:
            raise
        except Exception as exc:  # pragma: no cover - backend contract violation
            raise HarnessError(f"container backend failed unexpectedly: {exc}") from exc


def docker_available() -> bool:
    """True when a usable Docker CLI is present.

    Checked before a fan-out so an absent Docker is a clear setup error rather
    than a pile of harness errors attributed to the model.
    """
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def container_summary(pool: ContainerPool) -> dict[str, Any]:
    """Small dict for run manifests, so a run records its isolation settings."""
    return {
        "image": pool.spec.image,
        "memory_mb": pool.spec.memory_mb,
        "cpus": pool.spec.cpus,
        "read_only_rootfs": pool.spec.read_only_rootfs,
        "network": pool.spec.network,
        "pool_size": pool.size,
    }
