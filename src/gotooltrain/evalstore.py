"""Durable, content-addressed store for evaluation artifacts.

Parallel evaluation has two separate obligations, and this module keeps them
apart:

**Durability.** Results are append-only. A run is a directory of shards plus an
event log, never one big file. Writes are atomic (temp file + ``os.replace``) and
the log is fsynced, so a crash mid-run loses at most the attempt in flight and a
resume can reconstruct exactly what completed. A torn final line is a hard
error, never a silently skipped record.

**Idempotency.** Every attempt is addressed by a content key derived from *all*
inputs that can change the answer: task, sample index, stage, model identity and
revision, token-format and template versions, harness and dataset versions,
seed, decode parameters, tool catalogue, and the execution image digest.

The key doubles as a completeness assertion. If two different payloads land on
the same key, the fingerprint is missing a component and the store refuses it.
That catches the common bug -- "someone forgot to put ``model_revision`` in the
key" -- which would otherwise let a retrained model silently reuse stale results
and make a regression look like an improvement.

Determinism caveat: causal-LM sampling is not bit-reproducible (batching changes
the numerics), so re-running generation cannot be guaranteed to match. The store
therefore guarantees *keyed memoisation* -- a key always resolves to the stored
artifact, never a fresh sample -- plus byte-exact verification for the stages
where determinism is achievable (execution, and judge inputs).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from .errors import EvalStoreError, IdempotencyViolationError

STORE_VERSION: Final[str] = "evalstore-v1"
EVENT_LOG: Final[str] = "events.jsonl"
MANIFEST: Final[str] = "manifest.json"
OBJECTS_DIR: Final[str] = "objects"
RUNS_DIR: Final[str] = "runs"

STAGES: Final[tuple[str, ...]] = ("generate", "execute", "judge")

#: Every *input* that can change an attempt's result. Adding a field here is a
#: format change: old keys stay valid, new ones do not collide with them.
#: ``store_version`` is deliberately absent -- it is a constant this build
#: injects into the key payload, not something a caller supplies.
FINGERPRINT_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "sample_index",
    "stage",
    "model_id",
    "model_revision",
    "format_version",
    "template_sha",
    "harness_version",
    "dataset_version",
    "seed",
    "decode_params",
    "tool_catalog_sha",
    "image_digest",
)


__all__ = [
    "EVENT_LOG",
    "FINGERPRINT_FIELDS",
    "MANIFEST",
    "OBJECTS_DIR",
    "RUNS_DIR",
    "STAGES",
    "STORE_VERSION",
    "Fingerprint",
    "ResultStore",
    "attempt_key",
    "canonical_json",
    "iter_fingerprints",
    "sha256_file",
    "sha256_text",
]


def canonical_json(payload: Any) -> str:
    """Stable serialisation used for hashing and for byte comparison."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    """Hex sha256 of a string's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file's bytes; used for template and image identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """The complete input identity of one attempt."""

    task_id: str
    sample_index: int
    stage: str
    model_id: str
    model_revision: str
    format_version: str
    template_sha: str
    harness_version: str
    dataset_version: str
    seed: int
    decode_params: Mapping[str, Any]
    tool_catalog_sha: str
    image_digest: str

    def to_dict(self) -> dict[str, Any]:
        """The full key payload: every input plus this build's store version."""
        return {"store_version": STORE_VERSION, **asdict(self)}

    @property
    def key(self) -> str:
        """Content key: the identity of this attempt."""
        return sha256_text(canonical_json(self.to_dict()))

    def short(self) -> str:
        """Human-readable prefix for log lines."""
        return self.key[:16]


def _fingerprint(**kwargs: Any) -> Fingerprint:
    missing = [name for name in FINGERPRINT_FIELDS if name not in kwargs]
    if missing:
        raise EvalStoreError(
            f"fingerprint is missing required field(s): {missing}. Every field that can "
            f"change the result must be keyed, otherwise stale results are silently reused."
        )
    unknown = [name for name in kwargs if name not in FINGERPRINT_FIELDS]
    if unknown:
        raise EvalStoreError(
            f"unknown fingerprint field(s): {unknown}. Add them to FINGERPRINT_FIELDS if they "
            f"can change the result, or remove them."
        )
    return Fingerprint(**{name: kwargs[name] for name in FINGERPRINT_FIELDS})


def attempt_key(**kwargs: Any) -> str:
    """Content key for an attempt. See :func:`_fingerprint` for required fields."""
    return _fingerprint(**kwargs).key


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader never observes a partial file.

    A crash mid-write leaves at most a ``.tmp`` file, which the next write
    overwrites; the destination is only ever replaced by a complete, fsynced
    file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        staging = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


class ResultStore:
    """Content-addressed objects plus per-run append-only event logs.

    Layout::

        <root>/objects/<aa>/<key>.json     immutable, one per attempt
        <root>/runs/<run_id>/manifest.json
        <root>/runs/<run_id>/events.jsonl  append-only
    """

    def __init__(self, root: str | Path) -> None:
        """Open (or create on first write) a store rooted at ``root``."""
        self.root = Path(root)

    # ------------------------------------------------------------------ objects

    def object_path(self, key: str) -> Path:
        """Filesystem location of a key, validated as sha256 hex."""
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise EvalStoreError(f"not a sha256 key: {key!r}")
        return self.root / OBJECTS_DIR / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        """True when an object exists for ``key``."""
        return self.object_path(key).is_file()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored object, or None when the key is unknown."""
        path = self.object_path(key)
        if not path.is_file():
            return None
        return self._read_object(path)

    def put(
        self,
        key: str,
        payload: Mapping[str, Any],
        *,
        allow_identical_rewrite: bool = True,
    ) -> Path:
        """Store a result under ``key``. Write-once.

        Re-writing identical content is a no-op (idempotent replay). Writing
        *different* content to an existing key means the fingerprint is missing a
        component, and is refused: that is the bug this whole scheme exists to
        catch.
        """
        body = canonical_json(payload)
        path = self.object_path(key)
        if path.is_file():
            existing = self._read_object(path)
            if canonical_json(existing) == body:
                if allow_identical_rewrite:
                    return path
                raise IdempotencyViolationError(
                    f"key {key[:16]} already exists with identical content; rewrite refused"
                )
            stored_sha = sha256_text(canonical_json(existing))[:16]
            raise IdempotencyViolationError(
                f"key {key[:16]} already holds different content. The fingerprint is missing an "
                f"input that changed the result (stored sha={stored_sha}, "
                f"new sha={sha256_text(body)[:16]}). Add that input to FINGERPRINT_FIELDS."
            )
        _atomic_write(path, body)
        return path

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        """Read and decode an object file, reporting corruption explicitly."""
        try:
            decoded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return decoded
        except json.JSONDecodeError as exc:
            raise EvalStoreError(f"corrupt object {path}: {exc}") from exc

    # --------------------------------------------------------------------- runs

    def run_dir(self, run_id: str) -> Path:
        """Directory of a run, rejecting ids that could escape the store root."""
        if not run_id or "/" in run_id or "\\" in run_id:
            raise EvalStoreError(f"invalid run id: {run_id!r}")
        return self.root / RUNS_DIR / run_id

    def start_run(self, run_id: str, manifest: Mapping[str, Any]) -> Path:
        """Create a run with its manifest. Re-starting an identical run is fine."""
        directory = self.run_dir(run_id)
        path = directory / MANIFEST
        body = canonical_json({**dict(manifest), "store_version": STORE_VERSION})
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(existing) != body:
                raise IdempotencyViolationError(
                    f"run {run_id} exists with a different manifest. Use a new run id for a "
                    f"different configuration, or delete the run directory deliberately."
                )
        else:
            _atomic_write(path, body)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def rewrite_manifest(self, run_id: str, manifest: Mapping[str, Any]) -> Path:
        """Replace a run's manifest, for a change that is known to be legitimate.

        The normal path is :meth:`start_run`, which is write-once: a run's identity
        must not drift under it. There is exactly one sanctioned change -- a run
        that started with no judge records which judge later graded it -- and this
        method exists so that change is explicit at the call site instead of being
        smuggled in through a flag that every other caller could also set.
        """
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        body = canonical_json({**dict(manifest), "store_version": STORE_VERSION})
        _atomic_write(directory / MANIFEST, body)
        return directory

    def load_manifest(self, run_id: str) -> dict[str, Any]:
        """Read a run manifest, failing if the run was never started."""
        path = self.run_dir(run_id) / MANIFEST
        if not path.is_file():
            raise EvalStoreError(f"run {run_id} has no manifest at {path}")
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return manifest

    def append_event(self, run_id: str, record: Mapping[str, Any]) -> None:
        """Append one event and fsync it, so a crash cannot lose it silently."""
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        line = canonical_json(record) + "\n"
        with (directory / EVENT_LOG).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_events(self, run_id: str) -> Iterator[dict[str, Any]]:
        """Stream the event log. A torn final line is an error, not a skip."""
        path = self.run_dir(run_id) / EVENT_LOG
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise EvalStoreError(
                        f"{path}:{number} is torn (no trailing newline); the writer crashed "
                        f"mid-append. Inspect and repair the log before resuming."
                    )
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise EvalStoreError(f"{path}:{number} is not valid JSON: {exc}") from exc

    def completed_keys(self, run_id: str, stage: str | None = None) -> set[str]:
        """Keys already recorded as done, for resuming a run."""
        done: set[str] = set()
        for event in self.read_events(run_id):
            if event.get("event") != "completed":
                continue
            key = event.get("key")
            if not isinstance(key, str):
                raise EvalStoreError(f"event without a key: {event!r}")
            if stage is None or event.get("stage") == stage:
                done.add(key)
        return done

    def new_run_id(self, prefix: str = "run") -> str:
        """Collision-free run id; recorded in the log so runs never interleave."""
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------- verification

    def verify_idempotency(
        self,
        fingerprints: Sequence[Fingerprint],
        recompute: Any,
        *,
        require_byte_equal: bool = True,
    ) -> list[str]:
        """Re-run stored attempts and report keys whose results changed.

        ``recompute(fingerprint) -> payload`` must produce the artifact. For
        stages that are genuinely deterministic this is a byte-exact guarantee.
        Returns the list of violated keys (empty when the run is idempotent).
        """
        violations: list[str] = []
        for fingerprint in fingerprints:
            key = fingerprint.key
            stored = self.get(key)
            if stored is None:
                violations.append(f"{key[:16]}: missing from the store")
                continue
            fresh = recompute(fingerprint)
            if require_byte_equal:
                if canonical_json(fresh) != canonical_json(stored):
                    violations.append(f"{key[:16]}: recomputed result differs from the stored one")
            else:
                fresh_hash = sha256_text(canonical_json(fresh))
                if fresh_hash != sha256_text(canonical_json(stored)):
                    violations.append(f"{key[:16]}: recomputed result is not stable")
        return violations

    def stats(self, run_id: str) -> dict[str, int]:
        """Counts per stage, for build logs and resume decisions."""
        counts: dict[str, int] = dict.fromkeys(STAGES, 0)
        for event in self.read_events(run_id):
            if event.get("event") != "completed":
                continue
            stage = str(event.get("stage", "unknown"))
            counts[stage] = counts.get(stage, 0) + 1
        return counts


def iter_fingerprints(records: Iterable[Mapping[str, Any]]) -> Iterator[Fingerprint]:
    """Build fingerprints from plain dicts (e.g. loaded from a config file)."""
    for record in records:
        yield _fingerprint(**dict(record))
