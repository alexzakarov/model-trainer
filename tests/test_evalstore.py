"""Durable, idempotent evaluation artifact store.

The properties under test are the ones that keep a parallel eval honest:
results survive a crash, a key never silently maps to two answers, and a
missing fingerprint field is an error rather than a stale-reuse bug.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain import (
    FINGERPRINT_FIELDS,
    STAGES,
    EvalStoreError,
    Fingerprint,
    IdempotencyViolationError,
    ResultStore,
    attempt_key,
    canonical_json,
    iter_fingerprints,
    sha256_file,
    sha256_text,
)

BASE: dict[str, Any] = {
    "task_id": "go-0001",
    "sample_index": 0,
    "stage": "generate",
    "model_id": "Qwen/Qwen3.5-4B-go",
    "model_revision": "rev-a",
    "format_version": "anthropic-tools-v1",
    "template_sha": "t" * 64,
    "harness_version": "0.1.0",
    "dataset_version": "go-ut-bench-holdout-1",
    "seed": 1234,
    "decode_params": {"temperature": 0.0, "max_tokens": 2048},
    "tool_catalog_sha": "c" * 64,
    "image_digest": "sha256:deadbeef",
}


def fp(**overrides: Any) -> Fingerprint:
    return Fingerprint(**{**BASE, **overrides})


@pytest.fixture
def store(tmp_path: pathlib.Path) -> ResultStore:
    return ResultStore(tmp_path / "store")


# ----------------------------------------------------------------- fingerprints


def test_key_is_stable_across_calls() -> None:
    assert attempt_key(**BASE) == attempt_key(**BASE)


def test_key_is_sha256_hex() -> None:
    key = attempt_key(**BASE)
    assert len(key) == 64
    assert all(c in "0123456789abcdef" for c in key)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "rev-b"),
        ("seed", 9999),
        ("stage", "execute"),
        ("sample_index", 1),
        ("task_id", "go-0002"),
        ("format_version", "anthropic-tools-v2"),
        ("template_sha", "e" * 64),
        ("harness_version", "0.2.0"),
        ("dataset_version", "go-ut-bench-holdout-2"),
        ("tool_catalog_sha", "d" * 64),
        ("image_digest", "sha256:cafe"),
        ("decode_params", {"temperature": 0.7}),
    ],
)
def test_every_fingerprint_field_changes_the_key(field: str, value: Any) -> None:
    """A field that does not affect the key is a field that can cause stale reuse."""
    assert attempt_key(**BASE) != attempt_key(**{**BASE, field: value})


def test_missing_fingerprint_field_is_an_error() -> None:
    broken = {k: v for k, v in BASE.items() if k != "model_revision"}
    with pytest.raises(EvalStoreError, match="missing required field"):
        attempt_key(**broken)


def test_unknown_fingerprint_field_is_an_error() -> None:
    with pytest.raises(EvalStoreError, match="unknown fingerprint field"):
        attempt_key(**{**BASE, "temperature": 0.7})


def test_every_documented_field_is_required() -> None:
    """Guards against a field being added to the list but not to a test input."""
    for field in FINGERPRINT_FIELDS:
        assert field in BASE, f"{field} is keyed but absent from the test fixture"


def test_store_version_is_keyed_but_not_caller_supplied() -> None:
    """The build's own store version is part of the key and cannot be overridden."""
    from dataclasses import asdict

    from gotooltrain import STORE_VERSION

    assert "store_version" not in FINGERPRINT_FIELDS
    payload = fp().to_dict()
    assert payload["store_version"] == STORE_VERSION
    assert set(asdict(fp())) == set(FINGERPRINT_FIELDS)
    with pytest.raises(EvalStoreError, match="unknown fingerprint field"):
        attempt_key(**{**BASE, "store_version": "other"})


def test_iter_fingerprints_builds_from_dicts() -> None:
    built = list(iter_fingerprints([BASE, {**BASE, "sample_index": 1}]))
    assert [f.sample_index for f in built] == [0, 1]
    assert built[0].key != built[1].key


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_sha256_helpers(tmp_path: pathlib.Path) -> None:
    assert sha256_text("x") == sha256_text("x")
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello" * 100_000)  # larger than the 1 MiB read chunk
    assert sha256_file(path) == sha256_text("hello" * 100_000)


# ---------------------------------------------------------------------- objects


def test_put_then_get_round_trip(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.put(key, {"text": "package parser"})
    assert store.get(key) == {"text": "package parser"}
    assert store.has(key)


def test_get_missing_returns_none(store: ResultStore) -> None:
    assert store.get(attempt_key(**BASE)) is None
    assert not store.has(attempt_key(**BASE))


def test_identical_rewrite_is_a_noop(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.put(key, {"a": 1})
    store.put(key, {"a": 1})
    assert store.get(key) == {"a": 1}


def test_identical_rewrite_can_be_refused(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.put(key, {"a": 1})
    with pytest.raises(IdempotencyViolationError, match="identical content"):
        store.put(key, {"a": 1}, allow_identical_rewrite=False)


def test_conflicting_rewrite_is_refused(store: ResultStore) -> None:
    """The core guard: one key must never hold two different answers."""
    key = attempt_key(**BASE)
    store.put(key, {"text": "from run A"})
    with pytest.raises(IdempotencyViolationError, match="different content"):
        store.put(key, {"text": "from run B"})


def test_object_path_rejects_a_non_key(store: ResultStore) -> None:
    with pytest.raises(EvalStoreError, match="not a sha256 key"):
        store.object_path("not-a-key")
    with pytest.raises(EvalStoreError, match="not a sha256 key"):
        store.has("ZZ" * 32)


def test_corrupt_object_is_reported(tmp_path: pathlib.Path) -> None:
    store = ResultStore(tmp_path / "store")
    key = attempt_key(**BASE)
    path = store.object_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(EvalStoreError, match="corrupt object"):
        store.get(key)


# ------------------------------------------------------------------------- runs


def test_start_run_writes_a_manifest(store: ResultStore) -> None:
    store.start_run("run-1", {"model_revision": "rev-a", "n_samples": 4})
    manifest = store.load_manifest("run-1")
    assert manifest["model_revision"] == "rev-a"
    assert manifest["store_version"]


def test_start_run_is_idempotent_for_the_same_manifest(store: ResultStore) -> None:
    store.start_run("run-1", {"a": 1})
    store.start_run("run-1", {"a": 1})


def test_start_run_refuses_a_different_manifest(store: ResultStore) -> None:
    store.start_run("run-1", {"a": 1})
    with pytest.raises(IdempotencyViolationError, match="different manifest"):
        store.start_run("run-1", {"a": 2})


def test_load_manifest_requires_a_run(store: ResultStore) -> None:
    with pytest.raises(EvalStoreError, match="no manifest"):
        store.load_manifest("missing")


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b"])
def test_invalid_run_id_rejected(store: ResultStore, bad: str) -> None:
    with pytest.raises(EvalStoreError, match="invalid run id"):
        store.run_dir(bad)


def test_new_run_id_is_unique(store: ResultStore) -> None:
    assert store.new_run_id() != store.new_run_id()
    assert store.new_run_id("eval").startswith("eval-")


# ------------------------------------------------------------------- event log


def test_events_append_and_read_back(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.append_event("run-1", {"event": "completed", "stage": "generate", "key": key})
    store.append_event("run-1", {"event": "completed", "stage": "execute", "key": key + "0"})
    events = list(store.read_events("run-1"))
    assert len(events) == 2
    assert events[0]["stage"] == "generate"


def test_read_events_on_missing_log_is_empty(store: ResultStore) -> None:
    assert list(store.read_events("nope")) == []


def test_torn_final_line_is_an_error(tmp_path: pathlib.Path) -> None:
    """A crash mid-append must not be mistaken for a short log."""
    store = ResultStore(tmp_path / "store")
    store.append_event("run-1", {"event": "completed", "stage": "generate", "key": "k"})
    log = store.run_dir("run-1") / "events.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "comp')  # no newline: torn write

    with pytest.raises(EvalStoreError, match="torn"):
        list(store.read_events("run-1"))


def test_invalid_json_line_is_an_error(store: ResultStore) -> None:
    store.run_dir("run-1").mkdir(parents=True, exist_ok=True)
    (store.run_dir("run-1") / "events.jsonl").write_text("nope\n", encoding="utf-8")
    with pytest.raises(EvalStoreError, match="not valid JSON"):
        list(store.read_events("run-1"))


def test_blank_lines_are_skipped(store: ResultStore) -> None:
    store.append_event("run-1", {"event": "completed", "stage": "generate", "key": "k"})
    log = store.run_dir("run-1") / "events.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(list(store.read_events("run-1"))) == 1


def test_completed_keys_supports_resume(store: ResultStore) -> None:
    gen = attempt_key(**BASE)
    exe = attempt_key(**{**BASE, "stage": "execute"})
    store.append_event("run-1", {"event": "started", "stage": "generate", "key": gen})
    store.append_event("run-1", {"event": "completed", "stage": "generate", "key": gen})
    store.append_event("run-1", {"event": "completed", "stage": "execute", "key": exe})

    assert store.completed_keys("run-1") == {gen, exe}
    assert store.completed_keys("run-1", stage="generate") == {gen}


def test_completed_keys_rejects_an_event_without_a_key(store: ResultStore) -> None:
    store.append_event("run-1", {"event": "completed", "stage": "generate"})
    with pytest.raises(EvalStoreError, match="without a key"):
        store.completed_keys("run-1")


def test_stats_counts_every_stage(store: ResultStore) -> None:
    for stage in STAGES:
        key = attempt_key(**{**BASE, "stage": stage})
        store.append_event("run-1", {"event": "completed", "stage": stage, "key": key})
    assert store.stats("run-1") == {"generate": 1, "execute": 1, "judge": 1}


def test_stats_ignores_non_completion_events(store: ResultStore) -> None:
    store.append_event("run-1", {"event": "started", "stage": "generate", "key": "k"})
    assert store.stats("run-1") == {"generate": 0, "execute": 0, "judge": 0}


def test_stats_counts_an_unknown_stage(store: ResultStore) -> None:
    store.append_event("run-1", {"event": "completed", "stage": "mystery", "key": "k"})
    assert store.stats("run-1")["mystery"] == 1


# ----------------------------------------------------------------- verification


def test_verify_idempotency_passes_when_stable(store: ResultStore) -> None:
    fingerprint = fp()
    store.put(fingerprint.key, {"result": "ok"})
    assert store.verify_idempotency([fingerprint], lambda _f: {"result": "ok"}) == []


def test_verify_idempotency_detects_drift(store: ResultStore) -> None:
    fingerprint = fp()
    store.put(fingerprint.key, {"result": "ok"})
    violations = store.verify_idempotency([fingerprint], lambda _f: {"result": "different"})
    assert len(violations) == 1
    assert "recomputed result differs" in violations[0]


def test_verify_idempotency_flags_missing_objects(store: ResultStore) -> None:
    violations = store.verify_idempotency([fp()], lambda _f: {"result": "ok"})
    assert len(violations) == 1
    assert "missing from the store" in violations[0]


def test_verify_idempotency_non_strict_mode_also_reports(store: ResultStore) -> None:
    fingerprint = fp()
    store.put(fingerprint.key, {"result": "ok"})
    violations = store.verify_idempotency(
        [fingerprint], lambda _f: {"result": "x"}, require_byte_equal=False
    )
    assert len(violations) == 1
    assert "not stable" in violations[0]


def test_verify_idempotency_non_strict_mode_accepts_a_stable_result(store: ResultStore) -> None:
    fingerprint = fp()
    store.put(fingerprint.key, {"result": "ok"})
    assert (
        store.verify_idempotency(
            [fingerprint], lambda _f: {"result": "ok"}, require_byte_equal=False
        )
        == []
    )


def test_fingerprint_short_key_is_readable() -> None:
    assert len(fp().short()) == 16


def test_atomic_write_cleans_up_on_failure(store: ResultStore) -> None:
    """A failed write must not leave a .tmp turd for the next reader to trip over."""
    import pathlib as _pathlib

    original = _pathlib.Path.replace

    def boom(self, target):  # type: ignore[no-untyped-def]
        raise OSError("simulated failure")

    _pathlib.Path.replace = boom  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError, match="simulated failure"):
            store.put(attempt_key(**BASE), {"a": 1})
    finally:
        _pathlib.Path.replace = original  # type: ignore[method-assign]
    assert [p.name for p in store.root.rglob("*.tmp")] == []
    assert not store.has(attempt_key(**BASE))


def test_verify_idempotency_handles_multiple_fingerprints(store: ResultStore) -> None:
    """One stable and one drifting key: the report must name only the offender."""
    good = fp()
    bad = fp(task_id="go-0002")
    store.put(good.key, {"result": "ok"})
    store.put(bad.key, {"result": "stored"})
    payloads = {good.key: {"result": "ok"}, bad.key: {"result": "changed"}}
    violations = store.verify_idempotency([good, bad], lambda f: payloads[f.key])
    assert len(violations) == 1
    assert violations[0].startswith(bad.key[:16])


# ------------------------------------------------------------------- durability


def test_writes_leave_no_temp_files(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.put(key, {"a": 1})
    store.start_run("run-1", {"a": 1})
    leftovers = [p.name for p in store.root.rglob("*.tmp")]
    assert leftovers == []


def test_events_survive_a_fresh_store_instance(tmp_path: pathlib.Path) -> None:
    """Resume must work from disk, not from in-memory state."""
    store = ResultStore(tmp_path / "store")
    key = attempt_key(**BASE)
    store.append_event("run-1", {"event": "completed", "stage": "generate", "key": key})
    store.put(key, {"a": 1})

    reopened = ResultStore(tmp_path / "store")
    assert reopened.completed_keys("run-1") == {key}
    assert reopened.get(key) == {"a": 1}


def test_unicode_payloads_round_trip(store: ResultStore) -> None:
    key = attempt_key(**BASE)
    store.put(key, {"text": "paket: `./parser`  çıktı ✓"})
    assert store.get(key) == {"text": "paket: `./parser`  çıktı ✓"}
    assert json.loads(store.object_path(key).read_text(encoding="utf-8"))["text"].endswith("✓")
