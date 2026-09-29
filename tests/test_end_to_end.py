"""End-to-end: a real Go repository, a real toolchain, a faked model endpoint.

Everything else stubs something. This test runs the whole loop against ``go test``
on an actual module: the model is asked to fix a failing test, calls the tools, sees
the compiler's own diagnostics, fixes the bug, and the trajectory is queued for
judging and then judged.

It is the only place the pieces are proven to fit together -- a template that
renders but a generator that sends the wrong shape would pass every unit test.

Two gates, and the second matters: Go must be present, and so must ``rtk``,
because the catalogue runs ``go test`` behind it. The full-loop tests are skipped
loudly without it rather than quietly degrading to raw ``go test``, which is
exactly the fallback this pipeline exists to forbid. The test that runs *anywhere*
is the one asserting the preflight refuses to start without rtk.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Any

import pytest

from gotooltrain.evalcli import main
from gotooltrain.evalstore import ResultStore
from gotooltrain.judge import read_judge_queue, write_verdicts

pytestmark = pytest.mark.skipif(
    shutil.which("go") is None, reason="the Go toolchain is not installed"
)

MODULE = "example.com/parser"

BROKEN = """package parser

// Count returns the number of entries in the table.
func Count(table map[string]int) int {
	return len(table)
}
"""

PASSING = """package parser

// Count returns the sum of every value in the table.
func Count(table map[string]int) int {
	total := 0
	for _, n := range table {
		total += n
	}
	return total
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


def go_available() -> bool:
    """True when the toolchain can actually build, not merely be on PATH."""
    binary = shutil.which("go")
    if binary is None:
        return False
    try:
        completed = subprocess.run(  # noqa: S603 - an absolute path from which()
            [binary, "version"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


requires_go = pytest.mark.skipif(not go_available(), reason="go cannot run here")

requires_rtk = pytest.mark.skipif(
    shutil.which("rtk") is None,
    reason=(
        "rtk is not installed; the catalogue runs go test behind it and the pipeline "
        "refuses to fall back to raw go test"
    ),
)


def make_module(root: pathlib.Path) -> pathlib.Path:
    """A Go module whose test passes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "go.mod").write_text(f"module {MODULE}\n\ngo 1.23\n", encoding="utf-8")
    (root / "parser.go").write_text(PASSING, encoding="utf-8")
    (root / "parser_test.go").write_text(TEST, encoding="utf-8")
    return root


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def chat(content: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> bytes:
    """One OpenAI-style completion body."""
    message: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return json.dumps({"choices": [{"message": message}]}).encode()


class ScriptedModel:
    """A fake endpoint that walks a model through one repair.

    The script is the interesting part: it only produces the fix *after* it has
    read a real failure from the harness. A loop that ignored tool results could
    not pass this test.
    """

    def __init__(self, broken: pathlib.Path) -> None:
        """Start at turn zero with no history recorded."""
        self.broken = broken
        self.turn = 0
        self.seen: list[list[dict[str, Any]]] = []

    def __call__(self, request, timeout=None) -> _Response:
        """Record the conversation and answer with the next scripted turn."""
        self.seen.append(json.loads(request.data)["messages"])
        turn = self.turn
        self.turn += 1
        bodies = [
            chat("First I will run the tests.", [tool_call("c1", "go_test", {"pkg": "./..."})]),
            chat(
                "Count returns len(table), but the test wants the sum. Rewriting it.",
                [tool_call("c2", "write_file", {"path": "parser.go", "content": PASSING})],
            ),
            chat("Re-running the suite.", [tool_call("c3", "go_test", {"pkg": "./..."})]),
            chat("Tests pass. Count returns the sum of the table's values."),
        ]
        return _Response(bodies[min(turn, len(bodies) - 1)])


class _Response:
    """Minimal ``urlopen`` result."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        """Return the canned payload."""
        return self._payload

    def __enter__(self) -> _Response:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> bool:
        """Leave the context manager without suppressing."""
        return False


COMMON = [
    "--model",
    "Qwen/Qwen3.5-4B-go",
    "--revision",
    "rev-a",
    "--dataset-version",
    "go-ut-bench-1",
    "--max-turns",
    "6",
]


@requires_rtk
def test_the_whole_loop_finds_and_fixes_a_bug(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = make_module(tmp_path / "broken")
    (broken / "parser.go").write_text(BROKEN, encoding="utf-8")

    model = ScriptedModel(broken)
    monkeypatch.setattr(urllib.request, "urlopen", model)

    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"id": "go-0001", "prompt": "The test suite fails. Fix it."}]),
        encoding="utf-8",
    )
    store = str(tmp_path / "store")
    queue = str(tmp_path / "queue.jsonl")

    code = main(
        [
            "queue",
            "--store",
            store,
            "--tasks",
            str(tasks),
            "--model-url",
            "http://model.invalid/v1",
            "--out",
            queue,
            "--sandbox",
            "local",
            "--allow-local-execution",
            "--fixture",
            str(broken),
            "--workspace",
            str(tmp_path / "ws"),
            *COMMON,
        ]
    )
    assert code == 0

    entries = read_judge_queue(queue)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.label == "go-0001#0"

    # The model saw the real diagnostic from RTK, not a stub. RTK compresses the
    # failure to one line per test, so the assertion is the actionable part.
    assert "Count = 2, want 3" in entry.trajectory
    assert "TestCount" in entry.trajectory
    assert "go_test" in entry.trajectory
    assert "write_file" in entry.trajectory
    # Four turns ran: test, edit, test, answer.
    assert entry.trajectory.count("## Turn") == 4

    # It graded every sample it was given, not just the last.
    assert len(model.seen) == 4
    assert any(m["role"] == "tool" for m in model.seen[-1]), "the final turn saw the results"


@requires_rtk
def test_a_judged_run_reports_the_session_judge(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = make_module(tmp_path / "broken")
    (broken / "parser.go").write_text(BROKEN, encoding="utf-8")
    monkeypatch.setattr(urllib.request, "urlopen", ScriptedModel(broken))

    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"id": "go-0001", "prompt": "The test suite fails. Fix it."}]),
        encoding="utf-8",
    )
    store = str(tmp_path / "store")
    queue = str(tmp_path / "queue.jsonl")
    args = ["--store", store, "--tasks", str(tasks), *COMMON]

    assert (
        main(
            [
                "queue",
                "--model-url",
                "http://model.invalid/v1",
                "--out",
                queue,
                "--sandbox",
                "local",
                "--allow-local-execution",
                "--fixture",
                str(broken),
                "--workspace",
                str(tmp_path / "ws"),
                "--run-id",
                "run-1",
                *args,
            ]
        )
        == 0
    )

    entries = read_judge_queue(queue)
    write_verdicts(
        str(tmp_path / "v.jsonl"), {e.key: {"score": 1.0, "reason": "fixed"} for e in entries}
    )
    code = main(
        [
            "judge",
            "--run-id",
            "run-1",
            "--queue",
            queue,
            "--verdicts",
            str(tmp_path / "v.jsonl"),
            *args,
        ]
    )
    assert code == 0


@requires_rtk
def test_a_second_run_reuses_the_finished_trajectory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running must not re-sample the model: the store already has the answer."""
    broken = make_module(tmp_path / "broken")
    (broken / "parser.go").write_text(BROKEN, encoding="utf-8")
    model = ScriptedModel(broken)
    monkeypatch.setattr(urllib.request, "urlopen", model)

    tasks = tmp_path / "tasks.json"
    tasks.write_text(
        json.dumps([{"id": "go-0001", "prompt": "The test suite fails. Fix it."}]),
        encoding="utf-8",
    )
    store = ResultStore(str(tmp_path / "store"))
    run_args = [
        "queue",
        "--store",
        str(store.root),
        "--tasks",
        str(tasks),
        "--model-url",
        "http://model.invalid/v1",
        "--out",
        str(tmp_path / "q.jsonl"),
        "--sandbox",
        "local",
        "--allow-local-execution",
        "--fixture",
        str(broken),
        "--workspace",
        str(tmp_path / "ws"),
        "--run-id",
        "run-1",
        *COMMON,
    ]
    assert main(run_args) == 0
    turns_before = model.turn

    assert main(run_args) == 0
    assert model.turn == turns_before, "the resumed run re-sampled the model"


def test_a_missing_rtk_stops_the_run_before_any_sample(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real behaviour on a host without rtk, which is most hosts.

    Every RTK-backed tool would fail as a harness error on every sample, after a
    full pass of wasted work, and the report would look like a broken harness. The
    run must refuse up front and name the missing command instead.
    """
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "rtk" else f"/usr/bin/{name}")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "go-0001", "prompt": "fix it"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model-url",
            "http://model.invalid/v1",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--sandbox",
            "local",
            "--allow-local-execution",
            "--workspace",
            str(tmp_path / "ws"),
            *COMMON,
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "needs ['rtk']" in err
    assert "is not on PATH" in err
    assert not (tmp_path / "q.jsonl").exists()


@requires_rtk
def test_a_dead_model_writes_no_queue(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that never produced a trajectory must not leave a queue behind."""
    broken = make_module(tmp_path / "broken")
    (broken / "parser.go").write_text(BROKEN, encoding="utf-8")

    def dying(request, timeout=None):  # type: ignore[no-untyped-def]
        """Lose the model server."""
        raise urllib.error.URLError("model server died")

    monkeypatch.setattr(urllib.request, "urlopen", dying)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "go-0001", "prompt": "fix it"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model-url",
            "http://model.invalid/v1",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--sandbox",
            "local",
            "--allow-local-execution",
            "--fixture",
            str(broken),
            "--workspace",
            str(tmp_path / "ws"),
            *COMMON,
        ]
    )
    assert code == 1
    assert "cannot reach" in capsys.readouterr().err
    assert not (tmp_path / "q.jsonl").exists()
