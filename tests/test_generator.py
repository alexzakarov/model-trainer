"""The generator and the CLI: what actually runs, and what refuses to.

The generator is the model's only voice in the loop, so its two failure modes are
tested hard: a reply that cannot be parsed, and arguments that are not the object
the tool expects. Both must stop the run rather than become a plausible-looking
turn.
"""

from __future__ import annotations

import json
import pathlib
import urllib.error
import urllib.request
from typing import Any

import pytest

from gotooltrain.errors import HarnessError, ToolTrainError
from gotooltrain.evalcli import main
from gotooltrain.evalrun import RunConfig
from gotooltrain.generator import SYSTEM_PROMPT, HttpGenerator, parse_reply
from gotooltrain.gotools import GO_TOOLS
from gotooltrain.judge import read_judge_queue


def completion(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str = "m",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": model, "choices": [{"message": message}]}


def body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode()


class FakeResponse:
    """Minimal ``urlopen`` result."""

    def __init__(self, payload: bytes) -> None:
        """Hold the canned payload."""
        self._payload = payload

    def read(self) -> bytes:
        """Return the canned payload."""
        return self._payload

    def __enter__(self) -> FakeResponse:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> bool:
        """Leave the context manager without suppressing."""
        return False


def returning(payload: bytes) -> Any:
    """Build a ``urlopen`` replacement returning ``payload``."""

    def opener(request, timeout=None):  # type: ignore[no-untyped-def]
        return FakeResponse(payload)

    return opener


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[urllib.request.Request]:
    """Capture outgoing requests, answering with a plain text reply."""
    seen: list[urllib.request.Request] = []

    def recorder(request, timeout=None):  # type: ignore[no-untyped-def]
        seen.append(request)
        return FakeResponse(body(completion(content="done")))

    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    return seen


def config() -> RunConfig:
    return RunConfig(
        model_id="Qwen/Qwen3.5-4B-go",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=7,
        decode_params={"temperature": 0.0},
    )


# ----------------------------------------------------------------- parsing


def test_a_plain_reply_becomes_an_empty_turn() -> None:
    turn = parse_reply(json.dumps(completion(content="I am done")), model="m")
    assert turn == {"text": "I am done", "tool_calls": []}


def test_a_tool_call_is_parsed_with_decoded_arguments() -> None:
    raw = json.dumps(
        completion(
            content="Reading first.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "main.go"}'},
                }
            ],
        )
    )
    turn = parse_reply(raw, model="m")
    assert turn["tool_calls"] == [
        {"id": "call_1", "name": "read_file", "arguments": {"path": "main.go"}}
    ]


def test_arguments_already_decoded_are_accepted() -> None:
    raw = json.dumps(
        completion(
            tool_calls=[{"id": "c", "function": {"name": "go_test", "arguments": {"pkg": "./..."}}}]
        )
    )
    assert parse_reply(raw, model="m")["tool_calls"][0]["arguments"] == {"pkg": "./..."}


def test_a_call_without_an_id_gets_a_stable_one() -> None:
    """The model must not be able to leave a tool result unaddressable."""
    raw = json.dumps(completion(tool_calls=[{"function": {"name": "go_test", "arguments": "{}"}}]))
    assert parse_reply(raw, model="m")["tool_calls"][0]["id"] == "call_0"


def test_unparseable_arguments_stop_the_run() -> None:
    """A truncated edit would apply unknown contents to the workspace."""
    raw = json.dumps(
        completion(
            tool_calls=[{"id": "c", "function": {"name": "edit_file", "arguments": '{"path": '}}]
        )
    )
    with pytest.raises(HarnessError, match="unparseable arguments"):
        parse_reply(raw, model="m")


def test_non_object_arguments_are_refused() -> None:
    raw = json.dumps(
        completion(tool_calls=[{"id": "c", "function": {"name": "grep", "arguments": '"x"'}}])
    )
    with pytest.raises(HarnessError, match="non-object arguments"):
        parse_reply(raw, model="m")


def test_a_call_without_a_function_is_refused() -> None:
    raw = json.dumps(completion(tool_calls=[{"id": "c"}]))
    with pytest.raises(HarnessError, match="without a function"):
        parse_reply(raw, model="m")


def test_a_non_json_body_is_refused() -> None:
    with pytest.raises(HarnessError, match="non-JSON response"):
        parse_reply("<html>", model="m")


def test_a_body_without_choices_is_refused() -> None:
    with pytest.raises(HarnessError, match="no choices"):
        parse_reply(json.dumps({"error": "overloaded"}), model="m")


# -------------------------------------------------------------- the request


def test_the_generator_advertises_the_catalogue(captured: list[urllib.request.Request]) -> None:
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "fix it"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
    )
    sent = json.loads(captured[0].data)
    # The advertised list is the catalogue itself: evaluating against a different
    # tool set or order than training would measure prompt-following, not tool use.
    assert [t["function"]["name"] for t in sent["tools"]] == [t.name for t in GO_TOOLS]
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_the_generator_can_withhold_the_catalogue(
    captured: list[urllib.request.Request],
) -> None:
    """Measuring the effect of the tool list needs a run without it."""
    HttpGenerator(base_url="http://x/v1", model="m", advertise_tools=False).generate(
        {"id": "t", "prompt": "fix it"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
    )
    assert "tools" not in json.loads(captured[0].data)


def test_tool_results_are_sent_back_as_tool_messages(
    captured: list[urllib.request.Request],
) -> None:
    """The loop is only real if turn 2 sees turn 1's output."""
    history = [
        {"role": "user", "content": "fix it"},
        {
            "role": "assistant",
            "content": "testing",
            "tool_calls": [{"id": "c1", "name": "go_test", "arguments": {"pkg": "./..."}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "FAIL: undefined: foo"},
    ]
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "fix it"},
        0,
        1,
        history,
        config().fingerprint("t", 0, "generate", 1),
    )
    sent = json.loads(captured[0].data)["messages"]
    assert sent[-1] == {"role": "tool", "tool_call_id": "c1", "content": "FAIL: undefined: foo"}
    assert json.loads(sent[-2]["tool_calls"][0]["function"]["arguments"]) == {"pkg": "./..."}


def test_samples_get_different_seeds(captured: list[urllib.request.Request]) -> None:
    """n_samples means different samples, not the same one decoded twice."""
    generator = HttpGenerator(base_url="http://x/v1", model="m")
    run = config()
    for index in (0, 1):
        generator.generate(
            {"id": "t", "prompt": "p"}, index, 0, [], run.fingerprint("t", index, "generate", 0)
        )
    seeds = [json.loads(r.data)["seed"] for r in captured]
    assert seeds[1] - seeds[0] == 1000


def test_the_identity_is_recorded() -> None:
    assert HttpGenerator(base_url="http://x/v1", model="m").identity == "m@http://x/v1"


def test_an_unpinned_endpoint_is_refused() -> None:
    with pytest.raises(HarnessError, match="base_url is required"):
        HttpGenerator(base_url="", model="m")
    with pytest.raises(HarnessError, match="model id is required"):
        HttpGenerator(base_url="http://x/v1", model="")


def test_a_non_http_endpoint_is_refused() -> None:
    with pytest.raises(HarnessError, match="must be http"):
        HttpGenerator(base_url="file:///etc/passwd", model="m").generate(
            {"id": "t", "prompt": "p"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
        )


def test_an_unreachable_model_is_reported(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(request, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HarnessError, match="cannot reach"):
        HttpGenerator(base_url="http://x/v1", model="m").generate(
            {"id": "t", "prompt": "p"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
        )


def test_no_auth_header_without_a_key(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("EVAL_API_KEY", raising=False)
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "p"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
    )
    assert captured[0].get_header("Authorization") is None


# ----------------------------------------------------------------------- CLI


def test_a_missing_task_file_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tmp_path / "nope.json"),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "task file not found" in capsys.readouterr().err


def test_a_task_without_an_id_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"prompt": "x"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "has no 'id'" in capsys.readouterr().err


def test_a_malformed_task_file_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("{nope", encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_jsonl_tasks_are_accepted(tmp_path: pathlib.Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id": "a", "prompt": "p"}\n\n{"id": "b", "prompt": "q"}\n', encoding="utf-8")
    assert (
        main(
            [
                "queue",
                "--store",
                str(tmp_path / "s"),
                "--tasks",
                str(tasks),
                "--model",
                "m",
                "--revision",
                "r",
                "--dataset-version",
                "d",
                "--out",
                str(tmp_path / "q.jsonl"),
                "--model-url",
                "http://x/v1",
            ]
        )
        == 1
    ), "it proceeds far enough to need a model, then fails on the sandbox"


def test_host_execution_is_refused_by_default(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Untrusted generated Go code on the host is what the container pool prevents."""
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
        ]
    )
    assert code == 1
    assert "refusing to execute model-authored code on the host" in capsys.readouterr().err


def test_a_missing_fixture_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
            "--fixture",
            str(tmp_path / "no-such-fixture"),
        ]
    )
    assert code == 1
    assert "fixture directory not found" in capsys.readouterr().err


def test_bad_decode_params_are_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        main(
            [
                "queue",
                "--store",
                str(tmp_path / "s"),
                "--tasks",
                str(tasks),
                "--model",
                "m",
                "--revision",
                "r",
                "--dataset-version",
                "d",
                "--out",
                str(tmp_path / "q.jsonl"),
                "--model-url",
                "http://x/v1",
                "--decode-params",
                "not json",
            ]
        )


def test_the_judge_command_never_executes(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reaching execution during judging means a trajectory was missing."""
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    queue = tmp_path / "q.jsonl"
    queue.write_text("", encoding="utf-8")
    verdicts = tmp_path / "v.jsonl"
    verdicts.write_text("", encoding="utf-8")
    code = main(
        [
            "judge",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--run-id",
            "run-1",
            "--queue",
            str(queue),
            "--verdicts",
            str(verdicts),
        ]
    )
    assert code == 1
    assert "never samples" in capsys.readouterr().err


def test_an_http_error_from_the_model_is_reported(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 mid-run is infrastructure, not a model that failed to answer."""
    import io

    def boom(request, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError("http://x", 503, "busy", {}, io.BytesIO(b"no capacity"))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HarnessError, match="HTTP 503"):
        HttpGenerator(base_url="http://x/v1", model="m").generate(
            {"id": "t", "prompt": "p"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
        )


def test_an_auth_header_is_sent_when_configured(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVAL_API_KEY", "k")
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "p"}, 0, 0, [], config().fingerprint("t", 0, "generate", 0)
    )
    assert captured[0].get_header("Authorization") == "Bearer k"


def test_an_assistant_turn_without_tool_calls_sends_no_field(
    captured: list[urllib.request.Request],
) -> None:
    """An empty tool_calls list would invite an unanswered tool result."""
    history = [{"role": "assistant", "content": "no tools this time"}]
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "p"},
        0,
        1,
        history,
        config().fingerprint("t", 0, "generate", 1),
    )
    assert "tool_calls" not in json.loads(captured[0].data)["messages"][-1]


def test_a_tool_result_message_is_forwarded(
    captured: list[urllib.request.Request],
) -> None:
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "name": "go_test", "arguments": {"pkg": "./..."}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "p"},
        0,
        1,
        history,
        config().fingerprint("t", 0, "generate", 1),
    )
    sent = json.loads(captured[0].data)["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]


def test_a_jsonl_task_line_that_is_not_an_object_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("[1, 2]\n", encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "must be a JSON object" in capsys.readouterr().err


def test_a_json_task_file_that_is_not_a_list_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps({"id": "a"}), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "must hold a list of task objects" in capsys.readouterr().err


def test_the_docker_sandbox_is_the_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Model-authored code runs in a container unless the host is opted into."""
    import gotooltrain.evalcli as cli

    built: list[dict[str, Any]] = []

    class FakePool:
        """Records its configuration without starting a container."""

        def __init__(self, backend: Any, spec: Any, size: int) -> None:
            """Capture the pool configuration."""
            built.append({"image": spec.image, "size": size, "cpus": spec.cpus})

    monkeypatch.setattr(cli, "docker_available", lambda: True)
    monkeypatch.setattr(cli, "ContainerPool", FakePool)
    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])
    monkeypatch.setattr(cli, "run_canary", lambda executor, cases: None)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--image",
            "golang:1.24",
            "--workers",
            "2",
        ]
    )
    assert code == 1, "the model endpoint is unreachable"
    assert built == [{"image": "golang:1.24", "size": 2, "cpus": 2.0}]
    assert "canary passed" in capsys.readouterr().err


def test_absent_docker_is_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing daemon is a setup error named as one, not a mystery."""
    import gotooltrain.evalcli as cli

    monkeypatch.setattr(cli, "docker_available", lambda: False)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "Docker was requested" in capsys.readouterr().err


def test_a_canary_can_be_skipped_explicitly(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch exists, but only when asked for by name."""
    import gotooltrain.evalcli as cli

    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_canary", lambda executor, cases: calls.append("canary"))
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
            "--skip-canary",
        ]
    )
    assert code == 1, "the model endpoint is unreachable, so the run fails"
    assert calls == [], "the canary was skipped and must not have run"


def test_a_canary_runs_by_default(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --skip-canary the probe runs before the fan-out."""
    import gotooltrain.evalcli as cli

    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_canary", lambda executor, cases: calls.append("canary"))
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
        ]
    )
    assert code == 1
    assert calls == ["canary"]


def test_a_failing_canary_stops_the_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A harness that cannot tell success from failure must never produce a score."""
    import gotooltrain.evalcli as cli

    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])

    def boom(executor, cases):  # type: ignore[no-untyped-def]
        raise HarnessError("canary failed")

    monkeypatch.setattr(cli, "run_canary", boom)
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
        ]
    )
    assert code == 1
    assert "canary failed" in capsys.readouterr().err


def test_the_never_executed_executor_refuses(
    tmp_path: pathlib.Path,
) -> None:
    from gotooltrain.evalcli import _NeverExecuted
    from gotooltrain.gorun import ExecRequest

    with pytest.raises(ToolTrainError, match="never executes"):
        _NeverExecuted().run(
            ["go", "test"],
            ExecRequest(
                task_id="t",
                sample_index=0,
                tool_name="go_test",
                arguments={},
                workspace=pathlib.Path(),
            ),
        )


def test_the_store_only_generator_refuses(tmp_path: pathlib.Path) -> None:
    from gotooltrain.evalcli import _StoreOnlyGenerator

    with pytest.raises(ToolTrainError, match="never samples"):
        _StoreOnlyGenerator().generate()


def test_a_queue_is_written_when_the_run_succeeds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path of the queue command, with the run itself stubbed."""
    import gotooltrain.evalcli as cli
    from gotooltrain.evalrun import RunReport
    from gotooltrain.gorun import TaskSummary
    from gotooltrain.judge import QueueEntry

    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])
    monkeypatch.setattr(cli, "run_canary", lambda executor, cases: None)
    entry = QueueEntry(key="a" * 64, task_id="go-0001", sample_index=0, trajectory="t", prompt="p")
    monkeypatch.setattr(cli, "judge_queue", lambda config, tasks, store: [entry])
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *a, **k: RunReport(
            run_id="r",
            summaries=[
                TaskSummary("go-0001", 1, 1, 0, 0, 0, 1.0, 1.0),
            ],
            judged={},
            stage_counts={},
            reused={},
        ),
    )
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "go-0001", "prompt": "p"}]), encoding="utf-8")
    out = tmp_path / "q.jsonl"
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(out),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
        ]
    )
    assert code == 0
    assert read_judge_queue(str(out)) == [entry]


def test_an_empty_queue_is_called_out(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero trajectories is a fact the operator needs, not a silent success."""
    import gotooltrain.evalcli as cli
    from gotooltrain.evalrun import RunReport

    monkeypatch.setattr(cli, "missing_tool_binaries", lambda: [])
    monkeypatch.setattr(cli, "run_canary", lambda executor, cases: None)
    monkeypatch.setattr(cli, "judge_queue", lambda config, tasks, store: [])
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *a, **k: RunReport(
            run_id="r",
            summaries=[],
            judged={},
            stage_counts={},
            reused={},
            unjudged_tasks=3,
        ),
    )
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    out = tmp_path / "q.jsonl"
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(out),
            "--model-url",
            "http://x/v1",
            "--sandbox",
            "local",
            "--allow-local-execution",
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    assert "nothing to judge" in err
    assert "3 task(s) unjudged" in err


def test_the_judge_command_prints_the_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import gotooltrain.evalcli as cli
    from gotooltrain.evalrun import RunReport
    from gotooltrain.judge import QueueEntry, write_verdicts

    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda *a, **k: RunReport(
            run_id="r",
            summaries=[],
            judged={},
            stage_counts={},
            reused={},
            scoring="session_judge",
        ),
    )
    entry = QueueEntry(key="b" * 64, task_id="go-0001", sample_index=0, trajectory="t", prompt="p")
    queue = tmp_path / "q.jsonl"
    write_verdicts(str(tmp_path / "v.jsonl"), {entry.key: {"score": 1.0}})
    from gotooltrain.judge import write_judge_queue

    write_judge_queue(str(queue), [entry])
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "go-0001", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "judge",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--run-id",
            "run-1",
            "--queue",
            str(queue),
            "--verdicts",
            str(tmp_path / "v.jsonl"),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["scoring"] == "session_judge"


def test_a_user_turn_in_the_history_is_forwarded(
    captured: list[urllib.request.Request],
) -> None:
    """A follow-up user turn reaches the model rather than being dropped."""
    history = [
        {"role": "user", "content": "also add an example"},
    ]
    HttpGenerator(base_url="http://x/v1", model="m").generate(
        {"id": "t", "prompt": "p"},
        0,
        1,
        history,
        config().fingerprint("t", 0, "generate", 1),
    )
    sent = json.loads(captured[0].data)["messages"]
    assert sent[-1] == {"role": "user", "content": "also add an example"}


def test_a_corrupt_jsonl_task_line_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("{not json}\n", encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert ":1 is not valid JSON" in err


def test_a_foreign_template_version_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run keyed by a token format this build does not use is not comparable."""
    import gotooltrain.evalcli as cli

    monkeypatch.setattr(cli, "template_format_version", lambda source: "anthropic-tools-v0")
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([{"id": "a", "prompt": "p"}]), encoding="utf-8")
    code = main(
        [
            "queue",
            "--store",
            str(tmp_path / "s"),
            "--tasks",
            str(tasks),
            "--model",
            "m",
            "--revision",
            "r",
            "--dataset-version",
            "d",
            "--out",
            str(tmp_path / "q.jsonl"),
            "--model-url",
            "http://x/v1",
        ]
    )
    assert code == 1
    assert "declares 'anthropic-tools-v0'" in capsys.readouterr().err


def test_an_unknown_role_in_the_history_is_refused(
    captured: list[urllib.request.Request],
) -> None:
    """Dropping it silently would hide a tool result from the model."""
    history = [{"role": "system_extra", "content": "surprise"}]
    with pytest.raises(HarnessError, match="not a role this harness produces"):
        HttpGenerator(base_url="http://x/v1", model="m").generate(
            {"id": "t", "prompt": "p"},
            0,
            1,
            history,
            config().fingerprint("t", 0, "generate", 1),
        )
    assert captured == [], "nothing was sent"
