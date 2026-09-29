"""The judge: pinning, parsing, and refusing to guess.

Every test here defends the same property: a score must mean something. A judge
that is silently substituted, that returns prose, or that times out must produce a
loud failure -- never a 0.0 that quietly drags the model's score down.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import pytest

from gotooltrain import EvalStoreError, HarnessError
from gotooltrain.judge import (
    JUDGE_SYSTEM_PROMPT,
    HttpJudge,
    judge_identity,
    judge_messages,
    parse_verdict,
    render_trajectory,
)


def response(*, content: str, model: str = "judge-1") -> dict[str, Any]:
    return {"model": model, "choices": [{"message": {"role": "assistant", "content": content}}]}


def ok_body(*, passed: bool = True) -> bytes:
    return json.dumps(response(content=json.dumps({"pass": passed, "reason": "fixed it"}))).encode()


def trajectory() -> dict[str, Any]:
    return {
        "task": {"id": "go-0001", "prompt": "fix the panic"},
        "truncated": False,
        "turns": [
            {
                "turn": 0,
                "generated": {
                    "text": "Running the tests first.",
                    "tool_calls": [{"name": "go_test", "arguments": {"pkg": "./parser"}}],
                },
                "executions": [
                    {
                        "tool_name": "go_test",
                        "status": "tool_error",
                        "exit_code": 1,
                        "stdout": "--- FAIL: TestParse\npanic: assignment to entry in nil map",
                        "harness_error": "",
                    }
                ],
            },
            {
                "turn": 1,
                "generated": {"text": "Added the guard.", "tool_calls": []},
                "executions": [],
            },
        ],
    }


# ------------------------------------------------------------------- pinning


def test_a_judge_must_name_its_model() -> None:
    with pytest.raises(EvalStoreError, match="model id is required"):
        HttpJudge(base_url="http://localhost:8000/v1", model="")


def test_a_judge_must_have_an_endpoint() -> None:
    with pytest.raises(EvalStoreError, match="base_url is required"):
        HttpJudge(base_url="", model="judge-1")


def test_identity_records_both_model_and_endpoint() -> None:
    identity = judge_identity(HttpJudge(base_url="http://localhost:8000/v1", model="judge-1"))
    assert "judge-1" in identity
    assert "localhost:8000" in identity


def test_an_unnamed_judge_falls_back_to_its_class() -> None:
    """Still identifiable, so two runs never look silently interchangeable."""

    class Mystery:
        def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
            """Unused."""
            return {"score": 1.0}

    assert judge_identity(Mystery()) == "Mystery"


# ----------------------------------------------------------------- rendering


def test_the_judge_sees_the_command_output_not_a_summary() -> None:
    """A judge that only sees "all OK" cannot tell a fix from a reformat."""
    text = render_trajectory(trajectory())
    assert "panic: assignment to entry in nil map" in text
    assert "exit=1" in text
    assert "go_test" in text


def test_the_task_prompt_reaches_the_judge() -> None:
    assert "fix the panic" in render_trajectory(trajectory())


def test_a_truncated_trajectory_is_flagged_to_the_judge() -> None:
    """Hitting the turn limit means incomplete work, not a quiet pass."""
    record = {**trajectory(), "truncated": True}
    assert "incomplete" in render_trajectory(record)


def test_a_harness_failure_is_shown_as_an_infrastructure_problem() -> None:
    record = trajectory()
    record["turns"][0]["executions"][0] = {
        "tool_name": "go_test",
        "status": "harness_error",
        "exit_code": None,
        "stdout": "",
        "harness_error": "container is not running",
    }
    assert "[infrastructure error] container is not running" in render_trajectory(record)


def test_the_prompt_pins_the_judge_to_evidence() -> None:
    messages = judge_messages(trajectory())
    assert messages[0]["role"] == "system"
    assert "printed nothing is not a pass" in messages[0]["content"]
    assert messages[0]["content"] == JUDGE_SYSTEM_PROMPT


# ------------------------------------------------------------------- parsing


def test_a_boolean_pass_is_parsed() -> None:
    verdict = parse_verdict('{"pass": true, "reason": "tests pass"}', judge_model="j")
    assert verdict.score == 1.0
    assert verdict.reason == "tests pass"
    assert verdict.model == "j"


def test_a_false_pass_is_zero() -> None:
    assert parse_verdict('{"pass": false}', judge_model="j").score == 0.0


def test_a_numeric_score_is_accepted() -> None:
    assert parse_verdict('{"score": 0.75}', judge_model="j").score == 0.75


def test_a_pass_given_as_a_number_is_accepted() -> None:
    assert parse_verdict('{"pass": 1}', judge_model="j").score == 1.0
    assert parse_verdict('{"pass": 0}', judge_model="j").score == 0.0


def test_a_fenced_reply_is_tolerated() -> None:
    raw = '```json\n{"pass": true, "reason": "ok"}\n```'
    assert parse_verdict(raw, judge_model="j").score == 1.0


def test_prose_is_refused_rather_than_scored_zero() -> None:
    """The whole point: a judge outage must not read as a model failure."""
    with pytest.raises(HarnessError, match="Refusing to score"):
        parse_verdict("I think the work looks good overall.", judge_model="j")


def test_an_empty_reply_is_refused() -> None:
    with pytest.raises(HarnessError, match="empty response"):
        parse_verdict("   ", judge_model="j")


def test_a_score_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(HarnessError, match="outside"):
        parse_verdict('{"score": 1.4}', judge_model="j")


def test_a_boolean_score_is_refused() -> None:
    """Bool is an int subclass; True would silently mean 1.0."""
    with pytest.raises(HarnessError, match="score=True"):
        parse_verdict('{"score": true}', judge_model="j")


def test_a_non_boolean_pass_is_refused() -> None:
    with pytest.raises(HarnessError, match="pass="):
        parse_verdict('{"pass": "yes"}', judge_model="j")


def test_a_reply_with_neither_field_is_refused() -> None:
    with pytest.raises(HarnessError, match="neither 'pass' nor 'score'"):
        parse_verdict('{"verdict": "good"}', judge_model="j")


def test_a_json_array_is_refused() -> None:
    with pytest.raises(HarnessError, match="expected an object"):
        parse_verdict("[1, 2]", judge_model="j")


def test_the_raw_reply_is_kept_for_audit() -> None:
    verdict = parse_verdict('{"pass": true, "reason": "ok"}', judge_model="j")
    assert "pass" in verdict.raw
    assert verdict.to_record()["judge_model"] == "j"


# ---------------------------------------------------------------- http judge


class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, body: bytes) -> None:
        """Hold the canned response body."""
        self._body = body

    def read(self) -> bytes:
        """Return the canned body."""
        return self._body

    def __enter__(self) -> FakeResponse:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> bool:
        """Leave the context manager without suppressing."""
        return False


def returning(body: bytes) -> Callable[..., FakeResponse]:
    """Build a ``urlopen`` replacement that returns ``body``."""

    def opener(request, timeout=None):  # type: ignore[no-untyped-def]
        return FakeResponse(body)

    return opener


def http_error(status: int, body: bytes) -> Callable[..., object]:
    """Build a ``urlopen`` replacement that raises ``status``."""

    def raiser(request, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            "http://x",
            status,
            "error",
            {},
            io.BytesIO(body),  # type: ignore[arg-type]
        )

    return raiser


def unreachable(reason: str) -> Callable[..., object]:
    """Build a ``urlopen`` replacement that fails to connect."""

    def raiser(request, timeout=None):  # type: ignore[no-untyped-def]
        raise urllib.error.URLError(reason)

    return raiser


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[urllib.request.Request]:
    """Capture outgoing requests and answer with a passing verdict."""
    seen: list[urllib.request.Request] = []

    def recorder(request, timeout=None):  # type: ignore[no-untyped-def]
        seen.append(request)
        return FakeResponse(ok_body())

    monkeypatch.setattr(urllib.request, "urlopen", recorder)
    return seen


def test_the_judge_posts_to_chat_completions(captured: list[urllib.request.Request]) -> None:
    verdict = HttpJudge(base_url="http://localhost:8000/v1/", model="judge-1").judge(
        trajectory(), None
    )
    assert verdict["score"] == 1.0
    sent = json.loads(captured[0].data)
    assert sent["model"] == "judge-1"
    assert sent["messages"][0]["content"] == JUDGE_SYSTEM_PROMPT
    assert sent["temperature"] == 0.0


def test_the_trailing_slash_is_not_doubled(captured: list[urllib.request.Request]) -> None:
    HttpJudge(base_url="http://localhost:8000/v1/", model="j").judge(trajectory(), None)
    assert captured[0].full_url == "http://localhost:8000/v1/chat/completions"


def test_decoding_is_deterministic(captured: list[urllib.request.Request]) -> None:
    """A sampling judge makes the eval non-reproducible for no benefit."""
    HttpJudge(base_url="http://x/v1", model="j", temperature=0.7).judge(trajectory(), None)
    assert json.loads(captured[0].data)["temperature"] == 0.7


def test_an_http_error_is_reported_with_its_status(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", http_error(503, b"no capacity"))
    with pytest.raises(HarnessError, match="HTTP 503"):
        HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)


def test_an_unreachable_judge_is_reported(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", unreachable("connection refused"))
    with pytest.raises(HarnessError, match="cannot reach judge"):
        HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)


def test_a_non_json_body_is_refused(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a non-JSON body.

    An HTML login page instead of JSON means the endpoint is wrong, rather than
    the model having scored zero.
    """
    monkeypatch.setattr(urllib.request, "urlopen", returning(b"<html>login</html>"))
    with pytest.raises(HarnessError, match="non-JSON body"):
        HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)


def test_a_body_without_choices_is_refused(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", returning(json.dumps({"error": "quota"}).encode())
    )
    with pytest.raises(HarnessError, match="no choices"):
        HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)


def test_a_non_http_endpoint_is_refused() -> None:
    with pytest.raises(EvalStoreError, match="must be http"):
        HttpJudge(base_url="file:///etc/passwd", model="j").judge(trajectory(), None)


def test_no_auth_header_without_a_key(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locally served judge needs no credentials; sending an empty one breaks it."""
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)
    assert captured[0].get_header("Authorization") is None


def test_an_auth_header_is_sent_when_configured(
    captured: list[urllib.request.Request], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "secret-token")
    HttpJudge(base_url="http://x/v1", model="j").judge(trajectory(), None)
    assert captured[0].get_header("Authorization") == "Bearer secret-token"


def test_a_turn_with_no_tool_calls_renders_without_calls() -> None:
    """The final turn of a trajectory has no calls, and saying so is noise."""
    record = trajectory()
    record["turns"][1]["generated"]["tool_calls"] = []
    text = render_trajectory(record)
    assert "calls " not in text.split("## Turn 1")[1]


def test_a_turn_with_no_prose_renders_the_call_without_an_empty_line() -> None:
    """A model that only calls tools must not leave a dangling "engineer:"."""
    record = trajectory()
    record["turns"][0]["generated"]["text"] = ""
    turn_zero = render_trajectory(record).split("## Turn 1")[0]
    assert "engineer:" not in turn_zero
    assert "calls go_test" in turn_zero


def test_a_silent_command_renders_only_its_status() -> None:
    """go_build succeeds silently; the exit status is the only evidence there is."""
    record = trajectory()
    record["turns"][0]["executions"][0] = {
        "tool_name": "go_build",
        "status": "ok",
        "exit_code": 0,
        "stdout": "",
        "harness_error": "",
    }
    text = render_trajectory(record)
    assert "go_build exit=0" in text
    assert "panic:" not in text
