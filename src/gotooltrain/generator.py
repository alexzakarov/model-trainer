"""The model under test: an OpenAI-compatible chat generator.

This is the piece that turns a conversation into the ``{"text", "tool_calls"}``
shape :mod:`gotooltrain.evalrun` drives. Three things in it are load-bearing.

**The prompt names the tools, in catalogue order.** Evaluating against a different
catalogue or ordering than the one the model was trained on would measure
prompt-following rather than tool use, and the difference would look like a
capability gap.

**Invalid tool calls are not pre-filtered here.** Unknown tools and arguments that
fail the schema are deliberately left for :mod:`gotooltrain.gorun` to reject at
execution time, so the failure comes back to the model as a ``<tool_result>`` it
can read and correct. Screening them here would hide the mistake from the one
place that can teach the model about it.

**A malformed reply is a harness error.** A model that answers with prose when the
protocol expects JSON has not produced a turn; recording that as "called nothing"
would file it as a model that chose not to use tools.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .errors import HarnessError
from .evalstore import Fingerprint
from .gotools import catalog

GENERATE_TIMEOUT_S: Final[int] = 600

#: Marker the chat template writes for a model-produced span. Present in the
#: system prompt so a served model reproduces the trained framing verbatim.
SYSTEM_PROMPT: Final[str] = (
    "You are an expert Go engineer working in a real repository.\n"
    "Use the provided tools to investigate before you edit, and verify your work by "
    "running tests or a build before you claim it is done.\n"
    "Prefer reading a file over guessing its contents, and prefer a targeted edit over "
    "rewriting a whole file.\n"
    "When a command fails, read the failure and address it; do not retry unchanged."
)


def _to_openai_messages(
    task: Mapping[str, Any], history: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Convert the harness conversation into OpenAI chat messages.

    The harness keeps tool results as ``role="tool"`` entries with an explicit
    ``tool_call_id``; the wire format wants the same thing under the same names, so
    this is a rename rather than a translation.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(task.get("prompt", ""))},
    ]
    for entry in history:
        role = entry.get("role")
        if role == "user":
            messages.append({"role": "user", "content": str(entry.get("content", ""))})
        elif role == "assistant":
            message: dict[str, Any] = {
                "role": "assistant",
                "content": str(entry.get("content", "")),
            }
            if entry.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": str(call.get("id") or f"call_{index}"),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name", "")),
                            "arguments": json.dumps(
                                call.get("arguments", {}), ensure_ascii=False, sort_keys=True
                            ),
                        },
                    }
                    for index, call in enumerate(entry["tool_calls"])
                ]
            messages.append(message)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(entry.get("tool_call_id", "")),
                    "content": str(entry.get("content", "")),
                }
            )
        else:
            # Silently dropping an entry the model was shown in training would make
            # it believe a tool result never arrived -- or worse, that a turn it just
            # wrote was never sent.
            raise HarnessError(
                f"conversation holds a {role!r} entry, which is not a role this harness "
                f"produces. Refusing to send a conversation the model was not trained on."
            )
    return messages


def parse_reply(raw: str, *, model: str) -> dict[str, Any]:
    """Turn a chat completion body into one assistant turn.

    Arguments arrive as a JSON string and are decoded here rather than being passed
    through: a model that emits ``{"path": }`` would otherwise be executed as if the
    arguments were empty, and the workspace edit would silently apply to nothing.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"{model} returned a non-JSON response: {raw[:200]!r}") from exc
    try:
        message = decoded["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HarnessError(f"{model} response has no choices[0].message: {raw[:200]!r}") from exc

    tool_calls: list[dict[str, Any]] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        try:
            function = call["function"]
            name = str(function["name"])
            arguments = function.get("arguments", "{}")
        except (KeyError, TypeError) as exc:
            raise HarnessError(f"{model} emitted a tool call without a function: {call!r}") from exc
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise HarnessError(
                    f"{model} called {name!r} with unparseable arguments {arguments!r}. "
                    "Executing it would apply an edit whose contents are unknown."
                ) from exc
        if not isinstance(arguments, dict):
            raise HarnessError(f"{model} called {name!r} with non-object arguments: {arguments!r}")
        tool_calls.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": name,
                "arguments": arguments,
            }
        )

    return {"text": str(message.get("content") or ""), "tool_calls": tool_calls}


@dataclass(frozen=True, slots=True)
class HttpGenerator:
    """Drives a served model through an OpenAI-compatible chat endpoint.

    The same backend works for vLLM, SGLang, llama.cpp, TGI's OpenAI route and
    hosted APIs, so evaluating a checkpoint does not depend on where it is served.
    """

    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    api_key_env: str = "EVAL_API_KEY"
    timeout_s: int = GENERATE_TIMEOUT_S
    advertise_tools: bool = True

    def __post_init__(self) -> None:
        """Refuse an unpinned endpoint; scores would not be attributable."""
        if not self.base_url:
            raise HarnessError("generator base_url is required")
        if not self.model:
            raise HarnessError("generator model id is required")

    @property
    def identity(self) -> str:
        """Recorded in run manifests so a served-model swap is visible."""
        return f"{self.model}@{self.base_url}"

    def generate(
        self,
        task: Mapping[str, Any],
        sample_index: int,
        turn_index: int,
        messages: Sequence[Mapping[str, Any]],
        fingerprint: Fingerprint,
    ) -> dict[str, Any]:
        """Produce one assistant turn for this sample."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(task, messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.advertise_tools:
            payload["tools"] = catalog()
        # The seed travels with every request: n_samples means genuinely different
        # samples, not the same one decoded repeatedly under different names.
        payload["seed"] = int(fingerprint.seed) + 1_000 * int(fingerprint.sample_index)

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        if not url.startswith(("http://", "https://")):
            raise HarnessError(f"generator base_url must be http(s), got {url!r}")
        request = urllib.request.Request(  # noqa: S310 - scheme checked above
            url=url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise HarnessError(f"{self.model} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise HarnessError(
                f"cannot reach {self.model} at {self.base_url}: {exc.reason}"
            ) from exc

        return parse_reply(raw, model=self.model)

    def _headers(self) -> dict[str, str]:
        """Auth header when a key is configured; a local server needs none."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            return {"Content-Type": "application/json"}
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
