"""LLM judge: a scored verdict on a completed trajectory, from a pinned backend.

Two rules shape this module, both from the no-fallback policy.

**The judge is pinned, not discovered.** A judge that silently resolves to whatever
model is reachable makes scores incomparable across days: last week's 0.62 and
today's 0.62 may come from different models, and a regression reads as an
improvement. The endpoint and model id are required, recorded in the report, and a
mismatched verdict is refused.

**A malformed verdict is a failure, not a zero.** If the judge returns prose where a
score was expected, scoring it 0.0 would fold a judge outage into the model's
report. It raises instead, so an infrastructure problem looks like one.

Scoring is binary by default (pass / fail) because a graded score invites
threshold-shopping: the same trajectory can be made to look better by moving the
cut. ``score`` is still returned as 1.0 or 0.0 so it fits the aggregation code.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import EvalStoreError, HarnessError

JUDGE_SYSTEM_PROMPT: Final[str] = (
    "You are grading an AI Go engineer's work on a real repository task.\n"
    "You are given the task, the engineer's turns, and the exact output of every "
    "command they ran.\n\n"
    "Judge only what the evidence shows:\n"
    "- Did the changes actually address the task as stated?\n"
    "- Did the engineer verify the work by running tests or a build?\n"
    "- If tests failed, did they engage with the failure rather than ignore it?\n"
    "- Did the engineer stop when done, or keep changing unrelated code?\n\n"
    "A command that printed nothing is not a pass on its own; check the exit status "
    "and the diagnostics. Do not reward plausible-sounding reasoning that the "
    "evidence contradicts.\n\n"
    "Reply with JSON only, no prose and no code fence:\n"
    '{"pass": true|false, "reason": "<one sentence>"}'
)

REQUEST_TIMEOUT_S: Final[int] = 300

#: The judge used when a served endpoint is not available or affordable: the coding
#: session itself. Not reproducible -- the same trajectory may be graded differently
#: later -- so runs scored this way are labelled as such in the report.
SESSION_JUDGE_MODEL: Final[str] = "opencode/space-bunny-free"

#: On-disk shapes. Versioned because a queue written by one build must not be read
#: by another that changed the prompt or the trajectory rendering.
QUEUE_VERSION: Final[str] = "judge-queue-v1"
VERDICTS_VERSION: Final[str] = "judge-verdicts-v1"


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One scored trajectory."""

    score: float
    reason: str
    model: str
    raw: str

    def to_record(self) -> dict[str, Any]:
        """Serialisable verdict, including which judge produced it."""
        return {"score": self.score, "reason": self.reason, "judge_model": self.model}


def render_trajectory(record: Mapping[str, Any]) -> str:
    """Flatten a stored trajectory into the text the judge reads.

    Execution records are rendered in full rather than summarised. A judge that
    sees "3 commands ran, all OK" cannot tell a real fix from a model that
    reformatted a comment, which is exactly the distinction being graded.
    """
    task = record.get("task", {})
    lines: list[str] = [
        "## Task",
        f"id: {task.get('id', 'unknown')}",
        f"prompt: {task.get('prompt', '')}",
        "",
    ]
    if record.get("truncated"):
        lines += [
            "## Note",
            "The engineer hit the turn limit while still calling tools; the work is incomplete.",
            "",
        ]
    for turn in record.get("turns", []):
        generated = turn.get("generated", {})
        lines.append(f"## Turn {turn.get('turn', '?')}")
        if generated.get("text"):
            lines.append(f"engineer: {generated['text']}")
        for call in generated.get("tool_calls", []):
            name = call.get("name", "?")
            args = call.get("arguments", {})
            lines.append(f"calls {name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})")
        for execution in turn.get("executions", []):
            lines.append(
                f"-> {execution.get('tool_name')} exit={execution.get('exit_code')} "
                f"status={execution.get('status')}"
            )
            output = str(execution.get("stdout", "")).strip()
            if output:
                lines.append(output)
            elif execution.get("harness_error"):
                lines.append(f"[infrastructure error] {execution['harness_error']}")
        lines.append("")
    return "\n".join(lines)


def parse_verdict(raw: str, *, judge_model: str) -> JudgeVerdict:
    """Parse a judge's reply into a score, refusing anything ambiguous.

    The reply must be a single JSON object with a boolean ``pass``. A number in
    ``[0, 1]`` is also accepted, because a capable judge will produce one and
    rejecting it would be a policy, not a safety check -- but a bare number where
    the prompt demanded a boolean is treated as the boolean it encodes. Anything
    else raises: a judge that returned prose has not graded the trajectory, and
    calling that a zero would corrupt the report.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Some judges wrap JSON in a fence despite the instruction.
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if not text:
        raise HarnessError(f"judge {judge_model} returned an empty response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HarnessError(
            f"judge {judge_model} did not return JSON: {text[:200]!r}. Refusing to score this "
            "as a failure; a judge outage must not look like a model regression."
        ) from exc
    if not isinstance(parsed, dict):
        raise HarnessError(
            f"judge {judge_model} returned {type(parsed).__name__}, expected an object"
        )

    if "pass" in parsed:
        verdict = parsed["pass"]
        if isinstance(verdict, bool):
            score = 1.0 if verdict else 0.0
        elif isinstance(verdict, (int, float)) and 0.0 <= float(verdict) <= 1.0:
            score = float(verdict)
        else:
            raise HarnessError(f"judge {judge_model} returned pass={verdict!r}")
    elif "score" in parsed:
        value = parsed["score"]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HarnessError(f"judge {judge_model} returned score={value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise HarnessError(f"judge {judge_model} returned score={value!r}, outside [0, 1]")
        score = float(value)
    else:
        raise HarnessError(
            f"judge {judge_model} returned neither 'pass' nor 'score': {sorted(parsed)}"
        )

    reason = parsed.get("reason", "")
    return JudgeVerdict(
        score=score,
        reason=str(reason)[:2000],
        model=judge_model,
        raw=text[:4000],
    )


def judge_messages(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """The exact prompt sent to the judge, for reproducibility."""
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": render_trajectory(record)},
    ]


@dataclass(frozen=True, slots=True)
class HttpJudge:
    """Judge backed by any OpenAI-compatible ``/chat/completions`` endpoint.

    Works against vLLM, SGLang, llama.cpp, TGI's OpenAI route, or a hosted API,
    with no code change -- which is deliberate. The judge must be a strong model,
    and insisting on one vendor would make the pipeline hostage to that vendor's
    pricing and availability for the least replaceable component in the loop.
    """

    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key_env: str = "JUDGE_API_KEY"

    def __post_init__(self) -> None:
        """Refuse a configuration whose scores could not be attributed."""
        if not self.base_url:
            raise EvalStoreError("judge base_url is required; refusing an unpinned judge")
        if not self.model:
            raise EvalStoreError("judge model id is required; a score needs an author")

    def judge(self, record: Mapping[str, Any], fingerprint: object) -> dict[str, Any]:
        """Score one trajectory and return the stored verdict record."""
        payload = {
            "model": self.model,
            "messages": judge_messages(record),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - scheme is validated below
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        if not request.full_url.startswith(("http://", "https://")):
            raise EvalStoreError(f"judge base_url must be http(s), got {request.full_url!r}")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise HarnessError(f"judge {self.model} returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise HarnessError(
                f"cannot reach judge {self.model} at {self.base_url}: {exc.reason}"
            ) from exc

        return parse_verdict(
            _first_message(raw, model=self.model), judge_model=self.model
        ).to_record()

    def _headers(self) -> dict[str, str]:
        """Auth header when a key is configured; judges served locally need none."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            return {"Content-Type": "application/json"}
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def _first_message(raw: str, *, model: str) -> str:
    """Extract the assistant text from an OpenAI-style response body."""
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"judge {model} returned non-JSON body: {raw[:200]!r}") from exc
    try:
        return str(decoded["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise HarnessError(
            f"judge {model} response has no choices[0].message.content: {raw[:200]!r}"
        ) from exc


def judge_identity(judge: object) -> str:
    """A stable identity for a judge, used in run manifests.

    Without this, two runs scored by different judges look directly comparable in
    the report, and the first thing a reader concludes from an improvement is wrong.
    """
    model = getattr(judge, "model", None)
    if isinstance(model, str) and model:
        base = getattr(judge, "base_url", "")
        return f"{model}@{base}" if base else model
    return type(judge).__name__


# ------------------------------------------------------------ judge queue


@dataclass(frozen=True, slots=True)
class QueueEntry:
    """One trajectory awaiting a verdict, addressable by its store key.

    The key travels with the entry so a verdict can be written back without
    recomputing the fingerprint, and so a verdict cannot be filed against the
    wrong trajectory by accident.
    """

    key: str
    task_id: str
    sample_index: int
    trajectory: str
    prompt: str

    def to_record(self) -> dict[str, Any]:
        """Serialisable queue line."""
        return {
            "version": QUEUE_VERSION,
            "key": self.key,
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "prompt": self.prompt,
            "trajectory": self.trajectory,
        }

    @property
    def label(self) -> str:
        """Human-readable identifier used in verdict files."""
        return f"{self.task_id}#{self.sample_index}"


def write_judge_queue(path: str, entries: Sequence[QueueEntry]) -> int:
    """Write the trajectories that still need grading; return how many.

    Deterministic order and one JSON object per line, so a diff of two queue files
    shows only the trajectories that actually changed.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(entry.to_record(), ensure_ascii=False, sort_keys=True) + "\n"
        for entry in entries
    )
    target.write_text(body, encoding="utf-8", newline="\n")
    return len(entries)


def read_judge_queue(path: str) -> list[QueueEntry]:
    """Read a queue file, refusing anything that is not the current shape."""
    source = Path(path)
    if not source.is_file():
        raise HarnessError(f"judge queue not found: {source}")
    entries: list[QueueEntry] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{source}:{number} is not valid JSON: {exc}") from exc
        if record.get("version") != QUEUE_VERSION:
            raise HarnessError(
                f"{source}:{number} declares queue version {record.get('version')!r}, expected "
                f"{QUEUE_VERSION!r}. The prompt or the trajectory rendering changed; regenerate "
                "the queue instead of grading against a stale format."
            )
        entries.append(
            QueueEntry(
                key=str(record["key"]),
                task_id=str(record["task_id"]),
                sample_index=int(record["sample_index"]),
                trajectory=str(record["trajectory"]),
                prompt=str(record["prompt"]),
            )
        )
    return entries


@dataclass(frozen=True, slots=True)
class SessionJudge:
    """Judge that replays verdicts produced in a coding session.

    This is the path that costs nothing and needs no GPU: the eval writes its
    trajectories to a queue, the session grades them, and the verdicts come back
    into the store under the keys the queue carried.

    The trade-off is stated rather than hidden. It is **not reproducible**: the same
    trajectory graded in a later session may get a different verdict. Runs scored
    this way are marked ``session_judge`` in the report so nobody compares them
    across checkpoints as if they were equally reliable as a served model.
    """

    verdicts: Mapping[str, Mapping[str, Any]]
    model: str = SESSION_JUDGE_MODEL

    @property
    def identity(self) -> str:
        """Recorded in the manifest and the report."""
        return f"{self.model}@session"

    def judge(self, record: Mapping[str, Any], fingerprint: Any) -> dict[str, Any]:  # noqa: ANN401 - Fingerprint
        """Return the recorded verdict for this trajectory."""
        key = fingerprint.key
        try:
            verdict = self.verdicts[key]
        except KeyError as exc:
            raise HarnessError(
                f"no session verdict for {record.get('task_id')}#{record.get('sample_index')} "
                f"(key {key[:16]}). Grade every queue entry; a missing verdict is not a zero."
            ) from exc
        return {
            "score": float(verdict["score"]),
            "reason": str(verdict.get("reason", "")),
            "judge_model": self.identity,
        }


def read_verdicts(path: str, queue: Sequence[QueueEntry]) -> dict[str, dict[str, Any]]:
    """Read verdicts and check them against the queue they answer.

    Every queue entry needs exactly one verdict, and no verdict may name an entry
    that was not queued. Both directions matter: a missing verdict would silently
    shrink the denominator, and an extra one means the queue and the verdicts came
    from different runs.
    """
    source = Path(path)
    if not source.is_file():
        raise HarnessError(f"verdict file not found: {source}")
    expected = {entry.key: entry for entry in queue}
    verdicts: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{source}:{number} is not valid JSON: {exc}") from exc
        if record.get("version") != VERDICTS_VERSION:
            raise HarnessError(
                f"{source}:{number} declares version {record.get('version')!r}, expected "
                f"{VERDICTS_VERSION!r}"
            )
        key = str(record.get("key", ""))
        if key not in expected:
            raise HarnessError(
                f"{source}:{number} grades key {key[:16]}, which is not in the queue. The queue "
                "and the verdicts are from different runs; regenerate the queue."
            )
        if key in verdicts:
            raise HarnessError(f"{source}:{number} grades {expected[key].label} twice")
        verdicts[key] = {"score": record["score"], "reason": record.get("reason", "")}

    missing = sorted(entry.label for key, entry in expected.items() if key not in verdicts)
    if missing:
        raise HarnessError(
            f"{len(missing)} queue entries have no verdict: {missing}. Every trajectory must be "
            "graded; an ungraded sample is a non-pass, not an absent one."
        )
    return verdicts


def write_verdicts(path: str, verdicts: Mapping[str, Mapping[str, Any]]) -> int:
    """Write a verdict file in the shape :func:`read_verdicts` expects."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(
            {
                "version": VERDICTS_VERSION,
                "key": key,
                "score": verdict["score"],
                "reason": verdict.get("reason", ""),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for key, verdict in sorted(verdicts.items())
    )
    target.write_text(body, encoding="utf-8", newline="\n")
    return len(verdicts)
