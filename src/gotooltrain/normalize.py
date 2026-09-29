"""Policy layer: turn raw records into validated :class:`Conversation` objects.

This is where every "is this record actually trainable?" decision lives. The
rules are deliberately strict and fail fast: a dataset is usually millions of
records, and a silently mis-validated record turns into hours of wasted training
on data that teaches the wrong behaviour.

Guarantees provided here
------------------------
* At most one system turn, always first.
* Every assistant turn has something to learn (text, thinking, or a tool call).
* Tool call ids are present, unique, and every result answers a call that was
  actually made; no dangling ids on either side.
* All results for one assistant turn are grouped into a single tool turn.
* Tool catalogue ordering is deterministic, so records are reproducible.
* Untrusted text cannot break out of its ``<tool_result>`` block.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Final

from .errors import (
    DanglingToolResultError,
    EmptyAssistantTurnError,
    ReservedTagError,
    ToolCallIdError,
    UnsupportedRoleError,
    ValidationError,
)
from .schema import (
    RESERVED_TAGS,
    Conversation,
    ImageSpec,
    Message,
    ToolCall,
    ToolResultPayload,
    ToolSpec,
)

_ALLOWED_ROLES: Final[frozenset[str]] = frozenset(
    {"system", "developer", "user", "assistant", "tool"}
)

#: Upper bound on images per turn. Vision pads are cheap relative to text but
#: not free, and an unbounded count can silently dominate a 32K context.
MAX_IMAGES_PER_TURN: Final[int] = 4

#: Replacement for '<' when neutralising reserved tags inside untrusted text.
#: A single guillemet is visually distinct and cannot reopen an XML-ish tag.
_ANGLE_REPLACEMENT: Final[str] = "‹"  # noqa: RUF001


# --------------------------------------------------------------------- escaping


def escape_reserved_tags(text: str) -> str:
    """Defuse reserved tags so untrusted text cannot close a block early.

    Only the ``<`` of each known tag is replaced, leaving the rest of the string
    byte-identical. This is deliberately *not* applied to assistant output: tool
    arguments are a training target, and rewriting targets would teach the model
    to emit mangled code.
    """
    out = text
    for tag in RESERVED_TAGS:
        if tag in out:
            out = out.replace(tag, tag.replace("<", _ANGLE_REPLACEMENT, 1))
    return out


def _check_untrusted(text: str, *, where: str, sanitize: bool) -> str:
    if not any(tag in text for tag in RESERVED_TAGS):
        return text
    if not sanitize:
        raise ReservedTagError(
            f"{where} contains a reserved tool tag, which could break out of its "
            f"block. Pass sanitize_reserved_tags=True to escape it, or fix the source data."
        )
    return escape_reserved_tags(text)


# ------------------------------------------------------------------------ tools


def _coerce_json_object(raw: Any, *, where: str) -> dict[str, Any]:
    """Accept either a JSON string or an already-decoded mapping."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{where} is not valid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValidationError(f"{where} must decode to an object, got {type(decoded).__name__}")
        return decoded
    raise ValidationError(f"{where} must be an object or JSON string, got {type(raw).__name__}")


def normalize_tools(
    tools: Sequence[dict[str, Any]] | None, *, sanitize: bool = False
) -> tuple[ToolSpec, ...]:
    """Convert OpenAI tool definitions into sorted :class:`ToolSpec` entries.

    Sorting by name is not cosmetic: the catalogue is part of the prompt, so a
    non-deterministic order makes the same record render differently across runs
    and breaks run-to-run comparability.
    """
    if not tools:
        return ()
    seen: dict[str, ToolSpec] = {}
    for index, entry in enumerate(tools):
        where = f"tools[{index}]"
        function = entry.get("function", entry)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{where}.name must be a non-empty string")
        if name in seen:
            raise ValidationError(f"{where}.name duplicates tool {name!r}")
        description = function.get("description") or ""
        if not isinstance(description, str):
            raise ValidationError(f"{where}.description must be a string")
        schema = _coerce_json_object(
            function.get("parameters") or {"type": "object", "properties": {}},
            where=f"{where}.parameters",
        )
        seen[name] = ToolSpec(
            name=name,
            description=_check_untrusted(
                description, where=f"{where}.description", sanitize=sanitize
            ),
            input_schema=schema,
        )
    return tuple(seen[name] for name in sorted(seen))


# --------------------------------------------------------------------- messages


def _extract_thinking(msg: dict[str, Any]) -> tuple[str, str]:
    """Return ``(reasoning_content, content)``.

    Accepts an explicit ``reasoning_content`` field or a ``<think>...</think>``
    prefix, matching the two conventions seen in the wild.
    """
    explicit = msg.get("reasoning_content") or msg.get("thinking") or ""
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), str(msg.get("content") or "")
    content = str(msg.get("content") or "")
    if "</think>" in content:
        head, _, tail = content.partition("</think>")
        thinking = head.rsplit("<think>", 1)[-1].strip()
        return thinking, tail.strip()
    return "", content


def _normalize_tool_calls(msg: dict[str, Any], *, where: str) -> tuple[ToolCall, ...]:
    raw_calls = msg.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ValidationError(f"{where}.tool_calls must be a list")
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        call_where = f"{where}.tool_calls[{index}]"
        function = raw.get("function", raw)
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError(f"{call_where}.function.name must be a non-empty string")
        call_id = raw.get("id") or function.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ToolCallIdError(
                f"{call_where} has no id. Every tool call needs a stable id so its "
                f"result can be matched; generate one in the data layer."
            )
        calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=_coerce_json_object(
                    function.get("arguments") or {}, where=f"{call_where}.function.arguments"
                ),
            )
        )
    return tuple(calls)


def _normalize_results(
    msg: dict[str, Any], *, where: str, sanitize: bool
) -> tuple[ToolResultPayload, ...]:
    """Read tool results from either supported shape.

    Canonical form is a ``results`` list. The flat OpenAI form
    (``role='tool'`` with ``tool_call_id`` and ``content``) is accepted too so
    that xLAM/ToolACE/BFCL exports need no preprocessing.
    """
    raw_results: Sequence[Any] = msg.get("results") or ()
    if not raw_results and msg.get("tool_call_id"):
        raw_results = [msg]
    results: list[ToolResultPayload] = []
    for index, raw in enumerate(raw_results):
        result_where = f"{where}.results[{index}]"
        if not isinstance(raw, dict):
            raise ValidationError(f"{result_where} must be an object")
        tool_use_id = raw.get("tool_use_id") or raw.get("tool_call_id")
        if not isinstance(tool_use_id, str) or not tool_use_id:
            raise ToolCallIdError(f"{result_where}.tool_use_id must be a non-empty string")
        results.append(
            ToolResultPayload(
                tool_use_id=tool_use_id,
                content=_check_untrusted(
                    str(raw.get("content") or ""),
                    where=f"{result_where}.content",
                    sanitize=sanitize,
                ),
                is_error=bool(raw.get("is_error", False)),
            )
        )
    return tuple(results)


def _is_positive_int(value: object) -> bool:
    """True for a real positive int.

    ``bool`` is an ``int`` subclass, so it is excluded explicitly: ``True`` as a
    patch count is a data bug, not a 1.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_image_ref(ref: str, *, where: str) -> None:
    """Reject an image reference that points outside the dataset's image root.

    ``ref`` is read from a data file, so it is untrusted input, and it is only
    dereferenced much later by the image loader. Catching it here means a poisoned
    dataset fails validation rather than quietly reading whatever the training host
    happens to have at that path.

    The check is on the *shape*, not on the filesystem: a symlink cannot be judged
    without opening it, which is why :func:`gotooltrain.vision.load_image` re-checks
    containment on the resolved path.
    """
    if "\\" in ref:
        raise ValidationError(
            f"{where} must use forward slashes; backslashes make the same path mean different "
            f"things on different hosts: {ref!r}"
        )
    if (
        ref.startswith("/")
        or PurePosixPath(ref).is_absolute()
        or PureWindowsPath(ref).is_absolute()
    ):
        raise ValidationError(
            f"{where} must be relative to the image root, got an absolute path: {ref!r}"
        )
    if PureWindowsPath(ref).drive:
        raise ValidationError(f"{where} must not carry a drive letter: {ref!r}")
    parts = PurePosixPath(ref).parts
    if ".." in parts:
        raise ValidationError(f"{where} must not traverse upwards out of the image root: {ref!r}")
    if not parts or parts[0] in (".", ""):
        raise ValidationError(f"{where} must name an image inside the root: {ref!r}")


def _normalize_images(
    raw_images: Sequence[Any], *, where: str, sanitize: bool
) -> tuple[ImageSpec, ...]:
    """Validate image attachments attached to a user turn.

    ``grid_thw`` must be three positive integers and ``merge_size`` at least 1,
    with the grid divisible by ``merge_size**2`` so the pad-token count is an
    exact integer. A guessable pad count would silently desynchronise the token
    layout from the vision tower's output.
    """
    if not isinstance(raw_images, Sequence) or isinstance(raw_images, (str, bytes)):
        raise ValidationError(f"{where}.images must be a list")
    if len(raw_images) > MAX_IMAGES_PER_TURN:
        raise ValidationError(
            f"{where} has {len(raw_images)} images, above the limit of "
            f"{MAX_IMAGES_PER_TURN} per turn"
        )
    images: list[ImageSpec] = []
    for index, raw in enumerate(raw_images):
        image_where = f"{where}.images[{index}]"
        if not isinstance(raw, dict):
            raise ValidationError(f"{image_where} must be an object")
        ref = raw.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            raise ValidationError(f"{image_where}.ref must be a non-empty string")
        _check_image_ref(ref, where=f"{image_where}.ref")
        grid_raw = raw.get("grid_thw")
        if not isinstance(grid_raw, Sequence) or isinstance(grid_raw, (str, bytes)):
            raise ValidationError(f"{image_where}.grid_thw must be a list of three integers")
        if len(grid_raw) != 3:
            raise ValidationError(
                f"{image_where}.grid_thw must have three entries (t, h, w), got {len(grid_raw)}"
            )
        if not all(_is_positive_int(dim) for dim in grid_raw):
            raise ValidationError(f"{image_where}.grid_thw entries must be positive integers")
        merge_size = raw.get("merge_size", 2)
        if not _is_positive_int(merge_size):
            raise ValidationError(f"{image_where}.merge_size must be an integer >= 1")
        spec = ImageSpec(
            ref=_check_untrusted(ref, where=f"{image_where}.ref", sanitize=sanitize),
            grid_thw=(grid_raw[0], grid_raw[1], grid_raw[2]),
            merge_size=merge_size,
        )
        patches = spec.grid_thw[0] * spec.grid_thw[1] * spec.grid_thw[2]
        per_merge = merge_size * merge_size
        if patches % per_merge != 0:
            # Positive dims plus exact divisibility guarantee at least one token,
            # so this is the only way the pad count can disagree with the tower.
            raise ValidationError(
                f"{image_where} has {patches} patches, which is not divisible by "
                f"merge_size**2 ({per_merge}); the vision token count would not match the "
                f"vision tower's output"
            )
        images.append(spec)
    return tuple(images)


def normalize_conversation(
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    *,
    sanitize_reserved_tags: bool = False,
) -> Conversation:
    """Validate and canonicalise one conversation.

    ``role="tool"`` messages are merged into the preceding group so that all
    results answering a single assistant turn land in one tool turn, which is
    what the wire format and the template both expect.
    """
    if not messages:
        raise ValidationError("conversation has no messages")

    # Images are only meaningful on user turns: the assistant generates tokens,
    # tool results are text, and Qwen's own template rejects images in a system
    # prompt. Rejecting them elsewhere keeps the vision path single-sourced.
    for index, msg in enumerate(messages):
        if msg.get("images") and msg.get("role") != "user":
            raise ValidationError(
                f"messages[{index}] carries images on role {msg.get('role')!r}; "
                f"images are only allowed on user turns"
            )

    tool_specs = normalize_tools(tools, sanitize=sanitize_reserved_tags)
    tool_names = {spec.name for spec in tool_specs}

    out: list[Message] = []
    pending_results: list[ToolResultPayload] = []
    seen_call_ids: set[str] = set()
    answered_call_ids: set[str] = set()
    user_turns = 0
    trainable_turns = 0

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            out.append(Message(role="tool", results=tuple(pending_results)))
            pending_results = []

    def record_result(payload: ToolResultPayload, *, index: int, count: int) -> None:
        if payload.tool_use_id not in seen_call_ids:
            raise DanglingToolResultError(
                f"result {index + 1}/{count} references unknown tool call "
                f"{payload.tool_use_id!r}; every result must answer a call made earlier"
            )
        if payload.tool_use_id in answered_call_ids:
            raise ToolCallIdError(f"tool call {payload.tool_use_id!r} has more than one result")
        answered_call_ids.add(payload.tool_use_id)
        pending_results.append(payload)

    for index, msg in enumerate(messages):
        where = f"messages[{index}]"
        role = msg.get("role")
        if role not in _ALLOWED_ROLES:
            raise UnsupportedRoleError(
                f"{where}.role={role!r} is not supported; expected one of {sorted(_ALLOWED_ROLES)}"
            )

        if role in ("system", "developer"):
            flush_results()
            if out:
                raise ValidationError(f"{where} is a system message but is not first")
            out.append(
                Message(
                    role="system",
                    content=_check_untrusted(
                        str(msg.get("content") or ""),
                        where=f"{where}.content",
                        sanitize=sanitize_reserved_tags,
                    ),
                )
            )
            continue

        if role == "user":
            flush_results()
            user_turns += 1
            out.append(
                Message(
                    role="user",
                    content=_check_untrusted(
                        str(msg.get("content") or ""),
                        where=f"{where}.content",
                        sanitize=sanitize_reserved_tags,
                    ),
                    images=_normalize_images(
                        msg.get("images") or (),
                        where=where,
                        sanitize=sanitize_reserved_tags,
                    ),
                )
            )
            continue

        if role == "assistant":
            thinking, content = _extract_thinking(msg)
            calls = _normalize_tool_calls(msg, where=where)
            # Whitespace-only text is empty: accepting it would emit a bare
            # turn marker that carries no supervision.
            if not (thinking or content.strip() or calls):
                raise EmptyAssistantTurnError(
                    f"{where} is an assistant turn with no text, no thinking and no tool call; "
                    f"it would contribute nothing but cost a turn marker"
                )
            unknown = {c.name for c in calls} - tool_names
            if unknown and tool_names:
                raise ValidationError(
                    f"{where} calls tools not present in the catalogue: {sorted(unknown)}"
                )
            for call in calls:
                if call.id in seen_call_ids:
                    raise ToolCallIdError(f"duplicate tool call id {call.id!r} at {where}")
                seen_call_ids.add(call.id)
            flush_results()
            out.append(
                Message(
                    role="assistant",
                    content=content,
                    reasoning_content=thinking,
                    tool_calls=calls,
                )
            )
            trainable_turns += 1
            continue

        # role == "tool"
        results = _normalize_results(msg, where=where, sanitize=sanitize_reserved_tags)
        if not results:
            raise ValidationError(f"{where} is a tool message with no results")
        count = len(results)
        for offset, payload in enumerate(results):
            record_result(payload, index=offset, count=count)

    # flush_results() empties the buffer, so no orphan check is needed here: a
    # result can only reach pending_results after its call was registered, which
    # record_result() already verifies.
    flush_results()

    if user_turns == 0:
        raise ValidationError("conversation has no user turn")
    if trainable_turns == 0:
        raise ValidationError(
            "conversation has no assistant turn, so it contributes no training signal"
        )
    unanswered = seen_call_ids - answered_call_ids
    if unanswered:
        raise ToolCallIdError(
            f"tool calls without a result: {sorted(unanswered)}; a training example must "
            f"show the full loop so the model learns to read tool output"
        )

    return Conversation(messages=out, tools=tool_specs)


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    sanitize_reserved_tags: bool = False,
    on_error: str = "raise",
) -> list[Conversation]:
    """Normalise a stream of raw records.

    ``on_error='skip'`` drops bad records and counts them; the count is attached
    to the returned list's ``skipped`` attribute by :func:`load_conversations`.
    """
    if on_error not in ("raise", "skip"):
        raise ValidationError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    kept: list[Conversation] = []
    for record in records:
        if on_error == "raise":
            kept.append(
                normalize_conversation(
                    record.get("messages", []),
                    record.get("tools"),
                    sanitize_reserved_tags=sanitize_reserved_tags,
                )
            )
        else:
            try:
                kept.append(
                    normalize_conversation(
                        record.get("messages", []),
                        record.get("tools"),
                        sanitize_reserved_tags=sanitize_reserved_tags,
                    )
                )
            except ValidationError:
                continue
    return kept
