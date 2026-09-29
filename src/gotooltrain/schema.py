"""Canonical message schema for the ``anthropic-tools-v1`` token format.

These dataclasses are the contract between the data layer and the Jinja chat
template. The template reads exactly these field names, so renaming a field here
is a breaking format change and must be accompanied by a new ``FORMAT_VERSION``.

Validation lives in :mod:`gotooltrain.normalize`, not here: this module is the
shape, that module is the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypedDict

FORMAT_VERSION: Final[str] = "anthropic-tools-v1"

#: Tags that delimit tool traffic. Untrusted content is checked against these.
TOOLS_OPEN: Final[str] = "<available_tools>"
TOOLS_CLOSE: Final[str] = "</available_tools>"
TOOL_USE_OPEN: Final[str] = "<tool_use>"
TOOL_USE_CLOSE: Final[str] = "</tool_use>"
TOOL_RESULT_OPEN: Final[str] = "<tool_result>"
TOOL_RESULT_CLOSE: Final[str] = "</tool_result>"

#: Substrings that would let injected text terminate a block early.
RESERVED_TAGS: Final[tuple[str, ...]] = (
    TOOLS_OPEN,
    TOOLS_CLOSE,
    TOOL_USE_OPEN,
    TOOL_USE_CLOSE,
    TOOL_RESULT_OPEN,
    TOOL_RESULT_CLOSE,
)

Role = Literal["system", "user", "assistant", "tool"]
ToolRole = Literal["tool"]


@dataclass(frozen=True, slots=True)
class ImageSpec:
    """An image attached to a user turn.

    ``grid_thw`` is the vision patch grid and ``merge_size`` the spatial merge
    factor, so the number of ``<|vision_pad|>`` tokens is *known* without the
    image processor. That matters: if token layout depended on the processor,
    the training-time and serving-time token counts could disagree and the
    format would stop being self-describing.
    """

    ref: str
    grid_thw: tuple[int, int, int]
    merge_size: int

    @property
    def vision_tokens(self) -> int:
        """Number of ``<|vision_pad|>`` tokens this image expands to."""
        t, h, w = self.grid_thw
        merged = t * h * w
        return merged // (self.merge_size * self.merge_size)

    @property
    def vision_sequence(self) -> str:
        """The exact token string for this image, without the wrapper tokens."""
        return "<|vision_pad|>" * self.vision_tokens

    def to_template(self) -> dict[str, Any]:
        """Shape the template reads: the reference plus the pad-token count."""
        return {
            "ref": self.ref,
            "grid_thw": list(self.grid_thw),
            "merge_size": self.merge_size,
            "vision_tokens": self.vision_tokens,
            "vision_sequence": self.vision_sequence,
        }


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool definition as advertised in the catalogue."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_template(self) -> dict[str, Any]:
        """Shape expected by the template (``name``/``description``/``input_schema``)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """An assistant request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_template(self) -> dict[str, Any]:
        """Shape consumed by the template's ``call.function.*`` lookups."""
        return {
            "id": self.id,
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True, slots=True)
class ToolResultPayload:
    """A single tool response, as it appears inside one ``<tool_result>`` block."""

    tool_use_id: str
    content: str
    is_error: bool = False

    def to_template(self) -> dict[str, Any]:
        """JSON body rendered inside a ``<tool_result>`` block."""
        return {
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


@dataclass(frozen=True, slots=True)
class Message:
    """One canonical message.

    ``role='tool'`` carries ``results`` (a list) because Anthropic delivers every
    result for one assistant turn inside a single user turn; grouping is the
    data layer's job and the template rejects ungrouped results.
    """

    role: Role
    content: str = ""
    reasoning_content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    results: tuple[ToolResultPayload, ...] = ()
    images: tuple[ImageSpec, ...] = ()


class ConversationDict(TypedDict, total=False):
    """JSON shape of one training record on disk."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(slots=True)
class Conversation:
    """A validated conversation: at most one leading system turn, then the turns."""

    messages: list[Message] = field(default_factory=list)
    tools: tuple[ToolSpec, ...] = ()

    @property
    def system_prompt(self) -> str:
        """The leading system message's text, or an empty string when absent."""
        if self.messages and self.messages[0].role == "system":
            return self.messages[0].content
        return ""

    def to_messages(self) -> list[dict[str, Any]]:
        """Serialise to plain dicts for ``apply_chat_template``."""
        out: list[dict[str, Any]] = []
        for message in self.messages:
            if message.role == "assistant":
                out.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "reasoning_content": message.reasoning_content,
                        "tool_calls": [c.to_template() for c in message.tool_calls],
                    }
                )
            elif message.role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "content": "",
                        "results": [r.to_template() for r in message.results],
                    }
                )
            else:
                out.append(
                    {
                        "role": message.role,
                        "content": message.content,
                        "images": [i.to_template() for i in message.images],
                    }
                )
        return out

    def to_tools(self) -> list[dict[str, Any]]:
        """Tool catalogue in the shape the template iterates over."""
        return [t.to_template() for t in self.tools]
