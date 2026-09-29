"""Template loading, rendering and assistant-mask construction.

The Jinja template in ``templates/anthropic-tools-v1.jinja`` is the single source
of truth. This module only *loads* it and turns the tokenizer's assistant mask
into training labels — it never re-implements the rendering.

Using the tokenizer's own ``{% generation %}`` bookkeeping (rather than
hand-rolled character offsets) is deliberate: it is the same code path the
inference stack will use, so a formatting mistake cannot hide in a Python-only
implementation that training uses but serving does not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import TemplateError
from .schema import FORMAT_VERSION, Conversation

TEMPLATE_PACKAGE: Final[str] = "gotooltrain"
TEMPLATE_DIRNAME: Final[str] = "templates"
TEMPLATE_FILENAME: Final[str] = "anthropic-tools-v1.jinja"
CHAT_TEMPLATE_FILENAME: Final[str] = "chat_template.jinja"
IGNORED_INDEX: Final[int] = -100

#: Marker every template revision must carry. A drifted or hand-edited template
#: fails here instead of silently training on a different token format.
_VERSION_MARKER: Final[str] = "format: "
_VERSION_PATTERN: Final[str] = r"\{#-\s*\n\s*format:\s*([a-z0-9.\-]+)\s*\n"


class SupportsChatTemplate(Protocol):
    """The slice of ``PreTrainedTokenizer`` this package depends on.

    Signatures stay loose on purpose: the real methods take many keyword
    arguments and are typed loosely upstream too.
    """

    chat_template: str | None

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
        """Render a conversation to text or to token ids."""

    def decode(self, *args: Any, **kwargs: Any) -> str:
        """Decode token ids back to text."""

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Write tokenizer files, including ``chat_template.jinja``."""


def template_path() -> Path:
    """Filesystem path of the shipped chat template."""
    try:
        root = resources.files(TEMPLATE_PACKAGE)
    except (ModuleNotFoundError, TypeError) as exc:  # pragma: no cover - packaging
        raise TemplateError(
            f"cannot locate the '{TEMPLATE_PACKAGE}' package; the chat template is missing "
            f"from the installation. Reinstall gotooltrain."
        ) from exc
    return Path(str(root)) / TEMPLATE_DIRNAME / TEMPLATE_FILENAME


@lru_cache(maxsize=1)
def load_template_source() -> str:
    """Read the chat template from disk and verify its format version.

    Callers get the shipped template or an exception. There is no silent
    fallback: a missing, unreadable or mis-versioned template is a build error,
    not something to paper over.
    """
    path = template_path()
    if not path.is_file():
        raise TemplateError(
            f"chat template not found at {path}. The package data is missing; "
            f"reinstall gotooltrain or pass an explicit source to install_template()."
        )
    source = path.read_text(encoding="utf-8")
    if "{% generation" not in source:
        raise TemplateError(
            f"{path} contains no {{% generation %}} block, so the tokenizer cannot emit an "
            f"assistant mask. Restore the template before training."
        )
    declared = template_format_version(source)
    if declared != FORMAT_VERSION:
        raise TemplateError(
            f"{path} declares format {declared!r} but this build expects "
            f"{FORMAT_VERSION!r}. Refusing to train on an unrecognised token format."
        )
    return source


def template_format_version(source: str) -> str:
    """Return the format version declared in a template's header marker."""
    match = re.search(_VERSION_PATTERN, source)
    if not match:
        raise TemplateError(
            "chat template has no format marker; it must start with "
            "'{#-\\n  format: <version>\\n'. Refusing an unversioned token format."
        )
    return match.group(1)


def require_installed_template(tokenizer: SupportsChatTemplate) -> str:
    """Return the tokenizer's template, or fail if it is absent or foreign.

    Falling back to the on-disk template here would hide the failure that
    matters most: a tokenizer saved without its chat template serves the base
    model's format, and the model appears to have lost its tool ability.
    """
    installed = tokenizer.chat_template
    if not installed:
        raise TemplateError(
            "tokenizer has no chat_template. Call install_template(tokenizer) before "
            "rendering, and ship the tokenizer via save_template_artifacts()."
        )
    declared = template_format_version(installed)
    if declared != FORMAT_VERSION:
        raise TemplateError(
            f"tokenizer chat template declares format {declared!r}, expected "
            f"{FORMAT_VERSION!r}. Training and serving would disagree about the token format."
        )
    return installed


def install_template(
    tokenizer: SupportsChatTemplate, source: str | None = None
) -> SupportsChatTemplate:
    """Attach the chat template to a tokenizer so it can be saved and served.

    This is how the format reaches inference: fine-tune with the template
    installed, then ship the tokenizer with the same template.
    """
    tokenizer.chat_template = source if source is not None else load_template_source()
    return tokenizer


def save_template_artifacts(tokenizer: SupportsChatTemplate, output_dir: str | Path) -> Path:
    """Persist the tokenizer with our chat template and verify the round trip.

    transformers >= 4.5x writes the template to a separate ``chat_template.jinja``
    rather than embedding it in ``tokenizer_config.json``. If that file is lost,
    a served model silently falls back to the base model's tool format, which
    looks like a capability regression rather than a packaging bug. So we verify
    the file exists and matches the source byte for byte.

    Returns the path of the written ``chat_template.jinja``.
    """
    source = load_template_source()
    install_template(tokenizer, source)
    directory = Path(output_dir)
    tokenizer.save_pretrained(str(directory))

    written = directory / CHAT_TEMPLATE_FILENAME
    if not written.is_file():
        raise TemplateError(
            f"{written} was not written. Without it the served model falls back to the base "
            f"chat format and tool calls break. Check the transformers version and that "
            f"output_dir is writable."
        )
    if written.read_text(encoding="utf-8") != source:
        raise TemplateError(
            f"{written} does not match the shipped template; serving and training would "
            f"disagree about the token format"
        )
    return written


@dataclass(frozen=True, slots=True)
class RenderedExample:
    """A tokenized, masked training example."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    assistant_mask: list[int]
    text: str

    @property
    def supervised_tokens(self) -> int:
        """Number of tokens contributing to the loss."""
        return sum(1 for label in self.labels if label != IGNORED_INDEX)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view of the example, without the decoded text."""
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "assistant_mask": self.assistant_mask,
        }


def render_text(
    tokenizer: SupportsChatTemplate,
    conversation: Conversation,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation to its exact training string."""
    rendered = tokenizer.apply_chat_template(
        conversation.to_messages(),
        tools=conversation.to_tools() or None,
        chat_template=require_installed_template(tokenizer),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise TemplateError(
            f"expected a rendered string from apply_chat_template, got {type(rendered).__name__}"
        )
    return rendered


def render_example(
    tokenizer: SupportsChatTemplate,
    conversation: Conversation,
    *,
    max_length: int | None = None,
) -> RenderedExample:
    """Tokenize a conversation and build completion-only labels.

    Labels come from the tokenizer's ``assistant_masks``: a token is a training
    target only when the template marked it as model-generated. The tool
    catalogue, user turns and ``<tool_result>`` blocks are therefore context by
    construction, not by a hand-maintained list that could drift.
    """
    encoded = tokenizer.apply_chat_template(
        conversation.to_messages(),
        tools=conversation.to_tools() or None,
        chat_template=require_installed_template(tokenizer),
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    try:
        input_ids: list[int] = list(encoded["input_ids"])
        attention_mask: list[int] = list(encoded["attention_mask"])
        assistant_mask: list[int] = list(encoded["assistant_masks"])
    except (KeyError, TypeError) as exc:
        raise TemplateError(
            "tokenizer did not return assistant_masks; the chat template must wrap "
            "assistant content in {% generation %} blocks"
        ) from exc

    labels = [
        token if mask else IGNORED_INDEX
        for token, mask in zip(input_ids, assistant_mask, strict=True)
    ]
    return RenderedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        assistant_mask=assistant_mask,
        text=tokenizer.decode(input_ids, skip_special_tokens=False),
    )


def assert_mask_sane(example: RenderedExample) -> None:
    """Fail loudly on the mask pathologies that silently ruin a run."""
    lengths = {
        len(example.input_ids),
        len(example.attention_mask),
        len(example.labels),
        len(example.assistant_mask),
    }
    if len(lengths) != 1:
        raise TemplateError(f"ragged example: field lengths differ ({sorted(lengths)})")
    if example.supervised_tokens == 0:
        raise TemplateError(
            f"example has no supervised tokens ({FORMAT_VERSION}); the assistant mask is empty"
        )
    for label, mask in zip(example.labels, example.assistant_mask, strict=True):
        if (label != IGNORED_INDEX) != bool(mask):
            raise TemplateError("labels and assistant_mask disagree")


def build_jsonl_record(example: RenderedExample, metadata: dict[str, Any] | None = None) -> str:
    """Serialise one example to a JSONL line, token ids only (portable, compact)."""
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        **example.to_dict(),
    }
    if metadata:
        payload["metadata"] = metadata
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
