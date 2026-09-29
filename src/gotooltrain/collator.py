"""Batching for training: tokens, labels and pixels as one consistent unit.

Three components, each of which has a way to be subtly wrong that never crashes.

**Padding.** Right padding with a real attention mask. A mask filled with ones over
the padding is a well-known way to make a run look like it is learning and quietly
destroy the batch instead.

**Label masking.** Only assistant tokens are targets. Two separate mechanisms apply:
the tokenizer's own ``{% generation %}`` mask, and the vision rule that a
``<|vision_pad|>`` position is never a target even when it falls inside an assistant
span. The second is not redundant -- without it a multi-image assistant turn trains
the model to emit image patches.

**Pixel-token agreement.** The vision encoder consumes one row per vision position,
paired by index. If the batch has more vision tokens than pixel rows, every image
after the first is attached to the wrong place: no error, lower loss, wrong model.
That count is checked rather than assumed.

The design keeps torch out of the module. A collator that cannot be exercised
without a GPU cannot be tested on a laptop, and this logic is exactly the kind that
needs testing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import DatasetError
from .schema import Conversation, ImageSpec
from .template import IGNORED_INDEX, RenderedExample, SupportsChatTemplate, render_example
from .vision import (
    VisionBatch,
    check_batch_vision_layout,
    count_vision_tokens,
    load_image,
    mask_vision_labels,
    process_images,
)

#: A batch this large cannot fit in memory. Refusing it up front turns an opaque
#: allocator failure into a sentence the operator can act on.
MAX_BATCH_TOKENS: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class Example:
    """One training example, tokenised and ready to be collated.

    ``images`` and ``image_specs`` are parallel: together they describe the pixel
    rows the vision encoder must receive, in the order the vision tokens appear.
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    vision_pad_id: int
    images: tuple[Any, ...] = ()
    image_specs: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a ragged example; padding cannot fix a ragged one."""
        lengths = {len(self.input_ids), len(self.attention_mask), len(self.labels)}
        if len(lengths) != 1:
            raise DatasetError(
                f"ragged example: {len(self.input_ids)} ids, {len(self.attention_mask)} mask, "
                f"{len(self.labels)} labels"
            )
        if len(self.images) != len(self.image_specs):
            raise DatasetError(
                f"{len(self.images)} images for {len(self.image_specs)} specs; pixels and "
                "declared layout would be paired by position, so the counts must match"
            )

    @property
    def length(self) -> int:
        """Token count before padding."""
        return len(self.input_ids)

    @property
    def supervised_tokens(self) -> int:
        """How many positions contribute to the loss."""
        return sum(1 for label in self.labels if label != IGNORED_INDEX)

    @property
    def vision_tokens(self) -> int:
        """How many vision positions this example declares."""
        return count_vision_tokens(self.input_ids, self.vision_pad_id)


@dataclass(frozen=True, slots=True)
class Batch:
    """A collated batch, plus the accounting the trainer needs to log it.

    Fields are backend-agnostic on purpose: whether these end up as ``torch.Tensor``
    or NumPy arrays is the trainer's decision, and a collator that pretended to know
    would be wrong in one of the two cases.
    """

    input_ids: Any
    attention_mask: Any
    labels: Any
    pixel_values: Any = None
    image_grid_thw: Any = None
    lengths: tuple[int, ...] = ()
    supervised_tokens: int = 0
    vision_tokens: int = 0
    examples: int = 0

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary, for a training log."""
        return {
            "examples": self.examples,
            "tokens": int(sum(self.lengths)),
            "supervised_tokens": self.supervised_tokens,
            "vision_tokens": self.vision_tokens,
            "padded_width": max(self.lengths) if self.lengths else 0,
        }


def pad_examples(examples: Sequence[Example], *, pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad token rows and build the mask, in input order.

    Order is preserved deliberately: the rows a trainer shuffles must stay
    traceable back to the examples that produced them, or a loss spike cannot be
    attributed to a specific record.
    """
    if not examples:
        raise DatasetError("cannot collate an empty batch")
    width = max(e.length for e in examples)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        padding = width - example.length
        input_ids.append([*example.input_ids, *([pad_id] * padding)])
        attention.append([*example.attention_mask, *([0] * padding)])
        labels.append([*example.labels, *([IGNORED_INDEX] * padding)])
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def collate(
    examples: Sequence[Example],
    *,
    pad_id: int,
    vision_pad_id: int,
    processor: Any = None,  # noqa: ANN401 - the image processor's own surface
) -> Batch:
    """Collate examples into a batch, enforcing the pixel/token agreement.

    Pixels are produced *after* padding so the vision position count is measured on
    the batch the model will actually see, not on the unpadded rows. Vision labels
    are masked here rather than at tokenisation time, so a batch that is assembled
    by some other route still gets the rule applied.
    """
    if vision_pad_id == pad_id:
        raise DatasetError(
            f"pad token id ({pad_id}) equals the vision pad token; padded positions would be "
            "read as image content"
        )
    padded = pad_examples(examples, pad_id=pad_id)
    expected_positions = sum(count_vision_tokens(row, vision_pad_id) for row in padded["input_ids"])

    declared_specs = [spec for e in examples for spec in e.image_specs]
    declared_images = [image for e in examples for image in e.images]

    pixel_values: Any = None
    image_grid_thw: Any = None
    if declared_images:
        if processor is None:
            raise DatasetError(
                f"{len(declared_images)} image(s) declared but no image processor was supplied; "
                "the vision tower would receive no pixels for the tokens it is given"
            )
        vision: VisionBatch = process_images(processor, declared_images, declared_specs)
        # The processor's output is checked against the *tokens*, not against the
        # declared specs: the specs are what produced the tokens, so comparing the
        # two spec-derived numbers would always agree and prove nothing.
        if vision.total_patches != expected_positions:
            raise DatasetError(
                f"the processor produced {vision.total_patches} merged patches but the tokenised "
                f"rows contain {expected_positions} vision positions. Pixels would be attached "
                "to the wrong tokens."
            )
        pixel_values = vision.pixel_values
        image_grid_thw = vision.image_grid_thw

    # The final guard, on the batch itself.
    check_batch_vision_layout(
        padded["input_ids"],
        vision_pad_id,
        pad_id=pad_id,
        expected=expected_positions,
    )

    masked_labels = [list(row) for row in padded["labels"]]
    for row, ids in zip(masked_labels, padded["input_ids"], strict=True):
        mask_vision_labels(row, ids, vision_pad_id)

    return Batch(
        input_ids=padded["input_ids"],
        attention_mask=padded["attention_mask"],
        labels=masked_labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        lengths=tuple(e.length for e in examples),
        supervised_tokens=sum(
            1 for row in masked_labels for label in row if label != IGNORED_INDEX
        ),
        vision_tokens=expected_positions,
        examples=len(examples),
    )


def declared_positions_from_tokens(specs: Sequence[Any]) -> int:
    """Total merged patches the declared specs describe."""
    from .vision import merged_patches

    return sum(merged_patches(spec) for spec in specs)


def example_from_rendered(
    rendered: RenderedExample,
    *,
    vision_pad_id: int,
    conversation: Conversation | None = None,
    image_root: str | Path | None = None,
) -> Example:
    """Turn a tokenized record into a collatable example, loading its images.

    This is the joint the pipeline was missing: :func:`template.render_example`
    produces token ids and a mask, :func:`collate` wants pixels and the declared
    layout alongside them, and nothing connected the two. Without it the images a
    conversation declares never reach the collator, and a multimodal batch would be
    assembled from tokens alone -- the vision tower would receive nothing for the
    vision positions the template emitted.

    Images are loaded here rather than at render time because a tokenized record is
    cacheable and a decoded image is not: a corpus can be tokenized once and the
    pixels read per epoch.
    """
    specs: list[ImageSpec] = []
    images: list[Any] = []
    if conversation is not None:
        for message in conversation.messages:
            for spec in message.images:
                specs.append(spec)
                images.append(load_image(spec.ref, image_root))
    return Example(
        input_ids=list(rendered.input_ids),
        attention_mask=list(rendered.attention_mask),
        labels=list(rendered.labels),
        vision_pad_id=vision_pad_id,
        images=tuple(images),
        image_specs=tuple(specs),
    )


def examples_from_conversations(
    conversations: Iterable[Conversation],
    tokenizer: SupportsChatTemplate,
    *,
    vision_pad_id: int,
    image_root: str | Path | None = None,
) -> Iterator[Example]:
    """Render and adapt a corpus, in order.

    A generator so a large corpus is not tokenized into a list before training
    starts, which would hold every token id in memory at once.
    """
    for conversation in conversations:
        rendered = render_example(tokenizer, conversation)
        yield example_from_rendered(
            rendered,
            vision_pad_id=vision_pad_id,
            conversation=conversation,
            image_root=image_root,
        )


def check_batch_budget(batch: Batch, *, max_tokens: int = MAX_BATCH_TOKENS) -> None:
    """Refuse a batch whose padded token count exceeds the budget.

    Padded width times batch size is what the model actually pays for, and it grows
    with the longest member rather than the average: one 32K example in a batch of
    short ones costs 32K slots each time.
    """
    total = max(batch.lengths) * len(batch.lengths) if batch.lengths else 0
    if total > max_tokens:
        raise DatasetError(
            f"batch of {len(batch.lengths)} padded to {max(batch.lengths)} tokens is {total} "
            f"tokens, over the {max_tokens} budget. Split it, or sort by length so short records "
            "batch together."
        )


def to_tensors(
    batch: Batch,
    *,
    pad_id: int,
    device: str | None = None,
) -> Mapping[str, Any]:
    """Convert a batch to torch tensors if torch is available.

    Kept separate from :func:`collate` so the collation logic stays testable without
    a GPU, and so a machine without torch gets a clear message rather than an
    ImportError from deep inside a training step.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DatasetError(
            "torch is required to build training tensors. Install with: pip install -e '.[train]'"
        ) from exc

    tensors: dict[str, Any] = {
        "input_ids": torch.tensor(batch.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch.attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch.labels, dtype=torch.long),
    }
    if batch.pixel_values is not None:
        tensors["pixel_values"] = torch.as_tensor(batch.pixel_values)
        tensors["image_grid_thw"] = torch.as_tensor(batch.image_grid_thw)
    if device is not None:
        tensors = {name: tensor.to(device) for name, tensor in tensors.items()}
    # `pad_id` is part of the signature so a caller wiring this into a trainer has
    # to state its padding choice where the tensors are built, not only where the
    # batch was collated.
    _ = pad_id
    return tensors
