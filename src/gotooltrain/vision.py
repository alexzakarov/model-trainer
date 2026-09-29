"""Vision pipeline: pixels in, batched tensors out, with the token layout checked.

The token format is deliberately self-describing: an :class:`~gotooltrain.schema.ImageSpec`
carries ``grid_thw`` and ``merge_size``, so the number of ``<|vision_pad|>`` positions is
known before the image processor ever runs. That is what makes this module possible.

The invariant it buys is the one that otherwise fails silently. The vision encoder consumes
one row of pixel values per ``<|vision_pad|>`` position, pairing them by index. If the
processor emits a different patch count than the template emitted tokens for, every
subsequent image is misaligned: training "works", loss goes down, and the model learns to
attend to the wrong pixels. Nothing crashes. So the count is compared and a mismatch is a
hard error, here, before a batch is built.

Labels never supervise a vision position. The vision tokens are input, not output: the
model is taught to *use* the image, not to reproduce it, and supervising them would spend
capacity on a task the model can already do.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from .errors import DatasetError, ValidationError
from .schema import ImageSpec

#: The token the Qwen-family templates use for one merged vision patch. Its id is
#: resolved from the tokenizer, not hard-coded, so a checkpoint that renumbers its
#: special tokens still masks correctly.
VISION_PAD_TOKEN: Final[str] = "<|vision_pad|>"  # noqa: S105 - a token name, not a secret

#: Bounding the decode side. A batch of a few dozen 1080p screenshots is tens of GB of
#: float32; the failure should be an explicit refusal, not an OOM kill mid-epoch.
MAX_PIXEL_VALUES_PER_BATCH: Final[int] = 512_000_000


class SupportsImageProcessing(Protocol):
    """The slice of ``AutoProcessor`` this module depends on.

    ``image_processor`` is what actually gets called: the multimodal wrapper also
    tokenises text, and text is rendered through the tokenizer's chat template
    instead. A plain callable is accepted too, via ``getattr`` fallback in
    :func:`process_images`.
    """

    image_processor: Any

    def __call__(self, images: Sequence[Any] | None = None, **kwargs: Any) -> Any:  # noqa: ANN401 - mirrors AutoProcessor
        """Run the processor."""


@dataclass(frozen=True, slots=True)
class VisionBatch:
    """Processor output for one batch's images, plus the layout it must match."""

    #: One row per ``<|vision_pad|>`` position across the whole batch.
    pixel_values: Any
    #: Per-image patch grid, in the same order the images were passed.
    image_grid_thw: Any
    #: Merged patch count per image; the unit ``pixel_values`` is checked in.
    patches_per_image: tuple[int, ...]
    #: Total merged patches, which must equal the number of vision token
    #: positions the template emitted. Asserted by
    #: :func:`check_batch_vision_layout`.
    total_patches: int


def merged_patches(spec: ImageSpec) -> int:
    """Merged patch count for one image.

    Duplicated from :attr:`ImageSpec.vision_tokens` on purpose: the *declared*
    layout and the *processor's* output are two independent sources, and the whole
    point of this module is to compare them. Calling the same function on both
    sides would make the check vacuously true.
    """
    t, h, w = spec.grid_thw
    if t < 1 or h < 1 or w < 1:
        raise DatasetError(f"image {spec.ref!r} has a degenerate grid {spec.grid_thw}")
    if h % spec.merge_size or w % spec.merge_size:
        raise DatasetError(
            f"image {spec.ref!r} grid {h}x{w} is not divisible by merge_size "
            f"{spec.merge_size}; the declared layout cannot be produced"
        )
    return (t * h * w) // (spec.merge_size * spec.merge_size)


def load_image(ref: str, root: str | Path | None = None) -> Any:  # noqa: ANN401 - PIL.Image
    """Open an image from ``ref``, refusing anything outside ``root``.

    ``ref`` comes from a data file, so it is untrusted. Resolution happens *after*
    the containment check, which means ``ref`` must be relative; a data file that
    names an absolute path is rejected rather than quietly reading whatever the
    training host happens to have.
    """
    from PIL import Image

    base = Path(root) if root is not None else Path()
    candidate = Path(ref)
    if candidate.is_absolute():
        raise ValidationError(f"image ref must be relative, got {ref!r}")
    resolved_root = base.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved_root not in resolved.parents:
        raise ValidationError(f"image ref escapes the image root: {ref!r}")
    if not resolved.is_file():
        raise ValidationError(f"image not found: {resolved}")
    with Image.open(resolved) as handle:
        return handle.convert("RGB")


def process_images(
    processor: SupportsImageProcessing,
    images: Sequence[Any],
    specs: Sequence[ImageSpec],
) -> VisionBatch:
    """Run the image processor and check it against the declared layout.

    The *image* half of the processor is called, not the multimodal wrapper: the
    wrapper also tokenises text, and text is rendered separately through the
    tokenizer's chat template. Going through the wrapper would tokenise the same
    content twice, under two different rules.

    Sizes and grids come from the data, because the template already committed to
    those token counts. If the processor would resize or re-grid the image
    differently, the two token layouts disagree and the batch is refused rather
    than mis-paired.
    """
    if len(images) != len(specs):
        raise DatasetError(f"{len(images)} images for {len(specs)} specs")
    if not images:
        return VisionBatch(
            pixel_values=None, image_grid_thw=None, patches_per_image=(), total_patches=0
        )

    runner = getattr(processor, "image_processor", processor)
    try:
        encoded = runner(images=list(images), return_tensors="pt")
    except Exception as exc:
        raise DatasetError(f"image processor failed: {exc}") from exc

    grid = _as_rows(encoded.get("image_grid_thw"), "image_grid_thw")
    declared = [merged_patches(spec) for spec in specs]
    if len(grid) != len(declared):
        raise DatasetError(
            f"processor returned {len(grid)} grids for {len(declared)} images; the batch "
            "would mis-pair pixels with vision tokens"
        )
    for index, (row, expected) in enumerate(zip(grid, declared, strict=True)):
        t, h, w = (int(v) for v in row)
        actual = (t * h * w) // (specs[index].merge_size ** 2)
        if actual != expected:
            raise DatasetError(
                f"image {specs[index].ref!r}: template emitted {expected} vision tokens but the "
                f"processor produced {actual} (grid {t}x{h}x{w}, merge {specs[index].merge_size}). "
                "Refusing to build a batch where pixels and tokens disagree."
            )
    return VisionBatch(
        pixel_values=encoded.get("pixel_values"),
        image_grid_thw=grid,
        patches_per_image=tuple(declared),
        total_patches=sum(declared),
    )


def _as_rows(value: Any, field: str) -> list[tuple[int, ...]]:  # noqa: ANN401 - tensor-ish
    """Normalise a tensor-ish grid into a list of integer triples."""
    if value is None:
        raise DatasetError(f"image processor returned no {field}")
    rows = [tuple(int(v) for v in row) for row in value]
    if any(len(row) != 3 for row in rows):
        raise DatasetError(f"{field} must have 3 columns per image, got {rows}")
    return rows


def count_vision_tokens(input_ids: Sequence[int], vision_pad_id: int) -> int:
    """Number of ``<|vision_pad|>`` positions in a token sequence."""
    return sum(1 for token in input_ids if token == vision_pad_id)


def mask_vision_labels(labels: list[int], input_ids: Sequence[int], vision_pad_id: int) -> int:
    """Force every vision position to ``IGNORE_INDEX``; return how many changed.

    A vision position inside an assistant span would otherwise be a training
    target. Supervising it teaches the model to emit image patches, which is not
    the task, and it dilutes the loss over the tokens that matter.
    """
    from .template import IGNORED_INDEX

    changed = 0
    for index, token in enumerate(input_ids):
        if token == vision_pad_id and labels[index] != IGNORED_INDEX:
            labels[index] = IGNORED_INDEX
            changed += 1
    return changed


def check_batch_vision_layout(
    batch_input_ids: Sequence[Sequence[int]],
    vision_pad_id: int,
    *,
    pad_id: int,
    expected: int,
) -> int:
    """Verify vision token positions across a padded batch; return the total.

    Two ways this goes wrong, both silent:

    * the padding id *is* the vision pad id, so every padded slot looks like a
      vision position and the encoder is handed features for empty space;
    * a row carries vision tokens the processor never produced (or the reverse),
      so the total no longer matches the pixel budget.

    So: padding must be distinguishable, and the count must equal the number of
    merged patches the processor reported.
    """
    if pad_id == vision_pad_id:
        raise DatasetError(
            f"the pad token id ({pad_id}) is the vision pad token; padded positions would be "
            "read as image content. The model's pad token is not a vision token."
        )
    total = sum(count_vision_tokens(row, vision_pad_id) for row in batch_input_ids)
    if total != expected:
        raise DatasetError(
            f"batch has {total} vision token positions but the processor produced {expected} "
            "merged patches. Pixels and tokens would be paired in the wrong order."
        )
    return total


@dataclass(frozen=True, slots=True)
class CollatedBatch:
    """A padded, ready-to-train batch.

    The tensor fields are typed loosely on purpose: whether they are ``torch.Tensor``
    or NumPy arrays depends on the backend the trainer configures, and pinning one
    here would make the collator lie about what it returns. The trainer owns device
    placement.
    """

    input_ids: Any
    attention_mask: Any
    labels: Any
    pixel_values: Any = None
    image_grid_thw: Any = None


def pad_batch(
    rows: Sequence[Mapping[str, Sequence[int]]],
    pad_id: int,
    *,
    label_pad: int,
) -> dict[str, list[list[int]]]:
    """Right-pad token rows and build the attention mask.

    Padding goes on the right so the assistant span a truncation produced stays
    contiguous, and the mask is built here rather than filled with ones, because
    a full-ones mask over padding is a silent, well-known way to corrupt a run.
    """
    if not rows:
        raise DatasetError("cannot collate an empty batch")
    width = max(len(row["input_ids"]) for row in rows)
    input_ids: list[list[int]] = []
    attention: list[list[int]] = []
    labels: list[list[int]] = []
    for row in rows:
        ids = list(row["input_ids"])
        mask = list(row.get("attention_mask") or [1] * len(ids))
        row_labels = list(row.get("labels") or ids)
        if not len(ids) == len(mask) == len(row_labels):
            raise DatasetError(
                f"ragged row: {len(ids)} ids, {len(mask)} mask, {len(row_labels)} labels"
            )
        padding = width - len(ids)
        input_ids.append(ids + [pad_id] * padding)
        attention.append(mask + [0] * padding)
        labels.append(row_labels + [label_pad] * padding)
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def budget_pixels(patches: Iterable[int], patch_values: int) -> int:
    """Total float32 elements a batch of merged patches would occupy.

    Used to refuse an oversized batch up front. A merged patch is
    ``merge_size**2`` raw patches, and each raw patch carries ``patch_values``
    channels, so the product is the encoder's actual input width.
    """
    return sum(patches) * patch_values
