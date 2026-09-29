"""The collator: padding, label masking, and the pixel/token agreement.

The failures guarded here do not crash. A mask filled with ones over padding, a
vision position left supervised, a pixel row attached to the wrong image -- each one
trains a model and each one is invisible until much later.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain.collator import (
    MAX_BATCH_TOKENS,
    Batch,
    Example,
    check_batch_budget,
    collate,
    declared_positions_from_tokens,
    example_from_rendered,
    examples_from_conversations,
    pad_examples,
    to_tensors,
)
from gotooltrain.errors import DatasetError
from gotooltrain.normalize import normalize_conversation
from gotooltrain.schema import ImageSpec
from gotooltrain.template import IGNORED_INDEX, RenderedExample, render_example

PAD_ID = 100
VISION_ID = 200
MERGE = 2


def spec(ref: str = "a.png", grid: tuple[int, int, int] = (1, 4, 4)) -> ImageSpec:
    return ImageSpec(ref=ref, grid_thw=grid, merge_size=MERGE)


class FakeProcessor:
    """Reports grids derived from the specs it is handed, like the real one."""

    def __init__(self, grids: list[tuple[int, int, int]] | None = None) -> None:
        """Adopt fixed grids, or report the spec's own."""
        self.grids = grids
        self.seen: list[int] = []

    def __call__(self, images=None, **kwargs: Any) -> dict[str, Any]:
        """Return canned pixel values and grids."""
        self.seen.append(len(images or []))
        rows = self.grids if self.grids is not None else [(1, 4, 4)] * len(images or [])
        return {
            "pixel_values": [[0.0]],
            "image_grid_thw": [list(row) for row in rows],
        }


def text_example(length: int = 6, supervised: int = 2) -> Example:
    labels = [IGNORED_INDEX] * (length - supervised) + [7] * supervised
    return Example(
        input_ids=list(range(length)),
        attention_mask=[1] * length,
        labels=labels,
        vision_pad_id=VISION_ID,
    )


def vision_example(positions: int = 4, supervised: int = 2) -> Example:
    ids = [1, *([VISION_ID] * positions), 2, 3]
    labels = [IGNORED_INDEX] * (len(ids) - supervised) + [9] * supervised
    return Example(
        input_ids=ids,
        attention_mask=[1] * len(ids),
        labels=labels,
        vision_pad_id=VISION_ID,
        images=("pixels",),
        image_specs=(spec(),),
    )


# ----------------------------------------------------------------- the example


def test_a_ragged_example_is_refused() -> None:
    with pytest.raises(DatasetError, match="ragged example"):
        Example(
            input_ids=[1, 2, 3],
            attention_mask=[1, 1],
            labels=[1, 2, 3],
            vision_pad_id=VISION_ID,
        )


def test_mismatched_image_counts_are_refused() -> None:
    """Pixels are paired by position, so the counts must agree before anything runs."""
    with pytest.raises(DatasetError, match="would be paired by position"):
        Example(
            input_ids=[1],
            attention_mask=[1],
            labels=[1],
            vision_pad_id=VISION_ID,
            images=("a", "b"),
            image_specs=(spec(),),
        )


def test_an_example_reports_its_accounting() -> None:
    example = vision_example()
    assert example.length == 7
    assert example.supervised_tokens == 2
    assert example.vision_tokens == 4


# -------------------------------------------------------------------- padding


def test_padding_goes_on_the_right_with_a_real_mask() -> None:
    padded = pad_examples([text_example(6), text_example(4)], pad_id=PAD_ID)
    assert padded["input_ids"][1] == [0, 1, 2, 3, PAD_ID, PAD_ID]
    assert padded["attention_mask"][1] == [1, 1, 1, 1, 0, 0]
    assert padded["labels"][1][-1] == IGNORED_INDEX


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(DatasetError, match="empty batch"):
        pad_examples([], pad_id=PAD_ID)


# ------------------------------------------------------------------- collate


def test_a_text_only_batch_needs_no_processor() -> None:
    batch = collate([text_example(), text_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID)
    assert batch.pixel_values is None
    assert batch.examples == 2
    assert batch.vision_tokens == 0


def test_the_padding_token_may_not_be_the_vision_token() -> None:
    """Otherwise every padded slot is read as image content."""
    with pytest.raises(DatasetError, match="equals the vision pad token"):
        collate([text_example()], pad_id=VISION_ID, vision_pad_id=VISION_ID)


def test_vision_positions_are_never_supervised() -> None:
    """The tokenizer's mask can leave a vision position inside an assistant span."""
    batch = collate(
        [vision_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID, processor=FakeProcessor()
    )
    labels = batch.labels[0]
    ids = batch.input_ids[0]
    for token, label in zip(ids, labels, strict=True):
        if token == VISION_ID:
            assert label == IGNORED_INDEX
    assert batch.supervised_tokens == 2, "the text targets survived"


def test_a_batch_with_images_but_no_processor_is_refused() -> None:
    """The encoder would be handed vision tokens and no pixels at all."""
    with pytest.raises(DatasetError, match="no image processor"):
        collate([vision_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID)


def test_a_processor_that_regrids_is_refused() -> None:
    """Different patch count, same call, no exception: the silent corruption case."""
    with pytest.raises(DatasetError, match="vision tokens but the processor produced"):
        collate(
            [vision_example()],
            pad_id=PAD_ID,
            vision_pad_id=VISION_ID,
            processor=FakeProcessor(grids=[(1, 2, 2)]),
        )


def test_a_processor_matching_the_specs_but_not_the_tokens_is_refused() -> None:
    """A batch whose declared specs and tokenised rows disagree on position count."""
    example = vision_example(positions=8, supervised=2)
    with pytest.raises(DatasetError, match="merged patches but the tokenised"):
        collate([example], pad_id=PAD_ID, vision_pad_id=VISION_ID, processor=FakeProcessor())


def test_the_processor_sees_every_image_in_order() -> None:
    processor = FakeProcessor()
    collate(
        [vision_example(), vision_example()],
        pad_id=PAD_ID,
        vision_pad_id=VISION_ID,
        processor=processor,
    )
    assert processor.seen == [2]
    assert (
        collate(
            [vision_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID, processor=processor
        ).vision_tokens
        == 4
    )


def test_batch_accounting_is_reported_for_the_log() -> None:
    batch = collate([text_example(6), text_example(4)], pad_id=PAD_ID, vision_pad_id=VISION_ID)
    record = batch.to_record()
    assert record["examples"] == 2
    assert record["tokens"] == 10
    assert record["padded_width"] == 6
    assert record["supervised_tokens"] == 4


# -------------------------------------------------------------------- budget


def test_an_oversized_batch_is_refused_with_advice() -> None:
    batch = Batch(input_ids=[], attention_mask=[], labels=[], lengths=(600_000, 600_000))
    with pytest.raises(DatasetError, match="sort by length"):
        check_batch_budget(batch, max_tokens=MAX_BATCH_TOKENS)


def test_a_normal_batch_passes_the_budget() -> None:
    batch = Batch(input_ids=[], attention_mask=[], labels=[], lengths=(8, 4), examples=2)
    check_batch_budget(batch, max_tokens=MAX_BATCH_TOKENS)


def test_an_empty_batch_costs_nothing() -> None:
    check_batch_budget(Batch(input_ids=[], attention_mask=[], labels=[]), max_tokens=1)


# ------------------------------------------------------------------ declared


def test_declared_positions_sum_the_specs() -> None:
    assert declared_positions_from_tokens([spec(), spec("b.png")]) == 8


# ------------------------------------------------------------------ tensors


def test_tensors_carry_padding_and_pixels() -> None:
    torch = pytest.importorskip("torch")
    batch = collate(
        [vision_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID, processor=FakeProcessor()
    )
    tensors = to_tensors(batch, pad_id=PAD_ID)
    assert tensors["input_ids"].shape == (1, 7)
    assert tensors["labels"].dtype is torch.long
    assert "pixel_values" in tensors
    assert tensors["attention_mask"].sum().item() == 7, "no position is masked out"


def test_tensors_refuse_to_guess_a_missing_torch() -> None:
    pytest.importorskip("torch")
    batch = collate([text_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID)
    assert "pixel_values" not in to_tensors(batch, pad_id=PAD_ID)


def test_tensors_can_be_moved_to_a_device() -> None:
    """Device placement belongs here so a trainer cannot forget it."""
    pytest.importorskip("torch")
    batch = collate(
        [vision_example()], pad_id=PAD_ID, vision_pad_id=VISION_ID, processor=FakeProcessor()
    )
    tensors = to_tensors(batch, pad_id=PAD_ID, device="cpu")
    assert all(str(tensor.device) == "cpu" for tensor in tensors.values())


# ------------------------------------------------- rendered -> collatable


def rendered(length: int = 5, supervised: int = 2) -> RenderedExample:
    """A tokenized row with ``supervised`` labels set and the rest ignored."""
    labels = [IGNORED_INDEX] * length
    for index in range(min(supervised, length)):
        labels[index] = 7
    return RenderedExample(
        input_ids=list(range(1, length + 1)),
        attention_mask=[1] * length,
        labels=labels,
        assistant_mask=[1 if label != IGNORED_INDEX else 0 for label in labels],
        text="x",
    )


def png(root: pathlib.Path, name: str = "shot.png") -> pathlib.Path:
    """A real image on disk, because the bridge opens what the record names."""
    from PIL import Image

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(path)
    return path


def image_record(ref: str = "shot.png", grid: list[int] | None = None) -> dict[str, Any]:
    return {
        "ref": ref,
        "grid_thw": grid if grid is not None else [1, 4, 4],
        "merge_size": MERGE,
    }


def conversation_with_images(*refs: str) -> Any:
    images = [image_record(ref) for ref in refs]
    messages = [
        {"role": "user", "content": "why is this failing?", "images": images},
        {"role": "assistant", "content": "The test asserts on a nil map."},
    ]
    return normalize_conversation(messages, [])


def test_the_bridge_carries_the_tokens_through() -> None:
    """It is an adapter, so it must not change what it adapts."""
    source = rendered(length=6, supervised=3)
    example = example_from_rendered(source, vision_pad_id=VISION_ID)

    assert example.input_ids == source.input_ids
    assert example.attention_mask == source.attention_mask
    assert example.labels == source.labels
    assert example.vision_pad_id == VISION_ID
    assert example.length == 6
    assert example.supervised_tokens == 3


def test_a_text_only_record_yields_no_images() -> None:
    """Nothing declared means nothing loaded, and no processor is needed."""
    example = example_from_rendered(rendered(), vision_pad_id=VISION_ID)
    assert example.images == ()
    assert example.image_specs == ()


def test_the_bridge_loads_the_images_the_record_declares(tmp_path: pathlib.Path) -> None:
    """Without this the vision positions would reach the tower with no pixels."""
    png(tmp_path)
    conversation = conversation_with_images("shot.png")
    example = example_from_rendered(
        rendered(), vision_pad_id=VISION_ID, conversation=conversation, image_root=tmp_path
    )

    assert len(example.images) == 1
    assert example.images[0].size == (8, 8)
    assert len(example.image_specs) == 1
    assert example.image_specs[0].ref == "shot.png"


def test_images_and_specs_are_collected_in_the_order_they_appear(
    tmp_path: pathlib.Path,
) -> None:
    """Pixels are paired with specs by position, so the order is load-bearing."""
    png(tmp_path, "a.png")
    png(tmp_path, "b.png")
    conversation = conversation_with_images("a.png", "b.png")
    example = example_from_rendered(
        rendered(), vision_pad_id=VISION_ID, conversation=conversation, image_root=tmp_path
    )

    assert [spec.ref for spec in example.image_specs] == ["a.png", "b.png"]


def test_a_load_failure_names_the_image(tmp_path: pathlib.Path) -> None:
    """A missing file must not silently become an example with no pixels."""
    conversation = conversation_with_images("absent.png")
    with pytest.raises(Exception, match="not found"):
        example_from_rendered(
            rendered(), vision_pad_id=VISION_ID, conversation=conversation, image_root=tmp_path
        )


def test_a_bridged_vision_record_reaches_a_batch(
    qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """The end of the joint: a rendered record with an image collates into a batch.

    This is the path that did not exist: render_example produced tokens, collate
    wanted pixels, and the images a conversation declared reached neither.
    """
    png(tmp_path)
    conversation = conversation_with_images("shot.png")  # grid (1, 4, 4)
    text = render_example(qwen_tokenizer, conversation)
    pad_id = qwen_tokenizer.pad_token_id
    vision_pad_id = qwen_tokenizer.convert_tokens_to_ids("<|vision_pad|>")

    example = example_from_rendered(
        text,
        vision_pad_id=vision_pad_id,
        conversation=conversation,
        image_root=tmp_path,
    )
    assert example.vision_tokens == declared_positions_from_tokens(example.image_specs)
    assert example.vision_tokens > 0

    batch = collate(
        [example],
        pad_id=pad_id,
        vision_pad_id=vision_pad_id,
        processor=FakeProcessor(grids=[(1, 4, 4)]),
    )
    assert batch.vision_tokens == example.vision_tokens
    assert batch.pixel_values is not None


def test_examples_are_rendered_in_order(qwen_tokenizer: Any) -> None:
    """A corpus is tokenized in order so a shuffled batch stays traceable."""
    conversations = [
        normalize_conversation(
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}], []
        ),
        normalize_conversation(
            [{"role": "user", "content": "longer prompt"}, {"role": "assistant", "content": "c"}],
            [],
        ),
    ]
    examples = list(
        examples_from_conversations(
            conversations,
            qwen_tokenizer,
            vision_pad_id=qwen_tokenizer.convert_tokens_to_ids("<|vision_pad|>"),
        )
    )
    assert len(examples) == 2
    assert examples[0].supervised_tokens > 0
    assert examples[0].length < examples[1].length
