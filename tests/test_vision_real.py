"""End-to-end multimodal check against the real Qwen3.5-4B tokenizer and processor.

Everything else in the suite uses fakes, which proves the logic but not the
contract. These tests bind to the actual checkpoint: the template that ships, the
tokenizer that will serve it, and the image processor that produces pixels. The
thing being verified is the agreement between three independently-written pieces
of code -- if the vision token count and the patch count ever drift apart, this is
where it shows.

Tokenizer and image-processor files are downloaded (tens of MB); weights are not.
"""

from __future__ import annotations

import pytest

from gotooltrain import (
    FORMAT_VERSION,
    ImageSpec,
    install_template,
    load_image,
    load_template_source,
    mask_vision_labels,
    merged_patches,
    normalize_conversation,
    process_images,
    render_example,
    template_path,
)
from gotooltrain.template import IGNORED_INDEX
from gotooltrain.vision import check_batch_vision_layout, count_vision_tokens

BASE_MODEL = "Qwen/Qwen3.5-4B"
VISION_PAD_TOKEN = "<|vision_pad|>"


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    """The real tokenizer with our template installed."""
    transformers = pytest.importorskip("transformers")
    loaded = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    return install_template(loaded, load_template_source())


@pytest.fixture(scope="module")
def processor():  # type: ignore[no-untyped-def]
    """The real multimodal processor for the same checkpoint.

    ``AutoProcessor`` resolves video-processor classes that import torch at module
    level, so without torch even loading the *image* half fails. That is reported
    as a skip with the install command rather than as a collection error, which
    would read like a broken package.
    """
    pytest.importorskip(
        "torch",
        reason="torch is required to load AutoProcessor; install with: pip install -e '.[train]'",
    )
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(BASE_MODEL)


@pytest.fixture
def image_file(tmp_path):  # type: ignore[no-untyped-def]
    """A real PNG on disk, sized so the vision grid is small."""
    from PIL import Image

    path = tmp_path / "fail.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(path)
    return path


def test_the_template_still_declares_the_format_it_is_given(tokenizer) -> None:  # type: ignore[no-untyped-def]
    assert tokenizer.chat_template is not None
    assert FORMAT_VERSION in tokenizer.chat_template
    assert template_path().is_file()


def test_the_shipped_template_renders(tokenizer) -> None:  # type: ignore[no-untyped-def]
    conversation = normalize_conversation(
        [
            {"role": "user", "content": "fix the panic"},
            {"role": "assistant", "content": "Reading the file first."},
        ],
        [],
    )
    example = render_example(tokenizer, conversation)
    assert example.supervised_tokens > 0
    assert all(
        (label != IGNORED_INDEX) == bool(mask)
        for label, mask in zip(example.labels, example.assistant_mask)
    )


def test_the_real_processor_agrees_with_the_declared_grid(processor, image_file, tokenizer) -> None:  # type: ignore[no-untyped-def]
    """The contract that cannot be faked: pixels in, token count agreed.

    The grid is measured from the processor, declared in the data, and the template
    is then asked to emit that many vision tokens. If any of the three disagreed,
    the model would be trained with vision positions the encoder cannot fill.
    """
    image = load_image("fail.png", image_file.parent)
    encoded = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(v) for v in encoded["image_grid_thw"][0]]

    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    spec = ImageSpec(ref="fail.png", grid_thw=(grid[0], grid[1], grid[2]), merge_size=merge_size)

    # The declared layout reproduces the processor's own patch count exactly.
    assert merged_patches(spec) == (grid[0] * grid[1] * grid[2]) // (merge_size * merge_size)

    # And the batch passes its own consistency check rather than being waved through.
    batch = process_images(processor, [image], [spec])
    assert batch.total_patches == merged_patches(spec)


def test_vision_pad_token_exists_in_the_real_vocabulary(tokenizer) -> None:  # type: ignore[no-untyped-def]
    """Masking and layout checks address the pad token by id; it must resolve."""
    token_id = tokenizer.convert_tokens_to_ids(VISION_PAD_TOKEN)
    assert token_id is not None and token_id >= 0
    assert tokenizer.convert_ids_to_tokens(token_id) == VISION_PAD_TOKEN


def test_vision_positions_are_masked_out_of_the_labels(tokenizer, processor, image_file) -> None:  # type: ignore[no-untyped-def]
    """A supervised vision position would train the model to emit image patches."""
    image = load_image("fail.png", image_file.parent)
    encoded = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(v) for v in encoded["image_grid_thw"][0]]
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    spec = ImageSpec(ref="fail.png", grid_thw=(grid[0], grid[1], grid[2]), merge_size=merge_size)

    conversation = normalize_conversation(
        [
            {
                "role": "user",
                "content": "the test fails on this screenshot",
                "images": [
                    {"ref": "fail.png", "grid_thw": list(spec.grid_thw), "merge_size": merge_size}
                ],
            },
            {"role": "assistant", "content": "The index is out of range at line 12."},
        ],
        [],
    )
    example = render_example(tokenizer, conversation)

    vision_pad_id = tokenizer.convert_tokens_to_ids(VISION_PAD_TOKEN)
    positions = count_vision_tokens(example.input_ids, vision_pad_id)
    assert positions == merged_patches(spec), (
        f"template emitted {positions} vision tokens, processor produced {merged_patches(spec)}"
    )

    labels = list(example.labels)
    changed = mask_vision_labels(labels, example.input_ids, vision_pad_id)
    assert changed == 0, "the template already excludes vision positions from the loss"

    # Sanity: real text is still supervised.
    assert example.supervised_tokens > 0


def test_the_batch_layout_check_accepts_a_correct_batch(processor, image_file, tokenizer) -> None:  # type: ignore[no-untyped-def]
    image = load_image("fail.png", image_file.parent)
    encoded = processor.image_processor(images=[image], return_tensors="pt")
    grid = [int(v) for v in encoded["image_grid_thw"][0]]
    merge_size = int(getattr(processor.image_processor, "merge_size", 2))
    spec = ImageSpec(ref="fail.png", grid_thw=(grid[0], grid[1], grid[2]), merge_size=merge_size)
    batch = process_images(processor, [image], [spec])

    conversation = normalize_conversation(
        [
            {
                "role": "user",
                "content": "look at this",
                "images": [
                    {"ref": "fail.png", "grid_thw": list(spec.grid_thw), "merge_size": merge_size}
                ],
            },
            {"role": "assistant", "content": "Understood."},
        ],
        [],
    )
    example = render_example(tokenizer, conversation)
    vision_pad_id = tokenizer.convert_tokens_to_ids(VISION_PAD_TOKEN)
    pad_id = tokenizer.pad_token_id

    total = check_batch_vision_layout(
        [example.input_ids], vision_pad_id, pad_id=pad_id, expected=batch.total_patches
    )
    assert total == batch.total_patches


def test_a_text_only_conversation_never_reaches_the_processor(tokenizer, processor) -> None:  # type: ignore[no-untyped-def]
    """Most examples have no images; the vision path must stay genuinely optional."""
    conversation = normalize_conversation(
        [
            {"role": "user", "content": "why does this test fail?"},
            {"role": "assistant", "content": "Because the fixture is nil."},
        ],
        [],
    )
    example = render_example(tokenizer, conversation)
    vision_pad_id = tokenizer.convert_tokens_to_ids(VISION_PAD_TOKEN)
    assert count_vision_tokens(example.input_ids, vision_pad_id) == 0
    assert process_images(processor, [], []).total_patches == 0
