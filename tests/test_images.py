"""Image attachments: declared layout, and the validation that keeps it honest.

The token format emits ``<|vision_pad|>`` positions from the *declared* grid, before
any image is opened. So a bad ``grid_thw`` is not a rendering nuisance: it is a
lie the template will faithfully encode, and the vision encoder will later be fed
a different number of patches. These tests pin that contract at both ends -- the
schema arithmetic and the normalisation policy.
"""

from __future__ import annotations

from typing import Any

import pytest

from gotooltrain import ImageSpec, ValidationError, normalize_conversation
from gotooltrain.normalize import MAX_IMAGES_PER_TURN


def image(
    grid: list[int] | None = None, merge: int = 2, ref: str = "shots/fail.png"
) -> dict[str, Any]:
    return {
        "ref": ref,
        "grid_thw": grid if grid is not None else [1, 28, 28],
        "merge_size": merge,
    }


def conversation_with(*images: dict[str, Any], role: str = "user") -> list[dict[str, Any]]:
    """A user turn carrying the images, plus an assistant turn to make it trainable."""
    return [
        {"role": role, "content": "why is this failing?", "images": list(images)},
        {"role": "assistant", "content": "The test asserts on a nil map."},
    ]


# ---------------------------------------------------------------- schema maths


def test_vision_tokens_are_the_merged_patch_count() -> None:
    assert ImageSpec(ref="a", grid_thw=(1, 28, 28), merge_size=2).vision_tokens == 196


def test_vision_sequence_is_one_pad_token_per_vision_token() -> None:
    spec = ImageSpec(ref="a", grid_thw=(1, 4, 4), merge_size=2)
    assert spec.vision_sequence == "<|vision_pad|>" * 4


def test_to_template_carries_the_count_the_jinja_template_reads() -> None:
    payload = ImageSpec(ref="shots/x.png", grid_thw=(2, 4, 4), merge_size=2).to_template()
    assert payload["ref"] == "shots/x.png"
    assert payload["grid_thw"] == [2, 4, 4]
    assert payload["merge_size"] == 2
    assert payload["vision_tokens"] == 8
    assert payload["vision_sequence"] == "<|vision_pad|>" * 8


def test_a_temporal_grid_counts_every_frame() -> None:
    assert ImageSpec(ref="a", grid_thw=(4, 2, 2), merge_size=2).vision_tokens == 4


# ------------------------------------------------------------- normalisation


def test_a_valid_image_survives_normalisation() -> None:
    conversation = normalize_conversation(conversation_with(image()), [])
    assert len(conversation.messages[0].images) == 1
    assert conversation.messages[0].images[0].vision_tokens == 196


def test_a_turn_may_carry_several_images() -> None:
    conversation = normalize_conversation(conversation_with(image(), image(ref="b.png")), [])
    assert len(conversation.messages[0].images) == 2


def test_too_many_images_are_refused() -> None:
    """Four is the ceiling; past it a turn can silently dominate a 32K context."""
    too_many = conversation_with(*[image(ref=f"{i}.png") for i in range(MAX_IMAGES_PER_TURN + 1)])
    with pytest.raises(ValidationError, match="above the limit"):
        normalize_conversation(too_many, [])


def test_images_as_a_mapping_are_refused() -> None:
    with pytest.raises(ValidationError, match="must be a list"):
        normalize_conversation(
            [
                {"role": "user", "content": "x", "images": {"ref": "a.png"}},
                {"role": "assistant", "content": "y"},
            ],
            [],
        )


def test_images_as_a_bare_string_are_refused() -> None:
    """A str is a Sequence, so the list check has to exclude it explicitly."""
    with pytest.raises(ValidationError, match="images must be a list"):
        normalize_conversation(
            [
                {"role": "user", "content": "x", "images": "one.png"},
                {"role": "assistant", "content": "y"},
            ],
            [],
        )


def test_images_reach_the_template_view() -> None:
    conversation = normalize_conversation(conversation_with(image()), [])
    rendered = conversation.to_messages()[0]
    assert rendered["images"][0]["vision_tokens"] == 196


# ------------------------------------------------------------ invalid layouts


def test_an_image_must_be_an_object() -> None:
    with pytest.raises(ValidationError, match="must be an object"):
        normalize_conversation(conversation_with("x.png"), [])  # type: ignore[arg-type]


def test_the_ref_must_be_a_non_empty_string() -> None:
    with pytest.raises(ValidationError, match="non-empty string"):
        normalize_conversation(conversation_with(image(ref="")), [])


def test_the_ref_must_be_relative() -> None:
    """An absolute ref would read whatever the training host happens to have."""
    with pytest.raises(ValidationError, match="absolute path"):
        normalize_conversation(conversation_with(image(ref="/etc/passwd")), [])


def test_a_windows_absolute_ref_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"absolute path|drive letter|forward slashes"):
        normalize_conversation(conversation_with(image(ref="C:/Windows/x.png")), [])


def test_a_drive_letter_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"drive letter|absolute path"):
        normalize_conversation(conversation_with(image(ref="C:x.png")), [])


def test_a_ref_that_names_no_file_is_refused() -> None:
    """A bare dot survives the non-empty check but resolves to a directory."""
    with pytest.raises(ValidationError, match="must name an image inside the root"):
        normalize_conversation(conversation_with(image(ref=".")), [])


def test_the_ref_must_not_escape_the_image_root() -> None:
    with pytest.raises(ValidationError, match="traverse upwards"):
        normalize_conversation(conversation_with(image(ref="../../secret.png")), [])


def test_a_backslash_ref_is_refused() -> None:
    """The same path means different things on different hosts; refuse the ambiguity."""
    with pytest.raises(ValidationError, match="forward slashes"):
        normalize_conversation(conversation_with(image(ref="shots\\\\fail.png")), [])


def test_grid_thw_must_have_three_entries() -> None:
    with pytest.raises(ValidationError, match="three entries"):
        normalize_conversation(conversation_with(image(grid=[1, 28])), [])


def test_grid_entries_must_be_positive_integers() -> None:
    with pytest.raises(ValidationError, match="positive integers"):
        normalize_conversation(conversation_with(image(grid=[1, 0, 28])), [])


def test_a_boolean_grid_entry_is_rejected() -> None:
    """Bool is an int subclass, so True would silently become a patch count of 1."""
    with pytest.raises(ValidationError, match="positive integers"):
        normalize_conversation(conversation_with(image(grid=[1, True, 28])), [])


def test_merge_size_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError, match="merge_size must be an integer"):
        normalize_conversation(conversation_with(image(merge=0)), [])


def test_a_patch_count_merge_size_cannot_produce_is_refused() -> None:
    """The template would emit a token count the encoder cannot match."""
    with pytest.raises(ValidationError, match="not divisible"):
        normalize_conversation(conversation_with(image(grid=[1, 30, 28], merge=4)), [])


# ----------------------------------------------------------------- role rules


def test_images_on_an_assistant_turn_are_refused() -> None:
    """The assistant generates tokens; it does not receive images."""
    with pytest.raises(ValidationError, match="only allowed on user turns"):
        normalize_conversation(conversation_with(image(), role="assistant"), [])


def test_images_on_a_system_turn_are_refused() -> None:
    with pytest.raises(ValidationError, match="only allowed on user turns"):
        normalize_conversation(conversation_with(image(), role="system"), [])


def test_a_user_turn_without_images_is_unaffected() -> None:
    conversation = normalize_conversation(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
        [],
    )
    assert conversation.messages[0].images == ()
