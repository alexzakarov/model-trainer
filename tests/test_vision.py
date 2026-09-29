"""Vision pipeline: the pixel/token contract, and the ways it silently breaks.

Every test here exists because the failure it guards is invisible. A mismatched
patch count does not crash; it trains. The model attends to the wrong pixels and
the loss still goes down.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from gotooltrain import DatasetError, ValidationError
from gotooltrain.schema import ImageSpec
from gotooltrain.template import IGNORED_INDEX
from gotooltrain.vision import (
    MAX_PIXEL_VALUES_PER_BATCH,
    budget_pixels,
    check_batch_vision_layout,
    count_vision_tokens,
    load_image,
    mask_vision_labels,
    merged_patches,
    pad_batch,
    process_images,
)

VISION_PAD_ID = 151655
PAD_ID = 151643


def spec(ref: str = "shots/fail.png", grid: tuple[int, int, int] = (1, 28, 28)) -> ImageSpec:
    return ImageSpec(ref=ref, grid_thw=grid, merge_size=2)


class FakeProcessor:
    """Returns grids derived from the images themselves, like a real one would."""

    def __init__(self, grids: list[tuple[int, int, int]] | None = None, fail: bool = False) -> None:
        """Derive grids from the images, or force the given ones / a failure."""
        self.grids = grids
        self.fail = fail
        self.calls: list[int] = []

    def __call__(self, images=None, **kwargs: Any) -> dict[str, Any]:
        """Return canned pixel values and grids."""
        self.calls.append(len(images or []))
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        rows = self.grids if self.grids is not None else [(1, 28, 28)] * len(images or [])
        return {
            "pixel_values": [[0.0]],
            "image_grid_thw": [list(row) for row in rows],
        }


def fake_image(name: str = "x") -> Any:
    return name


# ------------------------------------------------------------- declared layout


def test_merged_patches_matches_the_spec() -> None:
    assert merged_patches(spec()) == 196


def test_a_time_axis_multiplies_the_patch_count() -> None:
    """Video-style grids carry a temporal dimension that must be counted."""
    assert merged_patches(spec(grid=(4, 28, 28))) == 784


def test_a_degenerate_grid_is_refused() -> None:
    with pytest.raises(DatasetError, match="degenerate grid"):
        merged_patches(spec(grid=(0, 28, 28)))


def test_a_grid_that_merge_size_cannot_divide_is_refused() -> None:
    """The declared token count would not be an integer; the layout is impossible."""
    with pytest.raises(DatasetError, match="not divisible by merge_size"):
        merged_patches(spec(grid=(1, 29, 28)))


# ------------------------------------------------------- processor agreement


def test_a_matching_processor_is_accepted() -> None:
    batch = process_images(FakeProcessor(), [fake_image()], [spec()])
    assert batch.total_patches == 196
    assert batch.patches_per_image == (196,)


def test_a_processor_that_regrids_the_image_is_refused() -> None:
    """The dangerous case: different patch count, same call, no exception."""
    with pytest.raises(DatasetError, match="pixels and tokens disagree"):
        process_images(FakeProcessor(grids=[(1, 14, 14)]), [fake_image()], [spec()])


def test_a_processor_returning_the_wrong_number_of_grids_is_refused() -> None:
    with pytest.raises(DatasetError, match="would mis-pair pixels"):
        process_images(FakeProcessor(grids=[(1, 28, 28), (1, 28, 28)]), [fake_image()], [spec()])


def test_a_processor_with_no_grid_is_refused() -> None:
    class NoGrid:
        def __call__(self, images=None, **kwargs: Any) -> dict[str, Any]:
            """A processor that forgot to report a grid."""
            return {"pixel_values": [[0.0]]}

    with pytest.raises(DatasetError, match="no image_grid_thw"):
        process_images(NoGrid(), [fake_image()], [spec()])


def test_a_grid_without_three_columns_is_refused() -> None:
    with pytest.raises(DatasetError, match="3 columns"):
        process_images(FakeProcessor(grids=[(1, 28)]), [fake_image()], [spec()])


def test_a_processor_crash_is_reported_not_swallowed() -> None:
    """Silent pixel dropping would produce a batch missing images entirely."""
    with pytest.raises(DatasetError, match="image processor failed"):
        process_images(FakeProcessor(fail=True), [fake_image()], [spec()])


def test_image_and_spec_counts_must_match() -> None:
    with pytest.raises(DatasetError, match="1 images for 2 specs"):
        process_images(FakeProcessor(), [fake_image()], [spec(), spec("b.png")])


def test_an_empty_batch_short_circuits() -> None:
    batch = process_images(FakeProcessor(), [], [])
    assert batch.total_patches == 0
    assert batch.pixel_values is None


# ------------------------------------------------------------------ image load


def test_load_image_refuses_an_absolute_ref(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="must be relative"):
        load_image(str(tmp_path / "x.png"), tmp_path)


def test_load_image_refuses_an_escaping_ref(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="escapes the image root"):
        load_image("../../etc/passwd", tmp_path)


def test_load_image_reports_a_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValidationError, match="image not found"):
        load_image("nope.png", tmp_path)


def test_load_image_opens_a_real_png(tmp_path: pathlib.Path) -> None:
    from PIL import Image

    path = tmp_path / "a.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(path)
    assert load_image("a.png", tmp_path).size == (4, 4)


# --------------------------------------------------------------- label masking


def test_vision_positions_are_never_supervised() -> None:
    """The model should learn to use the image, not to emit patches."""
    input_ids = [1, VISION_PAD_ID, VISION_PAD_ID, 2, 3]
    labels = [IGNORED_INDEX, 42, 43, IGNORED_INDEX, 7]
    changed = mask_vision_labels(labels, input_ids, VISION_PAD_ID)
    assert changed == 2
    assert labels == [IGNORED_INDEX, IGNORED_INDEX, IGNORED_INDEX, IGNORED_INDEX, 7]


def test_masking_is_idempotent() -> None:
    input_ids = [VISION_PAD_ID]
    labels = [5]
    mask_vision_labels(labels, input_ids, VISION_PAD_ID)
    assert mask_vision_labels(labels, input_ids, VISION_PAD_ID) == 0


def test_counting_vision_tokens() -> None:
    assert count_vision_tokens([1, VISION_PAD_ID, 2, VISION_PAD_ID], VISION_PAD_ID) == 2


# ------------------------------------------------------------- batch integrity


def test_a_matching_batch_layout_passes() -> None:
    rows = [[1, VISION_PAD_ID, 2, 3], [1, 2, 3, PAD_ID]]
    assert check_batch_vision_layout(rows, VISION_PAD_ID, pad_id=PAD_ID, expected=1) == 1


def test_a_count_mismatch_is_refused() -> None:
    with pytest.raises(DatasetError, match="paired in the wrong order"):
        check_batch_vision_layout([[1, VISION_PAD_ID]], VISION_PAD_ID, pad_id=PAD_ID, expected=5)


def test_a_pad_id_equal_to_the_vision_pad_is_refused() -> None:
    """Otherwise every padded slot is read as image content."""
    with pytest.raises(DatasetError, match="is the vision pad token"):
        check_batch_vision_layout(
            [[VISION_PAD_ID]], VISION_PAD_ID, pad_id=VISION_PAD_ID, expected=1
        )


# ------------------------------------------------------------------ collation


def test_pad_batch_pads_on_the_right_with_a_real_mask() -> None:
    batch = pad_batch(
        [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [4, 5]},
        ],
        pad_id=PAD_ID,
        label_pad=IGNORED_INDEX,
    )
    assert batch["input_ids"] == [[1, 2, 3], [4, 5, PAD_ID]]
    assert batch["attention_mask"] == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"] == [[1, 2, 3], [4, 5, IGNORED_INDEX]]


def test_pad_batch_refuses_ragged_rows() -> None:
    with pytest.raises(DatasetError, match="ragged row"):
        pad_batch(
            [{"input_ids": [1, 2, 3], "attention_mask": [1, 1], "labels": [1, 2, 3]}],
            pad_id=PAD_ID,
            label_pad=IGNORED_INDEX,
        )


def test_pad_batch_refuses_an_empty_batch() -> None:
    with pytest.raises(DatasetError, match="empty batch"):
        pad_batch([], pad_id=PAD_ID, label_pad=IGNORED_INDEX)


def test_pad_batch_defaults_the_mask_when_absent() -> None:
    batch = pad_batch([{"input_ids": [1, 2]}], pad_id=PAD_ID, label_pad=IGNORED_INDEX)
    assert batch["attention_mask"] == [[1, 1]]
    assert batch["labels"] == [[1, 2]]


# ------------------------------------------------------------------- budget


def test_budget_counts_merged_patches() -> None:
    assert budget_pixels([196, 196], 1176) == 392 * 1176


def test_the_batch_budget_is_a_real_limit() -> None:
    """A documented ceiling that is never checked is decoration."""
    assert budget_pixels([1_000_000], 1176) > MAX_PIXEL_VALUES_PER_BATCH
    assert budget_pixels([10], 1176) < MAX_PIXEL_VALUES_PER_BATCH
