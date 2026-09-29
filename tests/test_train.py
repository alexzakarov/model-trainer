"""The training plan: what will be run, and the arithmetic that makes it affordable.

None of this needs a GPU. That is the point: the decisions with the largest cost
consequences -- padding waste, the step schedule, whether the tokenizer will
supervise the right tokens -- are decided and tested here, where a mistake costs a
second instead of a GPU day.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Any, ClassVar

import pytest

from gotooltrain.collator import examples_from_conversations
from gotooltrain.errors import DatasetError
from gotooltrain.hub import HubPushPolicy
from gotooltrain.schema import FORMAT_VERSION
from gotooltrain.train import (
    CONTEXT_LENGTH,
    DTYPES,
    OptimisationPlan,
    RunSummary,
    _apply_step,
    _build_optimizer,
    _history_entry,
    _maybe_push,
    _push_checkpoint,
    _record_push,
    _torch_dtype,
    _vision_pad_id,
    assert_examples_fit,
    assert_token_format,
    cosine_lr,
    freeze_vision_tower,
    is_multimodal,
    iter_epochs,
    length_grouped_batches,
    padding_waste,
    steps_for,
    train,
    warmup_steps,
)

TEMPLATE = (
    "{#-\n  format: anthropic-tools-v1\n#-}\n"
    "{% for m in messages %}{% generation %}x{% endgeneration %}{% endfor %}"
)


class FakeTokenizer:
    """Just enough of the tokenizer surface for the format assertions."""

    def __init__(
        self,
        *,
        template: str | None = TEMPLATE,
        pad_id: int | None = 151643,
        vision_id: int = 151655,
    ) -> None:
        """Adopt a template, a pad id, and a vision pad id."""
        self.chat_template = template
        self.pad_token_id = pad_id
        self._vision_id = vision_id

    def convert_tokens_to_ids(self, token: str) -> int:
        """Resolve the vision pad token, or a negative id when it is absent."""
        return self._vision_id if token == "<|vision_pad|>" else -1


class FakeParameter:
    """A stand-in for a real parameter, with a count and a rank."""

    def __init__(self, numel: int = 10, ndim: int = 2) -> None:
        """Hold a parameter count and a rank, like a real tensor would."""
        self._numel = numel
        self.ndim = ndim
        self.requires_grad = True

    def numel(self) -> int:
        """Element count."""
        return self._numel

    def requires_grad_(self, value: bool) -> None:
        """Set whether this parameter receives gradients."""
        self.requires_grad = value


class FakeModel:
    """A model whose parameters are named, for freeze tests."""

    def __init__(self, names: list[str]) -> None:
        """Build parameters under the given names."""
        self.named = [(n, FakeParameter()) for n in names]

    def named_parameters(self):  # type: ignore[no-untyped-def]
        """Iterate name/parameter pairs."""
        return iter(self.named)


class RecordingPusher:
    """A pusher that remembers what it was asked to publish."""

    def __init__(self) -> None:
        """Start with nothing published."""
        self.calls: list[tuple[str, int, int]] = []

    def push(self, directory: str | Path, step: int, total_steps: int) -> str:
        """Record the call and answer with a synthetic revision."""
        self.calls.append((str(directory), step, total_steps))
        return f"rev-{step}"


class SavingModel:
    """A model that records what it was asked to write, and nothing else."""

    def __init__(self) -> None:
        """Start with no saves."""
        self.saves: list[tuple[str, bool]] = []

    def save_pretrained(self, directory: str, safe_serialization: bool = True) -> None:
        """Record a save. No weights are written; the bookkeeping is the point."""
        self.saves.append((directory, safe_serialization))


class SavingTokenizer:
    """A tokenizer that records where it was written."""

    def __init__(self) -> None:
        """Start with no saves."""
        self.saves: list[str] = []

    def save_pretrained(self, directory: str) -> None:
        """Record a save."""
        self.saves.append(directory)


class Multimodal:
    """A config that has a vision tower."""

    vision_config: ClassVar[dict[str, int]] = {"hidden_size": 32}
    text_config: ClassVar[str] = "inner"


class TextOnly:
    """A config with no vision tower at all."""

    text_config: ClassVar[str] = "inner"


def plan(**overrides: Any) -> OptimisationPlan:
    base: dict[str, Any] = {"model_id": "Qwen/Qwen3.5-4B", "output_dir": "out"}
    base.update(overrides)
    return OptimisationPlan(**base)


class Example:
    """A training example with a length and a supervised count."""

    def __init__(self, length: int, supervised: int = 0) -> None:
        """Hold a token length and a supervised count."""
        self.length = length
        self.supervised_tokens = supervised


# ------------------------------------------------------------------- the plan


def test_a_plan_records_every_setting_that_changes_the_result() -> None:
    record = plan(learning_rate=2e-5, epochs=4, seed=11).to_record()
    for field in (
        "model_id",
        "learning_rate",
        "epochs",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "effective_batch_size",
        "warmup_ratio",
        "context_length",
        "seed",
        "train_vision_tower",
        "dtype",
        "token_format",
    ):
        assert field in record
    assert record["token_format"] == FORMAT_VERSION


def test_the_compute_dtype_is_named_and_checked() -> None:
    """A hard-coded bfloat16 is an unreviewed choice and untestable on a CPU."""
    assert plan().dtype == "bfloat16"
    assert set(DTYPES) == {"bfloat16", "float16", "float32"}


def test_an_unknown_dtype_is_refused() -> None:
    with pytest.raises(DatasetError, match="dtype must be one of"):
        plan(dtype="float64")


def test_an_unknown_dtype_is_named_against_the_installed_torch() -> None:
    """A name from the allow-list that this torch lacks is a torch problem."""
    torch = pytest.importorskip("torch")
    assert _torch_dtype("float32", torch) is torch.float32


def test_a_dtype_this_torch_does_not_have_is_reported() -> None:
    class Bare:
        """A torch with no dtypes at all."""

    with pytest.raises(DatasetError, match="no dtype"):
        _torch_dtype("bfloat16", Bare())


def test_effective_batch_size_multiplies_accumulation() -> None:
    assert plan(per_device_batch_size=2, gradient_accumulation_steps=8).effective_batch_size == 16


def test_the_context_is_32k() -> None:
    assert CONTEXT_LENGTH == 32_768
    assert plan().context_length == 32_768


def test_a_zero_learning_rate_is_refused() -> None:
    with pytest.raises(DatasetError, match="learning_rate must be"):
        plan(learning_rate=0.0)


def test_zero_epochs_is_refused() -> None:
    with pytest.raises(DatasetError, match="epochs must be"):
        plan(epochs=0)


def test_a_zero_batch_size_is_refused() -> None:
    with pytest.raises(DatasetError, match="batch size and gradient accumulation"):
        plan(per_device_batch_size=0)


def test_a_warmup_ratio_of_one_is_refused() -> None:
    with pytest.raises(DatasetError, match="warmup_ratio must be"):
        plan(warmup_ratio=1.0)


def test_an_unnamed_model_is_refused() -> None:
    with pytest.raises(DatasetError, match="model_id is required"):
        plan(model_id="")


def test_a_plan_is_written_next_to_the_checkpoint(tmp_path: pathlib.Path) -> None:
    path = plan(output_dir=str(tmp_path)).write()

    assert path.name == "training_plan.json"
    assert '"model_id"' in path.read_text(encoding="utf-8")


def test_rewriting_an_identical_plan_is_allowed(tmp_path: pathlib.Path) -> None:
    first = plan(output_dir=str(tmp_path))
    first.write()
    first.write()


def test_overwriting_a_different_plan_is_refused(tmp_path: pathlib.Path) -> None:
    """Refuse to resume into a directory whose recorded settings differ.

    The directory is where a reader looks for the truth about a checkpoint, so
    letting it hold a plan that is not the one that ran would make it a lie.
    """
    plan(output_dir=str(tmp_path), learning_rate=1e-5).write()
    with pytest.raises(DatasetError, match="already holds a different training plan"):
        plan(output_dir=str(tmp_path), learning_rate=9e-5).write()


# ------------------------------------------------------------ token format


def test_a_correct_tokenizer_passes() -> None:
    assert assert_token_format(FakeTokenizer()) == TEMPLATE


def test_a_tokenizer_without_a_template_is_refused() -> None:
    with pytest.raises(DatasetError, match="no chat_template"):
        assert_token_format(FakeTokenizer(template=None))


def test_a_foreign_template_version_is_refused() -> None:
    template = TEMPLATE.replace("anthropic-tools-v1", "anthropic-tools-v0")
    with pytest.raises(DatasetError, match="the data was rendered as"):
        assert_token_format(FakeTokenizer(template=template))


def test_a_template_without_generation_is_refused() -> None:
    """Without it there is no assistant mask, so the loss covers the prompt too."""
    template = "{#-\n  format: anthropic-tools-v1\n#-}\n{% for m in messages %}x{% endfor %}"
    with pytest.raises(DatasetError, match="no \\{% generation %\\} block"):
        assert_token_format(FakeTokenizer(template=template))


def test_a_missing_pad_token_is_refused() -> None:
    """An unmasked pad position is trained on as if it were content."""
    with pytest.raises(DatasetError, match="no pad_token_id"):
        assert_token_format(FakeTokenizer(pad_id=None))


def test_a_pad_equal_to_the_vision_pad_is_refused() -> None:
    with pytest.raises(DatasetError, match="equals the vision pad token"):
        assert_token_format(FakeTokenizer(pad_id=151655, vision_id=151655))


# ------------------------------------------------------------------ batching


def test_batches_group_similar_lengths() -> None:
    lengths = [10] * 4 + [10_000] * 4
    batches = length_grouped_batches(lengths, batch_size=4, bucket_multiplier=4)
    for batch in batches:
        widths = {lengths[i] for i in batch}
        assert len(widths) == 1, "a batch mixed a 10-token record with a 10k one"


def test_batches_are_shuffled_but_cover_every_index() -> None:
    lengths = list(range(100))
    batches = length_grouped_batches(lengths, batch_size=8, bucket_multiplier=8)
    assert sorted(i for b in batches for i in b) == list(range(100))


def test_a_bucket_smaller_than_the_batch_is_refused() -> None:
    with pytest.raises(DatasetError, match="bucket_multiplier"):
        length_grouped_batches([1, 2, 3], batch_size=8, bucket_multiplier=4)


def test_an_empty_corpus_yields_no_batches() -> None:
    assert length_grouped_batches([], batch_size=4) == []


def test_padding_waste_is_measured() -> None:
    """The number that decides whether a run is affordable."""
    lengths = [100, 110, 120]
    batches = [[0, 1, 2]]
    assert padding_waste(batches, lengths) == pytest.approx((120 * 3 - 330) / 360)
    assert padding_waste([], lengths) == 0.0


def test_grouping_actually_reduces_padding_waste() -> None:
    lengths = [100] * 64 + [32_000]
    grouped = length_grouped_batches(lengths, batch_size=8, bucket_multiplier=8)
    assert padding_waste(grouped, lengths) < 0.05, "the outlier should batch alone"


def test_shuffling_without_grouping_would_waste_mostly_padding() -> None:
    """Motivates the grouping: random batches of the same corpus are far worse.

    Fixed-size batches over a shuffled corpus -- the obvious implementation, and
    what a Trainer does by default -- put the 32K records wherever they land, and
    every batch they land in pays for them.
    """
    import random

    lengths = [100] * 64 + [32_000] * 8
    order = list(range(len(lengths)))
    random.Random(0).shuffle(order)
    naive = [order[start : start + 8] for start in range(0, len(order), 8)]
    assert padding_waste(naive, lengths) > 0.5


def test_the_outlier_is_never_padded_against_short_records() -> None:
    """The guarantee the ratio cut buys, asserted directly."""
    lengths = [100] * 40 + [32_000] * 4
    for batch in length_grouped_batches(lengths, batch_size=8, bucket_multiplier=8):
        widths = [lengths[i] for i in batch]
        assert max(widths) <= 4 * min(widths), "a batch mixed incompatible lengths"


# ------------------------------------------------------------------ schedule


def test_steps_scale_with_effective_batch_and_epochs() -> None:
    p = plan(per_device_batch_size=2, gradient_accumulation_steps=4, epochs=2)
    assert steps_for(100, p) == 24, "12 per epoch over an effective batch of 8"


def test_warmup_is_a_share_of_the_run() -> None:
    assert warmup_steps(1000, 0.03) == 30
    assert warmup_steps(1000, 0.0) == 0


def test_cosine_warms_up_then_decays_to_a_small_floor() -> None:
    assert cosine_lr(1e-5, 0, 100, 10) == pytest.approx(1e-6)
    assert cosine_lr(1e-5, 5, 100, 10) == pytest.approx(6e-6)
    mid = cosine_lr(1e-5, 55, 100, 10)
    end = cosine_lr(1e-5, 99, 100, 10)
    assert mid > end > 0
    assert end == pytest.approx(1e-5 * 0.05, rel=1e-2)


def test_a_zero_length_schedule_is_refused() -> None:
    with pytest.raises(DatasetError, match="total_steps must be"):
        cosine_lr(1e-5, 0, 0, 0)


def test_epoch_order_changes_between_epochs() -> None:
    """Identical order every epoch lets a model exploit sequence, not content."""
    batches = [[0, 1], [2, 3], [4, 5]]
    stream = list(iter_epochs(batches, epochs=2, seed=3))
    assert [e for e, _ in stream] == [0, 0, 0, 1, 1, 1], "the epoch index is yielded"
    first = [b for e, b in stream if e == 0]
    second = [b for e, b in stream if e == 1]
    assert first != second, "the order must be reshuffled between epochs"
    assert all(len(b) == 2 for b in first + second)


# ------------------------------------------------------------------- fitting


def test_examples_within_the_context_are_accepted() -> None:
    assert_examples_fit([Example(10), Example(CONTEXT_LENGTH)], plan())


def test_an_over_long_example_is_refused_and_named() -> None:
    """Truncation would supervise a span that ends mid-sentence."""
    with pytest.raises(DatasetError, match="over the 32768 context"):
        assert_examples_fit([Example(10), Example(CONTEXT_LENGTH + 1)], plan())


def test_an_example_without_a_length_is_skipped() -> None:
    assert_examples_fit([object()], plan())


# -------------------------------------------------------------- vision tower


def test_the_vision_tower_is_trainable_by_default() -> None:
    assert plan().train_vision_tower is True


def test_freezing_the_tower_reports_how_many_parameters() -> None:
    model = FakeModel(["visual.patch.weight", "model.layers.0.weight", "vision_gate"])
    _, frozen = freeze_vision_tower(model)
    assert frozen == 20, "two of the three parameters stopped training"


# -------------------------------------------------------------------- summary


def test_a_summary_serialises_for_the_record() -> None:
    record = RunSummary(
        steps=10,
        epochs=3,
        examples=240,
        supervised_tokens=1000,
        final_loss=0.42,
        duration_s=12.5,
        plan_path="p.json",
    ).to_record()
    assert record["final_loss"] == 0.42
    assert record["duration_s"] == 12.5
    assert record["history"] == []


def test_a_summary_reports_every_publication_it_made() -> None:
    """The durable record of a disposable machine: what already exists off-box."""
    from gotooltrain.hub import PushRecord

    record = RunSummary(
        steps=10,
        epochs=1,
        examples=10,
        supervised_tokens=100,
        final_loss=0.1,
        duration_s=1.0,
        plan_path="p.json",
        pushes=[PushRecord(step=5, total_steps=10, revision="abc", path="/out")],
    ).to_record()
    assert record["pushes"] == [
        {"step": 5, "total_steps": 10, "revision": "abc", "path": "/out", "dry_run": False}
    ]


# ------------------------------------------------------------- model loading


def test_a_checkpoint_with_a_vision_tower_is_recognised() -> None:
    """Loading the causal-LM class would silently drop the tower from training."""
    assert is_multimodal(Multimodal()) is True


def test_a_text_only_checkpoint_is_recognised() -> None:
    assert is_multimodal(TextOnly()) is False


# ------------------------------------------------------- hub publication


def test_the_publication_settings_belong_in_the_plan(tmp_path: pathlib.Path) -> None:
    """The destination is part of what a run produced.

    Two runs with identical hyperparameters and different destinations leave
    different evidence behind, so the plan that records the run records this too.
    """
    hub = HubPushPolicy(repo_id="a/b", every_steps=25)
    record = plan(hub=hub).to_record()
    assert record["hub"] == {
        "repo_id": "a/b",
        "every_steps": 25,
        "token_env": "HF_TOKEN",
        "private": False,
        "dry_run": False,
    }
    assert plan().to_record()["hub"] is None


def test_publishing_writes_the_checkpoint_and_describes_it(tmp_path: pathlib.Path) -> None:
    """The folder carries its own provenance.

    A checkpoint on the Hub is read long after the run directory that produced it
    is gone, so it has to say which settings made it.
    """
    from gotooltrain.hub import read_push_state

    output = tmp_path / "out"
    model, tokenizer, pusher = SavingModel(), SavingTokenizer(), RecordingPusher()
    plan_with_hub = plan(output_dir=str(output), hub=HubPushPolicy(repo_id="a/b", every_steps=2))

    record = _push_checkpoint(pusher, plan_with_hub, model, tokenizer, step=4, total_steps=10)

    assert record.revision == "rev-4"
    assert pusher.calls == [(str(output), 4, 10)]
    assert model.saves == [(str(output), True)]
    assert tokenizer.saves == [str(output)]

    state = read_push_state(output)
    assert state["step"] == 4
    assert state["total_steps"] == 10
    assert state["plan"]["hub"]["repo_id"] == "a/b"


def test_a_published_folders_dry_run_flag_is_recorded(tmp_path: pathlib.Path) -> None:
    output = tmp_path / "out"
    plan_dry = plan(
        output_dir=str(output), hub=HubPushPolicy(repo_id="a/b", every_steps=1, dry_run=True)
    )
    record = _push_checkpoint(RecordingPusher(), plan_dry, SavingModel(), SavingTokenizer(), 1, 1)
    assert record.dry_run is True


def test_a_step_off_the_schedule_publishes_nothing(tmp_path: pathlib.Path) -> None:
    pusher = RecordingPusher()
    plan_with_hub = plan(
        output_dir=str(tmp_path / "out"), hub=HubPushPolicy(repo_id="a/b", every_steps=5)
    )
    record = _maybe_push(pusher, plan_with_hub, SavingModel(), SavingTokenizer(), 3, 10)
    assert record is None
    assert pusher.calls == []


def test_a_step_on_the_schedule_publishes(tmp_path: pathlib.Path) -> None:
    pusher = RecordingPusher()
    plan_with_hub = plan(
        output_dir=str(tmp_path / "out"), hub=HubPushPolicy(repo_id="a/b", every_steps=5)
    )
    record = _maybe_push(pusher, plan_with_hub, SavingModel(), SavingTokenizer(), 5, 10)
    assert record is not None
    assert record.step == 5
    assert pusher.calls == [(str(tmp_path / "out"), 5, 10)]


def test_a_run_with_no_repo_publishes_nothing(tmp_path: pathlib.Path) -> None:
    """A pusher with no policy is a wiring mistake.

    The decision reads the plan, not just the pusher, so a plan without a repo
    publishes nothing even if something is wired to push.
    """
    pusher = RecordingPusher()
    unpushed = plan(output_dir=str(tmp_path))
    result = _maybe_push(pusher, unpushed, SavingModel(), SavingTokenizer(), 1, 1)
    assert result is None
    assert pusher.calls == []


def test_a_step_with_no_pusher_publishes_nothing(tmp_path: pathlib.Path) -> None:
    plan_with_hub = plan(
        output_dir=str(tmp_path / "out"), hub=HubPushPolicy(repo_id="a/b", every_steps=1)
    )
    assert _maybe_push(None, plan_with_hub, SavingModel(), SavingTokenizer(), 1, 1) is None


def test_the_log_line_names_the_revision() -> None:
    """A session that ends mid-run must leave behind what is already stored."""
    from gotooltrain.hub import PushRecord

    lines: list[str] = []
    pushes: list[PushRecord] = []
    record = PushRecord(step=5, total_steps=10, revision="abc", path="/out")

    _record_push(pushes, record, lines.append)

    assert pushes == [record]
    assert lines == ["pushed step 5/10: abc"]


def test_a_step_that_published_nothing_says_nothing() -> None:
    """A log line for each of 900 non-events buries the one that matters."""
    from gotooltrain.hub import PushRecord

    lines: list[str] = []
    pushes: list[PushRecord] = []
    _record_push(pushes, None, lines.append)
    assert lines == []
    assert pushes == []


# ------------------------------------------------------- the step accounting


class FakeOptimizer:
    """An optimizer that only counts its updates."""

    def __init__(self) -> None:
        """Start with no steps taken and gradients un-zeroed."""
        self.steps = 0
        self.zeroed = False
        self.param_groups = [{"lr": 0.0}]

    def step(self) -> None:
        """Record one optimiser update."""
        self.steps += 1

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Accept the call the real optimizer expects."""
        self.zeroed = True


class FakeScheduler:
    """A schedule that only counts its own advances."""

    def __init__(self) -> None:
        """Start with no steps taken."""
        self.steps = 0

    def step(self) -> None:
        """Record one schedule advance."""
        self.steps += 1

    def get_last_lr(self) -> list[float]:
        """Report the rate the optimizer is using."""
        return [0.5]


def test_a_step_clips_updates_and_advances_the_schedule() -> None:
    """The learning rate is written from the plan, not from a second schedule."""
    torch = pytest.importorskip("torch")
    del torch  # imported for the pragma-free clip call below

    optimizer = FakeOptimizer()
    scheduler = FakeScheduler()
    model = _TinyModel()
    _apply_step(model, optimizer, scheduler, plan(), step=0, total_steps=100, warmup=10)
    assert optimizer.steps == 1
    assert scheduler.steps == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert optimizer.zeroed


class _TinyModel:
    def __init__(self) -> None:
        self.zeroed = False
        import torch

        self._module = torch.nn.Linear(2, 2)

    def parameters(self):  # type: ignore[no-untyped-def]
        """Yield something clip_grad_norm can measure."""
        return self._module.parameters()


def test_the_history_entry_averages_the_window() -> None:
    """The rate logged is the one applied, not the scheduler's next value."""
    assert _history_entry(3, window_loss=2.0, window=4, rate=0.5) == {
        "step": 3.0,
        "loss": 0.5,
        "lr": 0.5,
    }


def test_an_empty_window_still_reports_a_loss() -> None:
    assert _history_entry(1, window_loss=0.0, window=0, rate=0.0)["loss"] == 0.0


def test_the_step_returns_the_rate_it_applied() -> None:
    """``scheduler.step()`` has already advanced past it, so reading it back lies."""
    pytest.importorskip("torch")
    rate = _apply_step(
        _TinyModel(), FakeOptimizer(), FakeScheduler(), plan(), step=0, total_steps=100, warmup=10
    )
    assert rate == cosine_lr(plan().learning_rate, 0, 100, 10)
    assert rate > 0


def test_the_optimizer_skips_frozen_parameters_and_biases_decay() -> None:
    """Freezing the tower must actually keep its parameters out of the update.

    Covered directly because a tiny text-only checkpoint has no tower to freeze, so
    the training loop alone would never reach this branch -- and a frozen tower that
    is silently still trained is the kind of bug that shows up as a changed model
    months later.
    """
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(4, 4)
    model.weight.requires_grad_(False)

    optimizer = _build_optimizer(model, plan(weight_decay=0.1))
    decay_params, no_decay = (group["params"] for group in optimizer.param_groups)

    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.weight) not in grouped, "a frozen parameter reached the optimizer"
    assert id(model.bias) in grouped

    # A bias is one-dimensional, so it belongs to the undecayed group.
    assert id(model.bias) in {id(p) for p in no_decay}
    assert id(model.bias) not in {id(p) for p in decay_params}
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


# ------------------------------------------------------- the vision pad lookup


def test_the_vision_pad_id_is_resolved() -> None:
    assert _vision_pad_id(FakeTokenizer()) == 151655


def test_a_tokenizer_without_a_vision_pad_is_refused() -> None:
    with pytest.raises(DatasetError, match="no <\\|vision_pad\\|> token"):
        _vision_pad_id(FakeTokenizer(vision_id=-1))


# --------------------------------------------------------- ratio and batching


def test_a_length_ratio_below_one_is_refused() -> None:
    with pytest.raises(DatasetError, match="max_length_ratio must be"):
        length_grouped_batches([1, 2, 3], batch_size=2, max_length_ratio=0)


def test_a_single_batch_covers_everything() -> None:
    batches = length_grouped_batches([5, 5, 5], batch_size=8)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_a_batch_stops_growing_past_its_limit() -> None:
    batches = length_grouped_batches([10] * 10, batch_size=4, bucket_multiplier=4)
    assert all(len(b) <= 4 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_a_zero_context_length_is_refused() -> None:
    with pytest.raises(DatasetError, match="context_length must be"):
        plan(context_length=0)


def test_a_zero_batch_size_is_refused_by_the_grouper() -> None:
    with pytest.raises(DatasetError, match="batch_size must be"):
        length_grouped_batches([1, 2], batch_size=0)


# ------------------------------------------------------------- the memory knobs


def test_the_memory_knobs_are_named_and_recorded() -> None:
    """They change what a run costs, so they belong in the plan that is written."""
    record = plan(gradient_checkpointing=True, optimizer="adafactor").to_record()
    assert record["gradient_checkpointing"] is True
    assert record["optimizer"] == "adafactor"
    assert plan().gradient_checkpointing is False
    assert plan().optimizer == "adamw"


def test_an_unknown_optimizer_is_refused() -> None:
    with pytest.raises(DatasetError, match="optimizer must be one of"):
        plan(optimizer="sgd")


def test_adafactor_is_used_when_the_plan_asks_for_it() -> None:
    """The optimiser is chosen, never substituted: a different one is a different run."""
    from transformers.optimization import Adafactor

    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(8, 8)

    adafactor = _build_optimizer(model, plan(optimizer="adafactor"))
    assert isinstance(adafactor, Adafactor)

    adamw = _build_optimizer(model, plan())
    assert not isinstance(adamw, Adafactor)


def test_gradient_checkpointing_is_off_unless_asked() -> None:
    """Silently recomputing everything would halve throughput on a big run."""
    assert plan().gradient_checkpointing is False
    assert plan(gradient_checkpointing=True).gradient_checkpointing is True


def test_a_batch_that_fills_exactly_leaves_no_trailing_batch() -> None:
    """The trailing-flush branch: a full final batch must not produce an empty one."""
    batches = length_grouped_batches([10] * 8, batch_size=4, bucket_multiplier=4)
    assert sorted(len(b) for b in batches) == [4, 4]
    assert all(b for b in batches), "no empty batch was emitted"


def test_a_schedule_past_its_end_sits_on_the_floor() -> None:
    assert cosine_lr(1e-5, 100, 100, 10) == pytest.approx(1e-5 * 0.05)


# ------------------------------------------------------------- the whole loop
#
# The end-to-end run. Everything above tests a decision in isolation; this is the
# only check that the decisions compose -- that the loader accepts the target
# architecture, that the collator's batch fits the model's forward signature, that
# a loss is produced, backpropagated and applied, and that a checkpoint and a plan
# come out the other side. It ran nowhere until it ran here, which is why it is
# done on the smallest coherent Qwen3.5 that can be built.


def test_a_run_trains_and_writes_a_checkpoint(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    sample_record: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """One optimiser step, on a real architecture, with a real tokenizer."""
    from gotooltrain import normalize_conversation

    output = tmp_path / "run"
    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(output),
        dtype="float32",
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
    )
    conversation = normalize_conversation(sample_record["messages"], sample_record["tools"])
    examples = list(
        examples_from_conversations(
            [conversation],
            qwen_tokenizer,
            vision_pad_id=_vision_pad_id(qwen_tokenizer),
        )
    )
    assert examples and examples[0].supervised_tokens > 0

    lines: list[str] = []
    summary = train(training_plan, examples, device="cpu", log=lines.append)

    assert summary.steps == 1
    assert summary.examples == 1
    assert summary.supervised_tokens > 0
    assert summary.final_loss is not None
    assert summary.final_loss == pytest.approx(summary.final_loss)  # finite
    assert len(summary.history) == 1
    assert summary.history[0]["lr"] > 0
    assert (output / "training_plan.json").is_file()
    assert (output / "config.json").is_file()
    # The plan's own settings, not a default: a run keyed to the wrong settings
    # cannot be reproduced.
    recorded = json.loads((output / "training_plan.json").read_text(encoding="utf-8"))
    assert recorded["dtype"] == "float32"
    assert recorded["model_id"] == str(tiny_checkpoint)


def test_a_run_with_gradient_checkpointing_still_trains(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    sample_record: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """Recomputing activations must not change the result, only the cost."""
    from gotooltrain import normalize_conversation

    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(tmp_path / "ckpt"),
        dtype="float32",
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        gradient_checkpointing=True,
    )
    conversation = normalize_conversation(sample_record["messages"], sample_record["tools"])
    examples = list(
        examples_from_conversations(
            [conversation],
            qwen_tokenizer,
            vision_pad_id=_vision_pad_id(qwen_tokenizer),
        )
    )
    lines: list[str] = []
    summary = train(training_plan, examples, device="cpu", log=lines.append)
    assert summary.steps == 1
    assert any("gradient checkpointing" in line for line in lines)


def test_adafactor_runs_end_to_end(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    sample_record: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """The low-memory optimiser has to survive a real step, not just construct."""
    from gotooltrain import normalize_conversation

    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(tmp_path / "adafactor"),
        dtype="float32",
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        optimizer="adafactor",
    )
    conversation = normalize_conversation(sample_record["messages"], sample_record["tools"])
    examples = list(
        examples_from_conversations(
            [conversation],
            qwen_tokenizer,
            vision_pad_id=_vision_pad_id(qwen_tokenizer),
        )
    )
    lines: list[str] = []
    summary = train(training_plan, examples, device="cpu", log=lines.append)
    assert summary.steps == 1
    assert any("adafactor" in line for line in lines)


def test_the_run_reports_what_it_did(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    sample_record: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """The log is the only account of a run that costs a GPU day."""
    from gotooltrain import normalize_conversation

    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(tmp_path / "run2"),
        dtype="float32",
        epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
    )
    conversation = normalize_conversation(sample_record["messages"], sample_record["tools"])
    examples = list(
        examples_from_conversations(
            [conversation],
            qwen_tokenizer,
            vision_pad_id=_vision_pad_id(qwen_tokenizer),
        )
    )
    lines: list[str] = []
    train(training_plan, examples, device="cpu", log=lines.append)

    joined = "\n".join(lines)
    assert "system prompt" in joined
    assert "batch" in joined
    assert "saved to" in joined


def test_a_run_refuses_an_example_over_the_context(
    tiny_checkpoint: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Refusing before the loader means an over-long corpus never starts a run."""
    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(tmp_path / "run3"),
        context_length=4,
    )
    with pytest.raises(DatasetError, match="over the"):
        train(training_plan, [Example(length=10)], device="cpu")


def test_the_vision_tower_is_frozen_only_when_asked(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    sample_record: dict[str, Any],
    tmp_path: pathlib.Path,
) -> None:
    """The escape hatch has to be reachable from the plan, not just the function."""
    from gotooltrain import normalize_conversation

    training_plan = OptimisationPlan(
        model_id=str(tiny_checkpoint),
        output_dir=str(tmp_path / "run4"),
        dtype="float32",
        epochs=1,
        learning_rate=1e-4,
        train_vision_tower=False,
    )
    conversation = normalize_conversation(sample_record["messages"], sample_record["tools"])
    examples = list(
        examples_from_conversations(
            [conversation],
            qwen_tokenizer,
            vision_pad_id=_vision_pad_id(qwen_tokenizer),
        )
    )
    lines: list[str] = []
    train(training_plan, examples, device="cpu", log=lines.append)
    assert any("vision tower frozen" in line for line in lines)
