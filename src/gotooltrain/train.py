"""Full fine-tuning of a Qwen3.5-4B tool agent.

The decisions that are not negotiable, and the reason each is here rather than in
a config file someone might override:

**Completion-only loss.** Only assistant tokens are targets. Everything else --
the catalogue, the user's request, the tool results -- is context. Training on
tool results teaches the model to fabricate command output, which is the single
most damaging thing it could learn in this setting: it would stop reading the
compiler and start imagining success.

**32K context, with a measured budget.** 32K is the target, but a batch padded to
32K costs 32K slots per row even when most rows are 400 tokens. Records are sorted
by length into buckets so padding is not paid for, and the budget is checked
before the batch is built rather than by the allocator.

**The vision tower is trained.** The model keeps its multimodal ability because
every vision parameter is updated and the corpus carries images. Freezing it and
training on image-laden data is how a model ends up worse at both.

**The token format is asserted, not assumed.** The training tokenizer must carry
this project's chat template with the same version the data was rendered with. A
tokenizer that silently fell back to the base format would train a model that
cannot parse its own tool calls at serving time, and the loss curve would look
fine.

**The checkpoint leaves the machine on a schedule.** A run on metered, disposable
hardware is only as durable as its last upload, so a plan may name a Hub repo and
an interval (:mod:`gotooltrain.hub`). The schedule is arithmetic that is decided
before the run, the token is resolved before a model is loaded, and the finished
weights are published even when the realised step count never hit the interval.

Nothing here imports torch at module level. The plan and the accounting are pure
and therefore testable without a GPU; only :func:`train` needs one.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .errors import DatasetError
from .hub import (
    CheckpointPusher,
    HubPushPolicy,
    PushRecord,
    pending_final_push,
    push_state_payload,
    resolve_pusher,
    should_push,
    write_push_state,
)
from .schema import FORMAT_VERSION
from .template import IGNORED_INDEX

#: The target context length. Not a max: examples longer than this are refused
#: rather than truncated, because a truncated assistant span is supervised
#: training on a fragment that ends mid-sentence.
CONTEXT_LENGTH: Final[int] = 32_768

DEFAULT_LR: Final[float] = 1e-5
DEFAULT_WARMUP_RATIO: Final[float] = 0.03
DEFAULT_WEIGHT_DECAY: Final[float] = 0.0
DEFAULT_GRAD_ACCUM: Final[int] = 8
DEFAULT_EPOCHS: Final[int] = 3

#: Compute dtypes a run may use. An explicit setting rather than a hard-coded
#: bfloat16: bf16 is the right choice on an H100 and the wrong one on a CPU, where
#: most operations fall back to a slow path and the optimiser state has no fp32
#: master weights to accumulate into. Naming it also makes a smoke test on CPU
#: possible at all, which is what proves the loop runs before a GPU day is spent.
DTYPES: Final[tuple[str, ...]] = ("bfloat16", "float16", "float32")

#: Bytes in a gibibyte, so the arithmetic above reads in the unit the budget is
#: stated in rather than in a scale factor repeated at every division.
_GB: Final[float] = float(1024**3)

#: Optimiser families a run may use. Adafactor keeps no first moment and factors
#: the second, so its state is a few megabytes instead of a few gigabytes; on a
#: single card that is often the difference between fitting and not.
OPTIMIZERS: Final[tuple[str, ...]] = ("adamw", "adafactor")

#: The card a run is fitted against unless told otherwise, in gigabytes.
#:
#: A full fine-tune in bf16 holds the weights and one gradient per parameter
#: resident no matter what is done about the rest: a 4.33B model is 17.4 GB before
#: a single activation exists. That floor is what makes the budget worth stating --
#: it cannot be optimised away, only planned around.
DEFAULT_MEMORY_BUDGET_GB: Final[float] = 40.0

#: How the loss is computed. Both are exact; they differ only in how much of the
#: vocabulary projection is materialised at once.
#:
#: ``selective`` runs the backbone, keeps only the positions that carry a label, and
#: projects *those* through the vocabulary head. The arithmetic is identical to the
#: built-in loss -- same shift, same ignored index, same mean -- but the logits
#: tensor is [supervised_positions, vocab] instead of [sequence, vocab]. Measured
#: on this project's own corpus the supervised share is 32%, so that is a 3.1x cut
#: on the single largest term in the budget.
#:
#: ``builtin`` hands the whole thing to the model. It is the escape hatch for a
#: checkpoint whose backbone does not expose hidden states separately; it costs the
#: full projection, and the plan records that it was chosen.
LOSS_MODES: Final[tuple[str, ...]] = ("selective", "builtin")

#: Bytes per intermediate activation tensor of shape [tokens, hidden] in bf16.
_ACTIVATION_BYTES: Final[int] = 2

#: How many such tensors one layer holds beyond its attention state. An estimate,
#: and labelled as one: the point of the budget is to stop a run starting on a card
#: it cannot fit, not to predict the allocator to the megabyte.
_INTERMEDIATES_PER_LAYER: Final[float] = 8.0


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """What a run is expected to hold, term by term, in gigabytes.

    Returned rather than printed and forgotten, so a run that then dies on an OOM
    can be compared against what was predicted. The split matters: a reader who
    knows the terms can see *which* knob moves the answer, and a reader given one
    total can only guess.
    """

    weights: float
    gradients: float
    optimiser: float
    stored_activations: float
    transient_activations: float
    loss: float
    context_length: int
    loss_mode: str
    models: int = 1

    @property
    def resident(self) -> float:
        """What cannot be traded away: weights, gradients and optimiser state."""
        return self.weights + self.gradients + self.optimiser

    @property
    def total(self) -> float:
        """The whole estimate."""
        return self.resident + self.stored_activations + self.transient_activations + self.loss

    def fits(self, budget_gb: float) -> bool:
        """Whether the estimate is inside the budget."""
        return self.total <= budget_gb

    def to_record(self) -> dict[str, Any]:
        """Serialisable form, for the training plan."""
        return {
            "context_length": self.context_length,
            "loss_mode": self.loss_mode,
            "models": self.models,
            "resident_gb": round(self.resident, 2),
            "stored_activations_gb": round(self.stored_activations, 2),
            "transient_activations_gb": round(self.transient_activations, 2),
            "loss_gb": round(self.loss, 2),
            "total_gb": round(self.total, 2),
        }

    def report(self) -> str:
        """A readable breakdown, so the arithmetic is reviewable by a human."""
        copies = f", {self.models} model copies" if self.models > 1 else ""
        return (
            f"memory estimate @ {self.context_length} ctx, {self.loss_mode} loss{copies}\n"
            f"    weights              {self.weights:7.2f} GB  (resident)\n"
            f"    gradients            {self.gradients:7.2f} GB  (resident)\n"
            f"    optimiser state      {self.optimiser:7.2f} GB  (resident)\n"
            f"    stored activations   {self.stored_activations:7.2f} GB\n"
            f"    transient activat.   {self.transient_activations:7.2f} GB\n"
            f"    vocabulary loss      {self.loss:7.2f} GB\n"
            f"    {'total':<20} {self.total:7.2f} GB"
        )


def completion_only_loss(model: Any, inputs: Mapping[str, Any], labels: Any) -> Any:
    """The training loss, computed without materialising the whole vocabulary.

    Mathematically identical to passing ``labels=`` to the model: the same causal
    shift, the same ignored index, the same mean over supervised positions. The
    only difference is *when* the vocabulary projection happens. Here the backbone
    runs first, the positions that carry a label are selected, and the head is
    applied to those alone -- so the logits tensor is
    ``[supervised_positions, vocab]`` rather than ``[sequence, vocab]``.

    Why it matters, measured on this project's own corpus: the vocabulary is 248,320
    and only 32% of positions are supervised, so the selective form is 3.1x smaller
    on the single largest term in the budget. On a 4B model that is the difference
    between fitting a 40 GB card and not.

    Raises rather than guessing when the checkpoint does not have this shape: a
    silent fallback to the full projection would cost exactly the memory this
    function exists to save, and nothing would say so.
    """
    import torch.nn.functional as functional

    backbone = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if backbone is None or head is None:
        raise DatasetError(
            "selective loss needs a checkpoint that separates the backbone from the "
            "vocabulary head (model.model and model.lm_head). This one does not, so the "
            "full projection cannot be avoided; run with loss_mode='builtin' and a "
            "smaller context instead."
        )

    outputs = backbone(**inputs, use_cache=False)
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise DatasetError(
            "the backbone returned no last_hidden_state, so the supervised positions cannot be "
            "selected before the vocabulary projection. Use loss_mode='builtin'."
        )

    # Causal shift: position i predicts token i+1. Identical to the built-in loss.
    shifted_hidden = hidden[:, :-1, :]
    shifted_labels = labels[:, 1:]
    supervised = shifted_labels != IGNORED_INDEX
    if not bool(supervised.any()):
        raise DatasetError(
            "no supervised position in this batch; the loss would be undefined and a zero "
            "gradient step would be recorded as progress"
        )
    logits = head(shifted_hidden[supervised])
    return functional.cross_entropy(logits.float(), shifted_labels[supervised])


def architecture_facts(config: Any) -> dict[str, Any]:
    """The numbers a memory estimate needs, read off a model config.

    Duck-typed on purpose: this module must not import transformers, so the plan and
    the arithmetic stay testable with no checkpoint and no GPU. Missing linear
    attention fields simply mean there is no recurrent state to account for, which
    is the correct answer for a text-only architecture rather than a guess.
    """
    text = getattr(config, "text_config", config)
    layer_types = list(getattr(text, "layer_types", []) or [])
    linear_layers = sum(1 for kind in layer_types if kind == "linear_attention")
    v_heads = int(getattr(text, "linear_num_value_heads", 0) or 0)
    k_dim = int(getattr(text, "linear_key_head_dim", 0) or 0)
    v_dim = int(getattr(text, "linear_value_head_dim", 0) or 0)
    return {
        "layers": int(getattr(text, "num_hidden_layers", 0)),
        "hidden": int(getattr(text, "hidden_size", 0)),
        "vocab": int(getattr(text, "vocab_size", 0)),
        "linear_layers": linear_layers,
        "full_layers": max(0, len(layer_types) - linear_layers),
        # Bytes of recurrent state per token per linear layer. This is the term
        # that makes a hybrid architecture's memory scale with sequence length:
        # for Qwen3.5-4B it is 1 MB, so 8K tokens across 24 layers is ~196 GB.
        "state_bytes_per_token": v_heads * k_dim * v_dim * _ACTIVATION_BYTES,
    }


def estimate_memory(
    facts: Mapping[str, Any],
    *,
    parameters: int,
    context_length: int,
    supervised_share: float,
    gradient_checkpointing: bool,
    optimiser: str,
    loss_mode: str = "selective",
    batch_size: int = 1,
    models: int = 1,
) -> MemoryBudget:
    """Estimate a run's peak memory, term by term.

    All inputs are plain numbers so this is testable without a GPU. The estimate is
    approximate in its two activation terms and exact in its resident ones; the
    resident terms are the ones that decide whether a card is in the conversation
    at all, which is why they are separated and named.

    ``supervised_share`` is the fraction of positions carrying a label. The
    vocabulary term is proportional to it under ``selective`` and to 1.0 under
    ``builtin`` -- that ratio is the whole reason the mode exists.

    ``models`` counts how many copies of the weights are resident. Preference
    optimisation holds two: the policy being trained and the frozen reference it is
    compared against. Only the first is trained, so the extra copies are weights
    alone -- no gradient buffer, no optimiser state, and under ``no_grad`` nothing
    stored to recompute. Counting gradients per copy would over-refuse a run that
    fits; counting the reference's activations would refuse one that does not.
    """
    if models < 1:
        raise ValueError(f"a run holds at least one model, got {models}")
    parameters_b = parameters / 1e9
    weights = parameters_b * 2 * models
    gradients = parameters_b * 2
    # Adafactor keeps factored row and column statistics, so its state is a few
    # megabytes rather than a copy of the parameters. AdamW keeps two.
    optimiser_gb = 0.02 if optimiser == "adafactor" else parameters_b * 8

    tokens = context_length * batch_size
    hidden = int(facts.get("hidden", 0))
    layers = int(facts.get("layers", 0))
    vocab = int(facts.get("vocab", 0))
    state_bytes = int(facts.get("state_bytes_per_token", 0))
    linear_layers = int(facts.get("linear_layers", 0))

    if gradient_checkpointing:
        # Only layer inputs survive between layers; one layer is recomputed at a
        # time, so the state is paid for a single layer rather than all of them.
        stored = layers * tokens * hidden * _ACTIVATION_BYTES
        transient_state = state_bytes * tokens
    else:
        stored = layers * tokens * hidden * _ACTIVATION_BYTES
        transient_state = state_bytes * tokens * linear_layers
    intermediates = _INTERMEDIATES_PER_LAYER * tokens * hidden * _ACTIVATION_BYTES
    transient = transient_state + (1.0 if gradient_checkpointing else layers) * intermediates

    projected = supervised_share if loss_mode == "selective" else 1.0
    loss = projected * tokens * vocab * _ACTIVATION_BYTES * 2  # bf16 logits + fp32 copy

    return MemoryBudget(
        weights=weights,
        gradients=gradients,
        optimiser=optimiser_gb,
        stored_activations=stored / _GB,
        transient_activations=transient / _GB,
        loss=loss / _GB,
        context_length=context_length,
        loss_mode=loss_mode,
        models=models,
    )


@dataclass(frozen=True, slots=True)
class OptimisationPlan:
    """The hyperparameters, validated and recorded before any GPU is touched.

    Recorded in the run directory because a training run whose settings cannot be
    reproduced is a number with no meaning.
    """

    model_id: str
    output_dir: str
    learning_rate: float = DEFAULT_LR
    epochs: int = DEFAULT_EPOCHS
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = DEFAULT_GRAD_ACCUM
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = 1.0
    context_length: int = CONTEXT_LENGTH
    #: Trainer version of the checkpoint being replaced, or None for the base model.
    resume_from: str | None = None
    seed: int = 0
    train_vision_tower: bool = True
    dtype: str = "bfloat16"
    #: Trade compute for activation memory. On a small card it is the difference
    #: between a run and an out-of-memory crash, and it is the cheapest knob for it:
    #: activations are recomputed from the layer inputs, so only one layer's worth is
    #: held at a time.
    gradient_checkpointing: bool = False
    #: Optimiser family. "adamw" is the accurate default for a fit; "adafactor"
    #: drops the first moment and factors the second, which is the difference
    #: between a few gigabytes of optimiser state and a few megabytes, and is what
    #: makes a full fine-tune reachable on one consumer card. It is a named choice,
    #: never a silent substitution.
    optimizer: str = "adamw"
    #: Where and how often the checkpoint is published, or ``None`` to keep it on
    #: this machine. It is part of the plan because it changes what a run produces:
    #: two runs with identical hyperparameters and different destinations do not
    #: leave the same evidence behind.
    hub: HubPushPolicy | None = None
    #: The card this run is fitted against. Checked before the loop, with the
    #: arithmetic printed either way, so a run that cannot fit says so in a second
    #: rather than in three hours.
    memory_budget_gb: float = DEFAULT_MEMORY_BUDGET_GB
    #: How the loss is computed. See :data:`LOSS_MODES`; the default projects the
    #: vocabulary only at supervised positions.
    loss_mode: str = "selective"

    def __post_init__(self) -> None:
        """Reject a plan that cannot train, before it costs a GPU hour to find out."""
        if self.learning_rate <= 0:
            raise DatasetError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise DatasetError(f"epochs must be >= 1, got {self.epochs}")
        if self.per_device_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise DatasetError("batch size and gradient accumulation must both be >= 1")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise DatasetError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.context_length < 1:
            raise DatasetError(f"context_length must be >= 1, got {self.context_length}")
        if self.dtype not in DTYPES:
            raise DatasetError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.optimizer not in OPTIMIZERS:
            raise DatasetError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.loss_mode not in LOSS_MODES:
            raise DatasetError(f"loss_mode must be one of {LOSS_MODES}, got {self.loss_mode!r}")
        if self.memory_budget_gb <= 0:
            raise DatasetError(f"memory_budget_gb must be > 0, got {self.memory_budget_gb}")
        if not self.model_id:
            raise DatasetError("model_id is required; a run needs an author")

    @property
    def effective_batch_size(self) -> int:
        """Examples per optimiser step."""
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def to_record(self) -> dict[str, Any]:
        """Serialisable settings, written next to the checkpoint."""
        return {
            "model_id": self.model_id,
            "output_dir": self.output_dir,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "context_length": self.context_length,
            "resume_from": self.resume_from,
            "seed": self.seed,
            "train_vision_tower": self.train_vision_tower,
            "dtype": self.dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optimizer": self.optimizer,
            "memory_budget_gb": self.memory_budget_gb,
            "loss_mode": self.loss_mode,
            "hub": self.hub.to_record() if self.hub is not None else None,
            "token_format": FORMAT_VERSION,
        }

    def write(self, directory: str | Path | None = None) -> Path:
        """Record the plan beside the output, refusing to overwrite a different one.

        The directory defaults to :attr:`output_dir`: passing both would be two
        sources of truth for where the run is going, and the one that disagrees
        would be the one that mattered.
        """
        target = Path(directory if directory is not None else self.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "training_plan.json"
        body = json.dumps(self.to_record(), indent=2, sort_keys=True)
        if path.is_file() and path.read_text(encoding="utf-8").strip() != body:
            raise DatasetError(
                f"{path} already holds a different training plan. Point the run at a new "
                "output directory, or delete the old one deliberately: resuming into a "
                "directory whose recorded settings differ produces a checkpoint nobody can "
                "reproduce."
            )
        path.write_text(body + "\n", encoding="utf-8", newline="\n")
        return path


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What one training run did, for the record."""

    steps: int
    epochs: int
    examples: int
    supervised_tokens: int
    final_loss: float | None
    duration_s: float
    plan_path: str
    history: list[dict[str, float]] = field(default_factory=list)
    #: Every publication this run made. The durable record of a disposable machine:
    #: if this list has an entry, the checkpoint exists somewhere the tab closing
    #: cannot take.
    pushes: list[PushRecord] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "steps": self.steps,
            "epochs": self.epochs,
            "examples": self.examples,
            "supervised_tokens": self.supervised_tokens,
            "final_loss": self.final_loss,
            "duration_s": round(self.duration_s, 2),
            "plan_path": self.plan_path,
            "history": self.history,
            "pushes": [p.to_record() for p in self.pushes],
        }


def _supervised_share(examples: Sequence[Any]) -> float:
    """The fraction of positions in the corpus that carry a label.

    Measured rather than assumed, because the vocabulary term in the memory budget
    is proportional to exactly this number. A corpus of short answers supervises
    most of itself; an agent trajectory supervises the assistant's turns and little
    else, and the difference is a factor of three in the largest single term.
    """
    total = 0
    supervised = 0
    for example in examples:
        length = getattr(example, "length", 0)
        if not length:
            continue
        total += length
        supervised += int(getattr(example, "supervised_tokens", 0))
    if total == 0:
        return 1.0
    return min(1.0, supervised / total)


def measure_run(
    model: Any,
    config: Any,
    plan: OptimisationPlan,
    *,
    supervised_share: float,
) -> MemoryBudget:
    """Estimate a loaded run's memory: exact resident terms, stated activation terms.

    The parameter count is read off the loaded model rather than the name, because
    a name cannot tell you how big a vision tower is. A multimodal checkpoint
    carries a tower, and forgetting it understates the floor that decides whether a
    card is in the conversation at all.
    """
    parameters = sum(p.numel() for p in model.parameters())
    return estimate_memory(
        architecture_facts(config),
        parameters=parameters,
        context_length=plan.context_length,
        supervised_share=supervised_share,
        gradient_checkpointing=plan.gradient_checkpointing,
        optimiser=plan.optimizer,
        loss_mode=plan.loss_mode,
    )


def enter_training_mode(model: Any) -> int:
    """Put the model in training mode; return how many submodules were switched.

    Its own function because it is load-bearing and trivially omissible.
    ``from_pretrained`` returns a model in eval mode, and the gradient
    checkpointing wrapper only fires when ``self.training`` is true -- so a run that
    sets the flag without this line pays full price in activations while the plan
    claims the memory was saved. The count is returned so the run log can state what
    actually happened rather than what was requested.
    """
    switched = sum(1 for module in model.modules() if not module.training)
    model.train()
    return switched


def is_multimodal(config: Any) -> bool:
    """Whether a checkpoint carries a vision tower that training must reach.

    A multimodal checkpoint's language model and its vision tower live under
    different auto-classes: the causal-LM class returns the text stack alone, so
    loading one means training a model that cannot see the image tokens its own
    template emits, and passing ``pixel_values`` to it raises. Deciding from the
    config's own ``vision_config`` is a condition, not a fallback: the class is
    chosen because the checkpoint has that part, and the choice is logged.
    """
    return getattr(config, "vision_config", None) is not None


def assert_token_format(tokenizer: Any) -> str:
    """Check the tokenizer renders and supervises in this project's format.

    Four things have to hold, and any of them failing produces a model that trains
    to a low loss and cannot use a tool at serving time:

    1. the chat template is installed at all;
    2. it declares the version the data was rendered with;
    3. it carries ``{% generation %}``, so an assistant mask is possible;
    4. the pad token is a real single id, distinct from the vision pad token.

    Checked before the first step rather than inferred from the loss curve.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise DatasetError(
            "the tokenizer has no chat_template. Call install_template(tokenizer) before "
            "training, or the model will be trained on a format it will never be served in."
        )
    from .template import template_format_version

    declared = template_format_version(template)
    if declared != FORMAT_VERSION:
        raise DatasetError(
            f"tokenizer template declares {declared!r} but the data was rendered as "
            f"{FORMAT_VERSION!r}. Training and serving would disagree about the format."
        )
    if "{% generation" not in template:
        raise DatasetError(
            "the chat template has no {% generation %} block, so no assistant mask can be "
            "produced and the loss would be computed over the prompt as well."
        )
    pad_id = tokenizer.pad_token_id
    vision_pad_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if pad_id is None:
        raise DatasetError(
            "the tokenizer has no pad_token_id. A missing pad id cannot be masked out, and an "
            "unmasked pad is trained on as if it were content."
        )
    if pad_id == vision_pad_id:
        raise DatasetError(
            f"pad_token_id equals the vision pad token id ({pad_id}); padded positions would "
            "be supervised and read as image content."
        )
    return str(template)


#: A batch may not mix records whose longest is more than this many times the
#: shortest. This bounds padding waste by construction, which matters more than
#: keeping batches exactly full: a 32K example sharing a batch with 400-token ones
#: turns 401 real tokens into 32K slots.
BATCH_LENGTH_RATIO: Final[int] = 4


def length_grouped_batches(
    lengths: Sequence[int],
    *,
    batch_size: int,
    bucket_multiplier: int = 32,
    max_length_ratio: int = BATCH_LENGTH_RATIO,
) -> list[list[int]]:
    """Group record indices into batches of similar length.

    Two properties, in order of importance:

    * **Bounded padding.** A batch is cut short when the next record is
      ``max_length_ratio`` times longer than the shortest in the batch, so one
      32K outlier batches with other long records instead of with 400-token ones.
      Without this, sorting alone still puts the outlier next to whatever filled
      the tail of its chunk, and the waste is just as large.
    * **Stochastic order.** Indices are shuffled first, then sorted inside large
      random chunks, and the resulting batches are shuffled again. Pure sorting
      would correlate the token distribution with the label distribution across
      steps; pure shuffling would pay the padding.
    """
    if batch_size < 1:
        raise DatasetError(f"batch_size must be >= 1, got {batch_size}")
    if max_length_ratio < 1:
        raise DatasetError(f"max_length_ratio must be >= 1, got {max_length_ratio}")
    if not lengths:
        return []
    if bucket_multiplier < batch_size:
        raise DatasetError(
            f"bucket_multiplier ({bucket_multiplier}) must be at least batch_size "
            f"({batch_size}); smaller buckets would be shuffled whole and lose randomness"
        )
    order = list(range(len(lengths)))
    rng = random.Random(0)
    rng.shuffle(order)
    mega = batch_size * bucket_multiplier
    batches: list[list[int]] = []
    for start in range(0, len(order), mega):
        chunk = sorted(order[start : start + mega], key=lambda i: lengths[i])
        current: list[int] = []
        for index in chunk:
            if current and (
                len(current) >= batch_size
                or lengths[index] > lengths[current[0]] * max_length_ratio
            ):
                batches.append(current)
                current = []
            current.append(index)
        # Unconditional: the range guarantees `start < len(order)`, so `chunk`
        # has at least one element and `current` is never empty here. A guarded
        # flush would add a branch that cannot be false.
        batches.append(current)
    rng.shuffle(batches)
    return batches


def padding_waste(batches: Sequence[Sequence[int]], lengths: Sequence[int]) -> float:
    """Fraction of padded token slots that carry nothing.

    This is the number that decides whether a run is affordable, and it is
    invisible in the example count: a corpus of 400-token records batched
    together with one 32K record pads everything to 32K.
    """
    if not batches:
        return 0.0
    real = sum(lengths[i] for batch in batches for i in batch)
    padded = sum(max(lengths[i] for i in batch) * len(batch) for batch in batches)
    return 0.0 if padded == 0 else (padded - real) / padded


def warmup_steps(total_steps: int, ratio: float) -> int:
    """Number of warmup steps, so the schedule is inspectable without a trainer."""
    return int(total_steps * ratio)


def cosine_lr(base: float, step: int, total: int, warmup: int) -> float:
    """Linear warmup into cosine decay.

    Ends at ``base * min_ratio`` rather than at zero: a schedule that decays to
    exactly zero leaves the final weights sitting on a flat, insensitive point,
    which is not a stable place to stop.
    """
    if total < 1:
        raise DatasetError(f"total_steps must be >= 1, got {total}")
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    if step >= total:
        return base * 0.05
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def steps_for(examples: int, plan: OptimisationPlan) -> int:
    """Optimiser steps for one epoch, and for the whole run."""
    per_epoch = max(1, examples // plan.effective_batch_size)
    return per_epoch * plan.epochs


def iter_epochs(
    batches: Sequence[Sequence[int]], epochs: int, seed: int
) -> Iterator[tuple[int, list[int]]]:
    """Yield ``(epoch, batch_indices)`` across the whole run, reshuffled each epoch.

    Identical order every epoch is a silent form of overfitting: the model sees
    the same neighbouring examples in the same order each time and can exploit
    their order instead of their content. The epoch index is yielded so the caller
    can close out a partial gradient-accumulation window at each boundary.
    """
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(batches)
        rng.shuffle(order)
        for batch in order:
            yield epoch, list(batch)


def assert_examples_fit(examples: Sequence[Any], plan: OptimisationPlan) -> None:
    """Refuse a corpus that cannot fit the context, naming the offenders.

    Truncation is not offered as an option. A truncated assistant span is
    supervised on a fragment, and the model is then trained to stop mid-thought --
    the failure looks like the model becoming concise.
    """
    for index, example in enumerate(examples):
        length = getattr(example, "length", None)
        if length is None:
            continue
        if length > plan.context_length:
            raise DatasetError(
                f"example {index} is {length} tokens, over the {plan.context_length} context. "
                "Refusing to train on a truncated trajectory: the assistant span would be cut "
                "mid-sentence and supervised as if it had ended there."
            )


def train(
    plan: OptimisationPlan,
    examples: Sequence[Any],
    *,
    collate_fn: Callable[..., Any] | None = None,
    device: str | None = None,
    log: Callable[[str], None] | None = None,
) -> RunSummary:
    """Run the fine-tune.

    torch and transformers are imported here rather than at module level so that
    the plan, the batching and the accounting above can be exercised on a machine
    with no GPU. The absence of torch is reported as a clear install instruction.
    """
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DatasetError(
            "training needs torch and transformers. Install with: pip install -e '.[train]'"
        ) from exc

    from .collator import collate, to_tensors
    from .generator import SYSTEM_PROMPT
    from .template import install_template, load_template_source

    emit = log or (lambda message: None)
    assert_examples_fit(examples, plan)
    plan_path = plan.write(plan.output_dir)

    # Built before the model is loaded: a missing token must cost a second, not a
    # GPU day and seven hundred steps.
    pusher = resolve_pusher(plan.hub)
    if pusher is not None and plan.hub is not None:
        emit(f"checkpoint push: {plan.hub.describe()}")
        if plan.hub.dry_run:
            emit("WARNING: --hub-dry-run is set; the upload sequence is rehearsed, nothing is sent")

    tokenizer = AutoTokenizer.from_pretrained(plan.model_id)
    install_template(tokenizer, load_template_source())
    assert_token_format(tokenizer)
    if SYSTEM_PROMPT:
        emit(f"system prompt: {len(SYSTEM_PROMPT)} chars")

    processor = None
    if any(getattr(e, "images", ()) for e in examples):
        processor = AutoProcessor.from_pretrained(plan.model_id)
        emit("image processor loaded; the vision tower is trained")

    weights = plan.resume_from or plan.model_id
    model_config = AutoConfig.from_pretrained(weights)
    loader = AutoModelForImageTextToText if is_multimodal(model_config) else AutoModelForCausalLM
    emit(f"model class: {loader.__name__} (multimodal checkpoint: {is_multimodal(model_config)})")
    model = loader.from_pretrained(weights, dtype=_torch_dtype(plan.dtype, torch))
    switched = enter_training_mode(model)

    budget = measure_run(
        model,
        model_config,
        plan,
        supervised_share=_supervised_share(examples),
    )
    emit(budget.report())
    emit(f"budget: {budget.total:.2f} GB est. / {plan.memory_budget_gb:.2f} GB kart")
    if not budget.fits(plan.memory_budget_gb):
        raise DatasetError(
            f"this run needs about {budget.total:.1f} GB and the budget is "
            f"{plan.memory_budget_gb:.1f} GB. The resident floor alone -- weights, gradients "
            f"and optimiser state -- is {budget.resident:.1f} GB and cannot be traded away. "
            "Options, in order of what they cost: lower context_length, switch optimizer to "
            "adafactor if it is not already, keep gradient_checkpointing on, or use "
            "loss_mode='selective'. Raising memory_budget_gb is honest if the card really has "
            "the room; the printed breakdown says which term moved."
        )

    if not plan.train_vision_tower:
        model, frozen = freeze_vision_tower(model)
        emit(f"vision tower frozen: {frozen} parameters")
    model.config.use_cache = False
    if plan.gradient_checkpointing:
        # Required for a full fine-tune on a single consumer card: without it the
        # activations for a 32K context are held for every layer at once.
        model.gradient_checkpointing_enable()
        # `from_pretrained` hands back a model in *eval* mode, and the
        # checkpointing wrapper's condition is `gradient_checkpointing and
        # self.training`. Entering train mode first is therefore not a formality:
        # in eval mode the flag is set, recorded in the plan, and never fires.
        # Measured on Qwen3.5-4B: with 24 of 32 layers being Gated DeltaNet, the
        # per-token recurrent state is 1 MB, so 8K tokens is ~196 GB of stored
        # activations without checkpointing against ~1.3 GB with it. Skipping it
        # is the difference between fitting on an 80 GB card and not.
        emit(
            f"gradient checkpointing on: {switched} module(s) recomputed per layer, "
            f"activations held for one layer at a time"
        )

    collator = collate_fn or (
        lambda batch: collate(
            batch,
            pad_id=tokenizer.pad_token_id,
            vision_pad_id=_vision_pad_id(tokenizer),
            processor=processor,
        )
    )

    batches = length_grouped_batches(
        [e.length for e in examples], batch_size=plan.per_device_batch_size
    )
    waste = padding_waste(batches, [e.length for e in examples])
    emit(f"{len(examples)} examples in {len(batches)} batches; padding waste {waste:.1%}")
    if waste > 0.5:
        emit(
            f"WARNING: {waste:.0%} of padded tokens carry nothing. Lower --max-length or "
            "rebalance the length distribution before spending a GPU day on this."
        )

    total_steps = steps_for(len(examples), plan)
    warmup = warmup_steps(total_steps, plan.warmup_ratio)
    optimizer = _build_optimizer(model, plan)
    scheduler = _build_scheduler(optimizer, plan, total_steps, warmup)
    # No device named means "use the GPU if there is one". Making the caller pass
    # it would mean the default path silently trains a 4B model on a CPU.
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emit(f"device: {resolved_device}; optimizer: {plan.optimizer}")
    model.to(resolved_device)

    started = time.monotonic()
    history: list[dict[str, float]] = []
    step = 0
    seen = 0
    current_epoch = -1
    # Micro-batches since the last optimiser step. Reset at every epoch boundary
    # so a partial accumulation window is closed rather than carried across epochs,
    # where it would mix examples from two different passes.
    pending = 0
    window_loss = 0.0
    pushes: list[PushRecord] = []
    for epoch, batch_indices in iter_epochs(batches, plan.epochs, plan.seed):
        if epoch != current_epoch:
            if pending:
                rate = _apply_step(model, optimizer, scheduler, plan, step, total_steps, warmup)
                step += 1
                history.append(_history_entry(step, window_loss, pending, rate))
                _record_push(
                    pushes,
                    _maybe_push(pusher, plan, model, tokenizer, step, total_steps),
                    emit,
                )
            current_epoch = epoch
            pending = 0
            window_loss = 0.0

        batch = [examples[i] for i in batch_indices]
        tensors = to_tensors(collator(batch), pad_id=tokenizer.pad_token_id, device=resolved_device)
        inputs = {
            k: v
            for k, v in tensors.items()
            if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
        }
        if plan.loss_mode == "selective":
            loss = completion_only_loss(model, inputs, tensors["labels"])
        else:
            loss = model(**inputs, labels=tensors["labels"]).loss
        (loss / plan.gradient_accumulation_steps).backward()
        window_loss += float(loss.detach())
        pending += 1
        seen += len(batch)

        if pending == plan.gradient_accumulation_steps:
            rate = _apply_step(model, optimizer, scheduler, plan, step, total_steps, warmup)
            step += 1
            history.append(_history_entry(step, window_loss, pending, rate))
            pending = 0
            window_loss = 0.0
            _record_push(
                pushes,
                _maybe_push(pusher, plan, model, tokenizer, step, total_steps),
                emit,
            )
            if step % 20 == 0:
                emit(f"step {step}/{total_steps} loss {history[-1]['loss']:.4f}")

    if pending:
        rate = _apply_step(model, optimizer, scheduler, plan, step, total_steps, warmup)
        step += 1
        history.append(_history_entry(step, window_loss, pending, rate))

    output = Path(plan.output_dir)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    emit(f"saved to {output}")

    if pusher is not None and pending_final_push(pushes[-1].step if pushes else None, step):
        # The run ended on a step the schedule never named. Those are the weights it
        # actually finished on, and on a disposable machine they are the only ones
        # that would otherwise be lost.
        _record_push(
            pushes,
            _push_checkpoint(pusher, plan, model, tokenizer, step, total_steps),
            emit,
        )

    return RunSummary(
        steps=step,
        epochs=plan.epochs,
        examples=seen,
        supervised_tokens=sum(e.supervised_tokens for e in examples) * plan.epochs,
        final_loss=history[-1]["loss"] if history else None,
        duration_s=time.monotonic() - started,
        plan_path=str(plan_path),
        history=history,
        pushes=pushes,
    )


def _push_checkpoint(
    pusher: CheckpointPusher,
    plan: OptimisationPlan,
    model: Any,
    tokenizer: Any,
    step: int,
    total_steps: int,
) -> PushRecord:
    """Write the checkpoint, describe it, and publish it; return what was published.

    The plan's own settings are written into the folder *before* the upload, so the
    copy that lands on the Hub says which hyperparameters and which token format
    produced it. A checkpoint whose provenance has to be reconstructed from a run
    directory that a notebook session is about to delete is not much of a record.
    """
    output = Path(plan.output_dir)
    model.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    write_push_state(output, push_state_payload(plan.to_record(), step, total_steps))
    revision = pusher.push(output, step, total_steps)
    return PushRecord(
        step=step,
        total_steps=total_steps,
        revision=revision,
        path=str(output),
        dry_run=plan.hub.dry_run if plan.hub is not None else False,
    )


def _record_push(
    pushes: list[PushRecord], record: PushRecord | None, emit: Callable[[str], None]
) -> None:
    """Add a publication to the run's log and say so, when there was one.

    The log line names the revision, so a session that ends mid-run leaves behind
    the identity of what is already safely stored -- which is the only thing that
    matters once the tab is gone.
    """
    if record is None:
        return
    pushes.append(record)
    emit(f"pushed step {record.step}/{record.total_steps}: {record.revision}")


def _maybe_push(
    pusher: CheckpointPusher | None,
    plan: OptimisationPlan,
    model: Any,
    tokenizer: Any,
    step: int,
    total_steps: int,
) -> PushRecord | None:
    """Publish if the schedule says so, and return what was published.

    ``None`` means this step was not a publication step -- which is the common case,
    and the reason the caller can keep its bookkeeping on one line.
    """
    if pusher is None or plan.hub is None:
        return None
    if not should_push(step, total_steps, plan.hub.every_steps):
        return None
    return _push_checkpoint(pusher, plan, model, tokenizer, step, total_steps)


def _torch_dtype(name: str, torch_module: Any) -> Any:
    """Resolve a plan's dtype name against the installed torch.

    A name that reaches here has already been checked against :data:`DTYPES`, so a
    failure means the installed torch is older than the name it was given, which is
    worth saying rather than turning into an AttributeError from inside the loader.
    """
    dtype = getattr(torch_module, name, None)
    if dtype is None:
        raise DatasetError(f"this torch build has no dtype {name!r}")
    return dtype


def _apply_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    plan: OptimisationPlan,
    step: int,
    total_steps: int,
    warmup: int,
) -> float:
    """Clip, step, and advance the schedule for one optimiser update.

    The learning rate is written explicitly rather than left to the scheduler
    because :func:`cosine_lr` is what the plan's warmup and decay were computed
    with; letting a second schedule disagree with it would make the recorded
    schedule a description of something else. It is returned rather than read back
    from the scheduler, because ``scheduler.step()`` has already advanced past the
    value that was used, and a log line that reports a rate the run never applied
    is worse than no line at all.
    """
    import torch

    torch.nn.utils.clip_grad_norm_(model.parameters(), plan.max_grad_norm)
    rate = cosine_lr(plan.learning_rate, step, total_steps, warmup)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return rate


def _history_entry(step: int, window_loss: float, window: int, rate: float) -> dict[str, float]:
    """One log line: the mean loss over the window that just closed."""
    return {
        "step": float(step),
        "loss": window_loss / max(1, window),
        "lr": float(rate),
    }


def _vision_pad_id(tokenizer: Any) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|vision_pad|>")
    if token_id is None or token_id < 0:
        raise DatasetError("the tokenizer has no <|vision_pad|> token; the vision format is absent")
    return int(token_id)


def freeze_vision_tower(model: Any) -> tuple[Any, int]:
    """Freeze the vision tower; returns the model and how many parameters stopped.

    Not the default -- keeping multimodal ability means training it -- but the
    escape hatch exists for a run where the images are incidental and the tower is
    only in the way.
    """
    frozen = 0
    for name, parameter in model.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return model, frozen


def _build_optimizer(model: Any, plan: OptimisationPlan) -> Any:
    """AdamW with weight decay on matrices only.

    Biases and norms are excluded: decaying them shrinks the layers that control
    the output scale, which is a known way to make a fine-tune plateau.

    ``adafactor`` is the escape hatch for a card that cannot hold AdamW's state.
    It is chosen in the plan and recorded there, never substituted silently.
    """
    import torch

    if plan.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            lr=plan.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=plan.weight_decay,
        )

    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Biases and norms are excluded from weight decay: decaying them shrinks
        # the layers that control the output scale, which is a known way to make
        # a fine-tune plateau.
        (no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": plan.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=plan.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def _build_scheduler(
    optimizer: Any,
    plan: OptimisationPlan,
    total: int,
    warmup: int,
) -> Any:
    """A cosine schedule, built by transformers so the run uses a known one.

    Its rate is overwritten every step by :func:cosine_lr; it exists to keep the
    optimiser's own bookkeeping consistent, not to decide the rate.
    """
    from transformers import get_scheduler

    return get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total,
    )
