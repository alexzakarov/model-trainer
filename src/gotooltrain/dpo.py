"""Direct preference optimisation on execution reward.

The loss is implemented here rather than taken from a library for two reasons.
The first is dependency hygiene: the training stack in this project pins torch and
transformers, and a preference library would pin both again with different
bounds. The second is that the objective is short enough to read in full, and the
part that actually goes wrong is not the algebra -- it is the tokenisation of the
two members of a pair, and the reference model's frozen state. Those live here,
where they can be checked.

The objective, with ``beta`` controlling how far the policy may drift from the
reference:

    loss = -logsigmoid(beta * ((logp_c - logp_r) - (logref_c - logref_r)))

Both members of a pair are scored over their own assistant spans only. If the
chosen and rejected sides supervised different token sets -- because one is longer,
or because a tool result leaked into the labels -- the difference being optimised
is partly a difference in how much text each side has, which is not the preference.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .collator import Example, collate, to_tensors
from .errors import DatasetError
from .reward import PreferencePair
from .template import IGNORED_INDEX, SupportsChatTemplate
from .train import (
    OptimisationPlan,
    _build_optimizer,
    _supervised_share,
    _torch_dtype,
    _vision_pad_id,
    architecture_facts,
    cosine_lr,
    enter_training_mode,
    estimate_memory,
    iter_epochs,
    length_grouped_batches,
    steps_for,
    warmup_steps,
)

#: How far the policy may move from the reference. Too high and it memorises the
#: pairs; too low and nothing changes. 0.1 is the value the original work reports.
DEFAULT_BETA: Final[float] = 0.1


@dataclass(frozen=True, slots=True)
class DpoSummary:
    """What a preference run did."""

    steps: int
    examples: int
    pairs: int
    beta: float
    final_loss: float | None
    final_margin: float | None
    duration_s: float
    plan_path: str
    history: list[dict[str, float]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Serialisable summary."""
        return {
            "steps": self.steps,
            "examples": self.examples,
            "pairs": self.pairs,
            "beta": self.beta,
            "final_loss": self.final_loss,
            "final_margin": self.final_margin,
            "duration_s": self.duration_s,
            "plan_path": self.plan_path,
            "history": self.history,
        }


def pair_examples(
    pair: PreferencePair,
    tokenizer: SupportsChatTemplate,
    *,
    tool_specs: Sequence[dict[str, Any]] = (),
) -> tuple[Example, Example]:
    """Render both members of a pair into collatable examples.

    Only the assistant spans are supervised, on both sides, so the quantity the
    loss compares is the model's own text and nothing else.
    """
    from .normalize import normalize_conversation
    from .template import render_example

    vision_pad_id = _vision_pad_id(tokenizer)
    built: list[Example] = []
    for messages in (pair.chosen, pair.rejected):
        conversation = normalize_conversation([dict(m) for m in messages], list(tool_specs))
        rendered = render_example(tokenizer, conversation)
        built.append(
            Example(
                input_ids=list(rendered.input_ids),
                attention_mask=list(rendered.attention_mask),
                labels=list(rendered.labels),
                vision_pad_id=vision_pad_id,
            )
        )
    return built[0], built[1]


def sequence_logprobs(
    model: Any,
    tensors: dict[str, Any],
    *,
    mode: str = "selective",
) -> Any:
    """Sum of the log-probabilities of the supervised tokens, per sequence.

    ``labels`` carries the ignored index at every position that is not an assistant
    token, and the loss is gathered only there. Summing the whole sequence instead
    would make the number depend on how long the prompt is, and the prompt is
    identical on both sides of a pair -- the difference would then vanish into a
    quantity neither side controls.

    Under ``selective`` the vocabulary head is applied only at those positions, the
    same trick as :func:`completion_only_loss`, and for the same reason. It matters
    more here than in SFT: this runs on the policy *and* the reference, for the
    chosen *and* the rejected side, so a preference step builds four such tensors
    where a supervised one builds one. The vocabulary is 248,320 wide, so at 4096
    tokens the unprojected form is 4.07 GB per tensor before the float32 copy that
    ``log_softmax`` makes.

    Raises rather than falling back: a silent full projection would spend exactly
    the memory this avoids, and the run would die on an OOM with no explanation.
    """
    import torch

    if mode == "builtin":
        outputs = model(
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = tensors["labels"][:, 1:]
        # The ignored index is not a vocabulary entry, so it has to be masked out
        # before the gather -- and then zeroed in the result, or it would count.
        mask = (shifted_labels != IGNORED_INDEX).to(shifted_logits.dtype)
        safe_labels = shifted_labels.masked_fill(shifted_labels == IGNORED_INDEX, 0)
        gathered = _gathered_logprobs(shifted_logits, safe_labels)
        return (gathered * mask).sum(dim=-1)

    backbone = getattr(model, "model", None)
    head = getattr(model, "lm_head", None)
    if backbone is None or head is None:
        raise DatasetError(
            "selective loss needs a checkpoint that separates the backbone from the "
            "vocabulary head (model.model and model.lm_head). This one does not; a "
            "preference step builds four vocabulary tensors per micro-batch, so the "
            "full projection is not affordable. Run with loss_mode='builtin' and a "
            "much smaller context."
        )

    hidden = backbone(
        input_ids=tensors["input_ids"],
        attention_mask=tensors["attention_mask"],
        use_cache=False,
    )
    state = getattr(hidden, "last_hidden_state", None)
    if state is None:
        raise DatasetError(
            "the backbone returned no last_hidden_state, so the supervised positions "
            "cannot be selected before the vocabulary projection. Use loss_mode='builtin'."
        )

    shifted_hidden = state[:, :-1, :]
    shifted_labels = tensors["labels"][:, 1:]
    supervised = shifted_labels != IGNORED_INDEX
    if not bool(supervised.any()):
        # A side with nothing supervised sums to nothing, which is the honest
        # answer and what the full projection returns as well. Refusing here would
        # be a different contract from the one this function has always had, and the
        # degenerate pair is caught where the pair is built, not here.
        return torch.zeros(shifted_labels.shape[0], dtype=state.dtype, device=state.device)
    # One gather over the selected positions, then scatter the per-position sum back
    # to a [batch] vector so the caller keeps the shape it had before.
    per_position = _gathered_logprobs(head(shifted_hidden[supervised]), shifted_labels[supervised])
    summed = torch.zeros(shifted_labels.shape, dtype=per_position.dtype, device=per_position.device)
    summed[supervised] = per_position
    return summed.sum(dim=-1)


def _gathered_logprobs(logits: Any, labels: Any) -> Any:
    """The log-probability of each label under the logits, at the given positions."""
    import torch

    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def dpo_loss(
    policy_chosen: Any,
    policy_rejected: Any,
    reference_chosen: Any,
    reference_rejected: Any,
    *,
    beta: float,
) -> tuple[Any, Any, Any]:
    """The DPO objective, plus the two numbers that say whether it is working.

    Returns the loss, the implicit reward margin between chosen and rejected, and
    the fraction of the batch with a positive margin. The margin is what to watch:
    the loss goes down for a broken run too, but a negative margin means the policy
    prefers the rejected completions.
    """
    import torch.nn.functional as functional

    chosen = policy_chosen - reference_chosen
    rejected = policy_rejected - reference_rejected
    margin = chosen - rejected
    loss = -functional.logsigmoid(beta * margin).mean()
    accuracy = (margin > 0).to(margin.dtype).mean()
    return loss, margin.detach().mean(), accuracy.detach()


def _pair_batches(
    pairs: Sequence[tuple[Example, Example]],
    batch_size: int,
) -> list[list[int]]:
    """Group pairs by the longer side's length, so padding stays cheap."""
    lengths = [max(chosen.length, rejected.length) for chosen, rejected in pairs]
    return length_grouped_batches(lengths, batch_size=batch_size)


def _stack(examples: Sequence[Example], tokenizer: Any, *, device: str | None) -> dict[str, Any]:
    """Collate a list of examples into tensors.

    The tokenizer is `Any` because a transformers tokenizer has no single better
    annotation, and this function needs a field -- `pad_token_id` -- that the
    chat-template Protocol deliberately does not declare.
    """
    pad_id = int(tokenizer.pad_token_id)
    batch = collate(
        list(examples),
        pad_id=pad_id,
        vision_pad_id=_vision_pad_id(tokenizer),
    )
    return dict(to_tensors(batch, pad_id=pad_id, device=device))


def train_dpo(
    plan: OptimisationPlan,
    pairs: Sequence[PreferencePair],
    tokenizer: SupportsChatTemplate,
    *,
    tool_specs: Sequence[dict[str, Any]] = (),
    beta: float = DEFAULT_BETA,
    device: str | None = None,
    log: Any = None,
) -> DpoSummary:
    """Run preference optimisation against a frozen copy of the starting model.

    The reference is the checkpoint the run starts from, loaded once and never
    updated. It is not optional: without it the objective reduces to making the
    chosen side more likely, which is supervised fine-tuning with extra steps.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DatasetError(
            "preference training needs torch and transformers. Install with: "
            "pip install -e '.[train]'"
        ) from exc

    from .template import install_template, load_template_source

    if beta <= 0:
        raise DatasetError(f"beta must be > 0, got {beta}")
    if not pairs:
        raise DatasetError(
            "no preference pairs: a run with nothing to compare cannot learn a "
            "preference, and an empty run that reports success hides that."
        )

    emit = log or (lambda message: None)
    plan_path = plan.write(plan.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(plan.model_id)
    install_template(tokenizer, load_template_source())
    dtype = _torch_dtype(plan.dtype, torch)

    policy = AutoModelForCausalLM.from_pretrained(plan.model_id, dtype=dtype)
    reference = AutoModelForCausalLM.from_pretrained(plan.model_id, dtype=dtype)
    # The reference stays in eval mode: it is the frozen thing the policy is measured
    # against, so dropout in it would make the comparison depend on a seed.
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    policy.config.use_cache = False

    examples = [pair_examples(pair, tokenizer, tool_specs=tool_specs) for pair in pairs]
    rendered = [side for chosen, rejected in examples for side in (chosen, rejected)]

    # Two copies of the weights are resident, and only the policy carries gradients
    # and optimiser state. Estimated before the first step, because the alternative
    # is discovering the answer from a CUDA OOM with nothing to look at.
    budget = estimate_memory(
        architecture_facts(policy.config),
        parameters=sum(p.numel() for p in policy.parameters()),
        context_length=plan.context_length,
        supervised_share=_supervised_share(rendered),
        gradient_checkpointing=plan.gradient_checkpointing,
        optimiser=plan.optimizer,
        loss_mode=plan.loss_mode,
        models=2,
    )
    emit(budget.report())
    emit(f"budget: {budget.total:.2f} GB est. / {plan.memory_budget_gb:.2f} GB kart")
    if not budget.fits(plan.memory_budget_gb):
        raise DatasetError(
            f"a preference step needs two copies of the weights: the estimate is "
            f"{budget.total:.1f} GB against a {plan.memory_budget_gb:.0f} GB budget, and "
            f"{budget.resident:.1f} GB of that is resident and cannot be traded away. "
            f"Lower --max-length, or point the run at a card that large."
        )

    if plan.gradient_checkpointing:
        # `from_pretrained` hands back a model in *eval* mode and the transformers
        # wrapper checks `self.training` before it will checkpoint anything, so
        # enabling it on an eval-mode model is a silent no-op. That is what made a
        # preference run hold every layer's activations: ~101 GB of transient at 4096
        # tokens, before either model is counted. The reference is left in eval on
        # purpose; it runs under no_grad, so it stores nothing to recompute.
        enter_training_mode(policy)
        policy.gradient_checkpointing_enable()
        emit("gradient checkpointing on: activations recomputed per layer")

    batches = _pair_batches(examples, plan.per_device_batch_size)
    total_steps = steps_for(len(examples), plan)
    warmup = warmup_steps(total_steps, plan.warmup_ratio)
    optimizer = _build_optimizer(policy, plan)
    from transformers import get_scheduler

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total_steps,
    )
    if device:
        policy.to(device)
        reference.to(device)

    emit(f"{len(examples)} pair(s) in {len(batches)} batch(es); beta {beta}")
    started = time.monotonic()
    history: list[dict[str, float]] = []
    step = 0
    seen = 0
    pending = 0
    window_loss = 0.0
    window_margin = 0.0
    for _epoch, batch_indices in iter_epochs(batches, plan.epochs, plan.seed):
        batch = [examples[i] for i in batch_indices]
        chosen = _stack([c for c, _ in batch], tokenizer, device=device)
        rejected = _stack([r for _, r in batch], tokenizer, device=device)
        with torch.no_grad():
            reference_chosen = sequence_logprobs(reference, chosen, mode=plan.loss_mode)
            reference_rejected = sequence_logprobs(reference, rejected, mode=plan.loss_mode)
        # Outside no_grad: the policy's forward has to build a graph, and putting the
        # objective inside that block silently trains nothing while reporting a loss.
        loss, margin, _accuracy = dpo_loss(
            sequence_logprobs(policy, chosen, mode=plan.loss_mode),
            sequence_logprobs(policy, rejected, mode=plan.loss_mode),
            reference_chosen,
            reference_rejected,
            beta=beta,
        )
        (loss / plan.gradient_accumulation_steps).backward()
        window_loss += float(loss.detach())
        window_margin += float(margin)
        pending += 1
        seen += len(batch)

        if pending == plan.gradient_accumulation_steps:
            rate = cosine_lr(plan.learning_rate, step, total_steps, warmup)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), plan.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            history.append(
                {
                    "step": float(step),
                    "loss": window_loss / max(1, pending),
                    "margin": window_margin / max(1, pending),
                    "lr": float(rate),
                }
            )
            pending = 0
            window_loss = 0.0
            window_margin = 0.0

    # A partial accumulation window is closed rather than dropped. Without this a
    # run whose pair count is not a multiple of the accumulation window takes fewer
    # steps than `steps_for` promised -- and with a small corpus it takes none at
    # all, then reports success and saves an unchanged model.
    if pending:
        rate = cosine_lr(plan.learning_rate, step, total_steps, warmup)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), plan.max_grad_norm)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        history.append(
            {
                "step": float(step),
                "loss": window_loss / max(1, pending),
                "margin": window_margin / max(1, pending),
                "lr": float(rate),
            }
        )

    output = Path(plan.output_dir)
    policy.save_pretrained(str(output), safe_serialization=True)
    tokenizer.save_pretrained(str(output))
    emit(f"saved to {output}")

    return DpoSummary(
        steps=step,
        examples=seen,
        pairs=len(examples),
        beta=beta,
        final_loss=history[-1]["loss"] if history else None,
        final_margin=history[-1]["margin"] if history else None,
        duration_s=time.monotonic() - started,
        plan_path=str(plan_path),
        history=history,
    )
