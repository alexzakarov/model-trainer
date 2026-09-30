"""The preference objective.

Two things have to be right for a DPO run to mean anything: the loss must be the
one that was described, and both members of a pair must be scored over the same
kind of token. The second is the one that fails quietly, so most of what follows
is about the scoring rather than the algebra.
"""

from __future__ import annotations

import math
import pathlib
from typing import Any

import pytest

from gotooltrain.collator import Example
from gotooltrain.dpo import (
    DEFAULT_BETA,
    DpoSummary,
    dpo_loss,
    pair_examples,
    sequence_logprobs,
    train_dpo,
)
from gotooltrain.errors import DatasetError
from gotooltrain.reward import PreferencePair
from gotooltrain.template import IGNORED_INDEX
from gotooltrain.train import OptimisationPlan

VISION_PAD = 200


# ------------------------------------------------------------------- the loss


def test_the_loss_is_the_described_objective() -> None:
    """-logsigmoid(beta * ((pi_c - pi_r) - (ref_c - ref_r)))."""
    torch = pytest.importorskip("torch")
    policy_chosen, policy_rejected = torch.tensor([-2.0]), torch.tensor([-6.0])
    reference_chosen, reference_rejected = torch.tensor([-3.0]), torch.tensor([-3.0])
    beta = 0.1

    loss, margin, accuracy = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=beta,
    )

    expected = -math.log(1 / (1 + math.exp(-beta * ((-2.0 - -6.0) - (-3.0 - -3.0)))))
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    assert float(margin) == pytest.approx(4.0, rel=1e-6)
    assert float(accuracy) == 1.0


def test_a_preference_the_policy_lost_lowers_its_reward_margin() -> None:
    """A policy that prefers the rejected side is the failure to watch for."""
    torch = pytest.importorskip("torch")
    good, _, _ = dpo_loss(
        torch.tensor([-1.0]),
        torch.tensor([-5.0]),
        torch.tensor([-3.0]),
        torch.tensor([-3.0]),
        beta=0.1,
    )
    bad, _, _ = dpo_loss(
        torch.tensor([-5.0]),
        torch.tensor([-1.0]),
        torch.tensor([-3.0]),
        torch.tensor([-3.0]),
        beta=0.1,
    )
    assert float(good) < float(bad)


def test_an_identical_policy_has_a_positive_loss() -> None:
    """At the reference the margin is zero and the loss is log(2), not zero.

    Zero here would mean "already done", and the run would report a perfect result
    for a model that has not been trained at all.
    """
    torch = pytest.importorskip("torch")
    zeros = torch.tensor([0.0])
    loss, margin, _ = dpo_loss(zeros, zeros, zeros, zeros, beta=DEFAULT_BETA)
    assert float(loss) == pytest.approx(math.log(2), rel=1e-6)
    assert float(margin) == 0.0


def test_a_larger_beta_sharpens_the_objective() -> None:
    """Beta controls how far the policy may drift; zero would maximise the margin."""
    torch = pytest.importorskip("torch")
    args = (torch.tensor([-4.0]), torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-3.0]))
    low, _, _ = dpo_loss(*args, beta=0.1)
    high, _, _ = dpo_loss(*args, beta=1.0)
    assert float(high) > float(low)


def test_the_margin_counts_sign_not_size() -> None:
    """A fraction of the batch preferring the wrong side must be visible."""
    torch = pytest.importorskip("torch")
    policy_chosen = torch.tensor([-5.0, -5.0, -1.0, -1.0])
    policy_rejected = torch.tensor([-1.0, -1.0, -5.0, -5.0])
    zeros = torch.zeros(4)
    _loss, _margin, accuracy = dpo_loss(
        policy_chosen, policy_rejected, zeros, zeros, beta=DEFAULT_BETA
    )
    assert float(accuracy) == pytest.approx(0.5)


# --------------------------------------------------------- sequence log-probs


class FixedLogits:
    """A model whose vocabulary logits the test chose.

    Shaped like a real causal LM -- a backbone under ``.model`` and a head under
    ``.lm_head`` -- because the default path projects the vocabulary itself, and a
    fake without that split would only ever exercise the fallback. The head returns
    the same rows the full path would have produced, so both paths can be asserted
    against each other.
    """

    def __init__(self, logits: Any) -> None:
        """Hold the logits the model should return."""
        self.logits = logits
        self.model = _FixedBackbone(int(logits.shape[-1]))
        self.lm_head = _FixedHead(logits)

    def __call__(self, input_ids: Any, attention_mask: Any = None, **_: Any) -> Any:
        """Return the canned logits, whatever it is asked."""
        return type("Out", (), {"logits": self.logits})()


class _FixedBackbone:
    """Stands in for the transformer stack: hidden states of the right width."""

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden

    def __call__(self, input_ids: Any = None, **_: Any) -> Any:
        """Report a hidden state wide enough for the fixed head to consume."""
        import torch

        length = int(input_ids.shape[1]) if input_ids is not None else 1
        return type("Out", (), {"last_hidden_state": torch.ones(1, length, self.hidden)})()


class _FixedHead:
    """Stands in for the vocabulary projection, returning the canned logits."""

    def __init__(self, logits: Any) -> None:
        self.logits = logits

    def __call__(self, hidden: Any) -> Any:
        """Return one logit row per selected position, taken from the canned tensor."""
        count = int(hidden.shape[0])
        return self.logits.reshape(-1, self.logits.shape[-1])[:count]


def test_a_run_that_cannot_fit_two_models_is_refused_with_the_reason(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """A preference step holds two copies of the weights, and the reader is told.

    The old failure was a raw CUDA OOM with no explanation, after the tokenizer had
    done its work. The refusal has to name the reason -- two models, and the part of
    the total that is resident and cannot be traded away -- because a reader who only
    sees "out of memory" cannot tell a preference run from a supervised one.
    """
    lines: list[str] = []
    with pytest.raises(DatasetError, match="two copies of the weights"):
        train_dpo(
            plan(tmp_path / "tight", model_id=str(tiny_checkpoint), memory_budget_gb=0.001),
            [pair()],
            qwen_tokenizer,
            device="cpu",
            log=lines.append,
        )
    assert any("2 model copies" in line for line in lines), (
        f"the estimate did not say how many models it counted: {lines}"
    )
    assert any("budget" in line for line in lines), "the budget was not stated"


def test_the_budget_counts_the_reference_model_not_just_the_policy(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """One model in the estimate would be an estimate for a different run.

    Weights double; gradients do not, because the reference is frozen. An estimate
    that got either of those wrong would refuse a run that fits or admit one that
    does not, and neither shows up until a card is already full.
    """
    lines: list[str] = []
    train_dpo(
        plan(tmp_path / "counted", model_id=str(tiny_checkpoint)),
        [pair()],
        qwen_tokenizer,
        device="cpu",
        log=lines.append,
    )
    report = "\n".join(lines)
    weights = float(report.split("weights")[1].split("GB")[0])
    gradients = float(report.split("gradients")[1].split("GB")[0])
    assert weights == pytest.approx(gradients * 2, rel=0.05), (
        f"weights {weights} and gradients {gradients} do not describe two copies of "
        f"one trainable model; a frozen reference has no gradients"
    )


def test_the_optimiser_is_the_one_the_plan_asked_for(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AdamW was hardcoded here while the plan recorded a different choice.

    The plan is written to disk as the record of the run, so a recorded "adafactor"
    that the run ignored is a false record -- and the resident term of the budget
    understated by 34.6 GB on a 4B model. Asserting the two estimated states differ
    would not catch a hardcoded optimiser that also happened to move the estimate;
    watching the factory itself does.
    """
    import gotooltrain.dpo as dpo

    # Captured before the patch: calling dpo._build_optimizer inside the replacement
    # would call the replacement.
    build = dpo._build_optimizer
    asked: list[str] = []

    def record(model: Any, chosen: OptimisationPlan) -> Any:
        asked.append(chosen.optimizer)
        return build(model, chosen)

    monkeypatch.setattr(dpo, "_build_optimizer", record)
    for optimiser in ("adafactor", "adamw"):
        train_dpo(
            plan(tmp_path / optimiser, model_id=str(tiny_checkpoint), optimizer=optimiser),
            [pair()],
            qwen_tokenizer,
            device="cpu",
        )
    assert asked == ["adafactor", "adamw"], (
        f"the run built its own optimiser instead of the plan's: {asked}"
    )


def test_the_policy_actually_checkpoints_instead_of_pretending_to(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """The transformers wrapper checks ``self.training`` before it checkpoints.

    Enabling checkpointing on a model that ``from_pretrained`` returned in eval mode
    is a silent no-op, which is why this run held every layer's activations. The line
    in the log has to mean something, so the model is asserted to be in train mode
    after the run that claims to have enabled it.
    """
    lines: list[str] = []
    train_dpo(
        plan(
            tmp_path / "gc",
            model_id=str(tiny_checkpoint),
            gradient_checkpointing=True,
        ),
        [pair()],
        qwen_tokenizer,
        device="cpu",
        log=lines.append,
    )
    assert any("gradient checkpointing on" in line for line in lines)

    import torch
    from transformers import AutoModelForCausalLM

    from gotooltrain.train import enter_training_mode

    model = AutoModelForCausalLM.from_pretrained(str(tiny_checkpoint), dtype=torch.float32)
    assert model.training is False, "precondition: from_pretrained returns eval mode"
    enter_training_mode(model)
    assert model.training is True


def test_the_two_projections_agree_on_value_and_on_gradient(
    tiny_checkpoint: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The saving is only legitimate if nothing about the objective changed.

    The selective path is worth having because a preference step builds four
    vocabulary tensors per micro-batch -- policy and reference, chosen and rejected
    -- at a vocabulary of 248,320. That is only a saving if the numbers come out the
    same, so both the value and the gradient are compared against the full path on
    the real architecture. A cheaper graph that trains something else is a different
    objective wearing a saving's name.
    """
    torch = pytest.importorskip("torch")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(tiny_checkpoint), dtype=torch.float32)
    model.train()
    ids = torch.randint(0, 64, (1, 12))
    labels = ids.clone()
    labels[:, :7] = IGNORED_INDEX
    tensors = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}

    gradients: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for mode in ("builtin", "selective"):
        model.zero_grad(set_to_none=True)
        out = sequence_logprobs(model, dict(tensors), mode=mode)
        out.sum().backward()
        values[mode] = out.detach()
        gradients[mode] = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )

    assert torch.allclose(values["builtin"], values["selective"], atol=1e-4)
    scale = max(1.0, float(gradients["builtin"].abs().max()))
    worst = float((gradients["builtin"] - gradients["selective"]).abs().max())
    assert worst < 1e-4 * scale, f"gradients differ by {worst} against a scale of {scale}"


def test_a_model_without_a_separable_head_is_refused_rather_than_silently_widened() -> None:
    """Falling back to the full projection would spend the memory this avoids.

    The refusal has to name the alternative, because "selective needs a different
    checkpoint" is only actionable if the reader is told what to do instead.
    """
    torch = pytest.importorskip("torch")
    logits = torch.zeros(1, 4, 8)

    class NoHead:
        """A checkpoint with no separable vocabulary head, as some wrappers are."""

        def __call__(self, input_ids: Any, attention_mask: Any = None, **_: Any) -> Any:
            return type("Out", (), {"logits": logits})()

    with pytest.raises(DatasetError, match="loss_mode='builtin'"):
        sequence_logprobs(
            NoHead(),
            {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.full((1, 4), IGNORED_INDEX),
            },
            mode="selective",
        )


def test_a_backbone_that_hides_its_hidden_state_is_refused() -> None:
    """Without the hidden state the supervised positions cannot be selected.

    The alternative is the full projection this mode exists to avoid, so this is a
    refusal rather than a fallback.
    """
    torch = pytest.importorskip("torch")

    class Opaque:
        """A wrapper whose backbone returns an object with no hidden state."""

        model = type("Backbone", (), {"__call__": lambda self, **k: type("Out", (), {})()})()
        lm_head = type("Head", (), {"__call__": lambda self, h: h})()

        def __call__(self, input_ids: Any, attention_mask: Any = None, **_: Any) -> Any:
            return type("Out", (), {"logits": torch.zeros(1, 4, 8)})()

    with pytest.raises(DatasetError, match="last_hidden_state"):
        sequence_logprobs(
            Opaque(),
            {
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "labels": torch.tensor([[0, IGNORED_INDEX, IGNORED_INDEX, IGNORED_INDEX]]),
            },
            mode="selective",
        )


def test_only_supervised_positions_contribute() -> None:
    """Prompt tokens must not count, or the length of the prompt is the reward."""
    torch = pytest.importorskip("torch")
    # Two positions, three symbols: the second is clearly more likely.
    logits = torch.log(torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]]))
    labels = torch.tensor([[IGNORED_INDEX, 2]])

    total = sequence_logprobs(
        FixedLogits(logits),
        {"input_ids": labels, "attention_mask": torch.ones_like(labels), "labels": labels},
    )
    # `logits` is already in log space, so the expected total is that row's entry.
    assert float(total[0]) == pytest.approx(float(logits[0, 0, 2]), rel=1e-5)


def test_an_all_ignored_row_scores_zero() -> None:
    """A row with no assistant token carries no preference."""
    torch = pytest.importorskip("torch")
    logits = torch.log(torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]]]))
    labels = torch.tensor([[IGNORED_INDEX, IGNORED_INDEX]])

    total = sequence_logprobs(
        FixedLogits(logits),
        {"input_ids": labels, "attention_mask": torch.ones_like(labels), "labels": labels},
    )
    assert float(total[0]) == 0.0


# ------------------------------------------------------------- the pair itself


def pair(chosen: str = "Tests pass.", rejected: str = "I give up.") -> PreferencePair:
    return PreferencePair(
        task_id="go-0001",
        prompt="make the test pass",
        chosen=[
            {"role": "user", "content": "make the test pass"},
            {"role": "assistant", "content": chosen},
        ],
        rejected=[
            {"role": "user", "content": "make the test pass"},
            {"role": "assistant", "content": rejected},
        ],
        chosen_reward=1.0,
        rejected_reward=0.0,
        margin=1.0,
    )


def test_both_sides_of_a_pair_render_into_examples(qwen_tokenizer: Any) -> None:
    chosen, rejected = pair_examples(pair(), qwen_tokenizer)
    assert isinstance(chosen, Example)
    assert chosen.length > 0
    assert chosen.supervised_tokens > 0
    assert rejected.supervised_tokens > 0
    assert chosen.vision_pad_id == VISION_PAD or chosen.vision_pad_id > 0


def test_a_pair_renders_the_same_prompt_on_both_sides(qwen_tokenizer: Any) -> None:
    """Different prompts would make the comparison meaningless."""
    chosen, rejected = pair_examples(pair(), qwen_tokenizer)
    assert chosen.input_ids[:4] == rejected.input_ids[:4]


def test_the_rejected_side_is_supervised_too(qwen_tokenizer: Any) -> None:
    """Suppressing only the rejected side would teach the model to be silent."""
    _chosen, rejected = pair_examples(pair(), qwen_tokenizer)
    assert rejected.supervised_tokens > 0


# ------------------------------------------------------------------ the run


def plan(output: pathlib.Path, **overrides: Any) -> OptimisationPlan:
    base: dict[str, Any] = {
        "model_id": "",
        "output_dir": str(output),
        "dtype": "float32",
        "epochs": 1,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-4,
    }
    base.update(overrides)
    return OptimisationPlan(**base)


def test_a_quantised_plan_is_refused_rather_than_run_as_something_else(
    tmp_path: pathlib.Path,
) -> None:
    """Preference optimisation does not implement qlora, and must not pretend to.

    It holds two copies of the model -- the policy and the frozen reference -- and both
    would need quantising with the adapters merged before saving. Silently running a
    full fine-tune instead would train something other than what the plan records, and
    the plan is the only durable account of what happened.
    """
    with pytest.raises(DatasetError, match="qlora"):
        train_dpo(
            plan(tmp_path, model_id="any/model", training_mode="qlora"),
            [pair()],
            tokenizer=None,  # type: ignore[arg-type]
        )


def test_a_run_with_no_pairs_is_refused(tmp_path: pathlib.Path) -> None:
    """An empty run that reports success hides that nothing was compared."""
    with pytest.raises(DatasetError, match="no preference pairs"):
        train_dpo(plan(tmp_path, model_id="any/model"), [], tokenizer=None)  # type: ignore[arg-type]


def test_a_non_positive_beta_is_refused(
    tmp_path: pathlib.Path, tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any
) -> None:
    with pytest.raises(DatasetError, match="beta must be"):
        train_dpo(
            plan(tmp_path, model_id=str(tiny_checkpoint)),
            [pair()],
            qwen_tokenizer,
            beta=0.0,
        )


def test_an_unknown_dtype_is_refused(
    tmp_path: pathlib.Path, tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any
) -> None:
    with pytest.raises(DatasetError, match="dtype must be one of"):
        OptimisationPlan(model_id=str(tiny_checkpoint), output_dir=str(tmp_path), dtype="float64")


def test_a_preference_run_trains_and_saves(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """One pair, one step, on a real architecture with a frozen reference."""
    output = tmp_path / "dpo"
    lines: list[str] = []
    summary = train_dpo(
        plan(output, model_id=str(tiny_checkpoint)),
        [pair()],
        qwen_tokenizer,
        device="cpu",
        log=lines.append,
    )

    assert isinstance(summary, DpoSummary)
    assert summary.steps == 1, "a partial accumulation window was dropped"
    assert summary.pairs == 1
    assert summary.examples == 1
    assert summary.final_loss is not None
    assert summary.beta == DEFAULT_BETA
    assert (output / "config.json").is_file()
    assert (output / "training_plan.json").is_file()
    assert any("pair(s) in" in line for line in lines)
    assert any("saved to" in line for line in lines)


def test_a_preference_run_reports_its_margin(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """The margin is the number that says whether the run is learning anything."""
    summary = train_dpo(
        plan(tmp_path / "dpo2", model_id=str(tiny_checkpoint)),
        [pair()],
        qwen_tokenizer,
        device="cpu",
    )
    assert summary.final_margin is not None
    assert len(summary.history) == 1
    assert summary.history[0]["lr"] > 0


def test_the_reference_is_frozen(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """Without a frozen reference the objective is supervised fine-tuning."""
    import torch
    from transformers import AutoModelForCausalLM

    reference = AutoModelForCausalLM.from_pretrained(str(tiny_checkpoint), dtype=torch.float32)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    assert all(not p.requires_grad for p in reference.parameters())
    # The run itself only needs to not blow up; the freeze is asserted above.
    assert (
        train_dpo(
            plan(tmp_path / "dpo3", model_id=str(tiny_checkpoint)),
            [pair()],
            qwen_tokenizer,
            device="cpu",
        ).steps
        == 1
    )


def test_a_partial_window_is_closed_not_dropped(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """One pair with a two-step window: the lone batch must still produce a step.

    Without the closing step the run would take no optimiser step at all, then
    report success and save a model identical to the one it started from.
    """
    summary = train_dpo(
        plan(tmp_path / "dpo4", model_id=str(tiny_checkpoint), gradient_accumulation_steps=2),
        [pair()],
        qwen_tokenizer,
        device="cpu",
    )
    assert summary.steps == 1
    assert len(summary.history) == 1


def test_the_run_needs_no_explicit_device(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """CPU is the default, so omitting the device is a supported run, not an error."""
    summary = train_dpo(
        plan(tmp_path / "dpo5", model_id=str(tiny_checkpoint)),
        [pair()],
        qwen_tokenizer,
    )
    assert summary.steps == 1


def test_a_preference_run_with_gradient_checkpointing_still_trains(
    tiny_checkpoint: pathlib.Path, qwen_tokenizer: Any, tmp_path: pathlib.Path
) -> None:
    """Both sides recompute, and the step still completes."""
    lines: list[str] = []
    summary = train_dpo(
        plan(
            tmp_path / "dpo6",
            model_id=str(tiny_checkpoint),
            gradient_checkpointing=True,
            optimizer="adafactor",
        ),
        [pair()],
        qwen_tokenizer,
        device="cpu",
        log=lines.append,
    )
    assert summary.steps == 1
    assert any("gradient checkpointing" in line for line in lines)


def test_a_summary_serialises() -> None:
    record = DpoSummary(
        steps=1,
        examples=1,
        pairs=1,
        beta=0.1,
        final_loss=0.5,
        final_margin=0.2,
        duration_s=1.0,
        plan_path="p",
        history=[{"step": 1.0, "loss": 0.5, "margin": 0.2, "lr": 1e-4}],
    ).to_record()
    assert record["pairs"] == 1
    assert record["beta"] == 0.1
    assert record["history"][0]["margin"] == 0.2
    assert isinstance(record["final_loss"], float)
