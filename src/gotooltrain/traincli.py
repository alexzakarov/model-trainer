"""The training driver: a mined corpus in, a checkpoint out.

Two commands, because they take different inputs and answer different questions:

``sft``  supervised fine-tuning on a mined corpus (``--dataset``)
``dpo``  preference optimisation on execution-rewarded pairs (``--pairs``)

Both refuse an input that would produce a meaningless run rather than a small one:
a corpus with no supervised tokens, a record over the context, a preference set
with no contrast. Those are the failures that look like a successful run and cost
the most to discover later.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from .collator import examples_from_conversations
from .dataset import read_jsonl
from .dpo import DEFAULT_BETA, train_dpo
from .errors import ToolTrainError
from .gotools import GO_TOOLS
from .hub import HUB_TOKEN_ENV, HubPushPolicy
from .normalize import normalize_conversation
from .reward import PreferencePair
from .template import load_template_source
from .train import CONTEXT_LENGTH, OptimisationPlan, train
from .vision import VISION_PAD_TOKEN

DEFAULT_TOKENIZER: str = "Qwen/Qwen3.5-4B"

#: Publishing interval used when a repo is named without one. Short enough that a
#: dropped Colab session costs minutes rather than the whole run, and it is stated
#: here rather than defaulted inside the trainer so the number is visible in --help.
DEFAULT_PUSH_EVERY: int = 50


def _tokenizer(name: str) -> Any:  # noqa: ANN401 - transformers is an optional dependency
    """Load the base tokenizer with this repository's chat template installed."""
    try:
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ToolTrainError(
            "training needs transformers; install the 'train' extra: pip install -e '.[train]'"
        ) from exc
    from .template import install_template

    return install_template(
        transformers.AutoTokenizer.from_pretrained(name), load_template_source()
    )


def _hub_policy(args: argparse.Namespace) -> HubPushPolicy | None:
    """Build the publication policy, refusing a half-specified one.

    An interval without a repo is refused rather than ignored: on a disposable
    machine the operator who typed ``--hub-push-every 25`` believed the weights were
    leaving the machine, and finding that out after the tab closed is the exact loss
    this feature exists to prevent.
    """
    if not args.hub_repo_id:
        if args.hub_push_every != DEFAULT_PUSH_EVERY:
            raise ToolTrainError(
                f"--hub-push-every {args.hub_push_every} was given without --hub-repo-id, so "
                "there would be nowhere to push to. Name the repo, or drop the interval."
            )
        return None
    return HubPushPolicy(
        repo_id=args.hub_repo_id,
        every_steps=args.hub_push_every,
        token_env=args.hub_token_env,
        private=args.hub_private,
        dry_run=args.hub_dry_run,
    )


def _plan(args: argparse.Namespace) -> OptimisationPlan:
    return OptimisationPlan(
        model_id=args.model,
        output_dir=args.output,
        dtype=args.dtype,
        epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        context_length=args.max_length,
        resume_from=args.resume_from,
        seed=args.seed,
        train_vision_tower=not args.freeze_vision,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer=args.optimizer,
        hub=_hub_policy(args),
    )


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="base model or checkpoint id")
    parser.add_argument("--output", required=True, help="directory to write the checkpoint into")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="context; records longer than this are refused, never truncated",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        help="do not train the vision tower (multimodal ability is kept by default)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="recompute activations per layer; the difference between fitting and not",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=("adamw", "adafactor"),
        help="adafactor keeps no first moment; far less optimiser state",
    )
    parser.add_argument(
        "--device", default=None, help="e.g. cuda:0; the GPU is used when this is omitted"
    )
    # Publication to the Hub. Grouped because they are only meaningful together:
    # an interval without a repo is refused, and a repo without a token fails at
    # startup rather than at the first scheduled step.
    group = parser.add_argument_group("checkpoint publication")
    group.add_argument(
        "--hub-repo-id",
        default=None,
        help=(
            "Hugging Face model repo to publish to, e.g. alexzakarov/qwen3.5-4b-go. "
            "Omit to keep the checkpoint on this machine only."
        ),
    )
    group.add_argument(
        "--hub-push-every",
        type=int,
        default=DEFAULT_PUSH_EVERY,
        help=f"optimiser steps between publications (default: {DEFAULT_PUSH_EVERY})",
    )
    group.add_argument(
        "--hub-token-env",
        default=HUB_TOKEN_ENV,
        help=f"environment variable holding the token (default: {HUB_TOKEN_ENV})",
    )
    group.add_argument(
        "--hub-private",
        action="store_true",
        help="create the repo private if it does not exist yet",
    )
    group.add_argument(
        "--hub-dry-run",
        action="store_true",
        help="rehearse the push schedule and bookkeeping without uploading anything",
    )


def _load_conversations(path: str) -> list[Any]:
    """Read a mined dataset and validate every record, naming the first bad one.

    Strict: a corpus with one malformed record is a corpus whose size is unknown,
    and a silently dropped record is indistinguishable from a small corpus.
    """
    records = list(read_jsonl(path))
    if not records:
        raise ToolTrainError(f"dataset is empty: {path}")
    catalogue = [tool.to_openai() for tool in GO_TOOLS]
    conversations = []
    for index, record in enumerate(records):
        try:
            conversations.append(
                normalize_conversation(record.get("messages", []), record.get("tools") or catalogue)
            )
        except Exception as exc:
            raise ToolTrainError(f"{path}: record {index} is not usable: {exc}") from exc
    return conversations


def cmd_sft(args: argparse.Namespace) -> int:
    """Fine-tune on a mined corpus, refusing anything that would train on nothing."""
    tokenizer = _tokenizer(args.tokenizer)
    conversations = _load_conversations(args.dataset)
    vision_pad_id = tokenizer.convert_tokens_to_ids(VISION_PAD_TOKEN)
    if vision_pad_id is None or vision_pad_id < 0:
        raise ToolTrainError(
            f"the tokenizer has no {VISION_PAD_TOKEN} token; the vision format is absent"
        )

    examples = list(
        examples_from_conversations(
            conversations, tokenizer, vision_pad_id=vision_pad_id, image_root=args.image_root
        )
    )
    # Every record was rendered with completion-only labels, and the normaliser
    # refuses a conversation with no assistant turn, so the examples reaching the
    # trainer all carry supervised tokens. No second check here: an unreachable
    # guard is a claim that cannot be tested, and this project tests everything.
    plan = _plan(args)
    lines: list[str] = []
    summary = train(plan, examples, device=args.device, log=lines.append)
    for line in lines:
        print(line, file=sys.stderr)
    print(json.dumps(summary.to_record(), indent=2, sort_keys=True))
    return 0


def cmd_dpo(args: argparse.Namespace) -> int:
    """Preference-optimise on execution-rewarded pairs."""
    tokenizer = _tokenizer(args.tokenizer)
    pairs = _load_pairs(args.pairs)
    if not pairs:
        raise ToolTrainError(f"no preference pairs in {args.pairs}")

    plan = _plan(args)
    lines: list[str] = []
    summary = train_dpo(
        plan,
        pairs,
        tokenizer,
        tool_specs=[tool.to_openai() for tool in GO_TOOLS],
        beta=args.beta,
        device=args.device,
        log=lines.append,
    )
    for line in lines:
        print(line, file=sys.stderr)
    print(json.dumps(summary.to_record(), indent=2, sort_keys=True))
    return 0


def _load_pairs(path: str) -> list[PreferencePair]:
    """Read mined preference records back into pairs."""
    out: list[PreferencePair] = []
    for index, record in enumerate(read_jsonl(path)):
        try:
            metadata = record.get("metadata", {})
            out.append(
                PreferencePair(
                    task_id=str(metadata.get("task_id", f"pair-{index}")),
                    prompt=str(record.get("prompt", "")),
                    chosen=list(record["chosen"]),
                    rejected=list(record["rejected"]),
                    chosen_reward=float(metadata.get("chosen_reward", 1.0)),
                    rejected_reward=float(metadata.get("rejected_reward", 0.0)),
                    margin=float(metadata.get("margin", 1.0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolTrainError(f"{path}: pair {index} is not usable: {exc}") from exc
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``."""
    parser = argparse.ArgumentParser(prog="gotooltrain-train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sft = sub.add_parser("sft", help="supervised fine-tuning on a mined corpus")
    _add_shared(sft)
    sft.add_argument("--dataset", required=True, help="mined dataset JSONL")
    sft.add_argument(
        "--image-root",
        default=None,
        help="root the records' image refs resolve against",
    )
    sft.set_defaults(func=cmd_sft)

    dpo = sub.add_parser("dpo", help="preference optimisation on execution-rewarded pairs")
    _add_shared(dpo)
    dpo.add_argument("--pairs", required=True, help="preference JSONL")
    dpo.add_argument("--beta", type=float, default=DEFAULT_BETA)
    dpo.set_defaults(func=cmd_dpo)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ToolTrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
