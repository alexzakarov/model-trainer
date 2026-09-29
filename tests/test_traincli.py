"""The training command line.

What matters here is not that the command runs but what it refuses: a corpus that
would train on nothing, a record it cannot render, a preference set with no pairs.
Each of those is a run that would otherwise finish successfully and produce a
checkpoint identical to the one it started from.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain.dataset import write_jsonl
from gotooltrain.gotools import GO_TOOLS
from gotooltrain.traincli import main

TOOLS = [tool.to_openai() for tool in GO_TOOLS]


def record(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"messages": messages, "tools": TOOLS}


def tool_messages(path: str) -> list[dict[str, Any]]:
    """A small but complete agent loop, so a run supervises real tokens."""
    return [
        {"role": "user", "content": "make the test pass"},
        {
            "role": "assistant",
            "content": "Let me run the tests.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "go_test", "arguments": json.dumps({"pkg": "./..."})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok  parser  0.4s"},
        {"role": "assistant", "content": "Tests pass."},
    ]


def dataset(tmp_path: pathlib.Path, count: int = 1) -> pathlib.Path:
    path = tmp_path / "sft.jsonl"
    write_jsonl(
        path,
        (
            json.dumps(
                record(
                    [
                        {**m, "content": f"{m['content']} [{index}]"} if "content" in m else m
                        for m in tool_messages("")
                    ]
                )
            )
            for index in range(count)
        ),
    )
    return path


def pairs_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "pairs.jsonl"
    write_jsonl(
        path,
        [
            json.dumps(
                {
                    "prompt": "make the test pass",
                    "chosen": [
                        {"role": "user", "content": "make the test pass"},
                        {"role": "assistant", "content": "Fixed it; tests pass."},
                    ],
                    "rejected": [
                        {"role": "user", "content": "make the test pass"},
                        {"role": "assistant", "content": "I could not reproduce the failure."},
                    ],
                    "metadata": {
                        "task_id": "go-0001",
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "margin": 1.0,
                    },
                }
            )
        ],
    )
    return path


def sft_args(tmp_path: pathlib.Path, model: pathlib.Path, source: pathlib.Path) -> list[str]:
    return [
        "sft",
        "--model",
        str(model),
        "--output",
        str(tmp_path / "out"),
        "--tokenizer",
        "Qwen/Qwen3.5-4B",
        "--dataset",
        str(source),
        "--dtype",
        "float32",
        "--epochs",
        "1",
        "--grad-accum",
        "1",
    ]


# ------------------------------------------------------------------ loading


def test_an_empty_dataset_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(sft_args(tmp_path, pathlib.Path("any/model"), empty)) == 1
    assert "dataset is empty" in capsys.readouterr().err


def test_a_dataset_with_a_broken_record_names_it(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A silently dropped record is indistinguishable from a small corpus."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(record(tool_messages("")))
        + "\n"
        + json.dumps({"messages": [{"role": "banana", "content": "x"}], "tools": []})
        + "\n",
        encoding="utf-8",
    )
    code = main(sft_args(tmp_path, pathlib.Path("any/model"), bad))
    assert code == 1
    assert "record 1" in capsys.readouterr().err


def test_a_dataset_with_no_assistant_turns_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A prompt with nothing to predict trains on nothing while looking healthy."""
    path = tmp_path / "prompt_only.jsonl"
    write_jsonl(path, [json.dumps(record([{"role": "user", "content": "hi"}]))])
    code = main(sft_args(tmp_path, pathlib.Path("any/model"), path))
    assert code == 1
    assert "no assistant turn" in capsys.readouterr().err


# --------------------------------------------------------------- preferences


def dpo_args(tmp_path: pathlib.Path, model: pathlib.Path, source: pathlib.Path) -> list[str]:
    return [
        "dpo",
        "--model",
        str(model),
        "--output",
        str(tmp_path / "dpo-out"),
        "--pairs",
        str(source),
        "--dtype",
        "float32",
        "--epochs",
        "1",
        "--grad-accum",
        "1",
    ]


def test_an_empty_preference_file_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "nopairs.jsonl"
    empty.write_text("", encoding="utf-8")
    code = main(dpo_args(tmp_path, pathlib.Path("any/model"), empty))
    assert code == 1
    assert "no preference pairs" in capsys.readouterr().err


def test_a_preference_file_without_chosen_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text(json.dumps({"prompt": "x", "rejected": []}) + "\n", encoding="utf-8")
    code = main(dpo_args(tmp_path, pathlib.Path("any/model"), broken))
    assert code == 1
    assert "pair 0" in capsys.readouterr().err


# --------------------------------------------------------------- the run


def test_an_sft_run_trains_and_writes_a_checkpoint(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole command, on a real architecture, with the real tokenizer."""
    import gotooltrain.traincli as cli

    monkeypatch.setattr(cli, "_tokenizer", lambda name: qwen_tokenizer)
    source = dataset(tmp_path)

    code = main(sft_args(tmp_path, tiny_checkpoint, source))
    assert code == 0, capsys.readouterr().err
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps"] == 1
    assert summary["supervised_tokens"] > 0
    assert (tmp_path / "out" / "config.json").is_file()


def test_a_dpo_run_trains_and_writes_a_checkpoint(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gotooltrain.traincli as cli

    monkeypatch.setattr(cli, "_tokenizer", lambda name: qwen_tokenizer)
    source = pairs_file(tmp_path)

    code = main(dpo_args(tmp_path, tiny_checkpoint, source))
    assert code == 0, capsys.readouterr().err
    summary = json.loads(capsys.readouterr().out)
    assert summary["steps"] == 1
    assert summary["pairs"] == 1
    assert summary["final_loss"] is not None


def test_the_dataset_loader_validates_every_record(
    tmp_path: pathlib.Path,
) -> None:
    """Validation is a pre-flight, not something training discovers mid-run."""
    from gotooltrain.schema import Conversation
    from gotooltrain.traincli import _load_conversations

    path = dataset(tmp_path, count=2)
    conversations = _load_conversations(str(path))
    assert len(conversations) == 2
    assert all(isinstance(c, Conversation) for c in conversations)
    assert all(c.messages for c in conversations)


def test_a_tokenizer_without_the_vision_pad_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rendering a multimodal format on a tokenizer that lacks it is not a fallback."""
    import gotooltrain.traincli as cli

    class NoVision:
        def convert_tokens_to_ids(self, token):  # type: ignore[no-untyped-def]
            """Report the token as unknown."""
            return None

    monkeypatch.setattr(cli, "_tokenizer", lambda name: NoVision())
    code = main(sft_args(tmp_path, pathlib.Path("any/model"), dataset(tmp_path)))
    assert code == 1
    assert "vision format is absent" in capsys.readouterr().err


def test_a_missing_dataset_file_is_reported(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(sft_args(tmp_path, pathlib.Path("any/model"), tmp_path / "nope.jsonl")) == 1
    assert "dataset not found" in capsys.readouterr().err


# --------------------------------------------------------- hub publication


def parsed(tmp_path: pathlib.Path, extra: list[str]) -> Any:
    """Parse a full command line with the real parser, the way a shell would.

    Building the parser here rather than hand-assembling a ``Namespace`` is the
    point: a flag that exists in the test and not in ``--help`` is a flag nobody can
    pass to the command.
    """
    import argparse

    from gotooltrain import traincli

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sft = sub.add_parser("sft")
    traincli._add_shared(sft)
    sft.add_argument("--dataset", required=True)
    return parser.parse_args(
        ["sft", "--model", "m", "--output", str(tmp_path / "out"), "--dataset", "x.jsonl", *extra]
    )


def test_an_interval_without_a_repo_is_refused(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a disposable machine, a silently ignored interval is the loss this prevents."""
    args = [
        *sft_args(tmp_path, pathlib.Path("any/model"), dataset(tmp_path)),
        "--hub-push-every",
        "25",
    ]
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "--hub-push-every 25" in err
    assert "nowhere to push to" in err


def test_a_run_with_a_named_repo_publishes_on_schedule(
    tiny_checkpoint: pathlib.Path,
    qwen_tokenizer: Any,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole command with publication on: the checkpoint must actually leave.

    Run as a rehearsal -- ``--hub-dry-run`` -- so the schedule, the written state
    and the summary are all exercised with no network and no credential, which is
    the only way this can be asserted in CI at all.
    """
    import gotooltrain.traincli as cli

    monkeypatch.setattr(cli, "_tokenizer", lambda name: qwen_tokenizer)
    args = [
        *sft_args(tmp_path, tiny_checkpoint, dataset(tmp_path)),
        "--hub-repo-id",
        "alexzakarov/qwen3.5-4b-go",
        "--hub-push-every",
        "1",
        "--hub-dry-run",
    ]

    assert main(args) == 0, capsys.readouterr().err
    captured = capsys.readouterr()

    summary = json.loads(captured.out)
    assert [p["step"] for p in summary["pushes"]] == [1]
    assert summary["pushes"][0]["revision"] == "dry-run-step-1/1"
    assert summary["pushes"][0]["dry_run"] is True

    err = captured.err
    assert "DRY RUN" in err, "a rehearsal must not read like a publication in the log"
    assert "alexzakarov/qwen3.5-4b-go" in err
    # The published folder describes itself, so a checkpoint found on the Hub says
    # which hyperparameters produced it without a run directory to correlate.
    state = json.loads((tmp_path / "out" / "hub_push.json").read_text(encoding="utf-8"))
    assert state["step"] == 1
    assert state["plan"]["hub"]["repo_id"] == "alexzakarov/qwen3.5-4b-go"


def test_a_run_with_no_repo_records_no_publication_in_the_plan(tmp_path: pathlib.Path) -> None:
    from gotooltrain.traincli import _plan

    assert _plan(parsed(tmp_path, [])).hub is None
    assert _plan(parsed(tmp_path, [])).to_record()["hub"] is None


def test_the_dry_run_flag_reaches_the_policy(tmp_path: pathlib.Path) -> None:
    from gotooltrain.traincli import _plan

    args = parsed(tmp_path, ["--hub-repo-id", "a/b", "--hub-dry-run"])
    plan = _plan(args)
    assert plan.hub is not None
    assert plan.hub.dry_run is True


def test_an_interval_of_zero_is_refused_by_the_policy(tmp_path: pathlib.Path) -> None:
    """Handled by the policy, not the parser: 0 is a value, not a syntax error."""
    from gotooltrain.errors import HarnessError
    from gotooltrain.traincli import _plan

    args = parsed(tmp_path, ["--hub-repo-id", "a/b", "--hub-push-every", "0"])
    with pytest.raises(HarnessError, match="every_steps must be >= 1"):
        _plan(args)


def test_the_publication_flags_are_all_reachable_from_the_command_line(
    tmp_path: pathlib.Path,
) -> None:
    args = parsed(
        tmp_path,
        [
            "--hub-repo-id",
            "alexzakarov/qwen3.5-4b-go",
            "--hub-push-every",
            "7",
            "--hub-token-env",
            "COLAB_HF",
            "--hub-private",
            "--hub-dry-run",
        ],
    )
    assert args.hub_repo_id == "alexzakarov/qwen3.5-4b-go"
    assert args.hub_push_every == 7
    assert args.hub_token_env == "COLAB_HF"
    assert args.hub_private is True
    assert args.hub_dry_run is True


def test_a_named_repo_becomes_a_policy_recorded_in_the_plan(tmp_path: pathlib.Path) -> None:
    """The destination is part of what a run produced, so it is in the written plan."""
    from gotooltrain.traincli import _plan

    args = parsed(
        tmp_path,
        ["--hub-repo-id", "alexzakarov/qwen3.5-4b-go", "--hub-push-every", "25", "--hub-private"],
    )
    plan = _plan(args)
    assert plan.hub is not None
    assert plan.hub.repo_id == "alexzakarov/qwen3.5-4b-go"
    assert plan.hub.every_steps == 25
    assert plan.hub.private is True
    assert plan.hub.dry_run is False
    assert plan.to_record()["hub"] == plan.hub.to_record()
