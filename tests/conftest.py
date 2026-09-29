"""Shared fixtures.

Sample data lives here rather than in an importable module so tests depend on
fixtures, not on each other's import graph.

The tokenizer-dependent tests use the real ``Qwen/Qwen3.5-4B`` tokenizer. That
matters: the assistant mask is produced by the tokenizer from the template's
``{% generation %}`` bookkeeping, so a stub would verify our own reimplementation
instead of the code that actually runs. Only tokenizer files are downloaded
(a few MB), never weights.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from gotooltrain import Conversation, install_template, load_template_source, normalize_conversation

BASE_MODEL = "Qwen/Qwen3.5-4B"

GO_TEST_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "go_test",
        "description": "Run go test for a package.",
        "parameters": {
            "type": "object",
            "properties": {"pkg": {"type": "string", "description": "package path"}},
            "required": ["pkg"],
        },
    },
}

READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def build_sample_messages() -> list[dict[str, Any]]:
    """One full assistant -> tool -> assistant loop, in flat OpenAI form."""
    return [
        {"role": "system", "content": "You are a Go engineer."},
        {"role": "user", "content": "Add a test for the parser."},
        {
            "role": "assistant",
            "content": "Let me read the package first.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "parser/parser.go"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "package parser"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "go_test", "arguments": json.dumps({"pkg": "./parser"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "ok  parser  0.4s"},
        {"role": "assistant", "content": "Tests pass."},
    ]


@pytest.fixture(scope="session")
def template_source() -> str:
    return load_template_source()


@pytest.fixture(scope="session")
def qwen_tokenizer(template_source: str):  # type: ignore[no-untyped-def]
    """The real tokenizer with our chat template installed."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    return install_template(tokenizer, template_source)


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):  # type: ignore[no-untyped-def]
    """A tiny Qwen3.5 checkpoint whose vocabulary matches the real tokenizer.

    The architecture is the real one, shrunk: the loader path, the attention layer
    mix and the multimodal config are the same objects a full run uses, so a test
    that loads it exercises the loader rather than a stand-in. Session-scoped
    because building and instantiating it is the expensive part, and every training
    test wants the same one.
    """
    transformers = pytest.importorskip("transformers")

    directory = tmp_path_factory.mktemp("tiny") / "model"
    config = transformers.AutoConfig.from_pretrained(BASE_MODEL)
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 2
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.linear_key_head_dim = 16
    text.linear_value_head_dim = 16
    text.linear_num_key_heads = 4
    text.linear_num_value_heads = 4
    # One of each kind, so both attention implementations are constructed.
    text.layer_types = ["linear_attention", "full_attention"]
    text.mlp_only_layers = []
    text.mtp_num_hidden_layers = 0
    visual = config.vision_config
    visual.hidden_size = 32
    visual.num_hidden_layers = 1
    visual.num_attention_heads = 2
    visual.intermediate_size = 64

    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(directory))
    # A training loop loads its tokenizer from the model's directory, so the
    # tokenizer has to live there too.
    tokenizer = install_template(
        transformers.AutoTokenizer.from_pretrained(BASE_MODEL), load_template_source()
    )
    tokenizer.save_pretrained(str(directory))
    return directory


@pytest.fixture
def sample_tools() -> list[dict[str, Any]]:
    return [dict(GO_TEST_TOOL), dict(READ_FILE_TOOL)]


@pytest.fixture
def sample_messages() -> list[dict[str, Any]]:
    return build_sample_messages()


@pytest.fixture
def sample_record(sample_messages: list[dict[str, Any]], sample_tools: list[dict[str, Any]]):  # type: ignore[no-untyped-def]
    return {"messages": sample_messages, "tools": sample_tools}


@pytest.fixture
def conversation(sample_record: dict[str, Any]) -> Conversation:
    return normalize_conversation(sample_record["messages"], sample_record["tools"])
