"""The filtering decisions, which used to live in a notebook cell and be untested.

A notebook is a fine place to *show* a decision and a bad place to *make* one. The
logic here has a trap in it -- the one described in ``measure_dataset``'s docstring --
and for its whole life that trap was invisible to the test suite because it was three
lines of prose and a for-loop in a cell nobody could import.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain.pipeline import measure_dataset

pytestmark = pytest.mark.usefixtures("qwen_tokenizer")


def corpus(tmp_path: pathlib.Path, records: list[dict[str, Any]]) -> pathlib.Path:
    """A JSONL corpus file for the measurement to read."""
    path = tmp_path / "go-unit-tests.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def conversation(text: str) -> dict[str, Any]:
    """A minimal single-turn conversation, as the corpus stores it."""
    return {
        "messages": [
            {"role": "user", "content": "make the test pass"},
            {"role": "assistant", "content": text},
        ]
    }


def rendered_length(text: str, tokenizer_name: str) -> int:
    """How long a record actually renders, so the ceiling can be placed on purpose.

    The ceiling in these tests has to sit *between* two records to prove anything,
    and guessing it makes the test a statement about the guess.
    """
    from transformers import AutoTokenizer

    from gotooltrain import (
        catalog,
        install_template,
        load_template_source,
        normalize_conversation,
        render_example,
    )

    tokenizer = install_template(
        AutoTokenizer.from_pretrained(tokenizer_name), load_template_source()
    )
    record = conversation(text)
    example = render_example(
        tokenizer, normalize_conversation(record["messages"], catalog()), max_length=100_000
    )
    return len(example.input_ids)


def run(source: pathlib.Path, **overrides: Any) -> dict[str, Any]:
    """Measure with the real tokenizer at the real model's name."""
    options: dict[str, Any] = {
        "max_records": 10,
        "max_tokens_per_record": 4096,
        "tokenizer_name": "Qwen/Qwen3.5-4B",
    }
    options.update(overrides)
    return measure_dataset(source, **options)


def test_a_record_at_the_ceiling_is_dropped_rather_than_truncated(tmp_path: pathlib.Path) -> None:
    """Truncation keeps the leading prompt, so a cut record can still look supervised.

    Asking "is there anything to supervise" lets such a record through, and the
    trainer later re-renders it unbounded and refuses it as over-length. The pre-flight
    would then approve exactly what the run rejects. The length is the check.
    """
    tokenizer_name = "Qwen/Qwen3.5-4B"
    short = conversation("Tests pass.")
    long = conversation("def f() { return 1 }\n" * 400)
    ceiling = rendered_length("Tests pass.", tokenizer_name) + 1

    result = run(corpus(tmp_path, [short, long]), max_tokens_per_record=ceiling)

    assert result["kept"] == 1, "the ceiling did not separate the two records"
    assert result["dropped_truncated"] == 1, "the long record was not dropped for length"


def test_a_conversation_with_no_assistant_turn_is_rejected_before_rendering(
    tmp_path: pathlib.Path,
) -> None:
    """There is no such thing as a training example with no answer to learn from.

    It is refused at normalisation, so it counts as *invalid* rather than
    unsupervised. Both are refusals; which counter moves is worth pinning because a
    corpus that fails mostly on one of them has a different problem from a corpus that
    fails on the other.
    """
    only_user = {"messages": [{"role": "user", "content": "make the test pass"}]}
    result = run(corpus(tmp_path, [only_user, conversation("Tests pass.")]))
    assert result["kept"] == 1
    assert result["dropped_invalid"] == 1


def test_an_empty_assistant_turn_is_refused_rather_than_trained_on(
    tmp_path: pathlib.Path,
) -> None:
    """An answer of nothing supervises nothing, so it must not become a training row.

    Worth pinning *which* counter moves: with the current normaliser this is refused
    at normalisation, so it is counted invalid rather than unsupervised. Both refuse
    it, and a corpus failing mostly one way has a different problem from a corpus
    failing the other -- so the distinction is the assertion, not the outcome.
    """
    empty_answer = {
        "messages": [
            {"role": "user", "content": "make the test pass"},
            {"role": "assistant", "content": ""},
        ]
    }
    result = run(corpus(tmp_path, [empty_answer, conversation("Tests pass.")]))
    assert result["kept"] == 1, "the empty answer was trained on, or the real one was dropped"
    assert result["dropped_invalid"] == 1
    assert result["supervised_tokens"] > 0


def test_a_corpus_with_nothing_usable_stops_before_the_gpu_is_billed(
    tmp_path: pathlib.Path,
) -> None:
    """An empty run that reports success is the failure this exists to prevent."""
    from gotooltrain.errors import DatasetError

    with pytest.raises(DatasetError, match="No data to train on"):
        run(corpus(tmp_path, [{"messages": [{"role": "user", "content": "hi"}]}]))


def test_the_measurement_accounts_for_every_record_it_read(tmp_path: pathlib.Path) -> None:
    """The counts are the evidence that the filter did something, and which thing.

    One number, "kept: 12", cannot distinguish a corpus filtered for length from one
    filtered because it was malformed.
    """
    tokenizer_name = "Qwen/Qwen3.5-4B"
    good = conversation("Tests pass.")
    long = conversation("def f() { return 1 }\n" * 400)
    # Two different reasons for refusing: one is too long, the other is malformed.
    unsup: dict[str, Any] = {"messages": [{"role": "user", "content": "hi"}]}
    bad: dict[str, Any] = {"messages": [{"role": "nobody", "content": "hi"}]}
    ceiling = rendered_length("Tests pass.", tokenizer_name) + 1

    result = run(corpus(tmp_path, [good, long, unsup, bad]), max_tokens_per_record=ceiling)

    assert result["records_read"] == 4
    assert result["kept"] == 1, "the ceiling did not separate the short record from the long"
    assert result["dropped_truncated"] == 1
    # Both remaining records are refused at normalisation, where an assistant turn is
    # required. Pinned as a total rather than per-counter: the normaliser owns that
    # split, and this file's job is to prove no record escapes the accounting.
    assert result["dropped_invalid"] == 2
    total_dropped = result["dropped_truncated"] + result["dropped_invalid"]
    assert result["kept"] + total_dropped == result["records_read"], (
        "a record was neither kept nor accounted for"
    )


def test_prepare_data_writes_a_file_the_training_stage_can_read(tmp_path: pathlib.Path) -> None:
    """The measured corpus has to land in the shape the next stage reads.

    The stage that measures and the stage that trains are separated by a file on disk,
    which means the writing half has to be tested against the *reading* half rather
    than against its own idea of the format. This is the stage that produces the input
    the trainer consumes, so a shape mismatch here stops the run one stage later with
    a confusing error.
    """
    from gotooltrain.pipeline import prepare_data

    source = corpus(tmp_path, [conversation("Tests pass."), conversation("Also passing.")])
    destination = tmp_path / "sft.jsonl"
    lines: list[str] = []
    report = prepare_data(
        source,
        destination,
        max_records=10,
        max_tokens_per_record=8192,
        tokenizer_name="Qwen/Qwen3.5-4B",
        emit=lines.append,
    )

    assert report["kept"] == 2
    written = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2
    for record in written:
        assert set(record) == {"messages", "tools"}
        assert isinstance(record["messages"], list)
    # And the counts are on screen, because a reader who cannot see how many records
    # survived has no idea whether the run trains on the corpus they think it does.
    assert any("kept" in line for line in lines)


def test_the_measured_share_is_what_the_vocabulary_term_needs(tmp_path: pathlib.Path) -> None:
    """The memory budget multiplies the vocabulary term by this number.

    So it is measured rather than assumed: a corpus of short answers and an agent
    trajectory differ by a factor of three in the largest single term, and using the
    wrong one is the difference between fitting a card and not.
    """
    result = run(corpus(tmp_path, [conversation("Tests pass.")]))
    assert 0.0 < result["supervised_share"] <= 1.0
    assert result["supervised_tokens"] <= result["total_tokens"]
    assert result["longest"] <= result["total_tokens"]
    assert result["examples"], "the kept records are handed back for writing"


def test_the_record_limit_is_respected_before_anything_is_rendered(tmp_path: pathlib.Path) -> None:
    """Reading thousands to keep hundreds is not a free mistake."""
    records = [conversation(f"answer number {n}") for n in range(12)]
    result = run(corpus(tmp_path, records), max_records=4)
    assert result["records_read"] == 4
