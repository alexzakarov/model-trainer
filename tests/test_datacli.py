"""The data command line.

Two behaviours matter beyond "it ran": a corpus that cannot teach the catalogue
must not exit successfully, and a run that supports no training record must say so
rather than write an empty file and pass.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain import ResultStore
from gotooltrain.datacli import main
from gotooltrain.evalrun import RunConfig, run_evaluation

TASKS: list[dict[str, Any]] = [
    {"id": "go-0001", "repository": "acme/parser", "package": "parser", "prompt": "add a test"}
]


class Generator:
    """One tool call, then an answer."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Call go_test once, then finish."""
        if turn_index == 0:
            return {
                "text": "Running the tests.",
                "tool_calls": [{"id": "c0", "name": "go_test", "arguments": {"pkg": "./..."}}],
            }
        return {"text": "Done.", "tool_calls": []}


class Executor:
    """A command that succeeds."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Return a passing exit code."""
        return 0, "ok", ""


class Judge:
    """A judge with a fixed opinion."""

    def __init__(self, score: float = 1.0) -> None:
        """Fix the score."""
        self.score = score

    def judge(self, record, fingerprint):  # type: ignore[no-untyped-def]
        """Score without inspecting the trajectory."""
        return {"score": self.score}


def seed_run(tmp_path: pathlib.Path, *, score: float = 1.0) -> tuple[ResultStore, RunConfig]:
    """Put a real, judged run in a store the CLI can read."""
    store = ResultStore(tmp_path / "store")
    config = RunConfig(
        model_id="Qwen/Qwen3.5-4B-go",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=7,
        decode_params={"temperature": 0.0},
    )
    run_evaluation(
        config,
        TASKS,
        store,
        Generator(),
        Executor(),
        judge=Judge(score),
        workspace=tmp_path / "ws",
        run_id="run-cli",
    )
    return store, config


def write_tasks(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(TASKS), encoding="utf-8")
    return path


@pytest.fixture
def tokenizer(qwen_tokenizer: Any) -> Any:
    return qwen_tokenizer


def mine_args(tmp_path: pathlib.Path, store: ResultStore) -> list[str]:
    return [
        "mine",
        "--store",
        str(store.root),
        "--tasks",
        str(write_tasks(tmp_path)),
        "--out",
        str(tmp_path / "sft.jsonl"),
        "--report",
        str(tmp_path / "report.json"),
        "--model",
        "Qwen/Qwen3.5-4B-go",
        "--revision",
        "rev-a",
        "--dataset-version",
        "holdout-1",
        "--seed",
        "7",
    ]


def test_mine_writes_the_dataset_and_the_report(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    store, _ = seed_run(tmp_path)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    assert main(mine_args(tmp_path, store)) == 0

    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["messages"][0]["role"] == "user"
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["admitted"] == 1
    assert report["corpus"]["trajectories"] == 1


def test_mining_a_failed_run_says_so_and_returns_nonzero(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """Writing an empty file and exiting 0 would let a pipeline train on nothing."""
    store, _ = seed_run(tmp_path, score=0.0)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    assert main(mine_args(tmp_path, store)) == 2
    assert (tmp_path / "sft.jsonl").read_text() == ""
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["refusals"]["below_threshold"] == 1


def test_execution_only_mining_is_the_named_opt_in(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    store, _ = seed_run(tmp_path, score=0.0)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    args = [*mine_args(tmp_path, store), "--execution-only"]
    assert main(args) == 0
    assert (tmp_path / "sft.jsonl").read_text().strip() != ""


def test_an_unknown_judge_score_keeps_everything(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """A run judged below the threshold admits nothing, however high the threshold."""
    store, _ = seed_run(tmp_path, score=1.0)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)
    args = [*mine_args(tmp_path, store), "--min-score", "2.0"]
    assert main(args) == 2


def test_a_missing_task_file_is_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    store, _ = seed_run(tmp_path)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)
    args = mine_args(tmp_path, store)
    args[args.index("--tasks") + 1] = str(tmp_path / "nope.json")
    assert main(args) == 1


def healthy_rows(count: int = 400) -> list[dict[str, Any]]:
    """A corpus that clears every threshold, in the shape ``read_trajectories`` reads."""
    from gotooltrain.gotools import GO_TOOLS

    catalogue = sorted(t.name for t in GO_TOOLS)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        rows.append(
            {
                "id": f"t{index:05d}",
                "repository": f"repo{index % 40:03d}",
                "package": f"pkg{index:05d}",
                "tools": [
                    catalogue[index % len(catalogue)],
                    catalogue[(index + 1) % len(catalogue)],
                    catalogue[(index + 4) % len(catalogue)],
                ],
                "supervised_tokens": 200,
                "total_tokens": 800,
                "has_thinking": True,
                "has_images": False,
            }
        )
    return rows


def test_measure_passes_an_adequate_corpus(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corpus that covers the catalogue and the loop exits successfully."""
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in healthy_rows()), encoding="utf-8")

    assert main(["measure", "--corpus", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["adequate"] is True
    assert out["trajectories"] == 400


def test_measure_fails_an_inadequate_corpus(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A build step that reports a number and exits 0 gets its number ignored."""
    row = {
        "id": "t1",
        "repository": "r",
        "package": "p",
        "tools": ["go_test", "go_test"],
        "supervised_tokens": 10,
        "total_tokens": 100,
    }
    path = tmp_path / "corpus.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert main(["measure", "--corpus", str(path)]) == 1
    err = capsys.readouterr().err
    assert "not adequate" in err
    assert "next data targets" in err


def test_measure_reports_a_missing_corpus(tmp_path: pathlib.Path) -> None:
    assert main(["measure", "--corpus", str(tmp_path / "nope.jsonl")]) == 1


def test_tasks_drops_packages_whose_tests_already_pass(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A task the base model already solves measures nothing, so it is not written."""
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "go.mod").write_text("module example.com/x\n\ngo 1.23\n", encoding="utf-8")
    (pkg / "x.go").write_text(
        "package x\n\nfunc Add(a, b int) int { return a + b }\n", encoding="utf-8"
    )
    (pkg / "x_test.go").write_text(
        'package x\n\nimport "testing"\n\nfunc TestAdd(t *testing.T) {\n'
        '\tif Add(1, 2) != 3 {\n\t\tt.Fatal("nope")\n\t}\n}\n',
        encoding="utf-8",
    )

    code = main(
        [
            "tasks",
            "--repository",
            str(repo),
            "--repository-name",
            "acme/x",
            "--out",
            str(tmp_path / "tasks.jsonl"),
        ]
    )
    assert code == 2
    assert "already passing" in capsys.readouterr().err
    assert not (tmp_path / "tasks.jsonl").exists()


def test_tasks_writes_a_failing_package(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "go.mod").write_text("module example.com/x\n\ngo 1.23\n", encoding="utf-8")
    (pkg / "x.go").write_text(
        "package x\n\nfunc Add(a, b int) int { return a - b }\n", encoding="utf-8"
    )
    (pkg / "x_test.go").write_text(
        'package x\n\nimport "testing"\n\nfunc TestAdd(t *testing.T) {\n'
        '\tif Add(1, 2) != 3 {\n\t\tt.Fatal("nope")\n\t}\n}\n',
        encoding="utf-8",
    )

    code = main(
        [
            "tasks",
            "--repository",
            str(repo),
            "--repository-name",
            "acme/x",
            "--out",
            str(tmp_path / "tasks.jsonl"),
        ]
    )
    assert code == 0, capsys.readouterr().err
    rows = [json.loads(line) for line in (tmp_path / "tasks.jsonl").read_text().splitlines()]
    assert rows[0]["id"] == "acme/x:pkg"
    assert rows[0]["prompt"] == "Make the tests in pkg pass."


def test_tasks_refuses_a_missing_repository(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "tasks",
            "--repository",
            str(tmp_path / "nope"),
            "--repository-name",
            "acme/x",
            "--out",
            str(tmp_path / "tasks.jsonl"),
        ]
    )
    assert code == 1
    assert "repository not found" in capsys.readouterr().err


def test_tasks_refuses_a_repository_without_packages(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    code = main(
        [
            "tasks",
            "--repository",
            str(empty),
            "--repository-name",
            "acme/x",
            "--out",
            str(tmp_path / "tasks.jsonl"),
        ]
    )
    assert code == 1
    assert "no Go packages" in capsys.readouterr().err


# --------------------------------------------------------------- tokenizer use


def test_the_tokenizer_used_for_mining_carries_the_template() -> None:
    """Mining must count tokens under the same template the trainer will use."""
    from gotooltrain import normalize_conversation
    from gotooltrain.datacli import DEFAULT_TOKENIZER, _tokenizer
    from gotooltrain.template import render_text

    tokenizer = _tokenizer(DEFAULT_TOKENIZER)
    conversation = normalize_conversation(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}], []
    )
    assert "ok" in render_text(tokenizer, conversation)


def test_a_jsonl_task_file_is_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    """Both task formats the evaluator accepts must be minable."""
    store, _ = seed_run(tmp_path)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text("\n".join(json.dumps(t) for t in TASKS) + "\n", encoding="utf-8")
    args = mine_args(tmp_path, store)
    args[args.index("--tasks") + 1] = str(tasks_path)
    assert main(args) == 0


def test_a_task_file_that_is_not_a_list_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    store, _ = seed_run(tmp_path)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    args = mine_args(tmp_path, store)
    # Overwritten after mine_args, which itself writes a valid task file.
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text('{"not": "a list"}', encoding="utf-8")
    assert main(args) == 1


def test_a_task_without_an_id_in_the_file_is_refused(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Any
) -> None:
    store, _ = seed_run(tmp_path)
    monkeypatch.setattr("gotooltrain.datacli._tokenizer", lambda name: tokenizer)

    args = mine_args(tmp_path, store)
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text('[{"prompt": "no id"}]', encoding="utf-8")
    assert main(args) == 1


def test_measure_prints_no_targets_when_none_are_implied(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Some findings are not data-shape problems, so there is nothing to add.

    A corpus that is prompt-heavy is inadequate, but "add more records" would not
    fix it -- the records themselves need shorter prompts. Printing an empty target
    list would suggest otherwise.
    """
    rows = healthy_rows()
    for row in rows:
        row["supervised_tokens"] = 5
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main(["measure", "--corpus", str(path)]) == 1
    err = capsys.readouterr().err
    assert "not adequate" in err
    assert "next data targets" not in err


# ------------------------------------------------------------- preferences


PREF_TASKS: list[dict[str, Any]] = [
    {
        "id": "go-0001",
        "repository": "acme/parser",
        "package": "parser",
        "prompt": "make the test pass",
        "verification": ["go_test"],
    }
]


class PrefGenerator:
    """Calls go_test once, then answers."""

    def generate(self, task, sample_index, turn_index, messages, fingerprint):  # type: ignore[no-untyped-def]
        """Call go_test once, then answer."""
        if turn_index == 0:
            return {
                "text": "running",
                "tool_calls": [{"id": "c0", "name": "go_test", "arguments": {"pkg": "./..."}}],
            }
        return {"text": "done", "tool_calls": []}


class PrefExecutor:
    """Succeeds for sample 0 and fails for any other sample."""

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Sample 0 passes, sample 1 fails."""
        if request.sample_index == 0:
            return 0, "ok  parser  0.4s", ""
        return 1, "FAIL  parser", ""


def seed_pref_run(tmp_path: pathlib.Path, n_samples: int = 2) -> None:
    from gotooltrain.evalrun import run_evaluation

    store = ResultStore(tmp_path / "store")
    config = RunConfig(
        model_id="Qwen/Qwen3.5-4B-go",
        model_revision="rev-a",
        dataset_version="holdout-1",
        seed=7,
        decode_params={"temperature": 0.0},
        n_samples=n_samples,
    )
    run_evaluation(
        config,
        PREF_TASKS,
        store,
        PrefGenerator(),
        PrefExecutor(),
        workspace=tmp_path / "ws",
        run_id="run-pref",
    )


def pref_args(tmp_path: pathlib.Path, n_samples: int = 2) -> list[str]:
    tasks_path = tmp_path / "ptasks.json"
    tasks_path.write_text(json.dumps(PREF_TASKS), encoding="utf-8")
    return [
        "preferences",
        "--store",
        str(tmp_path / "store"),
        "--tasks",
        str(tasks_path),
        "--model",
        "Qwen/Qwen3.5-4B-go",
        "--revision",
        "rev-a",
        "--dataset-version",
        "holdout-1",
        "--seed",
        "7",
        "--n-samples",
        str(n_samples),
        "--out",
        str(tmp_path / "pairs.jsonl"),
        "--report",
        str(tmp_path / "pref-report.json"),
    ]


def test_preferences_are_written_from_a_real_run(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_pref_run(tmp_path)
    assert main(pref_args(tmp_path)) == 0
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["source"] == "execution"
    assert rows[0]["metadata"]["chosen_reward"] == 1.0
    report = json.loads((tmp_path / "pref-report.json").read_text())
    assert report["pairs"] == 1


def test_a_run_without_contrast_returns_nonzero(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One sample per task cannot contrast with itself, and saying so is the point."""
    seed_pref_run(tmp_path, n_samples=1)
    assert main(pref_args(tmp_path, n_samples=1)) == 2
    assert "no task produced a contrast" in capsys.readouterr().err
    assert (tmp_path / "pairs.jsonl").read_text() == ""


# ---------------------------------------------------------------- go corpus


def go_record(
    index: int, repository: str = "golang/go", directory: str = "src/pkg"
) -> dict[str, str]:
    """One Go-UT-Bench-shaped record."""
    return {
        "SHA256": f"{index:064x}",
        "Repository": repository,
        "File Name": f"f{index}.go",
        "File path in Repository": f"{directory}/f{index}.go",
        "Code": f"package p{index}\n",
        "Code Commit hash": "c" * 40,
        "File Path for Unit Test": f"{directory}/f{index}_test.go",
        "Unit Test - (Ground Truth)": f"package p{index}\n// test\n",
        "Unit Test Commit hash": "d" * 40,
    }


def go_split(tmp_path: pathlib.Path, records: list[dict[str, str]]) -> pathlib.Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_go_pairs_writes_records_and_refuses_review_repositories(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = [go_record(i, repository="moby/moby") for i in range(5)]
    records += [go_record(99, repository="hashicorp/terraform")]
    source = go_split(tmp_path, records)

    code = main(
        [
            "go-pairs",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    # Five files is far below the corpus floor, so the command reports inadequacy.
    assert code == 1
    rows = [json.loads(line) for line in (tmp_path / "dapt.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["files"] == 5
    assert report["refused"][0]["repository"] == "hashicorp/terraform"
    assert "inadequate" in capsys.readouterr().err


def test_go_pairs_can_admit_a_review_repository_deliberately(
    tmp_path: pathlib.Path,
) -> None:
    source = go_split(tmp_path, [go_record(i, repository="hashicorp/terraform") for i in range(3)])
    code = main(
        [
            "go-pairs",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--include-review",
        ]
    )
    assert code == 1  # still inadequate, but for size rather than licence
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["refused"] == []
    assert report["files"] == 3


def test_go_pairs_emits_unit_test_prompts(tmp_path: pathlib.Path) -> None:
    source = go_split(tmp_path, [go_record(i, repository="moby/moby") for i in range(4)])
    main(
        [
            "go-pairs",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
            "--unit-tests-out",
            str(tmp_path / "unit.jsonl"),
        ]
    )
    rows = [json.loads(line) for line in (tmp_path / "unit.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert [m["role"] for m in rows[0]["messages"]] == ["user", "assistant"]


def test_go_pairs_can_download_first(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The download path is exercised without touching the network."""
    from gotooltrain import datacli

    fetched = go_split(
        tmp_path,
        [go_record(0, repository="moby/moby")],
    )
    monkeypatch.setattr(datacli, "download_splits", lambda destination: [fetched])

    code = main(
        [
            "go-pairs",
            "--download-to",
            str(tmp_path / "download"),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 1
    assert (tmp_path / "dapt.jsonl").read_text().count("\n") == 1


def test_go_pairs_succeeds_on_an_adequate_corpus(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of the screening is that a good corpus exits clean."""
    names = [
        "gin-gonic/gin",
        "gohugoio/hugo",
        "golang/go",
        "moby/moby",
        "pingcap/tidb",
        "kserve/kserve",
    ]
    records = [
        go_record(i, repository=names[i % len(names)], directory=f"pkg/d{i}") for i in range(1000)
    ]
    source = go_split(tmp_path, records)

    code = main(
        [
            "go-pairs",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    assert code == 0, capsys.readouterr().err
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["adequate"] is True
    assert report["files"] == 1000


def test_go_pairs_removes_duplicate_sources(tmp_path: pathlib.Path) -> None:
    """The published splits overlap; the report must say how many were dropped."""
    records = [go_record(i, repository="moby/moby") for i in range(4)]
    records.append(go_record(0, repository="moby/moby"))  # same sha as the first
    source = go_split(tmp_path, records)

    main(
        [
            "go-pairs",
            "--source",
            str(source),
            "--out",
            str(tmp_path / "dapt.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["duplicates_removed"] == 1
    assert report["files"] == 4
    assert report["duplicate_share"] == 0.0


def test_go_pairs_needs_a_source_when_not_downloading(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No input is a usage mistake, not a finding about the data."""
    assert (
        main(
            [
                "go-pairs",
                "--out",
                str(tmp_path / "dapt.jsonl"),
                "--report",
                str(tmp_path / "report.json"),
            ]
        )
        == 1
    )
    assert "no corpus source given" in capsys.readouterr().err
