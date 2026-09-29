"""The Go domain-adaptation corpus.

Two claims get checked here rather than trusted: that the source dataset's
repositories are permissively licensed, and that a corpus of a given size can
teach the language. The first is checked against a licence table, the second by
measuring coverage rather than counting files.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from gotooltrain.errors import DatasetError
from gotooltrain.gocorpus import (
    LICENCES,
    MAX_DUPLICATE_SHARE,
    MAX_REPOSITORY_SHARE,
    MIN_DAPT_FILES,
    MIN_DAPT_PACKAGES,
    MIN_DAPT_REPOSITORIES,
    REVIEW_REQUIRED,
    GoPair,
    admit,
    dapt_records,
    measure_dapt,
    read_pairs,
    unit_test_messages,
)


def record(
    *,
    sha: str = "a" * 64,
    repository: str = "kubernetes/kubernetes",
    path: str = "pkg/util/backoff.go",
    code: str = "package util\n\nfunc Backoff() {}\n",
    commit: str = "c" * 40,
    test_path: str = "pkg/util/backoff_test.go",
    test: str = 'package util\n\nimport "testing"\n\nfunc TestBackoff(t *testing.T) {}\n',
) -> dict[str, str]:
    """One record in the source dataset's own field names."""
    return {
        "SHA256": sha,
        "Repository": repository,
        "File Name": "backoff.go",
        "File path in Repository": path,
        "Code": code,
        "Code Commit hash": commit,
        "File Path for Unit Test": test_path,
        "Unit Test - (Ground Truth)": test,
        "Unit Test Commit hash": "d" * 40,
    }


def pair(
    index: int = 0,
    *,
    repository: str = "kubernetes/kubernetes",
    code: str | None = None,
    directory: str | None = None,
    test: str | None = None,
) -> GoPair:
    """A synthetic pair, with the sha tied to the index so dedup can be exercised."""
    return read_pairs_from(
        [
            record(
                sha=f"{index:064x}",
                repository=repository,
                path=f"{directory or 'pkg'}/f{index}.go",
                code=code if code is not None else f"package p{index}\n",
                test=test if test is not None else f"package p{index}\n// test {index}\n",
            )
        ]
    )[0]


def read_pairs_from(records: list[dict[str, Any]]) -> list[GoPair]:
    """Round-trip records through the real parser so the tests use the real path."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "split.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return read_pairs(path)


@pytest.fixture
def split(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "split.json"
    path.write_text(json.dumps([record()]), encoding="utf-8")
    return path


# ------------------------------------------------------------------- licences


def test_every_documented_repository_has_a_licence() -> None:
    assert len(LICENCES) == 10, "Go-UT-Bench draws from ten repositories"


def test_the_review_repositories_are_among_the_documented_ones() -> None:
    """A review entry for an unknown repository would never fire."""
    assert set(REVIEW_REQUIRED) <= set(LICENCES)


def test_the_review_reasons_name_a_real_concern() -> None:
    for repository, reason in REVIEW_REQUIRED.items():
        assert repository in LICENCES
        assert len(reason) > 40, f"{repository} needs a reason, not a label"


# -------------------------------------------------------------------- parsing


def test_a_valid_split_is_read(split: pathlib.Path) -> None:
    pairs = read_pairs(split)
    assert len(pairs) == 1
    assert pairs[0].repository == "kubernetes/kubernetes"
    assert pairs[0].code.startswith("package util")
    assert pairs[0].test_commit == "d" * 40


def test_a_missing_split_is_reported(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        read_pairs(tmp_path / "nope.json")


def test_a_split_that_is_not_json_is_reported(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{oops}", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        read_pairs(path)


def test_a_split_that_is_not_a_list_is_reported(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "obj.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetError, match="must hold a list"):
        read_pairs(path)


def test_a_missing_field_names_itself(tmp_path: pathlib.Path) -> None:
    """The dataset's field names are irregular; guessing one yields a silent blank."""
    broken = record()
    del broken["Unit Test - (Ground Truth)"]
    path = tmp_path / "split.json"
    path.write_text(json.dumps([broken]), encoding="utf-8")
    with pytest.raises(DatasetError, match="Unit Test"):
        read_pairs(path)


def test_empty_code_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps([record(code="  \n")]), encoding="utf-8")
    with pytest.raises(DatasetError, match="empty code"):
        read_pairs(path)


def test_an_empty_unit_test_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps([record(test="\n")]), encoding="utf-8")
    with pytest.raises(DatasetError, match="empty unit test"):
        read_pairs(path)


def test_a_non_object_record_is_refused(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "split.json"
    path.write_text(json.dumps(["not an object"]), encoding="utf-8")
    with pytest.raises(DatasetError, match="not an object"):
        read_pairs(path)


# ------------------------------------------------------------------ properties


def test_the_declared_package_is_read_from_the_source() -> None:
    assert pair(code="package alpha\n\nfunc A() {}\n").package == "alpha"


def test_a_file_without_a_package_clause_reports_none() -> None:
    """`go` requires the clause, so its absence is a fact about the file."""
    assert pair(code="// just a comment\n").package == ""


def test_a_directory_at_the_repository_root_is_reported_as_dot() -> None:
    assert pair(directory=".").directory == "."


# ------------------------------------------------------------------- admitting


def test_only_permissive_repositories_are_kept_by_default() -> None:
    pairs = [pair(0, repository="gin-gonic/gin"), pair(1, repository="hashicorp/terraform")]
    kept, refused = admit(pairs)

    assert [p.repository for p in kept] == ["gin-gonic/gin"]
    assert [r.repository for r in refused] == ["hashicorp/terraform"]


def test_the_review_repositories_are_refused_by_name() -> None:
    pairs = [pair(0, repository="ethereum/go-ethereum")]
    kept, refused = admit(pairs)

    assert kept == []
    assert refused[0].repository == "ethereum/go-ethereum"
    assert "copyleft" in refused[0].reason


def test_a_review_repository_can_be_admitted_deliberately() -> None:
    pairs = [pair(0, repository="hashicorp/terraform")]
    kept, refused = admit(pairs, include_review=True)

    assert len(kept) == 1
    assert refused == []


def test_an_unknown_repository_is_refused() -> None:
    """A refreshed dataset adding an eleventh repository must not slip through."""
    pairs = [pair(0, repository="someone/else")]
    kept, refused = admit(pairs)

    assert kept == []
    assert "not one of the ten" in refused[0].reason


def test_refusals_are_ordered_deterministically() -> None:
    pairs = [
        pair(0, repository="someone/else"),
        pair(1, repository="hashicorp/terraform"),
        pair(2, repository="ethereum/go-ethereum"),
    ]
    _, refused = admit(pairs)
    assert [r.repository for r in refused] == [
        "ethereum/go-ethereum",
        "hashicorp/terraform",
        "someone/else",
    ]


def test_all_records_of_a_permissive_repository_survive() -> None:
    pairs = [pair(i, repository="moby/moby") for i in range(5)]
    kept, refused = admit(pairs)
    assert len(kept) == 5
    assert refused == []


# ------------------------------------------------------------------- measuring


def healthy(count: int = MIN_DAPT_FILES, repositories: int = 6) -> list[GoPair]:
    """A corpus that clears every threshold."""
    names = [
        "gin-gonic/gin",
        "gohugoio/hugo",
        "golang/go",
        "moby/moby",
        "pingcap/tidb",
        "kserve/kserve",
    ]
    return [
        pair(i, repository=names[i % repositories], directory=f"pkg/d{i}") for i in range(count)
    ]


def test_an_adequate_corpus_has_no_findings() -> None:
    report = measure_dapt(healthy())
    assert report.is_adequate, [f.name for f in report.findings]
    assert report.repositories == 6
    assert report.packages == MIN_DAPT_FILES


def test_an_empty_corpus_is_reported_not_blessed() -> None:
    report = measure_dapt([])
    assert not report.is_adequate
    assert report.findings[0].name == "no_go_sources"


def test_too_few_files_is_reported() -> None:
    report = measure_dapt(healthy(count=MIN_DAPT_FILES - 1))
    assert "too_few_files" in {f.name for f in report.findings}


def test_too_few_repositories_is_reported() -> None:
    report = measure_dapt(healthy(repositories=MIN_DAPT_REPOSITORIES - 1))
    assert "too_few_repositories" in {f.name for f in report.findings}


def test_a_dominant_repository_is_reported() -> None:
    """A 70/30 split is the shape a single-repository corpus actually has."""
    pairs = [
        pair(
            i,
            repository="kubernetes/kubernetes" if i < 700 else "gin-gonic/gin",
            directory=f"pkg/d{i}",
        )
        for i in range(MIN_DAPT_FILES)
    ]
    report = measure_dapt(pairs)
    assert report.largest_repository_share > MAX_REPOSITORY_SHARE
    assert "one_repository_dominates" in {f.name for f in report.findings}


def test_too_few_packages_is_reported() -> None:
    """Many files, one directory: the count looks fine and the variety is absent."""
    names = [
        "gin-gonic/gin",
        "gohugoio/hugo",
        "golang/go",
        "moby/moby",
        "pingcap/tidb",
        "kserve/kserve",
    ]
    pairs = [pair(i, repository=names[i % 6], directory="pkg") for i in range(MIN_DAPT_FILES)]
    report = measure_dapt(pairs)
    assert report.packages == 6 < MIN_DAPT_PACKAGES
    assert "too_few_packages" in {f.name for f in report.findings}


def test_duplicates_are_reported() -> None:
    """A tenth of the corpus repeated is a tenth that teaches nothing new."""
    pairs = healthy()
    for i in range(1, 101):
        pairs[i] = pairs[0]
    report = measure_dapt(pairs)
    assert report.duplicate_share > MAX_DUPLICATE_SHARE
    assert "too_many_duplicates" in {f.name for f in report.findings}


def test_a_few_duplicates_are_tolerated() -> None:
    report = measure_dapt(healthy())
    assert report.duplicate_share == 0.0
    assert report.is_adequate


def test_missing_package_clauses_are_counted() -> None:
    pairs = healthy(count=5)
    pairs[0] = pair(0, repository="golang/go", code="// no clause\n")
    report = measure_dapt(pairs)
    assert report.missing_package_clause == 1
    # Counted, and deliberately not a threshold: the field is informational.
    assert not any(f.name == "missing_package_clause" for f in report.findings)


def test_the_report_carries_the_numbers_it_measured() -> None:
    record_out = measure_dapt(healthy(), refused=[]).to_record()
    assert record_out["adequate"] is True
    assert record_out["files"] == MIN_DAPT_FILES
    assert record_out["bytes"] > 0
    assert set(record_out["repository_counts"]) == {
        "gin-gonic/gin",
        "gohugoio/hugo",
        "golang/go",
        "moby/moby",
        "pingcap/tidb",
        "kserve/kserve",
    }


def test_the_report_includes_what_was_refused() -> None:
    _, refused = admit([pair(0, repository="hashicorp/terraform")])
    record_out = measure_dapt(healthy(count=5), refused=refused).to_record()
    assert record_out["refused"][0]["repository"] == "hashicorp/terraform"


# ---------------------------------------------------------------- deduplicating


def test_duplicate_sources_are_removed() -> None:
    """The published splits overlap; a repeated file is the same file twice."""
    from gotooltrain.gocorpus import deduplicate

    pairs = [pair(0), pair(1), pair(0)]
    unique, removed = deduplicate(pairs)
    assert len(unique) == 2
    assert removed == 1
    assert [p.sha256 for p in unique] == [pairs[0].sha256, pairs[1].sha256]


def test_deduplication_keeps_the_first_occurrence() -> None:
    """Order decides which survives, so the result is reproducible."""
    from gotooltrain.gocorpus import deduplicate

    first = pair(0, repository="gin-gonic/gin")
    second = pair(0, repository="moby/moby")
    unique, _ = deduplicate([first, second])
    assert unique[0].repository == "gin-gonic/gin"


def test_deduplication_reports_nothing_removed_for_a_clean_corpus() -> None:
    from gotooltrain.gocorpus import deduplicate

    unique, removed = deduplicate(healthy(count=10))
    assert len(unique) == 10
    assert removed == 0


def test_the_report_carries_the_number_removed() -> None:
    """A removal the report does not show is an unexplained size difference."""
    from gotooltrain.gocorpus import deduplicate

    unique, removed = deduplicate([pair(0), pair(0)])
    record_out = measure_dapt(unique, duplicates_removed=removed).to_record()
    assert record_out["duplicates_removed"] == 1


# --------------------------------------------------------------------- output


def test_dapt_records_carry_the_provenance() -> None:
    records = dapt_records([pair(0)])
    assert set(records[0]) == {"text", "metadata"}
    metadata = records[0]["metadata"]
    assert metadata["repository"] == "kubernetes/kubernetes"
    assert metadata["commit"] == "c" * 40
    assert metadata["path"].endswith(".go")


def test_unit_test_messages_ask_for_the_test() -> None:
    messages = unit_test_messages(pair(0))
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert pair(0).code_path in messages[0]["content"]
    assert messages[1]["content"] == pair(0).test


def test_unit_test_messages_are_renderable() -> None:
    """A record the template cannot render is not a record."""
    from gotooltrain import normalize_conversation

    conversation = normalize_conversation(unit_test_messages(pair(0)))
    assert conversation.messages[0].role == "user"
    assert conversation.messages[1].role == "assistant"


# ----------------------------------------------------------------- downloading


def test_every_published_split_is_fetched(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning the dataset id is what makes an import reproducible."""
    import huggingface_hub

    from gotooltrain import gocorpus

    seen: list[tuple[str, str]] = []

    def fake_download(repo_id, filename, *, repo_type, local_dir):  # type: ignore[no-untyped-def]
        """Record the request and materialise a file, as the hub would."""
        seen.append((repo_id, filename))
        assert repo_type == "dataset"
        path = pathlib.Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([record()]), encoding="utf-8")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    paths = gocorpus.download_splits(tmp_path / "download")

    assert [f for _, f in seen] == list(gocorpus.SPLITS)
    assert all(repo == gocorpus.DATASET_ID for repo, _ in seen)
    assert len(paths) == len(gocorpus.SPLITS)
    assert all(p.is_file() for p in paths)


def test_a_missing_hub_is_a_setup_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dependency must name itself, not produce an empty corpus."""
    import sys

    from gotooltrain.gocorpus import download_splits

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(DatasetError, match="huggingface_hub"):
        download_splits(tmp_path / "download")


def test_splits_are_concatenated_in_order(tmp_path: pathlib.Path) -> None:
    from gotooltrain.gocorpus import read_splits

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps([record(repository="moby/moby", path="a/f.go")]), encoding="utf-8")
    second.write_text(json.dumps([record(repository="golang/go", path="b/g.go")]), encoding="utf-8")

    pairs = read_splits([first, second])
    assert [p.repository for p in pairs] == ["moby/moby", "golang/go"]
