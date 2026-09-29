"""Corpus measurement: what the data actually teaches, stated as numbers.

The failure this exists to prevent is picking a target like "50k examples" and
then measuring only the total, which stays green while the corpus teaches one
repository, never exercises `edit_file`, and consists of single turns.

Each test names the specific way a corpus can look adequate and be useless.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gotooltrain.corpus import (
    MIN_SUPERVISED_FRACTION,
    MIN_TOOL_COVERAGE,
    MIN_TOOL_EXAMPLES,
    Trajectory,
    measure,
    read_trajectories,
    summarise_targets,
)
from gotooltrain.errors import DatasetError
from gotooltrain.gotools import GO_TOOLS

CATALOGUE = sorted(t.name for t in GO_TOOLS)


def traj(
    index: int,
    *,
    tools: tuple[str, ...] = (),
    repository: str | None = None,
    package: str | None = None,
    supervised: int = 200,
    total: int = 800,
    thinking: bool = True,
    images: bool = False,
) -> Trajectory:
    return Trajectory(
        trajectory_id=f"t{index:05d}",
        repository=repository if repository is not None else f"repo{index % 40:03d}",
        package=package if package is not None else f"pkg{index:05d}",
        tools=tools,
        supervised_tokens=supervised,
        total_tokens=total,
        has_thinking=thinking,
        has_images=images,
    )


def healthy(count: int = 400) -> list[Trajectory]:
    """A corpus that clears every threshold: every tool, multi-turn, varied."""
    out: list[Trajectory] = []
    for index in range(count):
        # Rotate through the catalogue so each tool lands in ~1/9 of records,
        # comfortably over the coverage floor, with at least two calls each.
        first = CATALOGUE[index % len(CATALOGUE)]
        second = CATALOGUE[(index + 1) % len(CATALOGUE)]
        third = CATALOGUE[(index + 4) % len(CATALOGUE)]
        out.append(traj(index, tools=(first, second, third)))
    return out


# ------------------------------------------------------------------- the basics


def test_an_adequate_corpus_has_no_findings() -> None:
    report = measure(healthy())
    assert report.is_adequate, [f.name for f in report.findings]
    assert report.trajectories == 400


def test_the_report_carries_the_measured_numbers() -> None:
    record = measure(healthy(400)).to_record()
    assert record["trajectories"] == 400
    assert record["multi_turn_fraction"] == 1.0
    assert record["adequate"] is True
    assert record["findings"] == []


def test_a_corpus_too_small_to_teach_the_catalogue_is_reported() -> None:
    """Report every tool as under-taught when the call count is below the floor.

    A structurally clean corpus that is simply too small teaches nothing, and a
    clean-looking total is exactly what hides that.
    """
    report = measure(healthy(50))
    assert report.multi_turn_fraction == 1.0
    assert not report.is_adequate
    under = [f for f in report.findings if f.name.startswith("tool_undertrained:")]
    assert len(under) == len(CATALOGUE)


# ------------------------------------------------------- tools are the point


def test_a_tool_that_is_never_called_is_reported() -> None:
    """A catalogue member with no signal is worse than an absent one."""
    without = [traj(i, tools=("go_test", "go_test")) for i in range(200)]
    report = measure(without)
    names = {f.name for f in report.findings}
    assert "tool_undertrained:edit_file" in names
    assert "tool_undertrained:go_test" not in names


def test_every_catalogue_tool_is_accounted_for() -> None:
    report = measure(healthy())
    for name in CATALOGUE:
        assert report.tool_counts.get(name, 0) >= MIN_TOOL_EXAMPLES, name
        assert report.tool_trajectory_coverage.get(name, 0.0) >= MIN_TOOL_COVERAGE


def test_a_tool_called_often_in_one_trajectory_is_not_coverage() -> None:
    """Fifty calls in one record is one demonstration, not fifty."""
    report = measure([traj(i, tools=("go_test",) * 50 if i == 0 else ()) for i in range(100)])
    assert report.tool_counts["go_test"] == 50
    assert report.tool_trajectory_coverage["go_test"] == pytest.approx(0.01)


def test_a_tool_outside_the_catalogue_is_reported() -> None:
    """Training on a tool the model will never be offered teaches false affordances."""
    report = measure([traj(i, tools=("go_test", "curl_the_web")) for i in range(50)])
    assert any(f.name == "tools_outside_catalogue" for f in report.findings)


# ----------------------------------------------------------------- structure


def test_a_single_turn_corpus_is_reported() -> None:
    """Single-turn data teaches a single turn; the premise is an agent loop."""
    report = measure([traj(i, tools=("go_test",)) for i in range(300)])
    assert report.multi_turn_fraction == 0.0
    assert any(f.name == "too_few_multi_turn" for f in report.findings)


def test_a_prompt_heavy_corpus_is_reported() -> None:
    report = measure(
        [traj(i, tools=("go_test", "go_build"), supervised=10, total=900) for i in range(300)]
    )
    assert report.supervised_fraction < MIN_SUPERVISED_FRACTION
    assert any(f.name == "too_few_supervised_tokens" for f in report.findings)


def test_one_repository_dominating_is_reported() -> None:
    report = measure([traj(i, tools=("go_test", "go_build"), repository="one") for i in range(300)])
    assert report.largest_repository_fraction == 1.0
    assert any(f.name == "one_repository_dominates" for f in report.findings)


def test_too_few_packages_is_reported() -> None:
    report = measure(
        [
            traj(i, tools=("go_test", "go_build"), repository="one", package="same")
            for i in range(300)
        ]
    )
    assert report.packages == 1
    assert any(f.name == "too_few_packages" for f in report.findings)


def test_thinking_and_image_shares_are_measured() -> None:
    report = measure(
        [
            traj(i, tools=("go_test", "go_build"), thinking=i % 2 == 0, images=i % 10 == 0)
            for i in range(200)
        ]
    )
    assert report.thinking_fraction == pytest.approx(0.5)
    assert report.image_fraction == pytest.approx(0.1)


# ------------------------------------------------------------------- targets


def test_targets_are_counts_not_advice() -> None:
    report = measure([traj(i, tools=("go_test",)) for i in range(300)])
    targets = summarise_targets(report)
    assert targets, "a corpus with no edit_file must produce a target"
    worst = targets[0]
    assert isinstance(worst["trajectories_needed"], int)
    assert worst["trajectories_needed"] >= 0
    # Targets are ordered by how much data they need, most first.
    counts = [t["trajectories_needed"] for t in targets]
    assert counts == sorted(counts, reverse=True)


def test_an_adequate_corpus_needs_no_targets() -> None:
    assert summarise_targets(measure(healthy())) == []


def test_a_multi_turn_deficit_is_quantified() -> None:
    report = measure([traj(i, tools=("go_test",)) for i in range(100)])
    target = next(t for t in summarise_targets(report) if t["tool"] is None)
    assert target["trajectories_needed"] > 0
    assert target["current_coverage"] == 0.0


# ---------------------------------------------------------------------- io


def test_trajectories_are_read_from_jsonl(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "t1",
                "repository": "repo",
                "package": "pkg",
                "tools": ["go_test", "edit_file"],
                "supervised_tokens": 100,
                "total_tokens": 400,
            }
        )
        + "\n\n",
        encoding="utf-8",
    )
    items = read_trajectories(path)
    assert len(items) == 1
    assert items[0].is_multi_turn
    assert items[0].supervised_fraction == 0.25


def test_a_missing_field_is_named(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"id": "t1", "repository": "r"}) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="missing field 'package'"):
        read_trajectories(path)


def test_a_corrupt_line_is_named(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        read_trajectories(path)


def test_a_missing_corpus_file_is_reported(tmp_path: pathlib.Path) -> None:
    with pytest.raises(DatasetError, match="corpus file not found"):
        read_trajectories(tmp_path / "nope.jsonl")


def test_an_empty_corpus_is_reported_as_teaching_nothing() -> None:
    """Zero trajectories must never read as "adequate".

    Reporting an empty corpus as sufficient is the worst version of this failure:
    a pipeline that checks only ``is_adequate`` would train on nothing and call it
    a clean run.
    """
    report = measure([])
    assert report.trajectories == 0
    assert not report.is_adequate
    under = [f for f in report.findings if f.name.startswith("tool_undertrained:")]
    assert len(under) == len(CATALOGUE)
    assert {f.name for f in report.findings} >= {
        "too_few_multi_turn",
        "too_few_supervised_tokens",
        "too_few_packages",
    }
    assert report.to_record()["adequate"] is False


# ------------------------------------------------------------- boundary cases


def test_a_trajectory_with_no_tools_still_has_one_turn() -> None:
    """A record where the model answered directly produced one output, not zero."""
    assert Trajectory(trajectory_id="t", repository="r", package="p", tools=()).turns == 1


def test_a_record_with_no_token_count_has_no_supervised_share() -> None:
    """An unmeasured record must not silently claim a 0% or 100% share."""
    assert (
        Trajectory(
            trajectory_id="t", repository="r", package="p", total_tokens=0
        ).supervised_fraction
        == 0.0
    )


def test_enough_trajectories_but_too_few_calls_is_a_distinct_finding() -> None:
    """A tool can appear in plenty of records and still be rare in absolute terms.

    Coverage is calls divided by trajectories, so the two thresholds move
    independently: 40 calls over 180 records clears the 2% coverage floor and
    misses the 50-call floor, and that needs calls rather than new records.
    """
    many = [traj(i, tools=("go_test",)) for i in range(140)]
    report = measure(many + [traj(1000 + i, tools=("edit_file",)) for i in range(40)])
    assert report.tool_counts["edit_file"] == 40 < MIN_TOOL_EXAMPLES
    assert report.tool_trajectory_coverage["edit_file"] > MIN_TOOL_COVERAGE
    assert any(f.name == "tool_undertrained:edit_file" for f in report.findings)
    target = {t["tool"]: t for t in summarise_targets(report)}["edit_file"]
    assert target["blocked_by"] == "calls"
    assert target["missing_calls"] == MIN_TOOL_EXAMPLES - 40
    assert target["trajectories_needed"] == 0


def test_a_tool_used_in_too_few_records_needs_its_records_extended() -> None:
    """Enough calls in total, but confined to too few records.

    Adding more records without spreading the tool would not move the fraction, so
    the target counts records to *change*, not records to add.
    """
    many = [traj(i, tools=("go_test",)) for i in range(4000)]
    report = measure(many + [traj(9000 + i, tools=("edit_file",)) for i in range(60)])
    assert report.tool_counts["edit_file"] == 60 >= MIN_TOOL_EXAMPLES
    assert report.tool_trajectory_coverage["edit_file"] < MIN_TOOL_COVERAGE
    assert any(f.name == "tool_undertrained:edit_file" for f in report.findings)
    target = {t["tool"]: t for t in summarise_targets(report)}["edit_file"]
    assert target["blocked_by"] == "coverage"
    assert target["missing_calls"] == 0
    assert target["trajectories_needed"] == 0
    assert target["records_to_extend"] > 0


def test_the_catalogue_being_empty_is_not_a_reason_to_skip_findings() -> None:
    """Structural findings fire independently of the per-tool checks."""
    report = measure([traj(i, tools=("go_test",)) for i in range(20)])
    names = {f.name for f in report.findings}
    assert "too_few_multi_turn" in names
    assert "too_few_supervised_tokens" in names or report.supervised_fraction >= 0
    assert "too_few_packages" in names


def test_one_repository_dominating_does_not_fire_when_varied() -> None:
    report = measure([traj(i, tools=("go_test", "go_build")) for i in range(400)])
    assert report.largest_repository_fraction < 0.5
    assert not any(f.name == "one_repository_dominates" for f in report.findings)


def test_multi_turn_is_not_reported_when_the_corpus_has_it() -> None:
    report = measure(healthy(400))
    assert not any(f.name == "too_few_multi_turn" for f in report.findings)


def test_supervision_is_not_reported_when_there_is_enough() -> None:
    report = measure(healthy(400))
    assert not any(f.name == "too_few_supervised_tokens" for f in report.findings)


def test_reading_keeps_blank_lines_out_of_the_corpus(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"id": "a", "repository": "r", "package": "p"})
        + "\n\n"
        + json.dumps({"id": "b", "repository": "r", "package": "p"})
        + "\n\n",
        encoding="utf-8",
    )
    assert len(read_trajectories(path)) == 2
