"""Go execution harness: argv safety, status separation, canary, parallel, pass@k."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import pytest

from gotooltrain import HarnessError, ValidationError
from gotooltrain.gorun import (
    TIMEOUT_EXIT,
    CanaryCase,
    ExecRequest,
    LocalExecutor,
    Status,
    build_argv,
    execute,
    pass_at_k,
    plan,
    run_canary,
    run_many,
    summarise,
    validate_arguments,
)
from gotooltrain.gotools import get


class ScriptedExecutor:
    """Executor returning canned results, so no Go toolchain is needed."""

    def __init__(self, exit_code: int = 0, stdout: str = "ok", stderr: str = "") -> None:
        """Record every argv so tests can assert what would have run."""
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def run(self, argv, request):  # type: ignore[no-untyped-def]
        """Pretend to execute, recording the argv verbatim."""
        self.calls.append(list(argv))
        return self.exit_code, self.stdout, self.stderr


def request(tool: str = "go_test", timeout_s: int = 120, **arguments: Any) -> ExecRequest:
    return ExecRequest(
        task_id="t1",
        sample_index=0,
        tool_name=tool,
        arguments=arguments or {"pkg": "./..."},
        # An empty Path is the *current* directory. Most tests here only plan an
        # argv and never touch it, but a commandless tool such as write_file
        # mutates its workspace for real, so a test that shares this default
        # writes into the source tree. Anything that mutates passes its own
        # tmp_path instead.
        workspace=pathlib.Path(),
        timeout_s=timeout_s,
    )


def request_in(
    workspace: pathlib.Path, tool: str = "go_test", timeout_s: int = 120, **arguments: Any
) -> ExecRequest:
    """The same request, rooted at a workspace the test owns."""
    base = request(tool=tool, timeout_s=timeout_s, **arguments)
    return dataclasses.replace(base, workspace=workspace)


# ------------------------------------------------------------- argv building


def test_template_placeholders_become_separate_argv_entries() -> None:
    argv = build_argv(get("go_test"), {"pkg": "./parser"})
    assert argv == ["rtk", "go", "test", "./parser"]


def test_grep_uses_its_optional_path() -> None:
    # -r is not decoration: without it rtk hands the path to grep, which refuses a
    # directory outright, so every realistic grep call failed.
    assert build_argv(get("grep"), {"pattern": "func "}) == ["rtk", "grep", "-r", "func ", ""]


def test_argument_values_are_never_shell_interpreted() -> None:
    """A model-supplied value must stay one argv entry, whatever it contains."""
    hostile = "x; rm -rf / && $(whoami) `id` | tee /tmp/p"
    argv = build_argv(get("go_test"), {"pkg": hostile})
    assert argv == ["rtk", "go", "test", hostile]
    assert len(argv) == 4
    assert hostile not in " ".join(argv[:-1])


def test_commandless_tools_produce_no_argv() -> None:
    """Only the two workspace mutations have no command; they are applied directly."""
    assert build_argv(get("write_file"), {"path": "a.go", "content": "x"}) == []
    assert (
        build_argv(get("edit_file"), {"path": "a.go", "old_string": "a", "new_string": "b"}) == []
    )
    assert build_argv(get("go_mod_tidy"), {}) == ["go", "mod", "tidy"]
    assert build_argv(get("go_doc"), {"symbol": "errors.Is"}) == ["go", "doc", "errors.Is"]


def test_missing_required_argument_is_a_model_error() -> None:
    with pytest.raises(ValidationError, match="missing required argument"):
        validate_arguments(get("go_test"), {})


def test_unexpected_argument_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unexpected argument"):
        validate_arguments(get("go_test"), {"pkg": "./...", "race": True})


def test_non_string_argument_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be a string"):
        validate_arguments(get("go_test"), {"pkg": 3})


def test_validate_returns_only_declared_keys_in_schema_order() -> None:
    checked = validate_arguments(
        get("edit_file"), {"new_string": "b", "path": "a", "old_string": "a"}
    )
    assert list(checked) == ["path", "old_string", "new_string"]


def test_plan_rejects_an_unknown_tool() -> None:
    with pytest.raises(ValidationError, match="unknown tool"):
        plan(ExecRequest("t", 0, "rm_rf", {}, pathlib.Path()))


# ------------------------------------------------------------- classification


def test_success_is_ok() -> None:
    result = execute(request(), ScriptedExecutor(exit_code=0, stdout="ok parser"))
    assert result.status is Status.OK
    assert result.exit_code == 0
    assert result.stdout == "ok parser"
    assert result.is_model_visible


def test_failing_tests_are_tool_error_not_harness_error() -> None:
    """A red test suite is a legitimate result the model must learn to read."""
    result = execute(request(), ScriptedExecutor(exit_code=1, stdout="FAIL: undefined: foo"))
    assert result.status is Status.TOOL_ERROR
    assert result.is_model_visible
    assert "FAIL" in result.stdout


def test_model_error_carries_no_stdout() -> None:
    result = execute(request(tool="go_test"), ScriptedExecutor())
    assert result.status is Status.OK
    bad = execute(ExecRequest("t", 0, "go_test", {}, pathlib.Path()), ScriptedExecutor())
    assert bad.status is Status.MODEL_ERROR
    assert bad.stdout == ""
    assert not bad.is_model_visible


def test_unknown_tool_is_a_model_error() -> None:
    result = execute(request(tool="nope"), ScriptedExecutor())
    assert result.status is Status.MODEL_ERROR
    assert "unknown tool" in result.harness_error


def test_harness_error_is_separate_from_tool_error() -> None:
    class Broken:
        def run(self, argv, request):  # type: ignore[no-untyped-def]
            raise HarnessError("docker daemon unreachable")

    result = execute(request(), Broken())
    assert result.status is Status.HARNESS_ERROR
    assert "docker" in result.harness_error
    assert not result.is_model_visible


def test_stderr_is_used_when_stdout_is_blank() -> None:
    result = execute(request(), ScriptedExecutor(exit_code=2, stdout="   ", stderr="build failed"))
    assert result.stdout == "build failed"


def test_tool_output_is_truncated_to_the_declared_budget() -> None:
    noisy = "\n".join(f"log line {i}" for i in range(5000)) + "\nFAIL: undefined: foo"
    result = execute(request(), ScriptedExecutor(exit_code=1, stdout=noisy))
    budget = get("go_test").output_budget_chars
    assert len(result.stdout) <= budget
    assert "FAIL: undefined: foo" in result.stdout
    assert "truncated" in result.stdout


def test_commandless_tool_succeeds_without_shelling_out(tmp_path: pathlib.Path) -> None:
    executor = ScriptedExecutor()
    result = execute(
        request_in(tmp_path, tool="write_file", path="a.go", content="package a"), executor
    )
    assert result.status is Status.OK
    assert executor.calls == []
    assert (tmp_path / "a.go").read_text(encoding="utf-8") == "package a"


def test_result_serialises_for_the_store() -> None:
    record = execute(request(), ScriptedExecutor()).to_record()
    assert set(record) == {
        "task_id",
        "sample_index",
        "tool_name",
        "status",
        "exit_code",
        "stdout",
        "duration_ms",
        "harness_error",
        "call_index",
    }


# ------------------------------------------------------------------- canary


def _canary_cases() -> list[CanaryCase]:
    return [
        CanaryCase("passing", request(tool="go_test", pkg="./ok"), should_fail=False),
        CanaryCase("failing", request(tool="go_test", pkg="./bad"), should_fail=True),
    ]


def test_canary_passes_when_the_executor_discriminates() -> None:
    class Discriminating:
        def run(self, argv, request):  # type: ignore[no-untyped-def]
            return (0, "ok", "") if "./ok" in argv else (1, "FAIL", "")

    run_canary(Discriminating(), _canary_cases())


def test_canary_rejects_an_executor_that_always_succeeds() -> None:
    """This is the failure that would otherwise report 100% on every run."""
    with pytest.raises(HarnessError, match="cannot discriminate"):
        run_canary(ScriptedExecutor(exit_code=0), _canary_cases())


def test_canary_rejects_an_executor_that_always_fails() -> None:
    with pytest.raises(HarnessError, match="cannot discriminate"):
        run_canary(ScriptedExecutor(exit_code=1), _canary_cases())


def test_canary_requires_a_non_empty_probe_set() -> None:
    with pytest.raises(HarnessError, match="canary set is empty"):
        run_canary(ScriptedExecutor(), [])


# ------------------------------------------------------------------ parallel


def _requests(count: int) -> list[ExecRequest]:
    return [
        ExecRequest(f"t{i}", i, "go_test", {"pkg": f"./p{i}"}, pathlib.Path()) for i in range(count)
    ]


def test_run_many_preserves_input_order() -> None:
    results = run_many(_requests(6), ScriptedExecutor(), max_parallel=4)
    assert [r.task_id for r in results] == [f"t{i}" for i in range(6)]
    assert [r.sample_index for r in results] == list(range(6))


def test_run_many_reports_each_result_as_it_lands() -> None:
    seen: list[str] = []
    run_many(
        _requests(4),
        ScriptedExecutor(),
        max_parallel=2,
        on_result=lambda r: seen.append(r.task_id),
    )
    assert sorted(seen) == ["t0", "t1", "t2", "t3"]


def test_run_many_skips_completed_work_on_resume() -> None:
    executor = ScriptedExecutor()
    requests = _requests(4)
    run_many(requests, executor, should_run=lambda r: r.task_id not in {"t1", "t2"})
    assert len(executor.calls) == 2
    assert all("./p1" not in call and "./p2" not in call for call in executor.calls)


def test_run_many_with_nothing_left_to_do() -> None:
    executor = ScriptedExecutor()
    assert run_many(_requests(2), executor, should_run=lambda _r: False) == []
    assert executor.calls == []


def test_run_many_rejects_a_nonsense_concurrency() -> None:
    with pytest.raises(HarnessError, match="max_parallel must be"):
        run_many(_requests(1), ScriptedExecutor(), max_parallel=0)


# ------------------------------------------------------------------- pass@k


def test_pass_at_1_is_the_success_rate() -> None:
    assert pass_at_k(4, 2, 1) == 0.5
    assert pass_at_k(4, 0, 1) == 0.0
    assert pass_at_k(4, 4, 1) == 1.0


def test_pass_at_k_is_the_unbiased_estimator() -> None:
    # n=4, c=1, k=2 -> 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    # n=5, c=2, k=3 -> 1 - C(3,3)/C(5,3) = 1 - 1/10 = 0.9
    assert pass_at_k(5, 2, 3) == pytest.approx(0.9)


def test_pass_at_k_saturates_when_k_exceeds_samples() -> None:
    assert pass_at_k(2, 1, 8) == 1.0
    assert pass_at_k(2, 0, 8) == 0.0


def test_pass_at_k_rejects_nonsense() -> None:
    with pytest.raises(HarnessError, match="invalid pass@k"):
        pass_at_k(2, 3, 1)
    with pytest.raises(HarnessError, match="invalid pass@k"):
        pass_at_k(-1, 0, 1)
    with pytest.raises(HarnessError, match="k must be"):
        pass_at_k(2, 1, 0)


# --------------------------------------------------------------- aggregation


def test_summarise_groups_and_sorts_by_task() -> None:
    results = run_many(_requests(3), ScriptedExecutor(exit_code=0), max_parallel=2)
    summaries = summarise(results)
    assert [s.task_id for s in summaries] == ["t0", "t1", "t2"]
    assert all(s.pass_at_1 == 1.0 for s in summaries)


def test_summarise_counts_each_status_separately() -> None:
    def result(task: str, index: int, status: Status) -> Any:
        return type(
            "R",
            (),
            {
                "task_id": task,
                "sample_index": index,
                "status": status,
                "tool_name": "go_test",
                "exit_code": 0,
                "stdout": "",
                "duration_ms": 0,
                "harness_error": "",
            },
        )()

    summary = summarise(
        [
            result("t", 0, Status.OK),
            result("t", 1, Status.TOOL_ERROR),
            result("t", 2, Status.HARNESS_ERROR),
            result("t", 3, Status.MODEL_ERROR),
        ]
    )[0]
    assert summary.samples == 4
    assert summary.passed == 1
    assert summary.tool_errors == 1
    assert summary.harness_errors == 1
    assert summary.model_errors == 1
    assert summary.pass_at_1 == 0.25


def test_summarise_is_order_independent() -> None:
    """Parallel completion order must not leak into the report."""
    results = run_many(_requests(5), ScriptedExecutor(), max_parallel=5)
    assert summarise(results) == summarise(list(reversed(results)))


def test_summary_serialises() -> None:
    record = summarise(run_many(_requests(1), ScriptedExecutor()))[0].to_record()
    assert record["task_id"] == "t0"
    assert record["pass_at_n"] == 1.0


# ------------------------------------------------------------- local executor


def test_local_executor_times_out(tmp_path: pathlib.Path) -> None:
    """A hung command must be killed and reported, never left running."""
    import sys

    executor = LocalExecutor(cwd=tmp_path)
    code, out, err = executor.run(
        [sys.executable, "-c", "import time; time.sleep(30)"], request(timeout_s=1)
    )
    assert code == TIMEOUT_EXIT
    assert out == ""
    assert "timed out" in err


def test_local_executor_returns_a_real_command_result(tmp_path: pathlib.Path) -> None:
    import sys

    executor = LocalExecutor(cwd=tmp_path)
    code, out, _err = executor.run([sys.executable, "-c", "print('hello')"], request(timeout_s=30))
    assert code == 0
    assert "hello" in out


def test_local_executor_runs_a_real_command(tmp_path: pathlib.Path) -> None:
    if shutil_which("go") is None:
        pytest.skip("no Go toolchain on this machine")
    code, _out, _err = LocalExecutor(cwd=tmp_path).run(["go", "version"], request(timeout_s=30))
    assert code == 0


def test_local_executor_reports_a_missing_binary(tmp_path: pathlib.Path) -> None:
    executor = LocalExecutor(cwd=tmp_path)
    with pytest.raises(HarnessError, match="not found"):
        executor.run(["definitely-not-a-real-binary-xyz"], request())


def test_local_executor_handles_a_commandless_request(tmp_path: pathlib.Path) -> None:
    assert LocalExecutor(cwd=tmp_path).run([], request()) == (0, "", "")


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)
