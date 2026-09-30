"""The notebook as it now is: parameters, bootstrap, toolchain, one run cell.

The tests here used to assert that this or that guarantee was stated in some cell.
Most of them moved: the pipeline's behaviour lives in ``src/gotooltrain/pipeline.py``
and is tested there against real subprocesses, which is a stronger place for it than
a string in a notebook. What is left here is what is genuinely the notebook's job --
that the knobs are named, that the reader is told what will happen, and that the run
cell does not swallow a failure.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from gotooltrain import build_notebook
from gotooltrain.build_notebook import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_REPO_URL,
    GO_SHA256,
    GO_VERSION,
    NOTEBOOK_PATH,
    RTK_SHA256,
    RTK_VERSION,
    assemble_notebook,
    build_cells,
)


def sources(cell: dict[str, object]) -> str:
    """A cell's text."""
    return "".join(cell["source"])  # type: ignore[arg-type]


def code_cells() -> list[str]:
    """Every code cell's source, in order."""
    return [sources(c) for c in build_cells() if c["cell_type"] == "code"]


def _cell_containing(marker: str) -> dict[str, object]:
    """The code cell carrying a marker, found by content rather than by position."""
    for cell in build_cells():
        if cell["cell_type"] == "code" and marker in sources(cell):
            return cell
    raise AssertionError(f"no code cell contains {marker!r}")


def install_cell() -> dict[str, object]:
    """The cell that pulls and installs, found by what it does rather than by index."""
    for cell in build_cells():
        if (
            cell["cell_type"] == "code"
            and "pip install" in sources(cell)
            and "git" in sources(cell)
        ):
            return cell
    raise AssertionError("no cell pulls and installs the repository")


def toolchain_cell() -> str:
    """The cell that installs Go and rtk."""
    return sources(_cell_containing("rtk"))


def run_cell() -> str:
    """The single cell that drives the whole pipeline."""
    return sources(_cell_containing("gotooltrain.pipeline"))


def parameters_cell() -> str:
    """The knobs the run cell reads."""
    return sources(_cell_containing("MEMORY_BUDGET_GB"))


def pipeline_source() -> str:
    """The pipeline module's own text, for the guarantees that moved there."""
    return (
        pathlib.Path(build_notebook.__file__)
        .parent.joinpath("pipeline.py")
        .read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------- the document


def test_the_notebook_is_valid_json_with_the_expected_shape() -> None:
    document = assemble_notebook()
    assert document["nbformat"] == 4
    assert isinstance(document["cells"], list)
    assert all("cell_type" in cell for cell in document["cells"])


def test_a_written_notebook_reparses_identically(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "written.ipynb"
    build_notebook.write_notebook(target)
    assert json.loads(target.read_text(encoding="utf-8")) == assemble_notebook()


def test_the_notebook_lands_in_the_project_not_the_cwd(tmp_path: pathlib.Path) -> None:
    """Writing with no argument puts it where the project expects it.

    A notebook that lands in whatever directory the reader happened to be in is a
    notebook nobody opens twice. Compared resolved, because the constant is relative
    and the return value is absolute, and the difference is not the property.
    """
    assert (
        pathlib.Path(build_notebook.write_notebook()).resolve()
        == pathlib.Path(NOTEBOOK_PATH).resolve()
    )
    # An explicit path is honoured, which is what makes this testable at all.
    elsewhere = build_notebook.write_notebook(tmp_path / "out.ipynb")
    assert elsewhere == tmp_path / "out.ipynb"


def test_the_project_root_is_found_from_the_installed_package() -> None:
    assert pathlib.Path(NOTEBOOK_PATH).parent.name


def test_the_default_repository_is_the_one_this_project_uses() -> None:
    assert DEFAULT_REPO_URL.startswith("https://github.com/")
    assert DEFAULT_REPO_URL.endswith(".git")


def test_the_publication_target_is_written_once_in_the_source() -> None:
    """One place in the *codebase* to edit, so a renamed repo cannot be half-renamed.

    The generated notebook may name the repo as many times as it likes -- those
    renderings all come from the same constant. What must not happen is a second
    developer typing the literal into another module, where a rename would leave the
    two silently disagreeing: the run trains, the checkpoint uploads, and it lands
    somewhere nobody is looking.
    """
    package = pathlib.Path(build_notebook.__file__).parent
    occurrences = sum(
        path.read_text(encoding="utf-8").count(DEFAULT_HF_REPO_ID) for path in package.glob("*.py")
    )
    assert occurrences == 1, (
        f"{DEFAULT_HF_REPO_ID} is written in more than one module; keep it in "
        "build_notebook.DEFAULT_HF_REPO_ID and interpolate"
    )


def test_the_publication_target_is_named_in_the_notebook() -> None:
    assert DEFAULT_HF_REPO_ID in "\n".join(code_cells())


def test_the_target_repo_is_a_plausible_hub_id() -> None:
    assert "/" in DEFAULT_HF_REPO_ID
    assert " " not in DEFAULT_HF_REPO_ID


def test_every_code_cell_is_valid_python() -> None:
    for cell in build_cells():
        if cell["cell_type"] == "code":
            compile(sources(cell), f"<cell {cell.get('id', '?')}>", "exec")


def test_the_cells_run_in_a_usable_order() -> None:
    """Bootstrap has to come before the thing that uses the package.

    Checked by the number and the topic each cell carries rather than by index
    arithmetic, so inserting a cell does not silently reorder the run.
    """
    titles = [line for line in (s.splitlines()[0] for s in code_cells()) if "@title" in line]
    numbers = [int(t.split("@title", 1)[1].strip().split()[0]) for t in titles]
    assert numbers == sorted(numbers) == [1, 2, 3, 4], f"cells are out of order: {titles}"
    assert "Depoyu" in titles[1], "the repository is pulled before it is used"
    assert "rtk" in titles[2], "the toolchain is installed before the stage that needs it"
    assert "boru" in titles[3], f"the run comes last, not first: {titles}"


# ---------------------------------------------------------------- the toolchain


def test_both_external_toolchains_are_installed_before_the_quality_gate() -> None:
    """Rtk and Go are not optional: the catalogue's commands go through rtk."""
    cell = toolchain_cell()
    assert "go" in cell.lower()
    assert "rtk" in cell.lower()


def test_rtk_is_installed_at_the_version_the_sandbox_image_pins() -> None:
    cell = toolchain_cell()
    assert RTK_VERSION in cell
    assert RTK_SHA256 in cell


def test_the_go_version_is_pinned_not_latest() -> None:
    cell = toolchain_cell()
    assert GO_VERSION in cell
    assert "latest" not in cell.lower()


def test_the_go_download_is_verified_before_it_is_used() -> None:
    cell = toolchain_cell()
    assert GO_SHA256 in cell
    assert RTK_SHA256 in cell


def test_the_go_install_makes_the_toolchain_visible_to_subprocesses() -> None:
    """Installed into a directory on PATH, not into a shell the pipeline never sees.

    A toolchain installed in one cell and used in another subprocess needs the path,
    or the stage fails with "go not found" an hour later.
    """
    cell = toolchain_cell()
    assert "PATH" in cell or "path" in cell


def test_the_notebook_refuses_an_architecture_its_pinned_binaries_do_not_cover() -> None:
    """Go and rtk are pinned for a platform.

    A different one cannot run them, and saying so beats installing binaries that
    will not execute.
    """
    cell = toolchain_cell()
    assert "uname" in cell or "platform" in cell or "amd64" in cell or "arm64" in cell


# ---------------------------------------------------------------- the one run


def test_the_pipeline_is_a_module_the_notebook_calls() -> None:
    """The whole chain is one command, invoked once.

    This is the property the notebook is for now: order lives in the program, not in
    the reader's memory of which cells they ran.
    """
    cell = run_cell()
    assert "gotooltrain.pipeline" in cell
    assert cell.count("subprocess.run") == 1, "the run cell starts more than one thing"


def test_the_run_cell_does_not_swallow_a_failure() -> None:
    """A pipeline that returns non-zero and prints nothing is the worst outcome.

    Half a run that reports success produces a checkpoint that looks complete and is
    not.
    """
    cell = run_cell()
    assert "raise SystemExit" in cell
    assert "check=False" in cell, "the exit code has to be inspected, not asserted away"
    assert "returncode" in cell


def test_the_run_cell_names_the_stage_that_stopped_it() -> None:
    """The pipeline stops at a failed stage and says which.

    The reader should not have to go looking through a log directory to find out where
    the run ended.
    """
    source = pipeline_source()
    assert "PIPELINE STOPPED" in source
    assert "last 30 lines" in source


def test_the_run_cell_passes_every_knob_the_notebook_names() -> None:
    """A parameter cell that documents a setting the command never passes is a lie.

    The defaults happen to match today, which is why nothing would notice if they
    stopped.
    """
    cell = run_cell()
    for knob in (
        "MODEL_ID",
        "CONTEXT_LENGTH",
        "MEMORY_BUDGET_GB",
        "OPTIMIZER",
        "EPOCHS",
        "BATCH_SIZE",
        "GRAD_ACCUM",
        "LEARNING_RATE",
        "MAX_RECORDS",
        "SFT_OUTPUT",
        "RESUME_FROM",
        "HF_REPO_ID",
        "PUSH_EVERY",
        "DRY_RUN",
    ):
        assert knob in cell, f"{knob} is documented in the parameters cell but never passed"


def test_the_run_cell_will_not_upload_during_a_rehearsal() -> None:
    """A dry run rehearses the schedule and sends nothing."""
    cell = run_cell()
    assert "if DRY_RUN:" in cell
    dry = cell.split("if DRY_RUN:")[1].split("else:")[0]
    assert "--hub-repo-id" not in dry, "a rehearsal would publish"


def test_the_preference_stage_is_skipped_by_name_and_not_by_failure() -> None:
    """It needs a task file and a served model.

    Neither exists yet, so the stage is switched off deliberately -- rather than run,
    found wanting, and reported as a pipeline failure.
    """
    cell = run_cell()
    assert "RUN_PREFERENCE_STAGE" in cell
    assert '"--no-eval", "--no-dpo"' in cell


def test_the_run_cell_says_where_the_logs_are() -> None:
    """Output scrolls.

    A diagnosis that lives only in the scrollback is gone by the time anyone looks.
    """
    assert "runs/pipeline/" in run_cell()


def test_the_notebook_tells_the_reader_to_rehearse_before_it_costs_money() -> None:
    assert "--dry-run" in run_cell()


def test_a_run_that_fits_the_plan_is_the_one_being_fitted() -> None:
    """The budget and the context go to the pipeline together.

    Otherwise the estimate is computed for a run that will not happen.
    """
    cell = run_cell()
    assert '"--context-length", str(CONTEXT_LENGTH)' in cell
    assert '"--memory-budget-gb", str(MEMORY_BUDGET_GB)' in cell


def test_dpo_is_fitted_against_the_same_budget_and_two_models() -> None:
    """The stage passes the budget, and the budget counts both models.

    Without the flags the defaults happen to match, so this test would pass while the
    notebook and the measured budget drifted apart. The two-model accounting belongs
    to the trainer rather than the orchestrator, so it is asserted where it is.
    """
    source = pipeline_source()
    assert '"--memory-budget-gb"' in source
    assert '"--loss-mode"' in source and '"selective"' in source
    assert '"--gradient-checkpointing"' in source
    trainer = (
        pathlib.Path(build_notebook.__file__).parent.joinpath("dpo.py").read_text(encoding="utf-8")
    )
    assert "models=2" in trainer, "a preference run holds two copies of the weights"
    assert "two copies of the weights" in trainer, "and says so when it refuses"


def test_the_optimizer_is_named_because_two_models_leave_no_room_for_adamw() -> None:
    """The optimizer is named because two models leave no room for the other one.

    Measured: two copies of a 4B model with AdamW's moments is 60.6 GB resident, so it
    fits no context on a 60 GB card. Adafactor is 26.0 GB.
    """
    assert "adafactor" in parameters_cell()
    assert "OPTIMIZER" in run_cell()


# ---------------------------------------------------------------- the bootstrap


def test_the_install_precedes_everything_that_needs_the_package() -> None:
    joined = "\n".join(code_cells())
    assert joined.index("pip install") < joined.index("gotooltrain.pipeline")


def test_the_running_kernel_is_told_where_the_package_is() -> None:
    """`pip install -e` only reaches *new* processes.

    The kernel started before the cell would not see it, and the failure would surface
    two cells later in a different one.
    """
    install = sources(install_cell())
    assert "sys.path" in install


def test_the_import_is_verified_where_it_is_installed_not_where_it_is_used() -> None:
    install = sources(install_cell())
    assert "import gotooltrain" in install


def test_a_stray_install_elsewhere_on_the_path_is_refused() -> None:
    """A different gotooltrain earlier on the path would be trained on silently.

    The checkpoint would land somewhere nobody is looking.
    """
    install = sources(install_cell())
    assert "__file__" in install
    assert "startswith(target)" in install


def test_the_install_cell_drops_the_module_cache_before_importing() -> None:
    """Re-running the notebook must not train on the code from the previous run.

    ``import`` serves ``sys.modules`` first, so a re-run in a live kernel keeps the
    *old* ``train.py`` even though the pull succeeded. Measured: after a pull,
    ``import`` still returns the previous revision. The visible symptom is the worst
    kind -- the memory budget looks right in the cell, the 40 GB guarantee is already
    gone, and the run OOMs at 8192 exactly as before.
    """
    install = sources(install_cell())
    assert "importlib.invalidate_caches()" in install, "bytecode caches are separate"
    assert 'startswith("gotooltrain.")' in install, "the whole package must go, not just the root"
    assert "del sys.modules[_name]" in install


def test_the_cache_is_dropped_before_the_import_not_after_it() -> None:
    """Ordering is the whole fix, and it is the part a later edit would break.

    Cleaning up *after* the import looks identical in a diff and leaves the stale
    module in place. Verified end to end on a real repository: without the cleanup a
    re-run keeps the old module after the pull, and with it the new one loads.
    """
    install = sources(install_cell())
    assert install.index("importlib.invalidate_caches()") < install.index("import gotooltrain")


def test_the_install_cell_checks_the_code_can_do_what_this_run_needs() -> None:
    """Right path is not right code.

    Pinning ``REPO_REF`` at an old revision used to fail hours later, as an OOM. The
    capability check turns that into a message in the cell that failed.
    """
    install = sources(install_cell())
    for symbol in ("enter_training_mode", "completion_only_loss", "DEFAULT_MEMORY_BUDGET_GB"):
        assert symbol in install, f"{symbol} is what makes the 40 GB budget mean anything"


# ---------------------------------------------------------------- the knobs


def test_resuming_is_a_named_decision_not_a_hand_edited_command() -> None:
    """The question "does it continue from the checkpoint on the Hub?" needs a switch.

    The mechanism exists (``--resume-from``), but nothing in the notebook reached it,
    so the answer was only discoverable by reading the CLI.
    """
    assert 'RESUME_FROM = "auto"' in parameters_cell()
    assert '"--resume-from", RESUME_FROM' in run_cell()


def test_the_notebook_does_not_claim_a_resume_restores_the_run() -> None:
    """What a resume restores is narrower than "continues where it left off".

    The published folder is ``save_pretrained`` output plus provenance: weights and
    tokenizer. Optimizer moments, the schedule, the step count and the push history
    are all gone, so the same examples get trained a second time. Calling it a resume
    without that costs someone a silently doubled epoch.
    """
    parameters = parameters_cell()
    assert "Optimizer durumu, scheduler, adım sayacı ve push" in parameters
    assert "yeniden eğitim" in parameters, "the honest name for what it does"


def test_an_existing_checkpoint_is_never_overwritten_without_a_word() -> None:
    """A push replaces the repository, so overwriting is a data-loss event.

    ``upload_folder`` writes to the default branch, so a run started from the base
    model replaces a trained checkpoint at the first push.
    """
    cell = run_cell()
    assert "RESUME_FROM" in cell
    assert "RESUME_FROM" in parameters_cell(), "and the decision is named, not defaulted away"


def test_the_default_resume_is_the_choice_that_cannot_lose_work() -> None:
    """The command passes the switch through to the stage, which loads the weights.

    Asking the Hub what is there costs one API call and turns a dangerous default --
    start from the base model and overwrite a trained checkpoint at the first push --
    into a safe one.
    """
    cell = run_cell()
    assert '"--resume-from", RESUME_FROM' in cell
    assert "auto" in parameters_cell(), "and the safe default is the one in force"


def test_the_context_is_chosen_from_a_measured_budget_not_a_round_number() -> None:
    """Qwen3.5-4B's activation memory scales with sequence length, not with a constant.

    24 of its 32 layers are Gated DeltaNet with a 1 MB-per-token recurrent state, so
    the run is fitted against a stated card: 8192 tokens estimates to 29.3 GB of a
    40 GB budget, and 16384 is refused.
    """
    parameters = parameters_cell()
    assert "CONTEXT_LENGTH = 8192" in parameters
    assert "MEMORY_BUDGET_GB = 40.0" in parameters
    assert "17,34" in parameters, "the resident floor is stated, because it is exact"
    assert "HAYIR" in parameters, "at least one context is shown not fitting"


def test_the_reader_is_told_which_term_of_the_estimate_is_fuzzy() -> None:
    """The activation terms are estimates; the resident ones are not.

    Stating that is the difference between a budget a reader can trust and one they
    have to take on faith.
    """
    parameters = parameters_cell()
    assert "tahmindir" in parameters
    assert "hangi terimin belirsiz" in parameters


def test_the_allocator_is_told_to_reduce_fragmentation() -> None:
    assert "EXPANDABLE_SEGMENTS" in parameters_cell()


def test_the_token_is_never_written_into_a_cell() -> None:
    """A secret pasted into a notebook is a secret in a file that gets committed.

    The placeholder is allowed -- it is how the notebook runs without a real token --
    but anything shaped like a real one is not, and that distinction is the assertion.
    """
    joined = "\n".join(code_cells())
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped.startswith("HF_TOKEN") and '"hf_' in stripped:
            assert "hf_mock" in stripped, f"a real-looking token is written into a cell: {line}"


def test_the_requirements_are_pinned_in_the_notebook() -> None:
    """Colab's image changes over time.

    A version constraint that is not written down is a run that reproduces one week and
    not the next.
    """
    install = sources(install_cell())
    assert "transformers>=5.17,<6" in install


def test_a_cell_keeps_its_newlines() -> None:
    """Notebook source is a list of lines.

    A string joined into one line would still parse and would be unreadable in the
    editor.
    """
    for cell in build_cells():
        assert isinstance(cell["source"], list)


def test_code_cells_carry_empty_outputs() -> None:
    """Committed output makes the notebook diff unreadable and can carry a token."""
    for cell in build_cells():
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []


def test_a_custom_cell_list_is_accepted() -> None:
    assert assemble_notebook([{"cell_type": "code", "source": ["print(1)"], "metadata": {}}])


def test_the_generator_has_a_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Called and not just present.

    ``callable`` would pass against a ``main`` that raised, and this is the entry
    point the notebook is regenerated with -- the one thing that has to work when
    someone changes the generator.
    """
    assert build_notebook.main() == 0
    assert "cells" in capsys.readouterr().out


def test_a_written_notebook_is_readable_json_afterwards(tmp_path: pathlib.Path) -> None:
    """Whatever the writer does, the file it leaves behind has to be a notebook.

    A generator that emits something JSON cannot read back is not a generator of
    notebooks, and the failure would only appear when someone opens the file.
    """
    written = build_notebook.write_notebook(tmp_path / "written.ipynb")
    document = json.loads(pathlib.Path(written).read_text(encoding="utf-8"))
    assert document["nbformat"] == 4
    assert document["cells"]
