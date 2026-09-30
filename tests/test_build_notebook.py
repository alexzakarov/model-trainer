"""The Colab notebook is generated, not hand-written, and the generator is tested.

A notebook is JSON: a missing comma produces a file Jupyter refuses to open, and
nothing else in the project would notice. So the properties that matter are checked
here rather than hoped for: it parses, every code cell is valid Python, the cells
appear in a runnable order, and the two things that make a run expensive to lose --
the repository it clones and the publication interval -- are stated rather than
implied.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from gotooltrain import build_notebook
from gotooltrain.build_notebook import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_REPO_URL,
    GO_SHA256,
    GO_VERSION,
    NOTEBOOK_PATH,
    RTK_ASSET,
    RTK_BASE_URL,
    RTK_SHA256,
    RTK_VERSION,
    assemble_notebook,
    build_cells,
    write_notebook,
)


def sources(cell: dict[str, object]) -> str:
    """A cell's text."""
    return "".join(cell["source"])  # type: ignore[arg-type]


def code_cells() -> list[str]:
    """Every code cell's source, in order."""
    return [sources(c) for c in build_cells() if c["cell_type"] == "code"]


def install_cell() -> dict[str, object]:
    """The cell that pulls and installs, found by what it does rather than by index.

    Cell order is a presentation choice; what this cell is for is not, so the test
    names it by its content and stays valid if the notebook gains a cell.
    """
    for cell in build_cells():
        if (
            cell["cell_type"] == "code"
            and "pip install" in sources(cell)
            and "git" in sources(cell)
        ):
            return cell
    raise AssertionError("no cell pulls and installs the repository")


# ---------------------------------------------------------------- the document


def test_the_notebook_is_valid_json_with_the_expected_shape() -> None:
    document = assemble_notebook()
    assert document["nbformat"] == 4
    assert document["cells"]
    assert json.loads(json.dumps(document)) == document


def test_a_written_notebook_reparses_identically(tmp_path: pathlib.Path) -> None:
    """The file on disk is the document, not a lossy rendering of it."""
    path = write_notebook(tmp_path / NOTEBOOK_PATH)
    assert json.loads(path.read_text(encoding="utf-8")) == assemble_notebook()


def test_the_notebook_lands_in_the_project_not_the_cwd(tmp_path: pathlib.Path) -> None:
    """A generator that writes into whatever directory it was run from litters."""
    assert write_notebook().parent.name == "notebooks"
    assert build_notebook.NOTEBOOK_PATH.parent == pathlib.Path("notebooks")
    assert not list(tmp_path.iterdir())


def test_the_project_root_is_found_from_the_installed_package() -> None:
    root = build_notebook._project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "gotooltrain").is_dir()


def test_the_default_repository_is_the_one_this_project_uses() -> None:
    """A notebook that clones a stale URL fails at the first cell, hours in."""
    assert DEFAULT_REPO_URL.endswith("model-trainer.git")
    assert any(DEFAULT_REPO_URL in source for source in code_cells())


def test_the_publication_target_is_written_once_in_the_source() -> None:
    """One place in the *codebase* to edit, so a renamed repo cannot be half-renamed.

    The generated notebook may name the repo as many times as it likes -- those
    renderings all come from the same constant. What must not happen is a second
    developer typing the literal into another module, where a rename would leave
    the two silently disagreeing: the run trains, the checkpoint uploads, and it
    lands somewhere nobody is looking.
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
    """``namespace/name`` -- an id without both halves is not a repo id at all."""
    namespace, _, name = DEFAULT_HF_REPO_ID.partition("/")
    assert namespace and name
    assert " " not in DEFAULT_HF_REPO_ID
    assert DEFAULT_HF_REPO_ID.strip() == DEFAULT_HF_REPO_ID


# ------------------------------------------------------------------ the cells


def test_every_code_cell_is_valid_python() -> None:
    """Executed at open time by Jupyter; a syntax error surfaces as a red cell."""
    for index, source in enumerate(code_cells()):
        compile(source, f"cell-{index}", "exec")


def test_the_cells_run_in_a_usable_order() -> None:
    """Parameters, hardware, install, Go, token, gate, data, training, verify.

    Checked by the marker each cell carries rather than by index arithmetic, so
    inserting a cell does not silently reorder the run.
    """
    titles = [line for line in (s.splitlines()[0] for s in code_cells()) if "@title" in line]
    order = [t.split("@title", 1)[1] for t in titles]
    assert order == [
        " 1 — Parameters",
        " 2 — GPU ön kontrolü",
        " 3 — Depoyu çek ve kur",
        " 4 — Go ve rtk toolchain'leri",
        " 5 — Token: Colab secret ya da mock",
        " 6 — Kalite kapısı (paket kendi testini koşar)",
        " 7 — Go korpusunu indir, süz, ölç",
        " 8 — SFT verisini ölç, süz, doğrula",
        " 9 — Token formatını doğrula (eğitimden ÖNCE)",
        " 10 — Eğitimi başlat (periyodik Hub yüklemesiyle)",
        " 11 — Ne olduğunu doğrula",
    ]


def test_both_external_toolchains_are_installed_before_the_quality_gate() -> None:
    """The gate runs real commands through both of them.

    Colab ships neither Go nor rtk, and the catalogue's tools *are* ``rtk go test``
    and friends. So without this cell a missing dependency is reported as a broken
    project -- indistinguishable from a real regression, and it costs a round trip
    through a metered machine to discover.
    """
    joined = "\n".join(code_cells())
    gate = joined.index('"pytest", "-q')
    for fragment in ("go.dev/dl", RTK_BASE_URL, RTK_ASSET):
        assert joined.index(fragment) < gate, fragment


def test_rtk_is_installed_at_the_version_the_sandbox_image_pins() -> None:
    """The catalogue's output format *is* rtk's output format.

    A notebook on a different build measures a different format than the eval
    harness will, which is the whole class of bug the pinned-toolchain rule exists
    to prevent. So the notebook and the Dockerfile must name the same build.
    """
    joined = "\n".join(code_cells())
    assert RTK_VERSION in joined
    assert RTK_SHA256 in joined
    dockerfile = (
        build_notebook._project_root() / "docker" / "go-sandbox" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert RTK_VERSION in dockerfile, "the sandbox image pins a different rtk release"
    assert RTK_SHA256 in dockerfile, "the sandbox image pins a different rtk digest"
    assert RTK_ASSET in dockerfile, "the sandbox image downloads a different asset"


def test_rtk_is_verified_against_the_publisher_as_well_as_the_pin() -> None:
    """Two sources, the way the Dockerfile does it.

    The pin says what was reviewed; the fetch proves the publisher still says so.
    """
    joined = "\n".join(code_cells())
    assert "checksums.txt" in joined
    install = joined.index('shutil.copy2(binary, "/usr/local/bin/rtk")')
    assert joined.index("checksums.txt") < install, "rtk was installed before it was cross-checked"


def test_the_notebook_refuses_an_architecture_its_pinned_binaries_do_not_cover() -> None:
    """A static x86_64 musl build on arm64 fails at exec time, confusingly.

    Colab's T4/L4/A100 are all x86_64, so this only fires on a machine the constants
    do not describe -- which is exactly when a named refusal beats a stack trace
    from deep inside a subprocess.
    """
    joined = "\n".join(code_cells())
    assert "platform.machine()" in joined
    assert joined.index("platform.machine()") < joined.index("go.dev/dl")


def test_the_go_download_is_verified_before_it_is_used() -> None:
    """An unverified toolchain is an unreviewed one; the digest is the review artefact."""
    joined = "\n".join(code_cells())
    assert GO_SHA256 in joined
    verify = joined.index("hashlib.sha256")
    unpack = joined.index("tarfile.open")
    assert verify < unpack, "the bytes were unpacked before they were checked"


def test_the_go_version_is_pinned_not_latest() -> None:
    """A drifting toolchain makes a difference in test output unattributable."""
    joined = "\n".join(code_cells())
    assert GO_VERSION in joined
    assert "latest" not in joined


def test_the_go_toolchain_is_pinned_here_and_only_here() -> None:
    """One place to update when the toolchain is bumped.

    For the same reason the repo id is single-sourced: two toolchain versions in the
    tree is a difference nobody chose, and a difference nobody chose is a
    difference nobody can attribute.
    """
    package = pathlib.Path(build_notebook.__file__).parent
    occurrences = sum(
        path.read_text(encoding="utf-8").count(GO_SHA256) for path in package.glob("*.py")
    )
    assert occurrences == 1


def test_rtk_is_pinned_here_and_only_here_too() -> None:
    """The same rule as the repo id and the Go digest, for the same reason."""
    package = pathlib.Path(build_notebook.__file__).parent
    occurrences = sum(
        path.read_text(encoding="utf-8").count(RTK_SHA256) for path in package.glob("*.py")
    )
    assert occurrences == 1


def test_the_go_install_makes_the_toolchain_visible_to_subprocesses() -> None:
    """PATH is set in os.environ, not in a shell, so the gate and the run inherit it.

    A Go installed into a shell's PATH and not the process environment is a Go the
    quality gate never sees -- the run would pass the gate and fail the training.
    """
    joined = "\n".join(code_cells())
    assert 'os.environ["PATH"]' in joined
    assert "export PATH" not in joined
    assert "GOTOOLCHAIN" in joined, "go.mod's toolchain directive must not reach the network"


def test_the_expensive_checks_run_before_the_expensive_step() -> None:
    """Hardware, format and the package's own suite all precede the training cell.

    Each of these is cheap and each catches a failure that would otherwise be found
    hours into a metered run, so their position is a correctness property, not a
    preference.
    """
    joined = "\n".join(code_cells())
    ordered = [
        "get_device_properties",  # hardware
        '"pytest", "-q',  # the package's own suite
        "AutoTokenizer.from_pretrained",  # rendering the corpus
        "assert_mask_sane",  # the loss mask, before any weights move
        "gotooltrain.traincli",  # the expensive part
    ]
    positions = [joined.index(fragment) for fragment in ordered]
    assert positions == sorted(positions), positions


def test_the_install_precedes_everything_that_needs_the_package() -> None:
    """`pip install -e` is what puts gotooltrain on the path; before it, cell 4 fails."""
    joined = "\n".join(code_cells())
    install = joined.index("[dev,train]")
    first_use = min(joined.index("from gotooltrain import"), joined.index('"gotooltrain.'))
    assert install < first_use


def test_the_running_kernel_is_told_where_the_package_is() -> None:
    """An editable install is invisible to the interpreter that performed it.

    Measured, not assumed: the .pth and editable-finder hooks are registered when an
    interpreter *starts*, so a kernel that began before the install cannot import
    what it just installed. Every subprocess does see it -- which is why the quality
    gate went green and the very next cell raised ``ModuleNotFoundError``. Pinning
    the source onto ``sys.path`` is what makes the kernel agree with its children.
    """
    joined = "\n".join(code_cells())
    assert 'sys.path.insert(0, os.path.join(target, "src"))' in joined


def test_the_import_is_verified_where_it_is_installed_not_where_it_is_used() -> None:
    """Five cells later, a missing import is a long way from its cause.

    The install cell imports the package itself and refuses to continue if the clone
    is not what answered, so a broken environment is named here rather than as a
    ``ModuleNotFoundError`` in a cell about tokenization.
    """
    joined = "\n".join(code_cells())
    install = joined.index("[dev,train]")
    check = joined.index("import gotooltrain", install)
    assert install < check
    assert "gotooltrain.__file__" in joined, "the import must be checked, not merely attempted"
    assert "raise SystemExit" in joined[check : check + 900]


def test_a_stray_install_elsewhere_on_the_path_is_refused() -> None:
    """A previously installed copy on sys.path is a common and silent trap.

    Training the wrong copy of the code while reading the right repository is worse
    than not training at all, and nothing else in the notebook would notice.
    """
    joined = "\n".join(code_cells())
    assert "startswith(target)" in joined


def test_the_training_cell_refuses_to_swallow_a_failure() -> None:
    """A non-zero exit stops the notebook where the error is.

    The measured failure this prevents: training died, the cell printed
    ``exit: 1`` and carried on, and the next cell then complained about a missing
    ``hub_push.json`` -- which is a *consequence* of the failure, not its cause. Two
    cells of misdirection, discovered on a metered machine.
    """
    joined = "\n".join(code_cells())
    training = joined.index("gotooltrain.traincli")
    check = joined.index("if result != 0:", training)
    assert training < check
    verify = joined.index("@title 11", training)
    assert check < verify, "the failure is raised before the verification cell runs"


def test_the_training_cell_names_which_stage_the_run_reached() -> None:
    """Four different outcomes, four different messages.

    An empty output directory, a run that started and never finished, a run that
    finished without publishing, and a run that published -- collapsing these into
    "hub_push.json yok" throws away the only information that says where to look.
    """
    joined = "\n".join(code_cells())
    training = joined[joined.index("@title 10") : joined.index("@title 11")]
    assert "training_plan.json" in training, "started-but-unfinished is a different failure"
    assert "hub_push.json" in training, "finished-but-unpublished is a different failure"
    assert "Çıktı dizini hiç oluşmadı" in training, "and so is 'never got as far as the plan'"


def test_the_verification_cell_checks_before_it_reads() -> None:
    """It reads a file the run is supposed to have written, so it looks first.

    A bare ``read_push_state`` on a missing path reports a data error; here the
    output directory is checked and its contents listed first, so the reader is
    told what exists rather than only what does not.
    """
    joined = "\n".join(code_cells())
    verify = joined[joined.index("@title 11") :]
    assert "is_dir()" in verify, "an absent output directory is a different failure"
    assert "eğitim hiç çıktı üretmedi" in verify
    listing = verify.index("iterdir()")
    reading = verify.index("read_push_state(output_dir)")
    assert listing < reading, "the directory is listed before it is read"


def test_the_notebook_tells_the_reader_to_rehearse_before_it_costs_money() -> None:
    """A metered run is not the place to discover that publication is misconfigured.

    The dry run exercises the schedule, the folder writing and the bookkeeping, and
    sends nothing -- so the first real run is the second run.
    """
    joined = "\n".join(code_cells())
    assert "DRY_RUN=True ile başla" in joined


def test_the_dataset_is_written_in_the_shape_the_training_cli_reads() -> None:
    """Messages, not token ids.

    The measured failure this prevents: cell 8 rendered examples and handed the
    token ids on, but the training CLI's loader expects messages and renders them
    itself. The run died with ``record 0 is not usable: conversation has no
    messages`` and an output directory that was never created, so the real cause
    and the reported symptom were two different files apart. There is exactly one
    render path in this project, and the notebook now goes through it.
    """
    joined = "\n".join(code_cells())
    # The record is built in cell 8 and written in cell 10; both halves matter,
    # because a wrong shape in either place is equally fatal.
    building = joined[joined.index("@title 8") : joined.index("@title 9")]
    writing = joined[joined.index("@title 10") : joined.index("@title 11")]
    assert 'kept.append({"messages": record["messages"], "tools": TOOLS})' in building
    assert "example.to_dict()" not in building, (
        "to_dict() is the token-id shape the CLI cannot read; measuring must not keep it"
    )
    assert "input_ids" not in writing, "the notebook must not hand token ids to the CLI"
    assert "json.dumps(row)" in writing, "the record is written exactly as it was built"
    assert "mesaj biçimi" in writing


def test_the_measurement_matches_what_training_will_render() -> None:
    """Lengths are measured with the catalogue, because training will render with it.

    Measuring without the catalogue under-counts every record by the size of the
    tool definitions, so a record the pre-flight accepted is the one training then
    refuses as over-length -- and the pre-flight was the thing that approved it.
    """
    joined = "\n".join(code_cells())
    measuring = joined[joined.index("@title 8") : joined.index("@title 9")]
    assert "TOOLS = catalog()" in measuring
    assert 'normalize_conversation(record["messages"], TOOLS)' in measuring


def test_an_emptied_corpus_stops_the_run_before_the_gpu_is_billed() -> None:
    """Zero usable records is a result, not a reason to start a metered run."""
    joined = "\n".join(code_cells())
    measuring = joined[joined.index("@title 8") : joined.index("@title 9")]
    assert "if not kept:" in measuring
    assert "Eğitilecek veri yok" in measuring


def test_the_failure_message_really_interpolates() -> None:
    """The escaped-brace slip: the cell printed ``{result.returncode}`` literally.

    A diagnostic that cannot say what happened is worse than no diagnostic, because
    it looks like one. Asserted by running the cell's own formatter rather than by
    reading it.
    """
    joined = "\n".join(code_cells())
    training = joined[joined.index("@title 10") : joined.index("@title 11")]
    assert "{{result.returncode}}" not in training, "an f-string with escaped braces prints braces"
    assert "{" in training and "}" in training, "the message must still interpolate something"


def test_a_record_that_reached_the_context_limit_is_dropped() -> None:
    """Supervised tokens are not evidence the record was not cut.

    Truncation keeps the head. If the assistant turn starts before the limit, the
    truncated record still has supervised tokens -- so the old test waved it
    through, and training, which re-renders without a limit, refused it. Measured
    on the first 400 records of the real corpus: 74 of the 364 kept sat on the
    8192 cap, with true lengths from 8.5K to 17.6K tokens. The pre-flight was
    approving exactly the records the trainer would reject.
    """
    joined = "\n".join(code_cells())
    measuring = joined[joined.index("@title 8") : joined.index("@title 9")]
    guard = "if len(example.input_ids) >= MAX_TOKENS_PER_RECORD:"
    assert guard in measuring
    truncate = measuring.index(guard)
    supervision = measuring.index("if example.supervised_tokens == 0:")
    assert truncate < supervision, "length is checked first: it is the stronger test"
    assert "dropped_truncated" in measuring, "the two refusal reasons are counted apart"


def test_the_failure_excerpt_cannot_be_scrolled_away() -> None:
    """The child's own error line was never visible, and we read the wrong cell.

    A multi-hour run's output scrolls; a diagnosis that depends on the reader
    spotting a line in the scrollback is a diagnosis that depends on luck. The
    output is therefore teed to a file and the tail is reprinted on failure, which
    costs nothing while the run is healthy.
    """
    joined = "\n".join(code_cells())
    training = joined[joined.index("@title 10") : joined.index("@title 11")]
    assert "subprocess.Popen" in training, "streaming, so a long run still shows progress"
    assert "stderr=subprocess.STDOUT" in training, "stderr must reach the log, not vanish"
    assert "train.log" in training
    assert "log_path.read_text" in training, "the excerpt is reprinted, not merely kept"


def test_the_context_is_chosen_from_a_measured_budget_not_a_round_number() -> None:
    """Qwen3.5-4B's activation memory scales with sequence length, not with a constant.

    24 of its 32 layers are Gated DeltaNet with a 1 MB-per-token recurrent state, so
    the run is fitted against a stated card: 8192 tokens estimates to 29.3 GB of a
    40 GB budget, and 16384 is refused. The notebook says which, and says that the
    activation term is the approximate one.
    """
    parameters = "\n".join(sources(c) for c in build_cells() if c["cell_type"] == "code")
    assert "CONTEXT_LENGTH = 8192" in parameters
    assert "MEMORY_BUDGET_GB = 40.0" in parameters
    assert "17,34" in parameters, "the resident floor is stated, because it is exact"
    assert "HAYIR" in parameters, "at least one context is shown not fitting"


def test_the_notebook_states_the_budget_it_is_fitted_against() -> None:
    """A budget that is not passed to the trainer is a comment, not a constraint."""
    joined = "\n".join(code_cells())
    assert '"--memory-budget-gb", str(MEMORY_BUDGET_GB)' in joined
    assert '"--loss-mode", "selective"' in joined


def test_the_reader_is_told_which_term_of_the_estimate_is_fuzzy() -> None:
    """The activation terms are estimates; the resident ones are not.

    Stating that is the difference between a budget a reader can trust and one they
    have to take on faith.
    """
    parameters = "\n".join(sources(c) for c in build_cells() if c["cell_type"] == "code")
    assert "tahmindir" in parameters
    assert "hangi terimin belirsiz" in parameters


def test_the_install_cell_drops_the_module_cache_before_importing() -> None:
    """Re-running the notebook must not train on the code from the previous run.

    ``import`` serves ``sys.modules`` first, so a re-run in a live Colab kernel
    keeps the *old* ``train.py`` even though the pull succeeded. Measured: after a
    pull, ``import`` still returns the previous revision. The visible symptom is
    the worst kind -- the memory budget looks right in the cell, the 40 GB
    guarantee is already gone, and the run OOMs at 8192 exactly as before.
    """
    install = sources(install_cell())
    assert "importlib.invalidate_caches()" in install, "bytecode caches are separate"
    assert 'startswith("gotooltrain.")' in install, "the whole package must go, not just the root"
    assert "del sys.modules[_name]" in install


def test_the_cache_is_dropped_before_the_import_not_after_it() -> None:
    """Ordering is the whole fix, and it is the part a later edit would break.

    Cleaning up *after* the import looks identical in a diff and leaves the stale
    module in place. Verified end to end on a real repository: without the cleanup
    a re-run keeps the old module after the pull, and with it the new one loads.
    """
    install = sources(install_cell())
    cleanup = install.index("importlib.invalidate_caches()")
    imported = install.index("import gotooltrain")
    assert cleanup < imported, "the import is served from the cache the cleanup is meant to clear"


def test_the_install_cell_checks_the_code_can_do_what_this_run_needs() -> None:
    """Right path is not right code.

    Pinning ``REPO_REF`` at an old revision used to fail hours later, as an OOM.
    The capability check turns that into a message in the cell that failed.
    """
    install = sources(install_cell())
    for symbol in ("enter_training_mode", "completion_only_loss", "DEFAULT_MEMORY_BUDGET_GB"):
        assert symbol in install, f"{symbol} is what makes the 40 GB budget mean anything"
    assert "yeteneekler" in install or "yetenekler" in install


def test_the_run_cell_names_the_stale_plan_trap_before_it_bites() -> None:
    """A re-run after a failed run hits a refusal, and the notebook says so.

    The refusal is correct -- a plan is a record of what a run was -- so the fix is
    to tell the reader the one command that clears it.
    """
    run_cell = "\n".join(c for c in code_cells() if "training_plan" in c)
    assert run_cell, "the run cell must discuss the plan file it writes"
    assert "rm -rf runs/colab-sft" in run_cell
    assert "training_plan.json" in run_cell


def test_the_reader_is_told_to_check_that_checkpointing_actually_fired() -> None:
    """The flag is silent when it is not working.

    It was inert for the entire life of this code -- set, recorded in the plan, and
    never triggered, because the model was in eval mode. A number in the log is the
    only way a reader can tell the difference.
    """
    joined = "\n".join(code_cells())
    assert "gradient checkpointing on:" in joined
    assert "gör" in joined, "the cell must instruct the reader to look for the line"


def test_the_allocator_is_told_to_reduce_fragmentation() -> None:
    """A long metered run accumulates reserved-but-unused blocks.

    This does not fix an over-budget, and the notebook says so; it stops the
    allocator handing back unusable space in the last hour of a run.
    """
    joined = "\n".join(code_cells())
    assert "expandable_segments:True" in joined
    assert "PYTORCH_CUDA_ALLOC_CONF" in joined


def test_a_dry_run_publishes_nothing() -> None:
    joined = "\n".join(code_cells())
    assert '"--hub-dry-run"' in joined
    assert "DRY_RUN = False" in joined, "the rehearsal switch must be visible in the parameters"


def test_the_publication_interval_is_a_named_parameter() -> None:
    joined = "\n".join(code_cells())
    assert "PUSH_EVERY" in joined
    assert '"--hub-push-every", str(PUSH_EVERY)' in joined


def test_the_notebook_says_what_the_run_does_not_measure() -> None:
    """The failure this project exists to avoid is a smoke test sold as a measurement.

    The notebook therefore says so in its own words, in the cells a reader sees
    before they spend GPU hours.
    """
    markdown = "\n".join(sources(c) for c in build_cells() if c["cell_type"] == "markdown")
    assert "Ölçmez" in markdown
    assert "EVAL.md" in markdown


def test_the_notebook_names_the_blocker_in_the_empty_remote() -> None:
    """A notebook that clones a stale ref fails at the first cell, hours in."""
    markdown = "\n".join(sources(c) for c in build_cells() if c["cell_type"] == "markdown")
    assert "REPO_REF" in markdown
    assert "95b1aa3" in markdown


def test_the_token_is_never_written_into_a_cell() -> None:
    """A literal credential in a notebook is a credential in the git history."""
    joined = "\n".join(code_cells())
    assert "hf_mock_replace_me" in joined, "the placeholder must be recognisably a placeholder"
    assert 'userdata.get("HF_TOKEN")' in joined, "the real path is a Colab secret"
    # Nothing that looks like a real token: hf_ followed by 30+ characters.
    import re

    assert not re.search(r"hf_[A-Za-z0-9]{30,}", joined)


def test_the_requirements_are_pinned_in_the_notebook() -> None:
    """Colab's image changes under you.

    An unpinned transformers means a 4.x install, where the tokenizer loads and the
    config does not -- a failure that would surface halfway through the run.
    """
    joined = "\n".join(code_cells())
    assert "transformers>=5.17,<6" in joined
    assert "[dev,train]" in joined


# ----------------------------------------------------------------- the pieces


def test_a_cell_keeps_its_newlines() -> None:
    """A cell stored as one string with embedded newlines is not a valid notebook."""
    cell = build_cells()[1]
    assert isinstance(cell["source"], list)
    assert all(line.endswith("\n") for line in cell["source"][:-1])


def test_code_cells_carry_empty_outputs() -> None:
    """Stale outputs in a committed notebook look like a run that already happened."""
    for cell in build_cells():
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_a_custom_cell_list_is_accepted() -> None:
    """Injected cells are how a caller varies the notebook without editing it."""
    custom = [build_cells()[1]]
    assert assemble_notebook(custom)["cells"] == custom


def test_the_generator_has_a_main() -> None:
    """``python -m gotooltrain.build_notebook`` is how the file is regenerated."""
    assert callable(build_notebook.main)
    assert isinstance(build_notebook.main(), int)


def test_a_generated_document_that_is_not_json_is_reported(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generator bug must fail loudly here, not in a browser tab an hour later."""
    monkeypatch.setattr(
        build_notebook,
        "json",
        types.SimpleNamespace(
            dumps=lambda *a, **k: "{not json",
            loads=json.loads,
            JSONDecodeError=json.JSONDecodeError,
        ),
    )
    with pytest.raises(build_notebook.ToolTrainError, match="not valid JSON"):
        build_notebook.write_notebook(tmp_path / "broken.ipynb")
