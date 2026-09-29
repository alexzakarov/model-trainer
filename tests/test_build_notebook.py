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
    DEFAULT_REPO_URL,
    NOTEBOOK_PATH,
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


# ------------------------------------------------------------------ the cells


def test_every_code_cell_is_valid_python() -> None:
    """Executed at open time by Jupyter; a syntax error surfaces as a red cell."""
    for index, source in enumerate(code_cells()):
        compile(source, f"cell-{index}", "exec")


def test_the_cells_run_in_a_usable_order() -> None:
    """Parameters, then the hardware check, then install, then data, then training.

    Checked by the marker each cell carries rather than by index arithmetic, so
    inserting a cell does not silently reorder the run.
    """
    titles = [line for line in (s.splitlines()[0] for s in code_cells()) if "@title" in line]
    order = [t.split("@title", 1)[1] for t in titles]
    assert order == [
        " 1 — Parameters",
        " 2 — GPU ön kontrolü",
        " 3 — Depoyu çek ve kur",
        " 4 — Token: Colab secret ya da mock",
        " 5 — Kalite kapısı (paket kendi testini koşar)",
        " 6 — Go korpusunu indir, süz, ölç",
        " 7 — SFT verisini token'la",
        " 8 — Token formatını doğrula (eğitimden ÖNCE)",
        " 9 — Eğitimi başlat (periyodik Hub yüklemesiyle)",
        " 10 — Ne olduğunu doğrula",
    ]


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
    """The remote has no commits yet, so the first cell would fail without saying why."""
    markdown = "\n".join(sources(c) for c in build_cells() if c["cell_type"] == "markdown")
    assert "No commits yet" in markdown
    assert "git push" in markdown


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
