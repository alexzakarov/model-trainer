PYTHON ?= python

# Some tests start a Docker worker that bind-mounts a directory. Under WSL, doing
# that invalidates getcwd() for every process whose working directory is on the
# same drive -- a DrvFs effect, reproducible with `docker run -v` alone -- so the
# shell make would otherwise hand to the next recipe line cannot run coverage or
# mypy and dies with ENOENT. Re-entering the directory through /tmp forces a fresh
# lookup and gives that line a working directory again.
#
# Only under WSL: Windows has no such effect, and the `cd /tmp` would be a path
# that does not exist there.
ifeq ($(OS),Windows_NT)
REENTER =
else
REENTER = cd /tmp && cd "$(CURDIR)" &&
endif

.PHONY: help install test coverage lint format typecheck check clean golden notebook

help:
	@echo "make install    - install the package with dev extras"
	@echo "make test       - run the test suite"
	@echo "make coverage   - run tests under coverage and enforce the 100% gate"
	@echo "make lint       - ruff check"
	@echo "make format     - ruff format"
	@echo "make typecheck  - mypy (strict)"
	@echo "make check      - the full gate: lint, format, typecheck, coverage"
	@echo "make golden     - regenerate the chat-template golden file"
	@echo "make notebook   - regenerate the Colab notebook from its generator"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(REENTER) $(PYTHON) -m pytest

# Coverage runs through `coverage run` rather than the pytest-cov plugin: the
# plugin's parallel-data behaviour varies with the ambient environment, and a
# subprocess started in a pytest tmp directory cannot read this config, which
# produced "Can't combine statement coverage data with branch data".
coverage:
	$(REENTER) $(PYTHON) -m coverage run -m pytest
	$(REENTER) $(PYTHON) -m coverage report

lint:
	$(REENTER) $(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

typecheck:
	$(REENTER) $(PYTHON) -m mypy

# The same commands as `coverage`, spelled out rather than invoked through
# `$(MAKE)`: a recursive make depends on the ambient `make` being usable from the
# shell make hands to its recipes, and on Windows that resolves to a shim that
# cannot start its own helper. The gate should not be the thing that breaks.
check: lint
	$(REENTER) $(PYTHON) -m ruff format --check .
	$(REENTER) $(PYTHON) -m mypy
	$(REENTER) $(PYTHON) -m coverage run -m pytest
	$(REENTER) $(PYTHON) -m coverage report

golden:
	$(PYTHON) -m gotooltrain.regenerate_golden

# The notebook is generated, never hand-edited: a 400-line JSON blob cannot be
# reviewed in a diff, and this project's premise is that a change to the training
# path has to be visible in one. See docs/COLAB.md.
notebook:
	$(PYTHON) -m gotooltrain.build_notebook

clean:
	-rm -f .coverage .coverage.*
	-$(PYTHON) -m pytest --cache-clear
	-rm -rf .pytest_cache .mypy_cache .ruff_cache
