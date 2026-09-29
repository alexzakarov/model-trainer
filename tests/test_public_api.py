"""The package's advertised surface must actually exist.

``__all__`` is a promise about what a consumer may import. A name listed there but
missing, or exported twice under different spellings, is the kind of breakage that
only shows up in someone else's code, so it is checked here.
"""

from __future__ import annotations

import gotooltrain


def test_every_exported_name_resolves() -> None:
    missing = [name for name in gotooltrain.__all__ if not hasattr(gotooltrain, name)]
    assert missing == [], f"__all__ advertises names that do not exist: {missing}"


def test_the_surface_has_no_duplicates() -> None:
    duplicates = sorted({n for n in gotooltrain.__all__ if gotooltrain.__all__.count(n) > 1})
    assert duplicates == []


def test_the_variant_modules_are_reachable() -> None:
    """The pipeline's stages are public, not buried behind submodule imports."""
    for name in ("measure", "mine", "train", "harvest_packages", "collate"):
        assert hasattr(gotooltrain, name), name


def test_the_two_trajectory_types_stay_distinguishable() -> None:
    """``Trajectory`` is the eval record; the measured one has its own name.

    Collapsing them under one name would silently mix an executed transcript with a
    set of corpus counts, which are different things with different lifetimes.
    """
    assert gotooltrain.Trajectory is not gotooltrain.MeasuredTrajectory
    assert gotooltrain.MeasuredTrajectory.__module__ == "gotooltrain.corpus"
    assert gotooltrain.Trajectory.__module__ == "gotooltrain.evalrun"
