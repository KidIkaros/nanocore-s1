"""noul placement diagnostic — the reference-eval finding, adapted.

arXiv:2609.37647 found binary P(yes) ranks well (AUROC high) but is
poorly placed at a fixed 0.5: mean predicted probability sits off the
observed positive rate, so threshold metrics suffer even when the
ranking is sound (their mean P(yes) 0.465 vs observed 0.518). We
already report F1@0.5 and F1 at a tuned threshold; ``noul_placement``
adds the placement gap itself, so a well-ranked-but-misplaced P(yes) is
visible before any threshold is fitted.
"""
import numpy as np
import pytest

from src.decision.readiness import noul_placement


def test_placement_reports_mean_vs_observed():
    p = np.array([0.2, 0.3, 0.4, 0.6])
    g = np.array([0, 0, 1, 1])
    out = noul_placement(p, g)
    assert out["mean_p_yes"] == pytest.approx(0.375)
    assert out["observed_rate"] == pytest.approx(0.5)
    assert out["gap"] == pytest.approx(-0.125)  # under-confident


def test_overconfidence_is_positive_gap():
    # predicts yes often while yes is rare — the over-prediction the
    # reference eval found on multi-label noul (mean P .209 vs rate .041)
    p = np.array([0.9, 0.9, 0.1, 0.1])
    g = np.array([1, 0, 0, 0])
    out = noul_placement(p, g)
    assert out["gap"] > 0


def test_well_placed_has_near_zero_gap():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    g = np.array([0, 0, 1, 1])
    out = noul_placement(p, g)
    assert abs(out["gap"]) < 1e-9


def test_base_rate_mirrors_observed_rate():
    out = noul_placement(np.array([0.5, 0.5]), np.array([1, 0]))
    assert out["base_rate"] == out["observed_rate"] == pytest.approx(0.5)


def test_empty_input_reports_none():
    out = noul_placement(np.array([]), np.array([]))
    assert out["mean_p_yes"] is None and out["gap"] is None


def test_length_mismatch_reports_none():
    out = noul_placement(np.array([0.5, 0.5]), np.array([1]))
    assert out["gap"] is None
