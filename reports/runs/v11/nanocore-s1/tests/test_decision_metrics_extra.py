"""Evaluation rigor — bootstrap CIs, risk-coverage, AURC."""
import numpy as np
import pytest

from src.decision.metrics import (aurc, bootstrap_ci, risk_coverage_curve,
                                  selective_at_coverage)


def test_bootstrap_ci_contains_point_and_scales():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    pred = (rng.random(400) < 0.7).astype(int) == y.astype(bool)
    ci = bootstrap_ci(lambda a: a.mean(), pred.astype(float), resamples=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["point"] == pytest.approx(pred.mean())
    # CI width shrinks with n
    ci_small = bootstrap_ci(lambda a: a.mean(), pred[:50].astype(float),
                            resamples=300)
    assert (ci["hi"] - ci["lo"]) < (ci_small["hi"] - ci_small["lo"])


def test_aurc_perfect_vs_random():
    # perfect ranker: correct exactly where confident -> low risk, low AURC
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    good = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    bad = np.array([0, 0, 0, 1, 1, 1], dtype=float)  # confident and wrong
    assert aurc(conf, good) < aurc(conf, bad)
    assert 0 <= aurc(conf, good) <= 1


def test_risk_coverage_monotone_for_good_ranker():
    conf = np.linspace(0.99, 0.01, 100)
    correct = (conf > 0.5).astype(float)
    curve = risk_coverage_curve(conf, correct)
    risks = [p["risk"] for p in curve]
    assert risks[0] == 0.0
    assert risks[-1] >= risks[0]


def test_selective_at_coverage():
    conf = np.array([0.9, 0.8, 0.7, 0.2])
    hit = np.array([1, 1, 0, 1], dtype=float)
    assert selective_at_coverage(conf, hit, 0.5) == 1.0   # top-2 both right
    assert selective_at_coverage(conf, hit, 1.0) == 0.75
