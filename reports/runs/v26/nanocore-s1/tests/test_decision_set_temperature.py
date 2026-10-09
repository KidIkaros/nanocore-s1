"""Fitting t_set — the fix for vacuous conformal sets on a trained head.

`t_set` was pinned at 1.0 because 1.0 "measured sane". It is sane for a cosine
scorer, whose similarities are spread. It is not sane for a fitted head: the
conformal score compresses into [0.99, 1.0], the quantile saturates, and every
prediction set becomes the whole label space. Coverage alone cannot see this —
it reads 1.0000, which looks like success. Set *size* is the observable.
"""
import numpy as np
import pytest

from src.decision.gate import ConformalGate
from src.decision.scoring import (aps_members, fit_set_temperature, mass_needed,
                                  softmax_rows)

N_CLASSES = 40


def overconfident_scores(seed: int = 0, n: int = 1200, acc: float = 0.97):
    """A head-like scorer: huge logit spread, so softmax is nearly one-hot."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    S = rng.normal(0, 1.0, (n, N_CLASSES))
    right = rng.random(n) < acc
    S[np.arange(n), y] += np.where(right, 14.0, -14.0)
    return S, y


def spread_scores(seed: int = 0, n: int = 1200):
    """A cosine-like scorer: narrow range, so softmax stays gentle."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    S = rng.normal(0, 1.0, (n, N_CLASSES))
    S[np.arange(n), y] += 2.0
    return S, y


def _set_stats(S, y, t_set, qhat):
    members = aps_members(softmax_rows(S, t_set), qhat)
    sizes = np.array([len(m) for m in members])
    covered = np.mean([y[i] in members[i] for i in range(len(members))])
    return sizes.mean(), covered


def test_the_degeneracy_is_real_and_coverage_cannot_see_it():
    """Documents the failure the fit exists to fix."""
    S, y = overconfident_scores()
    n = len(y); half = n // 2
    q = float(np.quantile(mass_needed(softmax_rows(S[:half], 1.0), y[:half]), 0.9))
    assert q > 0.99                       # the quantile has saturated
    size, cov = _set_stats(S[half:], y[half:], 1.0, q)
    assert cov >= 0.90                    # the guarantee is met — looks fine
    assert size > 10                      # but the sets are unusable


def test_fitted_temperature_shrinks_sets_and_keeps_coverage():
    S, y = overconfident_scores()
    n = len(y); half = n // 2
    t = fit_set_temperature(S[:half], y[:half], alpha=0.10)
    assert t > 1.0                        # softening, not sharpening
    q = float(np.quantile(mass_needed(softmax_rows(S[:half], t), y[:half]), 0.9))
    size, cov = _set_stats(S[half:], y[half:], t, q)
    assert cov >= 0.90
    assert size < 5


def test_selection_is_out_of_sample():
    """A pure in-sample search would pick the temperature that fits its own
    quantile; the internal split means the choice must transfer."""
    S, y = overconfident_scores(seed=7)
    t = fit_set_temperature(S[:600], y[:600], alpha=0.10)
    q = float(np.quantile(mass_needed(softmax_rows(S[:600], t), y[:600]), 0.9))
    _, cov = _set_stats(S[600:], y[600:], t, q)
    assert cov >= 0.90


def test_a_spread_scorer_is_left_at_the_default():
    """The cosine path was never broken; the grid must not 'fix' it."""
    S, y = spread_scores()
    assert fit_set_temperature(S[:600], y[:600], alpha=0.10) == 1.0


def test_gate_fits_t_set_and_shrinks_sets_end_to_end():
    S, y = overconfident_scores()
    n = len(y); cal, ev = slice(0, 800), slice(800, n)
    fitted = ConformalGate(alpha=0.10, min_n=200)
    fitted.calibrate(S[cal], y[cal])
    size_f, cov_f = _set_stats(S[ev], y[ev], fitted.t_set, fitted.qhat)

    pinned = ConformalGate(alpha=0.10, min_n=200, fit_t_set=False)
    pinned.calibrate(S[cal], y[cal])
    size_p, cov_p = _set_stats(S[ev], y[ev], pinned.t_set, pinned.qhat)

    assert fitted.t_set > 1.0 and pinned.t_set == 1.0
    assert cov_f >= 0.90 and cov_p >= 0.90
    assert size_f < size_p / 3


def test_fitted_temperature_survives_save_load(tmp_path):
    S, y = overconfident_scores()
    g = ConformalGate(alpha=0.10, min_n=200)
    g.calibrate(S[:800], y[:800])
    g.save(tmp_path / "gate.json")
    back = ConformalGate.load(tmp_path / "gate.json")
    assert back.t_set == g.t_set and back.qhat == pytest.approx(g.qhat)
