"""Contract tests for the shared evaluation protocol.

Two of these are regression tests for bugs that produced wrong published conclusions:

- ``test_temperature_not_bound_limited`` reproduces the grid-edge failure that made
  cosine scoring look catastrophically miscalibrated.
- ``test_compares_against_best_baseline_not_weakest`` reproduces the verdict that
  reported a Brier win which does not exist against the strongest baseline.

Numpy only; no model weights.
"""
import numpy as np
import pytest

from src.decision import protocol
from src.decision.protocol import (
    compare_to_best, coverage_at_precision, escalation_summary, fit_temperature,
    headroom_check, metric_block, selective_curve, softmax,
)

rng = np.random.default_rng(0)


def onehot(idx, k):
    m = np.zeros((len(idx), k))
    m[np.arange(len(idx)), idx] = 1.0
    return m


class TestTemperature:
    def test_recovers_a_small_temperature_that_a_narrow_grid_would_miss(self):
        """Regression: the old grid bottomed out at 10**-0.3 = 0.5012 and three
        baselines fitted exactly that bound, invalidating their log scores. Here the
        optimum sits far below it, so a bound-limited fitter would return 0.5012."""
        n, k = 400, 4
        y = rng.integers(0, k, size=n)
        scores = np.full((n, k), 0.0)
        scores[np.arange(n), y] = 0.6
        scores += rng.normal(0, 0.01, size=(n, k))
        targets = onehot(y, k)

        fitted = fit_temperature(scores, targets)
        assert fitted < 0.5012, f"fitter still bound-limited at {fitted}"
        # the bound-limited value is measurably worse
        from src.decision import metrics
        bound = metrics.log_score(softmax(scores, 0.5012), targets)
        best = metrics.log_score(softmax(scores, fitted), targets)
        assert best < bound * 0.5, f"no real improvement: {best} vs {bound}"

    def test_recovers_a_very_small_temperature(self):
        """Regression for the residual edge: the second version widened the coarse grid
        but refined over [t/3, 3t], so when the coarse winner was the coarse minimum the
        refinement edge still bound. The dispatch run fitted 0.000333 — the refinement
        edge — for five option-count groups."""
        n, k = 300, 3
        y = rng.integers(0, k, size=n)
        scores = np.zeros((n, k))
        scores[np.arange(n), y] = 0.05          # tiny margin -> optimum far below 1
        targets = onehot(y, k)
        fitted = fit_temperature(scores, targets)
        assert fitted < 0.01, f"still bound-limited at {fitted}"

    def test_never_worse_than_the_coarse_grid(self):
        n, k = 200, 5
        y = rng.integers(0, k, size=n)
        scores = rng.normal(size=(n, k)) + 1.5 * onehot(y, k)
        targets = onehot(y, k)
        from src.decision import metrics
        fitted = fit_temperature(scores, targets)
        coarse = [10.0 ** e for e in np.linspace(-3.0, 2.0, 51)]
        assert metrics.log_score(softmax(scores, fitted), targets) <= \
            min(metrics.log_score(softmax(scores, t), targets) for t in coarse) + 1e-9

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            fit_temperature(np.zeros((3, 4)), np.zeros((3, 5)))


class TestSoftmax:
    def test_rows_sum_to_one(self):
        p = softmax(rng.normal(size=(7, 5)))
        assert np.allclose(p.sum(axis=1), 1.0)

    def test_non_positive_temperature_rejected(self):
        with pytest.raises(ValueError):
            softmax(np.zeros((2, 2)), temperature=0.0)


class TestMetricBlock:
    def test_keys_and_shape_guard(self):
        probs = softmax(rng.normal(size=(10, 4)))
        targets = onehot(rng.integers(0, 4, size=10), 4)
        block = metric_block(probs, targets)
        assert set(block) >= {"accuracy", "log_score", "brier", "ece_report_only", "n"}
        assert block["n"] == 10
        with pytest.raises(ValueError):
            metric_block(probs, targets[:5])

    def test_metrics_move_when_the_model_changes(self):
        """The audit's lesson: a metric that cannot distinguish a broken model is not
        a metric."""
        targets = onehot(np.zeros(50, dtype=int), 3)
        good = np.tile([0.9, 0.05, 0.05], (50, 1))
        bad = np.tile([0.05, 0.9, 0.05], (50, 1))
        assert metric_block(good, targets)["accuracy"] > metric_block(bad, targets)["accuracy"]
        assert metric_block(good, targets)["log_score"] < metric_block(bad, targets)["log_score"]


class TestRaggedMetricBlock:
    """Variable option counts are the norm in dispatch (2-37 tools per request), so the
    metric block must handle a ragged set of decisions."""

    def test_handles_mixed_option_counts(self):
        scores = [np.array([2.0, 0.0]), np.array([0.0, 0.0, 3.0, 0.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0, 0.0])]
        out = protocol.ragged_metric_block(scores, targets)
        assert out["n"] == 2
        assert out["accuracy"] == 1.0
        assert set(out["temperatures_by_cardinality"]) == {2, 5}
        assert out["log_score"] < 0.5

    def test_weights_groups_by_size(self):
        # nine easy 2-way decisions and one 4-way decision
        scores = [np.array([3.0, 0.0])] * 9 + [np.array([3.0, 0.0, 0.0, 0.0])]
        targets = [np.array([1.0, 0.0])] * 9 + [np.array([1.0, 0.0, 0.0, 0.0])]
        out = protocol.ragged_metric_block(scores, targets)
        assert out["n"] == 10
        assert out["accuracy"] == 1.0

    def test_per_decision_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            protocol.ragged_metric_block([np.zeros(3)], [np.zeros(4)])

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            protocol.ragged_metric_block([np.zeros(3)], [])

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            protocol.ragged_metric_block([], [])

    def test_small_groups_reported(self):
        scores = [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]
        targets = [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])]
        out = protocol.ragged_metric_block(scores, targets, min_group=2)
        assert out["small_groups"] == [2, 4]


class TestCoverageAtPrecision:
    def test_refuses_to_report_when_saturated(self):
        """Regression: every candidate reported coverage 1.0 at 90% precision on a task
        with 93% base accuracy, and a verdict was drawn from that artifact."""
        n = 100
        y = np.zeros(n, dtype=int)
        probs = np.tile([0.9, 0.1], (n, 1))
        targets = onehot(y, 2)
        probs[:7] = [0.1, 0.9]                       # 7% wrong -> base accuracy 0.93
        out = coverage_at_precision(probs, targets, 0.90)
        assert out["saturated"] is True and out["coverage"] is None
        assert "uninformative" in out["note"]

    def test_reports_above_the_base_error_rate(self):
        n = 100
        probs = np.tile([0.9, 0.1], (n, 1))
        targets = onehot(np.zeros(n, dtype=int), 2)
        probs[:7] = [0.1, 0.9]
        out = coverage_at_precision(probs, targets, 0.95)
        assert out["saturated"] is False
        assert 0.0 <= out["coverage"] <= 1.0


class TestHeadroom:
    def test_banking77_is_flagged_as_unusable(self):
        """R1: 92.9% zero-shot leaves 7.1% headroom, so no head can show a difference."""
        out = headroom_check(0.9292)
        assert out["usable"] is False
        assert "insufficient headroom" in out["note"]

    def test_a_hard_task_is_usable(self):
        assert headroom_check(0.55)["usable"] is True


class TestEscalationSummary:
    def _data(self):
        # 80 correct at high confidence, 20 wrong at low confidence
        conf = np.concatenate([np.full(80, 0.9), np.full(20, 0.1)])
        correct = np.concatenate([np.ones(80), np.zeros(20)])
        return conf, correct

    def test_quality_target_is_met_by_what_is_auto_handled(self):
        conf, correct = self._data()
        out = escalation_summary(conf, correct, quality_target=0.95)
        assert out["feasible"] is True
        assert out["achieved_accuracy"] >= 0.95
        assert out["escalated"] > 0.0 and out["auto_handled"] > 0.0
        assert out["saved_vs_always_strong"] > 0.0

    def test_infeasible_when_no_fraction_reaches_the_target(self):
        conf = np.linspace(1.0, 0.0, 50)
        correct = np.zeros(50)
        out = escalation_summary(conf, correct, quality_target=0.95)
        assert out["feasible"] is False
        assert out["auto_handled"] == 0.0

    def test_bad_target_rejected(self):
        conf, correct = self._data()
        with pytest.raises(ValueError):
            escalation_summary(conf, correct, quality_target=0.0)


class TestCompareToBest:
    def test_compares_against_best_baseline_not_weakest(self):
        """Regression: a verdict hard-coded the weakest baseline and reported a Brier
        win that does not exist against the strongest."""
        candidates = {
            "cosine_tau": {"brier": 0.968, "log_score": 3.80, "accuracy": 0.9292},
            "knn5_tau": {"brier": 0.113, "log_score": 0.508, "accuracy": 0.9364},
            "head": {"brier": 0.116, "log_score": 0.377, "accuracy": 0.9311},
        }
        out = compare_to_best(candidates, baselines=["cosine_tau", "knn5_tau"], min_effect=0.0)
        assert out["per_metric"]["brier"]["best_baseline"] == "knn5_tau"
        # the head beats the weak baseline on Brier but NOT the strong one
        assert out["per_metric"]["brier"]["rows"]["head"]["beats_best_baseline"] is False
        assert out["per_metric"]["brier"]["rows"]["head"]["delta_vs_best_baseline"] < 0
        # and it does beat the best baseline on the log score
        assert out["per_metric"]["log_score"]["rows"]["head"]["beats_best_baseline"] is True

    def test_min_effect_suppresses_a_noise_sized_delta(self):
        candidates = {"base": {"accuracy": 0.9300}, "head": {"accuracy": 0.9311}}
        loose = compare_to_best(candidates, baselines=["base"], min_effect=0.0)
        strict = compare_to_best(candidates, baselines=["base"], min_effect=0.002)
        assert loose["winners"]["accuracy"] == ["head"]
        assert strict["winners"]["accuracy"] == []

    def test_missing_baseline_rejected(self):
        with pytest.raises(ValueError):
            compare_to_best({"a": {"accuracy": 0.5}}, baselines=["nope"])

    def test_negative_min_effect_rejected(self):
        with pytest.raises(ValueError):
            compare_to_best({"a": {"accuracy": 0.5}}, baselines=["a"], min_effect=-0.1)


def test_selective_curve_shape_and_monotonicity_of_coverage():
    probs = softmax(rng.normal(size=(200, 4)))
    targets = onehot(rng.integers(0, 4, size=200), 4)
    curve = selective_curve(probs, targets)
    assert len(curve) == 10
    assert curve[0]["coverage"] < curve[-1]["coverage"]
    assert all(0.0 <= p["selective_accuracy"] <= 1.0 for p in curve)
