"""Retrain cadence — scheduled and drift-triggered, with the anti-thrash floor."""
import pytest

from src.decision.cadence import CadenceConfig, should_retrain
from src.decision.monitor import DriftReport

CFG = CadenceConfig(scheduled_every=1000, drift_min_samples=500)


def _drift(alerts=("top_prob KS=0.247 > 0.15",)):
    return DriftReport(covariate_ks=0.247, action_shift={}, set_size_ks=0.01,
                       shift_types=["covariate"], alerts=list(alerts))


def test_drift_triggers_once_the_floor_is_cleared():
    t = should_retrain(600, _drift(), CFG)
    assert t.retrain and t.cadence == "drift"
    assert "KS=0.247" in t.reason


def test_drift_below_the_floor_holds_and_says_why():
    """The anti-thrash property: retraining on a shifted window can hurt
    (measured 0.305 unsafe vs static 0.143), so an alert alone is not a trigger."""
    t = should_retrain(120, _drift(), CFG)
    assert not t.retrain and t.cadence == "none"
    assert "floor" in t.reason and "120" in t.reason


def test_scheduled_fires_without_any_drift():
    t = should_retrain(1000, None, CFG)
    assert t.retrain and t.cadence == "scheduled"


def test_below_every_threshold_holds():
    t = should_retrain(50, None, CFG)
    assert not t.retrain and t.cadence == "none"
    assert "no cadence reached" in t.reason


def test_scheduled_path_can_be_disabled():
    cfg = CadenceConfig(scheduled_every=None, drift_min_samples=500)
    assert not should_retrain(50_000, None, cfg).retrain


def test_drift_outranks_scheduled_when_both_are_ready():
    """Faster reaction wins: a drift trigger at 600 samples is not reported as
    a scheduled one."""
    assert should_retrain(1000, _drift(), CFG).cadence == "drift"


def test_empty_drift_report_is_not_a_trigger():
    quiet = DriftReport(covariate_ks=0.01, action_shift={}, set_size_ks=0.01)
    assert not should_retrain(600, quiet, CFG).retrain


def test_negative_samples_rejected():
    with pytest.raises(ValueError, match="negative"):
        should_retrain(-1, None, CFG)
