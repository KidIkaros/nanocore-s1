"""Canary release — deterministic bucketing and the guardrail verdict.

The model owns the decision, not the transport: these are the pieces a
deployer's router cannot easily get right itself.
"""
import pytest

from src.decision.canary import GuardrailConfig, arm_for, canary_verdict

CFG = GuardrailConfig()


def test_bucketing_is_stable_for_the_same_key():
    """A counter-based split would send the same input to different arms across
    processes, and the comparison would be between two arbitrary slices."""
    for key in ("cancel my flight", "what is the weather", "renew membership"):
        assert len({arm_for(key, 0.3) for _ in range(5)}) == 1


def test_fraction_extremes():
    assert all(arm_for(f"k{i}", 0.0) == "control" for i in range(200))
    assert all(arm_for(f"k{i}", 1.0) == "canary" for i in range(200))


def test_fraction_is_honoured_within_tolerance():
    keys = [f"utterance {i}" for i in range(4000)]
    share = sum(arm_for(k, 0.25) == "canary" for k in keys) / len(keys)
    assert abs(share - 0.25) < 0.02


def test_salt_moves_the_assignment():
    keys = [f"k{i}" for i in range(500)]
    a = {k for k in keys if arm_for(k, 0.5, salt="run1") == "canary"}
    b = {k for k in keys if arm_for(k, 0.5, salt="run2") == "canary"}
    assert a != b


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError, match="fraction"):
        arm_for("k", 1.5)


def _arm(n=500, unsafe=0.05, escalate=0.30, lat=100.0):
    return {"n": n, "unsafe_rate": unsafe, "escalate_rate": escalate,
            "latency_p95": lat}


def test_parity_promotes():
    v = canary_verdict(_arm(), _arm(unsafe=0.051), CFG)
    assert v.decision == "promote"
    assert v.deltas["unsafe_rate"] == pytest.approx(0.001, abs=1e-9)


def test_unsafe_regression_rolls_back_and_names_the_metric():
    v = canary_verdict(_arm(), _arm(unsafe=0.12), CFG)
    assert v.decision == "rollback" and "unsafe_rate" in v.reason


def test_escalation_regression_rolls_back():
    """Safer-but-escalates-everything is a cost regression, not a win."""
    v = canary_verdict(_arm(), _arm(escalate=0.60), CFG)
    assert v.decision == "rollback" and "escalate_rate" in v.reason


def test_latency_regression_rolls_back():
    v = canary_verdict(_arm(), _arm(lat=400.0), CFG)
    assert v.decision == "rollback" and "latency" in v.reason


def test_insufficient_evidence_holds_rather_than_promotes():
    v = canary_verdict(_arm(n=20), _arm(n=20), CFG)
    assert v.decision == "hold" and "insufficient" in v.reason


def test_hold_dominates_a_breach_on_thin_evidence():
    v = canary_verdict(_arm(n=20), _arm(n=20, unsafe=0.5), CFG)
    assert v.decision == "hold"


def test_missing_metric_is_reported_unchecked_not_passed():
    """A guard that fails open is not a guard."""
    control = {"n": 500, "escalate_rate": 0.30, "latency_p95": 100.0}
    canary = {"n": 500, "escalate_rate": 0.30, "latency_p95": 100.0}
    v = canary_verdict(control, canary, CFG)
    assert "unsafe_rate" in v.unchecked
    assert "unsafe_rate" not in v.checked
    assert v.decision == "promote"


def test_disabled_check_is_skipped_entirely():
    cfg = GuardrailConfig(max_unsafe_delta=None)
    v = canary_verdict(_arm(), _arm(unsafe=0.9), cfg)
    assert "unsafe_rate" not in v.checked
    assert "unsafe_rate" not in v.unchecked
    assert v.decision == "promote"
