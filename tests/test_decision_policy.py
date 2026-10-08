"""Slow-state policy arms — the glial A/B, tested on synthetic streams."""
import numpy as np
import pytest

from src.decision.policy import (GlialPolicy, Observation, PolicyThresholds,
                                 RecalibrateConfig, RecalibratePolicy,
                                 SlowState, SlowStateConfig, StaticPolicy,
                                 Stream, StreamConfig, run_stream)


BASE = PolicyThresholds(tau_answer=0.60, k_clarify=3, tau_in_schema=0.70)
CFG = StreamConfig(t_prob=1.0, qhat=0.95, k_clarify=3, recovery_window=50)


def _run(policy, scores, y, shift_at):
    return run_stream(policy, Stream(scores, y, shift_at), CFG)


def _obs(top_prob: float, set_size: int = 1, max_score: float = 1.0) -> Observation:
    return Observation(top_prob=top_prob, set_size=set_size, entropy=0.5,
                       pred=0, max_score=max_score)


def _stream(n: int, shift_at: int, phase2_scale: float, k: int = 4,
            seed: int = 0) -> tuple:
    """Class-centered scores; phase 2 is uniformly less confident."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    scores = rng.standard_normal((n, k))
    scores[np.arange(n), y] += 3.0
    scores[shift_at:] *= phase2_scale
    return scores, y


def _novel_stream(n: int, shift_at: int, k: int = 6, seed: int = 0) -> tuple:
    """Phase 2 carries no class signal — its true intent is absent from the
    option set, which is the only thing 'novel input' means for the gate."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    scores = rng.standard_normal((n, k)) * 0.6
    scores[np.arange(shift_at), y[:shift_at]] += 3.0
    return scores, y


def test_static_thresholds_never_move():
    p = StaticPolicy(BASE)
    for _ in range(100):
        p.observe(_obs(0.1, set_size=9))
    assert p.thresholds() == BASE
    assert p.label_delay is None


def test_glial_ignores_labels():
    """The whole claim is that the glial arm needs no labels."""
    assert GlialPolicy(BASE).label_delay is None


def test_slow_state_activates_on_anomalies():
    state = SlowState(BASE, SlowStateConfig(window=20, activate_at=14,
                                            reference_max_score=0.8))
    assert state.thresholds() == BASE
    for _ in range(19):
        state.observe(_obs(0.10, set_size=9))      # would escalate
    assert state.thresholds() == BASE              # partial window is not evidence
    state.observe(_obs(0.10, set_size=9))
    moved = state.thresholds()
    assert moved.tau_answer > BASE.tau_answer
    assert moved.k_clarify < BASE.k_clarify
    assert moved.tau_in_schema > BASE.tau_in_schema


def test_slow_state_treats_out_of_schema_scores_as_anomalous():
    """Novel input is gated by τ_in_schema, so it must drive the slow state."""
    state = SlowState(BASE, SlowStateConfig(window=20, activate_at=14,
                                            reference_max_score=0.8))
    for _ in range(20):
        state.observe(_obs(0.95, set_size=1, max_score=0.40))
    assert state.thresholds().tau_in_schema > BASE.tau_in_schema


def test_slow_state_detects_drift_without_any_threshold_crossing():
    """Every decision looks fine individually; the population has moved."""
    cfg = SlowStateConfig(window=20, activate_at=14, reference_max_score=3.0)
    state = SlowState(BASE, cfg)
    for _ in range(20):
        state.observe(_obs(0.95, set_size=1, max_score=2.0))
    assert state.thresholds().tau_in_schema > BASE.tau_in_schema


def test_slow_state_holds_when_decisions_are_healthy():
    state = SlowState(BASE, SlowStateConfig(window=20, activate_at=14))
    for _ in range(200):
        state.observe(_obs(0.95, set_size=1))
    assert state.thresholds() == BASE


def test_slow_state_stays_bounded_under_sustained_drift():
    cfg = SlowStateConfig(window=20, activate_at=14, tau_step=0.04, max_shift=0.30)
    state = SlowState(BASE, cfg)
    for _ in range(500):
        state.observe(_obs(0.05, set_size=12))
    moved = state.thresholds()
    assert moved.tau_answer == pytest.approx(BASE.tau_answer + cfg.max_shift)
    assert moved.k_clarify == BASE.k_clarify - 1   # one step, never collapsed
    assert moved.tau_answer <= 0.999


def test_slow_state_holds_through_a_single_quiet_window():
    """Schmitt trigger: one calm window is not grounds to release.

    Measured on real data, a shift sitting near ``drift_margin`` otherwise
    flickers and the bar snaps back while the shift is still in force.
    """
    cfg = SlowStateConfig(window=20, activate_at=14, release_at=6,
                          reference_max_score=0.8)
    state = SlowState(BASE, cfg)
    for _ in range(14):
        state.observe(_obs(0.05, set_size=12))         # anomalies
    for _ in range(6):
        state.observe(_obs(0.95, set_size=1))          # window now full: 14 anomalies
    raised = state.thresholds().tau_answer
    assert raised > BASE.tau_answer

    for _ in range(6):
        state.observe(_obs(0.95, set_size=1))          # 8 anomalies left in window
    assert state.thresholds().tau_answer == raised     # held, not released

    for _ in range(8):
        state.observe(_obs(0.95, set_size=1))          # 0 anomalies → sustained calm
    assert state.thresholds() == BASE


def test_calibrate_sets_the_bar_above_the_worst_healthy_window():
    """The bar is data-derived, not a constant: healthy worst window, plus one."""
    state = SlowState(BASE, SlowStateConfig(window=20, activate_at=14))
    mixed = [_obs(0.05, set_size=9) if i % 5 == 0 else _obs(0.95, set_size=1)
             for i in range(100)]                    # exactly 4 anomalies per window
    assert state.calibrate(mixed) == 5
    for _ in range(20):
        state.observe(_obs(0.05, set_size=9))
    assert state.thresholds().tau_answer > BASE.tau_answer


def test_calibrated_state_stays_quiet_on_its_reference_stream():
    """Regression: an uncalibrated bar fires on healthy data.

    On cached CLINC150 the median healthy decision sits exactly on tau_answer,
    so ~half of healthy decisions look anomalous and a constant bar activates
    the policy on data that has not shifted.
    """
    reference = [_obs(0.05, set_size=9) if i % 5 == 0 else _obs(0.95, set_size=1)
                 for i in range(200)]
    state = SlowState(BASE, SlowStateConfig(window=20, activate_at=14))
    state.calibrate(reference)
    for obs in reference:
        state.observe(obs)
    assert state.thresholds() == BASE


def test_slow_state_rejects_impossible_config():
    with pytest.raises(ValueError):
        SlowState(BASE, SlowStateConfig(window=10, activate_at=11))
    with pytest.raises(ValueError):
        SlowState(BASE, SlowStateConfig(activate_at=10, release_at=10))


def test_recalibrate_raises_bar_when_confident_labels_are_wrong():
    """A refit must react to labels the frozen threshold got wrong."""
    p = RecalibratePolicy(BASE, RecalibrateConfig(delay=0, refit_every=40))
    for _ in range(120):
        p.observe_labeled(_obs(0.99), label=1)      # confident, always wrong
    assert p.thresholds().tau_answer > BASE.tau_answer


def test_recalibrate_refits_in_schema_only_with_an_option_count():
    p = RecalibratePolicy(BASE, RecalibrateConfig(delay=0, refit_every=60,
                                                  n_options=4))
    for _ in range(60):
        p.observe_labeled(_obs(0.99, max_score=0.95), label=0)   # in schema
    for _ in range(60):
        p.observe_labeled(_obs(0.99, max_score=0.30), label=9)   # outside it
    assert p.thresholds().tau_in_schema > BASE.tau_in_schema

    frozen = RecalibratePolicy(BASE, RecalibrateConfig(delay=0, refit_every=60))
    frozen.observe_labeled(_obs(0.99, max_score=0.30), label=9)
    assert frozen.thresholds().tau_in_schema == BASE.tau_in_schema


def test_recalibrate_requires_valid_window():
    with pytest.raises(ValueError):
        RecalibratePolicy(BASE, RecalibrateConfig(delay=-1, refit_every=10))
    with pytest.raises(ValueError):
        RecalibratePolicy(BASE, RecalibrateConfig(delay=0, refit_every=1))


def test_run_stream_reports_both_phases_with_uncertainty():
    scores, y = _stream(400, shift_at=200, phase2_scale=1.0)
    out = _run(StaticPolicy(BASE), scores, y, 200)
    assert set(out["phase1"]) == set(out["phase2"])
    assert out["phase1"]["n"] == out["phase2"]["n"] == 200
    ci = out["phase2_selective_ci"]
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert out["phase2"]["tau_answer_mean"] == pytest.approx(BASE.tau_answer)
    assert out["phase2"]["tau_in_schema_max"] == pytest.approx(BASE.tau_in_schema)


def test_run_stream_rejects_misaligned_stream():
    scores, y = _stream(100, shift_at=50, phase2_scale=1.0)
    with pytest.raises(ValueError):
        run_stream(StaticPolicy(BASE), Stream(scores, y[:90], 50), CFG)


def test_run_stream_rejects_bad_shift_point():
    scores, y = _stream(100, shift_at=50, phase2_scale=1.0)
    with pytest.raises(ValueError):
        _run(StaticPolicy(BASE), scores, y, 0)


def test_glial_escalates_more_than_static_after_a_confidence_drop():
    scores, y = _stream(600, shift_at=300, phase2_scale=0.35, seed=3)
    static = _run(StaticPolicy(BASE), scores, y, 300)
    glial = _run(GlialPolicy(BASE), scores, y, 300)
    assert glial["phase2"]["escalation_rate"] > static["phase2"]["escalation_rate"]


def test_glial_leaves_a_healthy_stream_alone():
    """No false activation: a stationary stream must not move the policy."""
    scores, y = _stream(600, shift_at=300, phase2_scale=1.0, seed=4)
    static = _run(StaticPolicy(BASE), scores, y, 300)
    glial = _run(GlialPolicy(BASE), scores, y, 300)
    assert glial["phase2"]["escalation_rate"] == pytest.approx(
        static["phase2"]["escalation_rate"])


def test_label_delay_changes_what_recalibrate_sees():
    """Delay is the arm's information budget, so the two arms must diverge."""
    scores, y = _stream(600, shift_at=300, phase2_scale=0.35, seed=5)
    oracle = _run(RecalibratePolicy(BASE, RecalibrateConfig(0, 50)), scores, y, 300)
    delayed = _run(RecalibratePolicy(BASE, RecalibrateConfig(250, 50)), scores, y, 300)
    assert oracle["policy"] != delayed["policy"]
    assert oracle["phase2"]["tau_answer_mean"] != pytest.approx(
        delayed["phase2"]["tau_answer_mean"])
    for arm in (oracle, delayed):
        assert 0 < arm["phase2"]["tau_answer_mean"] < 1


def test_glial_raises_the_in_schema_bar_only_with_a_reference_scale():
    """τ_in_schema lives in raw-score units, so it may only move relative to a
    known scale — otherwise the arm cannot know what a step means."""
    scores, y = _novel_stream(600, shift_at=300, seed=4)
    scaled = _run(GlialPolicy(BASE, SlowStateConfig(reference_max_score=0.8)),
                  scores, y, 300)
    unscaled = _run(GlialPolicy(BASE), scores, y, 300)
    assert scaled["phase2"]["tau_in_schema_max"] > BASE.tau_in_schema
    assert unscaled["phase2"]["tau_in_schema_max"] == BASE.tau_in_schema


def test_stream_records_wrong_answers_separately_from_escalations():
    """The safety metric the A/B is decided on: answered and incorrect."""
    scores, y = _novel_stream(400, shift_at=200, seed=8)
    phase2 = _run(StaticPolicy(BASE), scores, y, 200)["phase2"]
    assert phase2["wrong_answer_rate"] <= 1 - phase2["escalation_rate"] + 1e-9


def test_recovery_latency_is_zero_when_the_shift_is_harmless():
    scores, y = _stream(600, shift_at=300, phase2_scale=1.0, seed=6)
    assert _run(StaticPolicy(BASE), scores, y, 300)["recovery_latency"] == 0
