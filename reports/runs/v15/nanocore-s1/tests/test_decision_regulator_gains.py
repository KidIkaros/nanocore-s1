"""Learned glial-regulator shift gains (#13).

The SlowState moves tau_answer by a hand-set tau_step (0.04) bounded by
max_shift (0.30). #13 fits those from the trajectory log instead of constants —
the library + arxiv negative result: regulator gains must be LEARNED, never a
fixed global gain. The fit maximizes separation between healthy and drifted
windows' anomaly counts, so the step is scaled to the deployment's own noise.

Behavioral contract (unchanged, must still hold): injected drift moves
thresholds; healthy input returns them to base (no one-way ratchet); movement
stays bounded; E6 units fix holds (tau_in_schema in raw-score units).
"""
import numpy as np

from src.decision.slow import (SlowState, SlowStateConfig, PolicyThresholds,
                               fit_regulator_gains, Observation)


def _obs(top_prob, set_size, max_score):
    return Observation(top_prob=top_prob, set_size=set_size, entropy=0.3,
                       pred=0, max_score=max_score, margin=0.1)


def _healthy(n, seed=0):
    rng = np.random.default_rng(seed)
    return [_obs(float(rng.uniform(0.8, 0.99)), 1, float(rng.uniform(8, 12)))
            for _ in range(n)]


def _drifted(n, seed=1):
    rng = np.random.default_rng(seed)
    return [_obs(float(rng.uniform(0.1, 0.3)), 6, float(rng.uniform(0, 2)))
            for _ in range(n)]


def test_fit_gains_returns_bounded_step():
    healthy, drifted = _healthy(60), _drifted(60)
    tau_step, max_shift = fit_regulator_gains(healthy, drifted)
    assert 0.0 < tau_step <= 0.3
    assert tau_step <= max_shift <= 0.5


def test_learned_gains_separate_healthy_from_drifted():
    """The fitted step must be large enough that a drifted streak raises
    tau_answer meaningfully above base, while healthy windows do not."""
    base = PolicyThresholds(tau_answer=0.5, k_clarify=3, tau_in_schema=0.0)
    healthy, drifted = _healthy(60), _drifted(60)
    tau_step, max_shift = fit_regulator_gains(healthy, drifted)
    cfg = SlowStateConfig(window=5, activate_at=3, release_at=1,
                          tau_step=tau_step, max_shift=max_shift,
                          reference_max_score=10.0)
    s = SlowState(base, cfg)
    s.calibrate(healthy)
    for o in healthy[:5]:
        s.observe(o)
    assert s.thresholds().tau_answer == base.tau_answer  # healthy -> no move
    for o in drifted[:8]:
        s.observe(o)
    assert s.thresholds().tau_answer > base.tau_answer     # drift -> moves up


def test_regulator_returns_to_base_after_drift():
    """No one-way ratchet: after drift then healthy, thresholds return to base."""
    base = PolicyThresholds(tau_answer=0.5, k_clarify=3, tau_in_schema=0.0)
    healthy, drifted = _healthy(60), _drifted(60)
    tau_step, max_shift = fit_regulator_gains(healthy, drifted)
    cfg = SlowStateConfig(window=5, activate_at=3, release_at=1,
                          tau_step=tau_step, max_shift=max_shift,
                          reference_max_score=10.0)
    s = SlowState(base, cfg)
    s.calibrate(healthy)
    for o in drifted[:10]:
        s.observe(o)
    assert s.thresholds().tau_answer > base.tau_answer
    for o in healthy[:12]:
        s.observe(o)
    assert s.thresholds().tau_answer == base.tau_answer


def test_fixed_gain_rejected_gains_differ_by_regime():
    """The learned gains adapt to the deployment: a subtle-drift regime (hard to
    separate from healthy) gets a smaller step than an obvious-drift regime. A
    fixed global gain returns the same constant regardless — this asserts the
    fit is data-dependent. (Separation saturates at 1.0 when both regimes are
    fully anomalous, so the two regimes must differ in SEPARABILITY, not just
    both being 'drifted'.)"""
    # obvious drift: anomaly rate ~1.0, healthy ~0 -> separation ~1
    obvious = _drifted(60, seed=2)
    # subtle drift: only mildly anomalous (top_prob just under the bar) ->
    # lower separation from healthy
    rng = np.random.default_rng(11)
    subtle = [_obs(float(rng.uniform(0.45, 0.49)), 3, float(rng.uniform(4, 6)))
              for _ in range(60)]
    _, ms_obvious = fit_regulator_gains(_healthy(60), obvious)
    _, ms_subtle = fit_regulator_gains(_healthy(60), subtle)
    assert ms_obvious > ms_subtle, (
        f"obvious drift should earn a larger max_shift than subtle: "
        f"{ms_obvious} vs {ms_subtle}")
