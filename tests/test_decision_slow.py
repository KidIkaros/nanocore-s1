"""The shipped slow state — gate wiring, persistence, and detachability."""
import numpy as np
import pytest

from src.decision.gate import ConformalGate
from src.decision.slow import (Observation, PolicyThresholds, SlowState,
                               SlowStateConfig, observe_row)


def _gate(alpha: float = 0.10) -> ConformalGate:
    """A calibrated gate on synthetic, well-separated scores."""
    rng = np.random.default_rng(0)
    k, n = 4, 400
    y = rng.integers(0, k, n)
    scores = rng.standard_normal((n, k))
    scores[np.arange(n), y] += 3.0
    gate = ConformalGate(alpha=alpha, min_n=100)
    gate.calibrate(scores, y, seed=0)
    gate.fit_in_schema(scores, rng.standard_normal((50, k)) - 1.0)
    return gate


def _anomalous_row(k: int = 4) -> np.ndarray:
    """Flat scores: no class stands out, so the policy would escalate."""
    return np.zeros(k)


def test_gate_without_a_slow_state_is_unchanged():
    gate = _gate()
    assert gate.slow is None
    assert gate.base_thresholds().tau_answer == gate.tau_answer
    gate.decide(np.array([3.0, 0.0, 0.0, 0.0]), ["a", "b", "c", "d"])


def test_attached_slow_state_leaves_a_healthy_stream_alone():
    """Detachability is the contract: no drift, no movement."""
    gate = _gate()
    base = gate.base_thresholds()
    gate.attach_slow_state(SlowStateConfig(reference_max_score=3.0))
    assert gate.slow is not None
    rng = np.random.default_rng(1)
    for i in range(60):
        row = rng.standard_normal(4)
        row[i % 4] += 3.0
        gate.decide(row, ["a", "b", "c", "d"])
    assert gate.slow.thresholds().tau_answer == pytest.approx(base.tau_answer)


def test_gate_escalates_more_once_anomalies_sustain():
    gate = _gate()
    gate.attach_slow_state(SlowStateConfig(reference_max_score=3.0))
    healthy = np.array([3.0, 0.0, 0.0, 0.0])
    for _ in range(60):
        gate.decide(healthy, ["a", "b", "c", "d"])
    assert gate.slow.thresholds().tau_answer == pytest.approx(gate.tau_answer)

    for _ in range(60):
        gate.decide(_anomalous_row(), ["a", "b", "c", "d"])
    assert gate.slow.thresholds().tau_answer > gate.tau_answer


def test_gate_calibrates_the_bar_from_a_reference_stream():
    gate = _gate()
    gate.attach_slow_state(SlowStateConfig())
    reference = [Observation(0.9, 1, 0.1, 0, 3.0, 0.5) for _ in range(60)]
    assert gate.calibrate_slow(reference) == 1


def test_calibrating_without_a_state_fails_loudly():
    with pytest.raises(RuntimeError):
        _gate().calibrate_slow([Observation(0.9, 1, 0.1, 0, 3.0, 0.5)])


def test_bundle_roundtrip_preserves_the_slow_mode(tmp_path):
    gate = _gate()
    gate.attach_slow_state(SlowStateConfig(window=30, tau_step=0.05))
    reference = [Observation(0.9, 1, 0.1, 0, 3.0, 0.5) for _ in range(60)]
    gate.calibrate_slow(reference)
    fitted = gate.slow.fitted_config().activate_at

    gate.save(tmp_path / "gate.json")
    reloaded = ConformalGate.load(tmp_path / "gate.json")
    assert reloaded.slow is not None
    assert reloaded.slow.config.window == 30
    assert reloaded.slow.config.tau_step == pytest.approx(0.05)
    assert reloaded.slow.fitted_config().activate_at == fitted
    assert reloaded.tau_answer == pytest.approx(gate.tau_answer)


def test_concurrent_decisions_run_clean():
    """Smoke check, not a proof.

    The gate is served by a ThreadingHTTPServer, so attaching mutable state
    makes concurrency a real question. A lost update is a race, so no test can
    fail deterministically on it — the guarantee comes from the lock inside
    ``SlowState``; this only asserts that concurrent use raises nothing and
    leaves the invariants intact.
    """
    import threading

    gate = _gate()
    gate.attach_slow_state(SlowStateConfig(reference_max_score=3.0))
    row = np.array([3.0, 0.0, 0.0, 0.0])
    errors: list = []

    def decide_many():
        try:
            for _ in range(80):
                gate.decide(row, ["a", "b", "c", "d"])
        except Exception as exc:            # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=decide_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    # a healthy stream never activates, however the requests interleaved
    assert gate.slow.thresholds().tau_answer == pytest.approx(gate.tau_answer)


def test_experiment_and_gate_share_one_action_rule():
    """The stream must decide exactly as the shipped gate does."""
    from src.decision.policy import action_for
    from src.decision.slow import Observation, PolicyThresholds

    gate = _gate()
    thresholds = PolicyThresholds(tau_answer=0.30, k_clarify=2, tau_in_schema=0.70)
    for top_prob, max_score, size in ((0.99, 0.20, 1), (0.99, 0.90, 1),
                                      (0.10, 0.90, 1), (0.10, 0.90, 9)):
        obs = Observation(top_prob, size, 0.5, 0, max_score, 0.1)
        assert action_for(obs, thresholds) == gate._gate_action(
            top_prob, max_score, size, thresholds)


def test_action_rule_order_is_out_of_schema_first():
    """The documented order is the contract, so it is asserted directly."""
    from src.decision.slow import PolicyThresholds

    gate = _gate()
    confident = PolicyThresholds(tau_answer=0.01, k_clarify=3, tau_in_schema=0.70)
    assert gate._gate_action(0.99, 0.20, 1, confident) == "escalate"
    assert gate._gate_action(0.99, 0.90, 1, confident) == "answer"
    quiet = PolicyThresholds(tau_answer=0.99, k_clarify=3, tau_in_schema=None)
    assert gate._gate_action(0.50, 0.90, 2, quiet) == "clarify"
    assert gate._gate_action(0.50, 0.90, 9, quiet) == "escalate"


def test_bundle_without_a_slow_state_stays_frozen(tmp_path):
    gate = _gate()
    gate.save(tmp_path / "gate.json")
    assert ConformalGate.load(tmp_path / "gate.json").slow is None


def test_adapt_produces_a_bundle_carrying_the_slow_mode():
    """The deployable path: adapt(glial=True) attaches and calibrates it."""
    from src.decision.adapt import AdaptConfig, adapt

    class _Encoder:
        """Deterministic, separable embeddings — no model weights involved."""
        def encode(self, texts, **kw):
            return np.array([[1.0 if f"class{i}" in t else 0.0 for i in range(4)]
                             for t in texts], dtype=np.float32)
        def encode_options(self, texts):
            return np.eye(4, dtype=np.float32)

    texts = [f"class{i} sample{j}" for i in range(4) for j in range(150)]
    labels = [str(i) for i in range(4) for _ in range(150)]
    out = adapt(texts, labels, _Encoder(), cfg=AdaptConfig(glial=True,
                                                           min_cal=100))
    assert out.report["glial"] is not None
    assert out.report["glial"]["activate_at"] >= 1
    assert out.model.gate.slow is not None


def test_adapt_without_glial_leaves_the_gate_frozen():
    from src.decision.adapt import AdaptConfig, adapt

    class _Encoder:
        def encode(self, texts, **kw):
            return np.array([[1.0 if f"class{i}" in t else 0.0 for i in range(4)]
                             for t in texts], dtype=np.float32)
        def encode_options(self, texts):
            return np.eye(4, dtype=np.float32)

    texts = [f"class{i} sample{j}" for i in range(4) for j in range(150)]
    labels = [str(i) for i in range(4) for _ in range(150)]
    out = adapt(texts, labels, _Encoder(), cfg=AdaptConfig(min_cal=100))
    assert out.report["glial"] is None
    assert out.model.gate.slow is None


def test_slow_state_is_reexported_for_the_experiment_module():
    """policy.py imports the mechanism; the identity must be one object."""
    from src.decision import policy
    assert policy.SlowState is SlowState
    assert policy.PolicyThresholds is PolicyThresholds
