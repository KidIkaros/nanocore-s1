"""Contract tests for NanoCoreS1 assembly and vendored metrics.

The model is exercised with a stub encoder — the real StateEncoder only exists
on Kaggle. This is the seam that keeps the package testable with no weights.
"""
import numpy as np
import pytest
import torch

from src.decision import metrics
from src.decision.head import DecisionHead
from src.decision.legacy import NanoCoreS1
from src.decision.schema import DecisionExample, Question, State

DIM = 32
rng = np.random.default_rng(0)


class StubEncoder:
    """Deterministic vectors per item — stands in for EmbeddingGemma 2."""

    embedding_dim = DIM

    def _vec(self, key):
        g = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        v = g.normal(size=DIM).astype(np.float32)
        return torch.tensor(v / np.linalg.norm(v))

    def encode_state(self, items, prompt_name=None):
        return torch.stack([self._vec(str(sorted(i.items())) if isinstance(i, dict) else i)
                            for i in items])

    def encode_options(self, texts, prompt_name=None):
        return torch.stack([self._vec("opt:" + t) for t in texts])


def _model(mode="fingerprint"):
    return NanoCoreS1(encoder=StubEncoder(),
                      head=DecisionHead(dim=DIM, mode=mode, hidden=16))


class TestAssembly:
    def test_decide_returns_prediction(self):
        m = _model()
        p = m.decide("what is my balance", Question(qtype="choice",
                                                  options=["balance", "lost_card"]))
        assert set(p.probabilities) == {"balance", "lost_card"}
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_string_state_becomes_text_state(self):
        m = _model()
        assert m.state_vector("hello").shape == (DIM,)

    def test_mean_pool_is_the_baseline(self):
        """composer=None must equal a masked mean over item embeddings."""
        m = _model()
        items = ["alpha", "beta", "gamma"]
        expected = m.encoder.encode_state(items).mean(0)
        expected = expected / expected.norm()
        assert torch.allclose(m.state_vector(State(items=items)), expected, atol=1e-5)

    def test_media_dict_state_runs(self):
        m = _model()
        s = State(items=[{"image": "x.png"}, "a shoe"])
        assert m.state_vector(s).shape == (DIM,)

    def test_decide_batch_one_encode(self):
        m = _model()
        preds = m.decide_batch("state text", [
            Question(qtype="choice", options=["a", "b"]),
            Question(qtype="noul"),
        ])
        assert len(preds) == 2
        assert preds[1].noul is not None

    def test_interaction_mode_decide(self):
        m = _model(mode="interaction")
        p = m.decide("state text", Question(qtype="choice", options=["a", "b"]))
        assert p.choice in ("a", "b")


class TestMetricsContracts:
    """The audit's lesson: metrics must move when the model changes."""

    def test_log_score_prefers_truth(self):
        target = np.array([[1.0, 0.0, 0.0]])
        good = np.array([[0.9, 0.05, 0.05]])
        bad = np.array([[0.05, 0.9, 0.05]])
        assert metrics.log_score(good, target) < metrics.log_score(bad, target)

    def test_brier_prefers_truth(self):
        target = np.array([[1.0, 0.0]])
        assert (metrics.brier_score(np.array([[0.9, 0.1]]), target)
                < metrics.brier_score(np.array([[0.1, 0.9]]), target))

    def test_rps_respects_ordinality(self):
        """A near miss on an ordered scale must cost less than a far miss."""
        target = np.array([[1.0, 0.0, 0.0, 0.0]])
        near = np.array([[0.5, 0.5, 0.0, 0.0]])
        far = np.array([[0.0, 0.0, 0.5, 0.5]])
        assert metrics.ranked_probability_score(near, target) \
            < metrics.ranked_probability_score(far, target)

    def test_weighted_metric_handles_ragged(self):
        probs = [np.array([0.9, 0.1]), np.array([0.5, 0.3, 0.2])]
        targets = [np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        v = metrics.weighted_metric(probs, targets, metrics.log_score)
        assert np.isfinite(v)

    def test_ece_distinguishes_miscalibration(self):
        """ECE must be worse for a confidently-wrong model than an honest one —
        the audit found an evaluator where a zeroed model scored *better*."""
        conf_wrong = np.array([[0.05, 0.95]] * 100)
        honest = np.array([[0.55, 0.45]] * 100)
        target = np.array([[1.0, 0.0]] * 100)
        assert metrics.expected_calibration_error(conf_wrong, target) \
            > metrics.expected_calibration_error(honest, target)

    def test_selective_accuracy(self):
        acc, cov = metrics.selective_accuracy(
            [0.9, 0.1, 0.95], [1, 0, 1], threshold=0.5)
        assert acc == 1.0 and cov == pytest.approx(2 / 3)


class TestExampleSchema:
    def test_example_accepts_option_vectors(self):
        ex = DecisionExample(state_embedding=np.zeros(DIM), qtype="choice",
                             labels=["a", "b"], target=[1.0, 0.0],
                             option_embeddings=[np.zeros(DIM), np.ones(DIM)])
        assert len(ex.option_embeddings) == 2
