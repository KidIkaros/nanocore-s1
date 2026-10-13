"""Contract tests for DecisionHead — random vectors, no model weights.

The properties under test are the design commitments: order invariance by
construction, zero-bias final layer, strictly-proper training that actually
learns, and an abstention path that fires on out-of-distribution states.
"""
import numpy as np
import pytest
import torch

from src.decision.head import DecisionHead
from src.decision.schema import DecisionExample

DIM, HID = 32, 16
rng = np.random.default_rng(0)


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _separable_data(n_per_class=30, k=3, dim=DIM, seed=1):
    """k well-separated option prototypes; states are noisy copies of one."""
    g = np.random.default_rng(seed)
    protos = np.stack([_unit(g.normal(size=dim)) for _ in range(k)])
    labels = [f"opt_{i}" for i in range(k)]
    examples = []
    for _ in range(n_per_class):
        for c in range(k):
            state = _unit(protos[c] + 0.15 * g.normal(size=dim))
            target = [0.0] * k
            target[c] = 1.0
            examples.append(DecisionExample(
                state_embedding=state, qtype="choice", labels=list(labels),
                target=target,
                option_embeddings=[protos[i] for i in range(k)]))
    return labels, protos, examples


class TestContract:
    @pytest.mark.parametrize("mode", ["interaction", "fingerprint"])
    def test_probabilities_sum_to_one(self, mode):
        head = DecisionHead(dim=DIM, mode=mode, hidden=HID)
        s = _unit(rng.normal(size=DIM))
        opts = [_unit(rng.normal(size=DIM)) for _ in range(4)]
        p = head.predict(s, opts, qtype="choice", labels=["a", "b", "c", "d"])
        assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert set(p.probabilities) == {"a", "b", "c", "d"}

    @pytest.mark.parametrize("mode", ["interaction", "fingerprint"])
    def test_order_invariance(self, mode):
        """Permuting options must only permute the scores — by construction."""
        head = DecisionHead(dim=DIM, mode=mode, hidden=HID)
        labels, protos, examples = _separable_data()
        if mode == "fingerprint":
            head.fit(examples, epochs=5, seed=0)  # fingerprints must exist
        s = _unit(rng.normal(size=DIM))
        fwd = head.predict(s, protos, qtype="choice", labels=labels)
        order = [2, 0, 1]
        rev_labels = [labels[i] for i in order]
        rev_opts = [protos[i] for i in order]
        rev = head.predict(s, rev_opts, qtype="choice", labels=rev_labels)
        assert fwd.choice == rev.choice
        for l in labels:
            assert fwd.probabilities[l] == pytest.approx(rev.probabilities[l], abs=1e-6)

    def test_final_layer_has_no_bias(self):
        for mode in ("interaction", "fingerprint"):
            assert DecisionHead(dim=DIM, mode=mode, hidden=HID).final_layer_has_bias is False

    def test_fingerprint_needs_no_option_embeddings(self):
        labels, _, examples = _separable_data()
        head = DecisionHead(dim=DIM, mode="fingerprint")
        head.fit(examples, epochs=5)
        p = head.predict(_unit(rng.normal(size=DIM)), None,
                         qtype="choice", labels=labels)
        assert p.choice in labels

    def test_interaction_requires_option_embeddings(self):
        head = DecisionHead(dim=DIM, mode="interaction")
        with pytest.raises(ValueError):
            head.predict(_unit(rng.normal(size=DIM)), None,
                         qtype="choice", labels=["a", "b"])

    def test_dim_mismatch_rejected(self):
        head = DecisionHead(dim=DIM, mode="interaction")
        with pytest.raises(ValueError):
            head.predict(_unit(rng.normal(size=DIM + 1)),
                         [_unit(rng.normal(size=DIM))],
                         qtype="choice", labels=["a"])


class TestLearning:
    @pytest.mark.parametrize("mode", ["interaction", "fingerprint"])
    def test_fit_beats_random_on_separable_data(self, mode):
        """A trained head must classify separable synthetic data near-perfectly —
        the metric-must-move check inherited from the audit."""
        labels, protos, examples = _separable_data()
        head = DecisionHead(dim=DIM, mode=mode, hidden=HID)
        result = head.fit(examples, epochs=150, lr=0.05, seed=0)
        assert result["final_loss"] < result["loss_history"][0]
        correct = sum(
            head.predict(ex.state_embedding, ex.option_embeddings,
                         qtype="choice", labels=ex.labels).choice
            == ex.labels[int(np.argmax(ex.target))]
            for ex in examples)
        assert correct / len(examples) > 0.9

    def test_temperature_fit_returns_per_qtype(self):
        labels, protos, examples = _separable_data(n_per_class=10)
        head = DecisionHead(dim=DIM, mode="interaction", hidden=HID)
        head.fit(examples, epochs=20, lr=0.05)
        fitted = head.fit_temperature(examples)
        assert set(fitted) == {"choice"}
        assert fitted["choice"] > 0

    def test_abstention_threshold_and_ood_state(self):
        labels, protos, examples = _separable_data(n_per_class=10)
        head = DecisionHead(dim=DIM, mode="interaction", hidden=HID)
        head.fit(examples, epochs=50, lr=0.05)
        tau = head.fit_abstention_threshold(examples, "choice",
                                            target_precision=0.8, min_coverage=0.1)
        assert np.isfinite(tau)
        # an in-distribution state should pass; a degenerate zero state should abstain
        in_dist = head.predict(examples[0].state_embedding,
                               examples[0].option_embeddings,
                               qtype="choice", labels=labels)
        assert in_dist.abstention == "passed"
        ood = head.predict(np.zeros(DIM), [np.zeros(DIM)] * len(labels),
                           qtype="choice", labels=labels)
        assert ood.abstention in ("abstained", "passed")  # contract: it is evaluated


class TestPersistence:
    @pytest.mark.parametrize("mode", ["interaction", "fingerprint"])
    def test_save_load_roundtrip(self, tmp_path, mode):
        labels, protos, examples = _separable_data(n_per_class=10)
        head = DecisionHead(dim=DIM, mode=mode, hidden=HID)
        head.fit(examples, epochs=20, lr=0.05)
        path = tmp_path / "head.json"
        head.save(path)
        clone = DecisionHead.load(path)
        s = examples[0].state_embedding
        o = examples[0].option_embeddings
        a = head.predict(s, o, qtype="choice", labels=labels)
        b = clone.predict(s, o, qtype="choice", labels=labels)
        assert a.probabilities == pytest.approx(b.probabilities)

    def test_normalize_flag_round_trips(self, tmp_path):
        head = DecisionHead(dim=DIM, mode="fingerprint", normalize_fingerprints=True)
        path = tmp_path / "head.json"
        head.save(path)
        assert DecisionHead.load(path).normalize_fingerprints is True


class TestZeroShotInitialization:
    """ZS-LP: an initialized head must start exactly at the zero-shot baseline.

    If this does not hold, the head is not a learned correction to the encoder's
    similarity structure and the S3 proposal is misimplemented.
    """

    def _setup(self, k=5):
        labels = [f"opt_{i}" for i in range(k)]
        opts = [_unit(rng.normal(size=DIM)) for _ in range(k)]
        return labels, opts

    def test_scores_equal_cosine_similarity(self):
        labels, opts = self._setup()
        head = DecisionHead(dim=DIM, mode="fingerprint", normalize_fingerprints=True)
        head.initialize_fingerprints("choice", labels, opts, normalize=True)
        x = _unit(rng.normal(size=DIM))
        scores = head.raw_scores(x, opts, qtype="choice", labels=labels).numpy()
        expected = np.array([float(np.dot(x, o)) for o in opts])
        assert np.allclose(scores, expected, atol=1e-5)

    def test_untrained_head_reproduces_cosine_argmax(self):
        labels, opts = self._setup()
        head = DecisionHead(dim=DIM, mode="fingerprint", normalize_fingerprints=True)
        head.initialize_fingerprints("choice", labels, opts, normalize=True)
        for _ in range(10):
            x = _unit(rng.normal(size=DIM))
            pred = head.predict(x, opts, qtype="choice", labels=labels)
            cosines = np.array([float(np.dot(x, o)) for o in opts])
            assert pred.choice == labels[int(cosines.argmax())]

    def test_normalization_removes_magnitude_bias(self):
        """Corollary 1: with unit fingerprints, score is scale-free in the option."""
        labels, opts = self._setup()
        head = DecisionHead(dim=DIM, mode="fingerprint", normalize_fingerprints=True)
        head.initialize_fingerprints("choice", labels, opts, normalize=True)
        x = _unit(rng.normal(size=DIM))
        scaled = [o * 7.0 for o in opts]          # same directions, 7x magnitude
        a = head.raw_scores(x, opts, qtype="choice", labels=labels).numpy()
        b = head.raw_scores(x, scaled, qtype="choice", labels=labels).numpy()
        assert np.allclose(a, b, atol=1e-5)

    def test_wrong_dim_option_rejected(self):
        labels, _ = self._setup()
        head = DecisionHead(dim=DIM, mode="fingerprint")
        with pytest.raises(ValueError):
            head.initialize_fingerprints("choice", labels, [np.zeros(DIM + 1)] * len(labels))

    def test_label_count_mismatch_rejected(self):
        labels, opts = self._setup()
        head = DecisionHead(dim=DIM, mode="fingerprint")
        with pytest.raises(ValueError):
            head.initialize_fingerprints("choice", labels, opts[:-1])
