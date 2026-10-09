"""Adaptation harness — pure-logic tests (stub encoder, no weights)."""
import numpy as np
import pytest

from src.decision.adapt import AdaptConfig, adapt, split_indices, _validate


class StubEncoder:
    """Deterministic, weightless encoder. The last whitespace token seeds a
    class center; the full text adds small jitter — so texts sharing a last
    token cluster together (a learnable task, not noise)."""
    model_name = "stub"

    def __init__(self, dim=64):
        self.dim = dim

    def _vec(self, text):
        center = np.random.default_rng(
            abs(hash(text.split()[-1])) % (2**31)).standard_normal(self.dim)
        center /= np.linalg.norm(center)
        jitter = 0.05 * np.random.default_rng(
            abs(hash(text)) % (2**31)).standard_normal(self.dim)
        v = center + jitter
        return v / np.linalg.norm(v)

    def encode(self, texts, **kw):
        return np.stack([self._vec(t) for t in texts])

    def encode_options(self, texts, **kw):
        return np.stack([self._vec(t) for t in texts])


def _make_data(n=1200, k=4, seed=0):
    """k well-separated clusters — a learnable synthetic task."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, 64))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    texts, labels = [], []
    for i in range(n):
        c = i % k
        texts.append(f"example {i} of class {c}")
        labels.append(f"class_{c}")
    return texts, labels


def test_split_indices_disjoint_and_covering():
    y = np.repeat(np.arange(4), 300)
    fi, ci, ti = split_indices(1200, 0.5, 0.25, seed=0, y=y)
    allidx = np.concatenate([fi, ci, ti])
    assert len(np.unique(allidx)) == 1200
    # stratified: every class appears in every split
    for split in (fi, ci, ti):
        assert len(np.unique(y[split])) == 4


def test_split_indices_time_ordered_is_contiguous():
    fi, ci, ti = split_indices(900, 0.5, 0.25, seed=0, time_ordered=True)
    assert fi.max() < ci.min() < ci.max() < ti.min()


def test_validate_rejects_bad_input():
    cfg = AdaptConfig()
    with pytest.raises(ValueError, match="misaligned"):
        _validate(["a", "b"], ["x"], cfg)
    with pytest.raises(ValueError, match="empty"):
        _validate([], [], cfg)
    with pytest.raises(ValueError, match="2 label"):
        _validate(["a"] * 5000, ["x"] * 5000, cfg)
    with pytest.raises(ValueError, match="calibration split"):
        _validate(["t"] * 300, ["a"] * 150 + ["b"] * 150, cfg)


def test_adapt_end_to_end_separable():
    texts, labels = _make_data(1600)
    res = adapt(texts, labels, StubEncoder(),
                AdaptConfig(head_kwargs={"epochs": 30}), out_dir=None)
    r = res.report["test"]
    assert r["accuracy"] > 0.85            # learnable task must be learned
    assert 0.0 <= r["conformal_coverage"] <= 1.0
    assert res.report["calibration"]["t_prob"] is not None
    assert res.model.gate.tau_answer is not None


def test_adapt_saves_bundle(tmp_path):
    texts, labels = _make_data(1200)
    res = adapt(texts, labels, StubEncoder(),
                AdaptConfig(head_kwargs={"epochs": 15}), out_dir=tmp_path)
    assert (res.bundle_dir / "manifest.json").exists()
    assert (tmp_path / "adapt_report.json").exists()


def test_adapt_deterministic_seed():
    texts, labels = _make_data(1200)
    a = adapt(texts, labels, StubEncoder(), AdaptConfig(head_kwargs={"epochs": 10}))
    b = adapt(texts, labels, StubEncoder(), AdaptConfig(head_kwargs={"epochs": 10}))
    assert a.report["test"]["accuracy"] == b.report["test"]["accuracy"]
