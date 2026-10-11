"""TaskHead torch-free inference + npz persistence (ADR-0007).

The research recommendation (docs/research/scoring-improvement.md) promotes
TaskHead to the primary scorer. That only works if the shipped decide() path
runs without torch — so the head must export numpy weights and infer in numpy.
These tests pin that: numpy == torch, npz round-trips, load runs torch-blocked.
"""
import sys
import numpy as np
import pytest

from src.decision.scoring import TaskHead


def _synth(n=200, d=16, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, k, n)
    return X, y


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_numpy_inference_matches_torch(kind):
    X, y = _synth()
    head = TaskHead(kind=kind, hidden=32)
    head.fit(X, y, labels=["a", "b", "c"], seed=0)
    lg_torch = head.logits(X[:8])
    head._torch = None  # force the numpy path
    lg_np = head.logits(X[:8])
    assert np.allclose(lg_torch, lg_np, atol=1e-5)


@pytest.mark.parametrize("kind", ["linear", "mlp"])
def test_npz_roundtrip_and_torch_free_load(kind, tmp_path):
    X, y = _synth()
    head = TaskHead(kind=kind, hidden=32)
    head.fit(X, y, labels=["a", "b", "c"], seed=0)
    head._torch = None
    lg_before = head.logits(X[:8])
    head.save_np(tmp_path / "head.npz")

    # block torch, then load + infer — proves ADR-0007 for the head
    class _Block:
        def find_module(self, name, path=None):
            return self if name == "torch" or name.startswith("torch.") else None

        def load_module(self, name):
            raise ImportError("torch blocked")

    saved = {m: sys.modules.pop(m) for m in list(sys.modules)
             if m == "torch" or m.startswith("torch.")
             or m.startswith("src.decision")}
    sys.meta_path.insert(0, _Block())
    try:
        from src.decision.scoring import TaskHead as TH
        h2 = TH.load_np(tmp_path / "head.npz")
        lg_after = h2.logits(X[:8])
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved)
    assert np.allclose(lg_before, lg_after, atol=1e-6)
