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


def test_bundle_ships_npz_and_loads_torch_free(tmp_path):
    """Increment 2: a DecisionModel with a TaskHead scorer saves scorer.npz
    (no scorer.pt), and load() rebuilds a torch-free model with identical
    decisions — the ADR-0007 bundle guarantee for the promoted scorer."""
    from src.decision.model import DecisionModel
    from src.decision.backends import StubEncoder
    from src.decision.schema import Question

    enc = StubEncoder(dim=16)
    texts = [f"class {i % 3} sample {i}" for i in range(200)]
    labels = [i % 3 for i in range(200)]
    E = enc.encode(texts)
    head = TaskHead(kind="linear")
    head.fit(E, labels, labels=["0", "1", "2"], seed=0)
    m = DecisionModel(encoder=enc, scorer=head, gate=None, cache=None)
    q = Question(qtype="choice", options=["0", "1", "2"])
    p1 = m.decide("class 1 sample 5", q)

    m.save(tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert "scorer.npz" in files and "scorer.pt" not in files

    class _Block:
        def find_module(self, name, path=None):
            return self if name == "torch" or name.startswith("torch.") else None

        def load_module(self, name):
            raise ImportError("torch blocked")

    saved = {mm: sys.modules.pop(mm) for mm in list(sys.modules)
             if mm == "torch" or mm.startswith("torch.")
             or mm.startswith("src.decision")}
    sys.meta_path.insert(0, _Block())
    try:
        from src.decision.model import DecisionModel as DM
        m2 = DM.load(tmp_path, encoder=enc)
        p2 = m2.decide("class 1 sample 5", q)
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved)
    assert p1.probabilities == p2.probabilities


def test_auto_scorer_picks_head_when_head_wins(tmp_path):
    """Increment 3 (ADR-0008 + research): 'auto' measures head vs cosine on the
    cal split and ships the winner. On a head-favorable task it ships TaskHead
    and a scorer.npz bundle."""
    from src.decision.adapt import adapt, AdaptConfig
    from src.decision.backends import StubEncoder

    enc = StubEncoder(dim=32)
    texts = [f"sample {i}" for i in range(1000)]
    labels = [i % 4 for i in range(1000)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="auto", seed=0, min_cal=100),
              out_dir=tmp_path)
    assert r.report["scorer_choice"] == "taskhead"
    assert r.model.scorer.__class__.__name__ == "TaskHead"
    assert (tmp_path / "bundle" / "scorer.npz").exists()


def test_auto_scorer_choice_matches_cal_argmax(tmp_path):
    """The auto mechanism (ADR-0008): auto's shipped scorer is whichever of
    head/cosine wins on the cal split. We assert the LOGIC (choice == cal
    argmax) rather than a fragile stub outcome, and that the shipped scorer
    and bundle artifact follow the choice."""
    from src.decision.adapt import adapt, AdaptConfig, split_indices
    from src.decision.backends import StubEncoder
    from src.decision.scoring import TaskHead
    import numpy as np

    enc = StubEncoder(dim=32)
    texts = [f"topic{i % 3} text sample {i}" for i in range(900)]
    labels = [i % 3 for i in range(900)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="auto", seed=0, min_cal=100),
              out_dir=tmp_path)

    # recompute the cal argmax independently
    X = enc.encode(texts)
    y_idx = np.array(labels)
    fi, ci, ti = split_indices(900, 0.5, 0.25, 0, y=y_idx)
    classes = sorted(set(labels))
    LV = enc.encode_options(classes)
    cos_acc = float(((X[ci] @ LV.T).argmax(1) == y_idx[ci]).mean())
    h = TaskHead()
    h.fit(X[fi], y_idx[fi], labels=classes, seed=0)
    head_acc = float((h.logits(X[ci]).argmax(1) == y_idx[ci]).mean())
    expected = "taskhead" if head_acc >= cos_acc else "cosine"

    assert r.report["scorer_choice"] == expected
    if expected == "taskhead":
        assert r.model.scorer.__class__.__name__ == "TaskHead"
        assert (tmp_path / "bundle" / "scorer.npz").exists()
    else:
        assert r.model.scorer.__class__.__name__ == "CosineScorer"
        assert not (tmp_path / "bundle" / "scorer.npz").exists()


def test_explicit_scorer_overrides_auto(tmp_path):
    """An explicit 'taskhead'/'cosine' forces the choice regardless of cal."""
    from src.decision.adapt import adapt, AdaptConfig
    from src.decision.backends import StubEncoder

    enc = StubEncoder(dim=32)
    texts = [f"topic{i % 3} text sample {i}" for i in range(900)]
    labels = [i % 3 for i in range(900)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="taskhead", seed=0, min_cal=100),
              out_dir=tmp_path)
    assert r.model.scorer.__class__.__name__ == "TaskHead"
    assert (tmp_path / "bundle" / "scorer.npz").exists()
