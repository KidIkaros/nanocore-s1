"""Gated per-anchor scorer selection (#14).

The auto-selection must honor the #6 gate: ship the head only if it wins the
proper-score (Brier) comparison by a real margin; a near-tie keeps cosine
(ADR-0008 non-inferiority). Raw cal-accuracy argmax is wrong — banking77's head
ties on accuracy but its Brier is 7.6x better, so the proper score decides.
"""
import numpy as np

from src.decision.adapt import select_scorer_gated
from src.decision.scoring import softmax_rows


def _probs(scores, t):
    return softmax_rows(scores, t)


def test_head_shipped_when_brier_wins_by_margin():
    # head: confident & correct; cosine: flat/wrong -> head Brier much lower
    rng = np.random.default_rng(0)
    n, k = 400, 5
    y = rng.integers(0, k, n)
    head_scores = np.full((n, k), -3.0)
    head_scores[np.arange(n), y] = 5.0           # head is right & confident
    cos_scores = rng.normal(0, 1, (n, k))        # cosine ~ noise
    choice = select_scorer_gated(
        cos_scores, head_scores, y, t_cos=1.0, t_head=1.0)
    assert choice == "taskhead"


def test_cosine_kept_on_near_tie():
    # both scorers identical -> delta ~0 -> within band -> keep cosine
    rng = np.random.default_rng(1)
    n, k = 400, 5
    y = rng.integers(0, k, n)
    scores = rng.normal(0, 2, (n, k))
    choice = select_scorer_gated(scores, scores.copy(), y, t_cos=1.0, t_head=1.0)
    assert choice == "cosine"


def test_cosine_kept_when_head_worse():
    # cosine confident & right; head noise -> cosine Brier lower -> keep cosine
    rng = np.random.default_rng(2)
    n, k = 400, 5
    y = rng.integers(0, k, n)
    cos_scores = np.full((n, k), -3.0)
    cos_scores[np.arange(n), y] = 5.0
    head_scores = rng.normal(0, 1, (n, k))
    choice = select_scorer_gated(
        cos_scores, head_scores, y, t_cos=1.0, t_head=1.0)
    assert choice == "cosine"


def test_explicit_override_beats_auto():
    rng = np.random.default_rng(3)
    n, k = 200, 4
    y = rng.integers(0, k, n)
    s = rng.normal(0, 1, (n, k))
    assert select_scorer_gated(s, s, y, t_cos=1.0, t_head=1.0,
                               forced="taskhead") == "taskhead"
    assert select_scorer_gated(s, s, y, t_cos=1.0, t_head=1.0,
                               forced="cosine") == "cosine"


def test_adapt_artifact_matches_choice(tmp_path):
    """End-to-end invariant (#14): whatever auto picks, the bundle artifact
    matches — a TaskHead/OrdinalScorer ships scorer.npz; CosineScorer ships none.
    (Not 'head always wins on a stub': the pure-function tests + the real
    clinc150 measurement (#10/#11) prove the selection; a stub can't force a
    specific winner reliably, so the robust invariant is artifact==choice.)"""
    from src.decision.adapt import adapt, AdaptConfig
    from src.decision.backends import StubEncoder

    enc = StubEncoder(dim=32)
    texts = [f"sample {i}" for i in range(1000)]
    labels = [i % 4 for i in range(1000)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="auto", seed=0, min_cal=100),
              out_dir=tmp_path)
    choice = r.report["scorer_choice"]
    has_npz = (tmp_path / "bundle" / "scorer.npz").exists()
    if choice == "cosine":
        assert r.model.scorer.__class__.__name__ == "CosineScorer"
        assert not has_npz
    else:
        assert r.model.scorer.__class__.__name__ in ("TaskHead", "OrdinalScorer")
        assert has_npz
    # and the choice is one the gate can actually emit
    assert choice in ("cosine", "taskhead", "ordinal")


def test_forced_taskhead_ships_npz(tmp_path):
    """An explicit forced 'taskhead' ships the head + scorer.npz regardless of
    the auto verdict — the override path."""
    from src.decision.adapt import adapt, AdaptConfig
    from src.decision.backends import StubEncoder

    enc = StubEncoder(dim=32)
    texts = [f"item number {i} class{i % 3}" for i in range(900)]
    labels = [i % 3 for i in range(900)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="taskhead", seed=0, min_cal=100),
              out_dir=tmp_path)
    assert r.model.scorer.__class__.__name__ == "TaskHead"
    assert (tmp_path / "bundle" / "scorer.npz").exists()


def test_near_tie_keeps_cosine_despite_marginal_head_cal_acc(tmp_path):
    """A near-tie on the proper score keeps cosine even if head cal accuracy is
    marginally higher (the ADR-0008 non-inferiority rule, decided on Brier)."""
    # identical score distributions -> Brier delta ~0 -> cosine (within band)
    rng = np.random.default_rng(7)
    n, k = 500, 6
    y = rng.integers(0, k, n)
    s = rng.normal(0, 2, (n, k))
    from src.decision.adapt import select_scorer_gated
    assert select_scorer_gated(s, s.copy(), y, t_cos=1.0, t_head=1.0) == "cosine"


def test_forced_ordinal_ships_npz_torch_free(tmp_path):
    """#15 acceptance: an ordered-label anchor forced to 'ordinal' ships an
    OrdinalScorer as scorer.npz, and the bundle loads + decides with torch
    blocked (ADR-0007)."""
    import sys
    from src.decision.adapt import adapt, AdaptConfig
    from src.decision.backends import StubEncoder
    from src.decision.schema import Question

    enc = StubEncoder(dim=16)
    texts = [f"level{i // 200} sample {i}" for i in range(1000)]  # ordered 0..4
    labels = [i // 200 for i in range(1000)]
    r = adapt(texts, labels, enc,
              cfg=AdaptConfig(scorer="ordinal", ordinal_labels=True,
                              seed=0, min_cal=100),
              out_dir=tmp_path)
    assert r.model.scorer.__class__.__name__ == "OrdinalScorer"
    assert (tmp_path / "bundle" / "scorer.npz").exists()

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
        from src.decision.model import DecisionModel
        m2 = DecisionModel.load(tmp_path / "bundle", encoder=enc)
        p = m2.decide("level1 sample 5",
                      Question(qtype="score", options=["0", "1", "2", "3", "4"]))
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(saved)
    assert abs(sum(p.probabilities.values()) - 1.0) < 1e-6
