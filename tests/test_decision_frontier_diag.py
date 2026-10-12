"""Frontier bottleneck diagnostic (#17) — mechanism tests.

Each synthetic record has a known bottleneck, so the verdict is provable
without the real head. The real head runs on Kaggle (its embeddings aren't
cached locally — the diagnostic re-ranks a leg's per-example records, no new
encoder pass).
"""
import numpy as np

from src.decision.frontier_diag import diagnose_frontier, _bucket_by


def _rec(top1, top_prob, scores):
    return {"top1": list(top1), "top_prob": list(top_prob),
            "intent": [0] * len(top1)}, np.asarray(scores, dtype=float)


def test_bucket_by_is_balanced_quantiles():
    v = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    b = _bucket_by(v, 4)
    # 12 items, 4 buckets -> 3 each
    counts = [int((b == i).sum()) for i in range(4)]
    assert counts == [3, 3, 3, 3]
    # monotonic: bucket index rises with value
    assert list(b) == sorted(b)


def test_depth_bottleneck_when_accuracy_falls_with_difficulty():
    """Half easy+correct, half hard+wrong, flat volume -> depth bottleneck."""
    n = 200
    top_prob = np.concatenate([np.full(n // 2, 0.95), np.full(n // 2, 0.30)])
    top1 = np.concatenate([np.ones(n // 2, bool), np.zeros(n // 2, bool)])
    # flat volume: every example has exactly 3 near competitors regardless
    scores = np.tile(np.array([5.0, 4.2, 4.1, 3.9, 0.0]), (n, 1))  # 3 near winner
    rec, S = _rec(top1, top_prob, scores)
    v = diagnose_frontier(rec, S, n_buckets=4, slope_thresh=0.10)
    assert v.depth_bottleneck and not v.volume_bottleneck
    assert "predictive" in v.recommendation
    assert v.depth_slope >= v.volume_slope


def test_volume_bottleneck_when_accuracy_falls_with_crowd():
    """Flat difficulty, but accuracy falls with how many classes crowd the
    winner -> volume bottleneck (typed-memory organ)."""
    n = 200
    top_prob = np.full(n, 0.90)                       # flat confidence
    top1 = np.concatenate([np.ones(n // 2, bool), np.zeros(n // 2, bool)])
    # half have a lone winner (low crowd), half have a big crowd (high volume)
    lone = np.tile(np.array([5.0, 0.0, 0.0, 0.0, 0.0]), (n // 2, 1))
    crowd = np.tile(np.array([5.0, 4.9, 4.8, 4.7, 4.6]), (n // 2, 1))
    scores = np.vstack([lone, crowd])
    rec, S = _rec(top1, top_prob, scores)
    v = diagnose_frontier(rec, S, n_buckets=4, slope_thresh=0.10)
    assert v.volume_bottleneck and not v.depth_bottleneck
    assert "typed-memory" in v.recommendation


def test_neither_when_accuracy_flat_across_both_axes():
    """Uniform accuracy across difficulty and volume -> neither (the honest
    negative result: decide-on-embeddings is the ceiling)."""
    n = 200
    rng = np.random.default_rng(0)
    top_prob = rng.uniform(0.3, 0.95, n)
    top1 = np.ones(n, bool)                            # 100% correct everywhere
    scores = rng.normal(0, 2, (n, 8))
    rec, S = _rec(top1, top_prob, scores)
    v = diagnose_frontier(rec, S, n_buckets=4, slope_thresh=0.10)
    assert v.neither
    assert "neither" in v.recommendation
    assert v.depth_slope < 0.10 and v.volume_slope < 0.10


def test_buckets_report_n_and_accuracy():
    n = 120
    top_prob = np.linspace(0.2, 0.95, n)
    top1 = np.ones(n, bool)
    scores = np.abs(np.random.default_rng(1).normal(5, 1, (n, 6)))
    rec, S = _rec(top1, top_prob, scores)
    v = diagnose_frontier(rec, S, n_buckets=4)
    for axis in ("difficulty", "volume"):
        bks = v.buckets[axis]
        assert len(bks) == 4
        assert sum(b["n"] for b in bks) == n
        for b in bks:
            assert 0.0 <= b["accuracy"] <= 1.0
