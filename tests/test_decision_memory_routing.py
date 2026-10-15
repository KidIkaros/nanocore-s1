"""Type-routed memory plane — mechanism tests on small synthetic data.

The real run (emotion's 43k-row memory) belongs on Kaggle; these pin the
mechanism locally at a size that finishes in milliseconds.
"""
import numpy as np

from src.decision.memory_routing import (MemoryPlane, plain_knn,
                                         type_routed_knn, blend_with_head,
                                         _batched_topk)


def _synth(n_ref=600, n_q=120, K=8, dim=16, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    cent = rng.normal(0, 1, (K, dim))
    cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
    ref_y = rng.integers(0, K, n_ref)
    ref = cent[ref_y] + rng.normal(0, noise, (n_ref, dim))
    q_y = rng.integers(0, K, n_q)
    q = cent[q_y] + rng.normal(0, noise, (n_q, dim))
    hp = np.full((n_q, K), 0.02)
    hp[np.arange(n_q), q_y] = 0.9          # head knows the type
    hp = hp / hp.sum(1, keepdims=True)
    return ref, ref_y, q, q_y, hp, K


def test_memory_plane_normalizes():
    ref, ref_y, *_ = _synth()
    plane = MemoryPlane.build(ref, ref_y)
    norms = np.linalg.norm(plane.ref_vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.array_equal(plane.ref_y, ref_y)


def test_batched_topk_returns_sorted_top_k():
    sims = np.array([[0.1, 0.9, 0.5, 0.3], [0.8, 0.2, 0.7, 0.4]])
    top = _batched_topk(sims, 2)
    assert top.shape == (2, 2)
    assert list(top[0]) == [1, 2]           # 0.9 then 0.5
    assert list(top[1]) == [0, 2]           # 0.8 then 0.7


def test_plain_knn_beats_chance():
    ref, ref_y, q, q_y, hp, K = _synth(noise=0.2)
    plane = MemoryPlane.build(ref, ref_y)
    pred = plain_knn(plane, q, k=10, n_classes=K, batch=32)
    assert (pred == q_y).mean() > 1.0 / K * 2


def test_type_routed_uses_head_partitions():
    """When the head's top-T types are correct, type-routed retrieval beats
    plain pooled kNN — retrieval happens inside the right partition."""
    ref, ref_y, q, q_y, hp, K = _synth(noise=0.3)
    plane = MemoryPlane.build(ref, ref_y)
    plain = plain_knn(plane, q, k=10, n_classes=K, batch=32)
    routed = type_routed_knn(plane, q, hp, top_t=3, k=10, n_classes=K, batch=32)
    assert (routed == q_y).mean() >= (plain == q_y).mean()


def test_type_routed_survives_wrong_head_types():
    """When the head's top types are WRONG, type-routed retrieval degrades
    gracefully (falls back to whatever votes exist) — it must not crash or
    return garbage."""
    ref, ref_y, q, q_y, hp, K = _synth()
    # scramble the head's predictions to wrong types
    bad = np.full_like(hp, 0.02)
    bad[np.arange(len(q)), (q_y + 1) % K] = 0.9
    bad = bad / bad.sum(1, keepdims=True)
    plane = MemoryPlane.build(ref, ref_y)
    pred = type_routed_knn(plane, q, bad, top_t=3, k=10, n_classes=K, batch=32)
    assert pred.shape == (len(q),)
    assert ((pred >= 0) & (pred < K)).all()   # valid labels, no crash


def test_blend_keeps_confident_head():
    """blend_with_head is additive: a confident head answer is never overridden."""
    head_pred = np.array([0, 1, 2])
    knn_pred = np.array([2, 2, 2])
    hp = np.array([[0.9, 0.05, 0.05],      # confident -> keep head
                   [0.4, 0.35, 0.25],      # unsure -> take kNN
                   [0.34, 0.33, 0.33]])    # unsure -> take kNN
    out = blend_with_head(knn_pred, head_pred, hp, trust=0.5)
    assert out[0] == 0                      # confident head kept
    assert out[1] == 2 and out[2] == 2      # unsure -> kNN


def test_batched_matches_unbatched():
    """Batching must not change results — memory-bounding is transparent."""
    ref, ref_y, q, q_y, hp, K = _synth(n_ref=400, n_q=60, seed=3)
    plane = MemoryPlane.build(ref, ref_y)
    a = plain_knn(plane, q, k=5, n_classes=K, batch=60)     # one batch
    b = plain_knn(plane, q, k=5, n_classes=K, batch=7)      # many batches
    assert np.array_equal(a, b)


def test_run_memory_frontier_reports_all_variants():
    """The experiment orchestrator reports head/plain/type-routed/blend overall
    and on the hard quartile — the frontier comparison."""
    from src.decision.memory_routing import run_memory_frontier
    ref, ref_y, q, q_y, hp, K = _synth(n_ref=600, n_q=120, noise=0.25)
    # synthesize head logits from the head probs
    logits = np.log(np.clip(hp, 1e-9, None))
    r = run_memory_frontier(ref, ref_y, q, q_y, logits, temperature=1.0,
                            n_classes=K, top_t=3, k=10, batch=32)
    assert r["n"] == 120
    for variant in ("head", "plain_knn", "type_routed", "blend"):
        assert variant in r
        assert "overall" in r[variant] and "hard_quartile" in r[variant]
        assert "depth_slope" in r[variant]
    assert r["n_hard"] > 0


def test_run_memory_frontier_detects_type_routing_gain():
    """When the head knows the type and the memory is informative, type-routed
    retrieval should beat plain pooled kNN on overall accuracy."""
    from src.decision.memory_routing import run_memory_frontier
    ref, ref_y, q, q_y, hp, K = _synth(n_ref=1200, n_q=200, noise=0.3, seed=7)
    logits = np.log(np.clip(hp, 1e-9, None))
    r = run_memory_frontier(ref, ref_y, q, q_y, logits, temperature=1.0,
                            n_classes=K, top_t=3, k=10, batch=32)
    assert r["type_routed"]["overall"] >= r["plain_knn"]["overall"]
    assert r["type_routed"]["depth_slope"] <= r["plain_knn"]["depth_slope"] + 0.2
