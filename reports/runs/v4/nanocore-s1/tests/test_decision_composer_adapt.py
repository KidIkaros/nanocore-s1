"""Composer factory (component 38) — train a composer end-to-end with a head.

Objective (b): compose -> head -> cross-entropy, so the composer learns the
composition that improves the DECISION. Identity-at-init means epoch 0 is the
mean-pool baseline. The earn-the-slot margin vs pooled is the report.

No encoder runs — synthetic item embeddings prove the control flow locally;
the real fit is Kaggle's (ADR-0006).
"""
import numpy as np
import pytest

from src.decision.composer import ComposerConfig, EmbeddingComposer
from src.decision.composer_adapt import (NON_INFERIORITY_MARGIN,
                                         ComposerAdaptConfig,
                                         ComposerAdaptResult,
                                         adapt_composer, _compose, _pool)


def _synth(n=120, D=32, classes=3, seed=0, signal="class"):
    """Synthetic multi-item states with a learnable class structure.

    Each state is a variable-length list of item vectors. For signal='class',
    every item of a class-c state is drawn near a class center, so the MEAN
    already separates — pooled should do well. For signal='order', the class
    is encoded in the item ORDER (first-vs-last position), which mean-pool
    destroys but a composer can learn.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    item_embs, y = [], []
    for i in range(n):
        c = i % classes
        T = int(rng.integers(2, 5))
        items = []
        for t in range(T):
            if signal == "class":
                v = centers[c] + 0.3 * rng.standard_normal(D)
            else:  # order: class c has its marker in position (c % T)
                v = 0.3 * rng.standard_normal(D)
                if t == c % T:
                    v = v + centers[c]
            items.append(v / np.linalg.norm(v))
        item_embs.append(np.stack(items))
        y.append(c)
    return item_embs, np.array(y)


# ── the contract ───────────────────────────────────────────────────────────

def test_returns_composer_and_report():
    ie, y = _synth(n=40)
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=32, epochs=3))
    assert isinstance(r, ComposerAdaptResult)
    assert isinstance(r.composer, EmbeddingComposer)
    assert r.classes == [0, 1, 2]
    assert "margin_test" in r.report
    assert "non_inferior" in r.report and "earns_slot" in r.report


def test_report_carries_both_baselines():
    ie, y = _synth(n=60)
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=32, epochs=3))
    assert "composed" in r.report and "pooled" in r.report
    assert r.report["composed"]["test_acc"] is not None
    assert r.report["pooled"]["test_acc"] is not None


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one"):
        adapt_composer([], [])


def test_length_mismatch_raises():
    ie, y = _synth(n=10)
    with pytest.raises(ValueError, match="same length"):
        adapt_composer(ie, y[:-1])


# ── the earn-the-slot logic ────────────────────────────────────────────────

def _synth_pure_order(n=150, D=32, classes=3, T=4, seed=1):
    """Order-ONLY classes: +marker at position c, -marker at (c+2)%T, so the
    mean is class-invariant (pooled ~= chance) but the order encodes class.
    This is the task mean-pool provably cannot solve and a composer can."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    item_embs, y = [], []
    for i in range(n):
        c = i % classes
        items = []
        for t in range(T):
            v = 0.2 * rng.standard_normal(D)
            if t == c:
                v = v + 2.0 * centers[c]
            if t == (c + 2) % T:
                v = v - 2.0 * centers[c]
            items.append(v / np.linalg.norm(v))
        item_embs.append(np.stack(items))
        y.append(c)
    return item_embs, np.array(y)


def test_earns_slot_when_composer_beats_pooled():
    """The earn-the-slot case: on order-only classes pooled is at chance and
    the trained composer clears it — margin > 0, earns_slot True."""
    ie, y = _synth_pure_order()
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=64, epochs=60,
                                                  n_layer=2, n_head=4))
    m = r.report["margin_test"]
    assert m is not None
    assert r.report["pooled"]["test_acc"] < 0.5   # pooled near chance
    assert r.report["composed"]["test_acc"] > 0.8  # composer solved it
    assert m > 0 and r.report["earns_slot"] is True
    assert r.report["non_inferior"] is True


def test_non_inferiority_floor_is_reported():
    ie, y = _synth(n=60)
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=32, epochs=3))
    m = r.report["margin_test"]
    if m is not None:
        assert r.report["non_inferior"] == (m >= -NON_INFERIORITY_MARGIN)
        assert r.report["earns_slot"] == (m > 0)


# ── the composer is real and usable ────────────────────────────────────────

def test_trained_composer_composes_variable_length_states():
    ie, y = _synth(n=40)
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=32, epochs=2))
    out = _compose(r.composer, [ie[0], ie[1]])
    assert out.shape == (2, 32)
    assert np.isfinite(out.detach().numpy()).all()


def test_pool_baseline_is_mean_pool():
    ie, _ = _synth(n=10)
    pooled = _pool(ie)
    assert pooled.shape == (10, ie[0].shape[1])
    # row 0 equals the mean of its items
    assert np.allclose(pooled[0], ie[0].mean(axis=0))


def test_identity_at_init_within_tolerance():
    """An untrained composer is mean-pool up to the modality embed + final
    norm (measured ~3e-3), not machine-zero — the spec's corrected claim."""
    c = EmbeddingComposer(ComposerConfig(n_layer=2, n_head=6, n_embd=768,
                                         embd_dim=768))
    rng = np.random.default_rng(0)
    items = [rng.standard_normal((3, 768)) for _ in range(2)]
    composed = _compose(c, items).detach().numpy()
    pooled = _pool(items)
    composed = composed / np.linalg.norm(composed, axis=1, keepdims=True)
    pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
    drift = np.abs(composed - pooled).max()
    assert drift < 1e-2, f"identity-at-init drift {drift} exceeds tolerance"


def test_trained_composer_depends_on_item_order():
    """The property mean-pool cannot express, and the reason to train a
    composer at all: AFTER training, reversing the items changes the composed
    vector. (An UNTRAINED composer is order-invariant — it IS mean-pool at
    init — so this must be measured on a trained one.)"""
    ie, y = _synth_pure_order(n=60)
    r = adapt_composer(ie, y, ComposerAdaptConfig(n_embd=64, epochs=60,
                                                  n_layer=2, n_head=4))
    a = ie[0]
    v1 = _compose(r.composer, [a]).detach().numpy()
    v2 = _compose(r.composer, [a[::-1].copy()]).detach().numpy()
    assert not np.allclose(v1, v2, atol=1e-6)


def test_untrained_composer_is_order_invariant():
    """The corrected identity-at-init claim: an untrained composer IS mean-pool,
    which is order-invariant. Order-dependence is what training buys."""
    c = EmbeddingComposer(ComposerConfig(n_layer=2, n_head=4, n_embd=64,
                                         embd_dim=32))
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2, 32))
    v1 = _compose(c, [a]).detach().numpy()
    v2 = _compose(c, [a[::-1].copy()]).detach().numpy()
    assert np.allclose(v1, v2, atol=1e-6)


def test_pool_is_order_invariant():
    """The contrast that makes the previous test meaningful: mean-pool of a
    reversed list is identical — it destroys order."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 8))
    assert np.allclose(a.mean(0), a[::-1].mean(0))
