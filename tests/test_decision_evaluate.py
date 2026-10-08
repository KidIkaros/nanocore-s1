"""Evaluation legs — the exact code kernels call, tested on synthetic data."""
import numpy as np
import pytest

from src.decision.evaluate import (rigor_block, similarity_bands,
                                   tfidf_baseline)


def test_tfidf_baseline_column_alignment():
    """The v9 bug: dataset label ids != column ids must not corrupt the frame."""
    texts = [f"tokens about intent number{chr(97+i)} marker{chr(97+i)}"
             for i in range(6) for _ in range(30)]
    cols = np.tile(np.arange(6), 30)
    P = tfidf_baseline(texts, cols, texts[:12], n_classes=6)
    assert P.shape == (12, 6)
    assert np.allclose(P.sum(axis=1), 1.0)
    # first 12 texts are class 0 (see construction)
    assert (P.argmax(axis=1) == 0).all()


def test_tfidf_baseline_missing_class():
    """A class absent from train gets uniform mass — no crash, no misalign."""
    texts = [f"intent marker{chr(97+i)} tokens" for i in [0, 1, 3] for _ in range(40)]
    cols = np.tile(np.array([0, 1, 3]), 40)
    P = tfidf_baseline(texts, cols, texts[:3], n_classes=4)
    assert P.shape == (3, 4)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_rigor_block_keys_and_sanity():
    rng = np.random.default_rng(0)
    n, k = 500, 4
    y = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k)) * 0.3
    logits[np.arange(n), y] += 3.0
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    r = rigor_block(P, y, resamples=100)
    for key in ("acc", "log", "aurc", "acc_at_50", "acc_at_80", "ece15", "brier"):
        assert key in r
    assert r["acc"]["lo"] <= r["acc"]["point"] <= r["acc"]["hi"]
    assert r["aurc"] < 0.1          # strong model → small area under risk curve
    assert 0 <= r["ece15"] <= 1


def test_similarity_bands():
    rng = np.random.default_rng(0)
    tr = rng.standard_normal((400, 32)); tr /= np.linalg.norm(tr, axis=1, keepdims=True)
    te = np.vstack([tr[:200] + 0.01 * rng.standard_normal((200, 32)),
                    rng.standard_normal((200, 32))])
    te /= np.linalg.norm(te, axis=1, keepdims=True)
    preds = np.zeros(400, dtype=int); targets = np.zeros(400, dtype=int)
    out = similarity_bands(te, tr, preds, targets)
    assert "[0.95,1.01)" in out and out["[0.95,1.01)"]["n"] >= 100
