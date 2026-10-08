"""Evaluation legs — the exact code kernels call, tested on synthetic data."""
import numpy as np
import pytest

from src.decision.evaluate import (dataset_suite, rigor_block,
                                   similarity_bands, tfidf_baseline)


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


def test_rigor_block_binary_adds_threshold_free():
    """Binary tasks report AUROC + AUPRC per the eval standard (§4)."""
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, n)
    logits = np.zeros((n, 2))
    logits[:, 1] = rng.standard_normal(n) + 2.0 * y
    e = np.exp(logits); P = e / e.sum(axis=1, keepdims=True)
    r = rigor_block(P, y, resamples=50)
    assert 0.9 < r["auroc"] <= 1.0 and 0.9 < r["auprc"] <= 1.0
    # multi-class blocks omit the binary-only keys
    y3 = rng.integers(0, 3, n)
    P3 = np.full((n, 3), 1 / 3)
    r3 = rigor_block(P3, y3, resamples=50)
    assert "auroc" not in r3 and "auprc" not in r3


def test_dataset_suite_end_to_end():
    """The breadth leg's exact call path: cosine/head/tfidf on one dataset."""
    rng = np.random.default_rng(2)
    k, d = 4, 24
    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def emb(y):
        v = centers[y] + 0.15 * rng.standard_normal((len(y), d))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    y_tr = np.tile(np.arange(k), 120)          # 480 fit
    y_cal = np.tile(np.arange(k), 60)          # 240 cal (≥ min_n=100)
    y_te = np.tile(np.arange(k), 40)           # 160 test
    texts_tr = [f"class{i} marker{i} sample{j}" for i in range(k) for j in range(120)]
    texts_te = [f"class{i} marker{i} eval{j}" for i in range(k) for j in range(40)]

    out = dataset_suite(emb(y_tr), y_tr, emb(y_cal), y_cal, emb(y_te), y_te,
                        texts_tr, texts_te, centers, k, resamples=50)
    blocks = out["blocks"]
    assert set(blocks) == {"cosine", "taskhead", "tfidf_lr"}
    for b in blocks.values():
        assert b["acc"]["lo"] <= b["acc"]["point"] <= b["acc"]["hi"]
    # separable data → head and cosine should both be strong
    assert blocks["taskhead"]["acc"]["point"] > 0.8
    assert blocks["cosine"]["acc"]["point"] > 0.8
    assert out["calibration"]["cosine_t"] is not None
    assert "memorization" in out
