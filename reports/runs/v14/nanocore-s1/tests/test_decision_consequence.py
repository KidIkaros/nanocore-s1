"""Consequence predictor (#18) — mechanism tests on synthetic separable data.

The real fit runs on the v12 corpus (state_emb → recoverable); these pin the
mechanism: logistic fit separates, Brier/AUC are sensible, npz round-trips
torch-free, ranking orders by recoverability × alignment.
"""
import numpy as np

from src.decision.consequence import (ConsequenceModel, fit_consequence,
                                      brier_score, auc)


def _separable(n=200, dim=8, seed=0):
    """Two clusters: positives at +mu, negatives at -mu (linearly separable)."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim); mu[0] = 1.5
    Xp = rng.normal(mu, 0.5, (n // 2, dim))
    Xn = rng.normal(-mu, 0.5, (n // 2, dim))
    X = np.vstack([Xp, Xn])
    y = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
    idx = rng.permutation(n)
    return X[idx], y[idx]


def test_fit_separates_separable_data():
    X, y = _separable()
    m = fit_consequence(X, y, epochs=200, seed=0)
    p = m.predict_proba(X)
    # positives should score higher than negatives on average
    assert p[y == 1].mean() > 0.7
    assert p[y == 0].mean() < 0.3
    assert auc(p, y) > 0.9
    assert brier_score(p, y) < 0.2


def test_predict_proba_single_and_batch():
    X, y = _separable(n=40, seed=1)
    m = fit_consequence(X, y, epochs=100)
    single = m.predict_proba(X[0])
    batch = m.predict_proba(X[:5])
    assert isinstance(single, float)
    assert 0.0 <= single <= 1.0
    assert batch.shape == (5,)
    assert abs(batch[0] - single) < 1e-9   # consistent single vs batch


def test_auc_and_brier_on_imbalanced():
    # heavily imbalanced (like the real escalate set): AUC still meaningful,
    # accuracy would collapse to majority baseline.
    rng = np.random.default_rng(2)
    n_pos, n_neg = 20, 180
    Xp = rng.normal(1.0, 0.5, (n_pos, 6))
    Xn = rng.normal(-1.0, 0.5, (n_neg, 6))
    X = np.vstack([Xp, Xn]); y = np.array([1.0] * n_pos + [0.0] * n_neg)
    m = fit_consequence(X, y, epochs=200, seed=0)
    p = m.predict_proba(X)
    a = auc(p, y)
    assert a > 0.8, f"AUC should be high on separable imbalanced data: {a}"
    # Brier is finite and in [0,1]
    b = brier_score(p, y)
    assert 0.0 <= b <= 1.0


def test_save_load_roundtrip():
    import tempfile, os
    X, y = _separable(n=60, seed=3)
    m = fit_consequence(X, y, epochs=100)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cons.npz")
        m.save_np(path)
        m2 = ConsequenceModel.load_np(path)
    p1 = m.predict_proba(X)
    p2 = m2.predict_proba(X)
    assert np.allclose(p1, p2, atol=1e-10)


def test_rank_candidates_orders_by_recoverability_alignment():
    # a recoverable state (high P) with candidates aligned to it ranks the
    # aligned candidate first.
    dim = 8
    w = np.zeros(dim); w[0] = 3.0
    m = ConsequenceModel(w=w, b=0.0)
    state = np.zeros(dim); state[0] = 2.0   # strongly recoverable region
    cand_aligned = state.copy()             # aligns with state
    cand_ortho = np.zeros(dim); cand_ortho[1] = 1.0   # orthogonal
    cands = np.vstack([cand_ortho, cand_aligned])
    order = m.rank_candidates(state, cands)
    assert order[0] == 1   # aligned candidate ranks first


def test_auc_nan_when_one_class():
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([1.0, 1.0, 1.0])   # all positive
    assert np.isnan(auc(p, y))
