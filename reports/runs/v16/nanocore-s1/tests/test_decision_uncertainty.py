"""Uncertainty decomposition — aleatoric vs epistemic splits."""
import numpy as np
import pytest

from src.decision.uncertainty import decompose, fit_bootstrap_heads, member_probs


def test_decompose_known_answer():
    """Members that all agree → zero epistemic; members that disagree → high."""
    k = 4
    # all 5 members identical and confident: epistemic = 0, aleatoric ~0
    p = np.tile(np.eye(k)[0][None, None, :], (5, 3, 1))
    d = decompose(p)
    assert np.all(d["epistemic"] == 0)
    assert np.all(d["aleatoric"] < 1e-9)

    # members split 2-vs-3 on two classes: high epistemic, high share
    pa = np.eye(k)[0]; pb = np.eye(k)[1]
    split = np.stack([pa, pa, pb, pb, pb])[:, None, :]
    split = np.repeat(split, 2, axis=1)  # 2 examples
    d2 = decompose(split)
    assert np.all(d2["epistemic"] > 0.5)
    assert np.all(d2["epistemic_share"] > 0.5)


def test_decompose_pure_aleatoric():
    # every member is uniform: total = log k, all of it aleatoric
    p = np.tile(np.full((1, 4), 0.25), (5, 1, 1))
    d = decompose(p)
    assert d["epistemic"][0] == pytest.approx(0.0, abs=1e-9)
    assert d["aleatoric"][0] == pytest.approx(np.log(4), abs=1e-9)


def test_bootstrap_heads_disagree_on_boundary():
    rng = np.random.default_rng(0)
    n, k, dim = 600, 3, 32
    centers = rng.standard_normal((k, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = centers[rng.integers(0, k, n)] + 0.3 * rng.standard_normal((n, dim))
    y = np.tile(np.arange(k), n // k)

    heads = fit_bootstrap_heads(X, y, labels=[f"c{i}" for i in range(k)],
                                n_members=4, epochs=15)
    P = member_probs(heads, X)
    assert P.shape == (4, n, k)
    d = decompose(P)
    # a learnable task: epistemic should be small on most points
    assert (d["epistemic_share"] < 0.5).mean() > 0.5
