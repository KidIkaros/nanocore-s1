"""Aleatoric vs epistemic uncertainty decomposition.

Total uncertainty (what the conformal set sizes) splits into:

- **aleatoric** — irreducible ambiguity in the input itself ("apple" the fruit
  vs the company). More data cannot fix it.
- **epistemic** — model ignorance. It shrinks with data, so a high-epistemic
  decision is exactly the case where "collect more info / escalate to gather
  context" is the right action rather than "the input is genuinely ambiguous".

The estimator needs *several plausible models* of the same data — the
ensemble recipe: fit ``k`` TaskHeads on bootstrap resamples of the fit split
(cheap: heads train in seconds on cached embeddings), then:

    total     = H(mean prob)          entropy of the consensus
    aleatoric = mean H(prob_member)   average per-member entropy
    epistemic = total − aleatoric     the BALD gap; zero = members agree

Bootstrap is the honest choice here because our heads are cheap enough to
refit — no dropout approximations, no deep-ensembles machinery.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.decision.scoring import softmax_rows


def _entropy(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-12, 1.0)
    return -(P * np.log(P)).sum(axis=-1)


def decompose(probs_per_member: np.ndarray) -> Dict[str, np.ndarray]:
    """``(m, n, k)`` member probabilities → per-example uncertainty split.

    Returns dicts of ``(n,)`` arrays: ``total``, ``aleatoric``, ``epistemic``,
    and ``epistemic_share`` (epistemic/total — high means "learnable error,
    gather more data"; low means "intrinsically ambiguous input").
    """
    P = np.asarray(probs_per_member, dtype=np.float64)
    if P.ndim != 3 or len(P) < 2:
        raise ValueError("expected (m>=2, n, k) member probabilities")
    mean_p = P.mean(axis=0)
    total = _entropy(mean_p)
    aleatoric = _entropy(P).mean(axis=0)
    epistemic = np.clip(total - aleatoric, 0.0, None)
    share = np.divide(epistemic, total, out=np.zeros_like(total),
                      where=total > 1e-9)
    return {"total": total, "aleatoric": aleatoric,
            "epistemic": epistemic, "epistemic_share": share}


def fit_bootstrap_heads(X: np.ndarray, y: np.ndarray, labels,
                        n_members: int = 5, seed: int = 0,
                        head_kind: str = "linear", **fit_kw) -> list:
    """``n_members`` TaskHeads on bootstrap resamples of the fit split."""
    from src.decision.scoring import TaskHead
    rng = np.random.default_rng(seed)
    heads = []
    for m in range(n_members):
        idx = rng.integers(0, len(X), len(X))
        h = TaskHead(kind=head_kind)
        h.fit(X[idx], y[idx], labels=labels, seed=seed + m, **fit_kw)
        heads.append(h)
    return heads


def member_probs(heads, X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """``(m, n, k)`` probabilities from a list of fitted heads."""
    return np.stack([softmax_rows(h.logits(X), temperature) for h in heads])
