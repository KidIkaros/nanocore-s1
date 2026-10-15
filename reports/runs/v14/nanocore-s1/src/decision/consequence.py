"""Consequence predictor (#18) — the RWM-shaped predictive organ.

Predicts, from the composed state embedding at an escalate verdict, the
probability that the question is *recoverable* (the trace eventually resolves).
This is the consequence/ranking signal path A unlocked: the state embedding
carries recoverability information (corr ~0.333 on the hard cases) that a linear
probe extracts. It does NOT predict "answer resolves" (that's definitional) —
it predicts recoverability of the *hard* cases, which the gate cannot see.

Non-generative: a linear model on the frozen encoder's state embedding, torch-
free numpy inference (the TaskHead template, ADR-0007). Used as an **additive
re-ranker**: at an escalate verdict, rank the gate's survivor candidates by
predicted recoverability, so the loop re-proposes the most recoverable option
instead of the next arbitrary one.

Scored by Brier / AUC, not accuracy — the training set is imbalanced (most
escalations don't recover), so accuracy collapses to the majority baseline.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class ConsequenceModel:
    """A linear recoverability probe over state embeddings.

    ``w`` (d,) and ``b`` (scalar): P(recover) = sigmoid(w·state + b). Fitted by
    logistic regression (gradient descent) on escalate-step (state → resolved)
    pairs. numpy-only; ``save_np``/``load_np`` persist it torch-free.
    """
    w: np.ndarray
    b: float

    def predict_proba(self, state_emb: np.ndarray) -> np.ndarray:
        """P(recover) for one or many state embeddings. ``state_emb`` is (d,) or
        (n, d). Returns scalar or (n,) array of probabilities."""
        s = np.asarray(state_emb, dtype=np.float64)
        single = s.ndim == 1
        if single:
            s = s[None, :]
        z = s @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return float(p[0]) if single else p

    def rank_candidates(self, state_emb: np.ndarray,
                        candidate_vecs: np.ndarray) -> List[int]:
        """Rank candidate indices by predicted recoverability of re-proposing
        from ``state_emb``. Uses the state's own recoverability as the base
        score, modulated by each candidate's alignment with the state (a
        candidate closer to a recoverable region of state space ranks higher).
        Returns candidate indices, best first.

        (A full per-(state,candidate) model needs candidate-conditioned training
        pairs the corpus doesn't yet have one-to-one; v1 uses state
        recoverability × candidate-state alignment, which is the ranking signal
        the embeddings support. Upgrade path noted on #18.)"""
        base = self.predict_proba(state_emb)
        cv = np.asarray(candidate_vecs, dtype=np.float64)
        s = np.asarray(state_emb, dtype=np.float64).reshape(-1)
        s = s / max(np.linalg.norm(s), 1e-12)
        cvn = cv / np.clip(np.linalg.norm(cv, axis=1, keepdims=True), 1e-12, None)
        align = cvn @ s                      # candidate alignment with state
        score = base * (0.5 + align)         # recoverability × alignment
        return list(np.argsort(score)[::-1])

    def save_np(self, path) -> None:
        np.savez(path, w=self.w, b=np.array([self.b]))

    @classmethod
    def load_np(cls, path) -> "ConsequenceModel":
        d = np.load(path, allow_pickle=True)
        return cls(w=d["w"], b=float(d["b"][0]))


def fit_consequence(state_embs: np.ndarray, labels: np.ndarray,
                    lr: float = 0.1, epochs: int = 300, l2: float = 1e-3,
                    seed: int = 0) -> ConsequenceModel:
    """Fit the recoverability probe by logistic regression (gradient descent).

    ``state_embs`` (n, d), ``labels`` (n,) in {0,1} (1 = trace recovered).
    L2-regularized; deterministic given ``seed``. Returns a ConsequenceModel.
    """
    X = np.asarray(state_embs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        gw = X.T @ (p - y) / n + l2 * w
        gb = float((p - y).mean())
        w -= lr * gw
        b -= lr * gb
    return ConsequenceModel(w=w, b=float(b))


def brier_score(pred: np.ndarray, labels: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities). Lower is better."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def auc(pred: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (probability a random positive outscores a random
    negative). Robust to class imbalance — the right metric here."""
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank all predictions; AUC = (sum of positive ranks - n_pos(n_pos+1)/2)/(n_pos*n_neg)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
