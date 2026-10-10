"""PairScorer — cosine(u, v) with the SentenceSimilarity prompt.

The v27 jev anchor root cause: pair tasks (paws .440, stsb .215,
boolq .583) scored a *pooled* state against options, destroying the
comparison signal before the scorer ran. Item splitting (commit
c37f49a) gave per-item vectors; this is the scorer that reads them.

EG2 is an embedding model — cosine(u, v) IS its native similarity
primitive, and the ``SentenceSimilarity`` task prompt trains exactly
this pairing. The architecture never computed it. A pair question asks
"how do these two items relate?", and the answer is the cosine between
their vectors — no option vectors, no softmax over labels.

Two pair operations, both zero-shot and training-free:

- ``similarity`` — the raw cosine in [-1, 1]. For a Score question
  over a rubric, this maps to a level by the rubric's scale; for a
  Noul it becomes P(true) through a fitted-or-fixed threshold.
- ``noul_probability`` — P(the two items match) from the cosine via a
  logistic map. The intercept/slope are the pair analogue of the
  temperature: one scalar pair, fitted on held-out data, or a
  zero-knowledge default.

The gate is bypassed by construction: the out-of-schema check is a
raw-score rule with no meaning on a cosine, and the answer bar is a
probability rule a pair cosine doesn't produce. Pair decisions route
through ``noul_probability`` and a caller-supplied threshold — the
paper's own placement finding (arXiv:2609.37647) says the fixed 0.5 is
where binary P(yes) fails, so the threshold is explicit, not hidden.

No model weights run here beyond the encoder the caller supplies; the
math is numpy.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: The task prompt that trains the symmetric similarity pairing. EG2
#: exposes it (encoder.TASK_PROMPTS["sentence_similarity"]); the scalar
#: scorers never used it because they score options, not pairs.
SIMILARITY_PROMPT = "sentence_similarity"


def pair_cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine between two vectors, in [-1, 1]. The native pair operation."""
    a = np.asarray(u, dtype=np.float64).reshape(-1)
    b = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"vector dim mismatch: {a.shape} vs {b.shape}")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def noul_probability(cosine: float, slope: float = 8.0,
                     intercept: float = -2.0) -> float:
    """P(the pair matches) from a cosine, via a logistic map.

    The zero-knowledge default places the decision boundary at
    cosine = 0.25 (intercept/slope = 2/8): EG2 text-pair cosines for
    related sentences sit well above 0.25 and unrelated ones well
    below, so 0.25 is a defensible zero-shot cut. ``slope``/``intercept``
    are the pair analogue of the temperature — one scalar pair, fitted
    on held-out data when labels exist (the threshold the placement
    diagnostic says to fit explicitly).
    """
    z = slope * float(cosine) + intercept
    # numerically stable sigmoid
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    e = np.exp(z)
    return float(e / (1.0 + e))


class PairScorer:
    """Scores a question whose answer is the relation between two items.

    Wraps an encoder's per-item vectors; the caller supplies the two
    item embeddings (or the texts to encode). Zero-shot: no fitted
    parameters beyond the optional logistic map's two scalars.

    Args:
        encoder: object with ``encode(texts, prompt_name=...)`` returning
            ``(n, dim)``. The ``SentenceSimilarity`` prompt is used for
            both sides so the pairing is symmetric — the measured-best
            configuration for similarity.
        slope, intercept: logistic map parameters (see
            ``noul_probability``).
    """

    def __init__(self, encoder=None, slope: float = 8.0,
                 intercept: float = -2.0):
        self.encoder = encoder
        self.slope = float(slope)
        self.intercept = float(intercept)

    def _encode(self, text: str) -> np.ndarray:
        if self.encoder is None:
            raise ValueError("PairScorer needs an encoder to encode text")
        v = self.encoder.encode([text], prompt_name=SIMILARITY_PROMPT)
        v = np.asarray(v, dtype=np.float64)
        return v[0] if v.ndim == 2 else v.reshape(-1)

    def similarity(self, u, v) -> float:
        """Cosine between two items — vectors or texts.

        Texts are encoded with the SentenceSimilarity prompt; vectors
        are used as given (the caller pre-encoded them).
        """
        a = self._encode(u) if isinstance(u, str) else np.asarray(u, float)
        b = self._encode(v) if isinstance(v, str) else np.asarray(v, float)
        return pair_cosine(a, b)

    def noul(self, u, v) -> float:
        """P(the two items match) — the pair binary judgment."""
        return noul_probability(self.similarity(u, v),
                                self.slope, self.intercept)

    def noul_probability_map(self, cosine: float) -> float:
        """P(match) for a precomputed cosine under the current map."""
        return noul_probability(cosine, self.slope, self.intercept)

    def fit_map(self, cosines: Sequence[float], gold: Sequence[bool],
                grid: int = 41) -> "PairScorer":
        """Fit slope/intercept on held-out (cosine, label) pairs.

        Grid search over slope in [1, 20] and intercept in [-6, 2],
        maximizing log-loss — the same objective the scalar temperature
        fit uses, so the two readouts stay comparable. Returns self for
        chaining. No labels: keep the zero-knowledge default.
        """
        c = np.asarray(cosines, dtype=np.float64).reshape(-1)
        g = np.asarray(gold, dtype=float).reshape(-1)
        if len(c) == 0 or len(c) != len(g):
            raise ValueError("fit_map needs matched cosine/label arrays")
        best, best_ll = (self.slope, self.intercept), np.inf
        for slope in np.linspace(1.0, 20.0, grid):
            for intercept in np.linspace(-6.0, 2.0, grid):
                p = np.array([noul_probability(x, slope, intercept)
                              for x in c])
                ll = float(-np.mean(
                    g * np.log(np.clip(p, 1e-9, 1))
                    + (1 - g) * np.log(np.clip(1 - p, 1e-9, 1))))
                if ll < best_ll:
                    best_ll, best = ll, (float(slope), float(intercept))
        self.slope, self.intercept = best
        return self
