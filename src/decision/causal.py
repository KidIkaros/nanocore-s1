"""Did the change cause it, or did the traffic?

Both models decide the *same* items, so the difference in outcome is
attributable to the model swap rather than to a different traffic mix. That
paired design is the tighter of the two the release process has: the canary leg
is randomized *between* arms, this is paired *within* item.

The scope boundary is narrow and worth stating plainly: this identifies the
effect of the model change **conditional on the observed traffic**. It says
nothing about a different traffic mix. And the labels come from the evaluation
harness — a production prediction log carries no gold, so an outcome readout is
only available where the truth is known.

Escalating, clarifying, or abstaining is safe by construction; only an
*answer* can be unsafe.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from src.decision import metrics as M
from src.decision.schema import Prediction, answer_labels  # noqa: F401  (re-export)


def unsafe_answers(preds: Sequence[Prediction], gold: Sequence[str],
                   oos_label: Optional[str] = None) -> np.ndarray:
    """``1.0`` where the model *answered* and the answer was wrong.

    An out-of-schema item (``gold == oos_label``) is unsafe to answer however
    confident the model is: no in-schema label is correct.
    """
    if len(preds) != len(gold):
        raise ValueError(f"{len(preds)} predictions vs {len(gold)} gold labels")
    out = np.zeros(len(preds))
    for i, (p, g) in enumerate(zip(preds, gold)):
        if p.action != "answer":
            continue
        if oos_label is not None and g == oos_label:
            out[i] = 1.0
        else:
            out[i] = float(max(p.probabilities, key=p.probabilities.get) != g)
    return out


def escalations(preds: Sequence[Prediction]) -> np.ndarray:
    """``1.0`` where the model escalated — the cost side of the trade."""
    return np.array([float(p.action == "escalate") for p in preds])


def paired_readout(metric: str, incumbent: np.ndarray, candidate: np.ndarray,
                   resamples: int = 500, confidence: float = 0.95,
                   seed: int = 0) -> dict:
    """Δ (candidate − incumbent) on a paired outcome, with a percentile CI.

    The resample index is shared across both arrays, so the interval is on the
    *difference* rather than on two independent means. A CI that straddles zero
    is the honest null: on this traffic the change is indistinguishable.
    """
    inc = np.asarray(incumbent, dtype=float).reshape(-1)
    cand = np.asarray(candidate, dtype=float).reshape(-1)
    if inc.shape != cand.shape:
        raise ValueError("paired outcomes must align item-for-item")
    delta = M.bootstrap_ci(lambda a, b: float(b.mean() - a.mean()), inc, cand,
                           resamples=resamples, confidence=confidence, seed=seed)
    if delta["hi"] < 0:
        verdict = "improved"
    elif delta["lo"] > 0:
        verdict = "regressed"
    else:
        verdict = "indistinguishable"
    return {"metric": metric, "n": int(inc.size), "delta": delta,
            "verdict": verdict, "direction": "lower_is_better"}
