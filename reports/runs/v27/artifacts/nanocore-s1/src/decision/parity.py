"""Backend parity — the measurement Phase 8 turns on.

The right primary metric is **cosine agreement between embeddings**, not argmax
agreement. Two backends can permute no decisions at all while shifting every
embedding; argmax agreement would hide that behind the head's own tolerance and
the shift would surface later as unexplained calibration drift — a gate whose
`qhat` no longer matches the scores it was fitted on.

So this measures both, and reports the embedding-level one first: a backend swap
is only safe when the vectors moved less than the decision layer's margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ParityTolerance:
    """What "close enough to swap backends" means. Both are tunable, not laws."""

    min_cosine: float = 0.99
    min_action_agreement: float = 0.95


def cosine_agreement(A: np.ndarray, B: np.ndarray) -> Dict:
    """Per-item cosine between two backends' embeddings, aligned item-for-item.

    Also reports the worst item: a mean of 0.999 hiding one vector at 0.4 is a
    real incompatibility, and the mean is exactly the statistic that would hide it.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError(f"embeddings must align: {A.shape} vs {B.shape}")
    a = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    b = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    per_item = (a * b).sum(axis=1)
    worst = int(np.argmin(per_item))
    return {"n": int(len(per_item)), "mean": float(per_item.mean()),
            "min": float(per_item.min()), "worst_index": worst,
            "p05": float(np.quantile(per_item, 0.05))}


def decision_agreement(preds_a: Sequence, preds_b: Sequence) -> Dict:
    """Action agreement and mean prediction-set overlap across two backends."""
    if len(preds_a) != len(preds_b):
        raise ValueError(f"{len(preds_a)} vs {len(preds_b)} predictions")
    same_action, jaccard = 0, []
    for pa, pb in zip(preds_a, preds_b):
        same_action += int(pa.action == pb.action)
        sa, sb = set(pa.prediction_set), set(pb.prediction_set)
        jaccard.append(len(sa & sb) / max(len(sa | sb), 1))
    n = max(len(preds_a), 1)
    return {"n": len(preds_a), "action_agreement": same_action / n,
            "mean_set_jaccard": float(np.mean(jaccard)) if jaccard else 0.0}


def latency_summary(times_ms: Sequence[float]) -> Dict:
    """Percentiles, never a mean alone — the monitor's rule, and the reason for it.

    A p99 three times the p50 is a different product from a flat distribution
    with the same mean, and only the percentiles say which one you have.
    """
    t = np.asarray(list(times_ms), dtype=np.float64)
    if t.size == 0:
        return {"n": 0}
    return {"n": int(t.size), "mean": float(t.mean()),
            "p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "max": float(t.max())}


def parity_report(cos: Dict, dec: Dict, lat_a: Optional[Dict] = None,
                  lat_b: Optional[Dict] = None,
                  tol: Optional[ParityTolerance] = None) -> Dict:
    """Verdicts over the two metrics, with the numbers that produced them."""
    tol = tol or ParityTolerance()
    cosine_ok = cos["mean"] >= tol.min_cosine
    action_ok = dec["action_agreement"] >= tol.min_action_agreement
    out = {"cosine": cos, "decisions": dec,
           "tolerance": {"min_cosine": tol.min_cosine,
                         "min_action_agreement": tol.min_action_agreement},
           "cosine_ok": bool(cosine_ok), "actions_ok": bool(action_ok),
           "parity_ok": bool(cosine_ok and action_ok)}
    if lat_a is not None or lat_b is not None:
        out["latency"] = {"reference": lat_a, "target": lat_b}
        if lat_a and lat_b and lat_a.get("p50"):
            out["latency"]["p50_ratio"] = lat_b["p50"] / lat_a["p50"]
    return out
