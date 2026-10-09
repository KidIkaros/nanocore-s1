"""Calibration and accuracy metrics for typed decisions.

Vendored from jev-stack `src/metrics.py` (same author, Apache-consistent usage)
so the decision model stays self-contained. Keep dependency-light (numpy only):
the eval harness and the head share one implementation. Every metric is defined
over a *distribution* where the target is a distribution — typed-decision
training uses the teacher's soft distributions, not hard labels, so the metrics
must accept the same shape.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def _as_2d(probs) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float64)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def group_by_length(probs, targets) -> Dict[int, List[tuple]]:
    """Bucket (prediction, target) pairs by option count.

    Questions have different cardinalities — 2 options for `noul`, 4 or 5 for `choice` and
    `score` — so a flat array cannot hold them: `np.asarray` raises "inhomogeneous shape".
    """
    groups: Dict[int, List[tuple]] = {}
    for p, t in zip(probs, targets):
        groups.setdefault(len(p), []).append((p, t))
    return groups


def weighted_metric(probs, targets, fn, **kwargs) -> float:
    """Compute ``fn`` per option-count group and aggregate weighted by decision count.

    This is the shared fix for the ragged-array problem, which appeared in three separate
    places — the benchmark notebook, the evaluation report, and temperature fitting — each
    time costing a debugging round. Anything that scores a *set* of decisions must go
    through here rather than calling a metric on the whole list.
    """
    groups = group_by_length(probs, targets)
    total = sum(len(pairs) for pairs in groups.values())
    if total == 0:
        return float("nan")
    return sum(
        float(fn(np.asarray([p for p, _ in pairs], dtype=np.float64),
                 np.asarray([t for _, t in pairs], dtype=np.float64), **kwargs)) * len(pairs)
        for pairs in groups.values()
    ) / total


def accuracy(pred_labels: Sequence, true_labels: Sequence) -> float:
    """Fraction of exact matches."""
    if len(pred_labels) == 0:
        return float("nan")
    return float(np.mean([p == t for p, t in zip(pred_labels, true_labels)]))


def soft_accuracy(pred_probs, target_probs) -> float:
    """Agreement between the predicted argmax and the target's argmax."""
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {t.shape}")
    return float(np.mean(p.argmax(axis=1) == t.argmax(axis=1)))


def brier_score(pred_probs, target_probs) -> float:
    """Multi-class Brier score: mean over samples of sum_i (p_i - o_i)^2.

    A strictly proper scoring rule — its unique minimum is the true
    distribution, which is why Laya's RLCD trains against it.
    """
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    return float(np.mean(np.sum((p - t) ** 2, axis=1)))


def log_score(pred_probs, target_probs) -> float:
    """Negative log likelihood of the target under the prediction."""
    p = np.clip(_as_2d(pred_probs), 1e-12, 1.0)
    t = _as_2d(target_probs)
    return float(-np.mean(np.sum(t * np.log(p), axis=1)))


def ranked_probability_score(pred_probs, target_probs) -> float:
    """RPS for *ordinal* questions — respects the ordering of the levels.

    Cross-entropy does not know that "critical" is further from "low" than
    "medium" is; RPS accumulates the squared difference of the cumulative
    distributions, so distance on the scale matters.
    """
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    cp = np.cumsum(p, axis=1)
    ct = np.cumsum(t, axis=1)
    # Drop the final cumulative entry: it is 1.0 for both by construction.
    return float(np.mean(np.sum((cp - ct)[:, :-1] ** 2, axis=1)))


def expected_value(probs, scale: Sequence[float] | None = None) -> float:
    """Expected scale value of a distribution over ordered levels."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    values = np.arange(len(p), dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)
    return float(np.dot(p, values))


def score_mae(pred_probs, target_probs) -> float:
    """Mean absolute error of the expected scale value."""
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    diffs = [abs(expected_value(a) - expected_value(b)) for a, b in zip(p, t)]
    return float(np.mean(diffs)) if diffs else float("nan")


def within_one_level(pred_probs, target_probs) -> float:
    """Fraction whose expected value is within one level of the target's."""
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    hits = [abs(expected_value(a) - expected_value(b)) <= 1.0 for a, b in zip(p, t)]
    return float(np.mean(hits)) if hits else float("nan")


def confidence_of(probs) -> np.ndarray:
    """``max(p)`` per sample — the quantity temperature scaling fits."""
    return _as_2d(probs).max(axis=1)


def correctness(probs, target_probs) -> np.ndarray:
    """1.0 where the predicted argmax matches the target's, else 0.0."""
    p = _as_2d(probs)
    t = _as_2d(target_probs)
    return (p.argmax(axis=1) == t.argmax(axis=1)).astype(np.float64)


def reliability_curve(probs, target_probs, n_bins: int = 10) -> Dict[str, List[float]]:
    """Bin confidences and report observed accuracy per bin.

    Returns bin centers, mean confidence, observed accuracy and counts. Empty
    bins are omitted rather than reported as zero-accuracy points.
    """
    conf = confidence_of(probs)
    correct = correctness(probs, target_probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mean_conf, acc, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        centers.append(float((lo + hi) / 2))
        mean_conf.append(float(conf[mask].mean()))
        acc.append(float(correct[mask].mean()))
        counts.append(n)
    return {
        "bin_center": centers,
        "mean_confidence": mean_conf,
        "accuracy": acc,
        "count": counts,
    }


def expected_calibration_error(probs, target_probs, n_bins: int = 10) -> float:
    """ECE: weighted mean gap between confidence and observed accuracy.

    Lower is better; 0 is perfect. Laya reports 0.081 after temperature scaling.
    """
    conf = confidence_of(probs)
    correct = correctness(probs, target_probs)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        gap = abs(float(conf[mask].mean()) - float(correct[mask].mean()))
        total += (n / len(conf)) * gap
    return float(total)


def maximum_calibration_error(probs, target_probs, n_bins: int = 10) -> float:
    """MCE: the worst per-bin gap, which ECE can average away."""
    curve = reliability_curve(probs, target_probs, n_bins=n_bins)
    if not curve["count"]:
        return float("nan")
    gaps = [abs(c - a) for c, a in zip(curve["mean_confidence"], curve["accuracy"])]
    return float(max(gaps))


def kl_divergence(pred_probs, target_probs) -> float:
    """Mean KL(target || pred)."""
    p = np.clip(_as_2d(pred_probs), 1e-12, 1.0)
    t = np.clip(_as_2d(target_probs), 1e-12, 1.0)
    return float(np.mean(np.sum(t * np.log(t / p), axis=1)))


def total_variation(pred_probs, target_probs) -> float:
    """Mean total variation distance between predicted and target distributions."""
    p = _as_2d(pred_probs)
    t = _as_2d(target_probs)
    return float(np.mean(0.5 * np.sum(np.abs(p - t), axis=1)))


def selective_accuracy(confidences, correct, threshold: float) -> Tuple[float, float]:
    """Accuracy over, and fraction of, answers whose confidence clears ``threshold``.

    This is the gate's own metric: a decision model that abstains should be
    judged on the answers it kept, not on all of them.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    mask = conf >= threshold
    if not mask.any():
        return float("nan"), 0.0
    return float(hit[mask].mean()), float(mask.mean())


# ── uncertainty on the metrics themselves ────────────────────────────────────

def bootstrap_ci(metric_fn, *arrays, resamples: int = 500,
                 confidence: float = 0.95, seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for a metric over aligned arrays.

    ``metric_fn(*sliced_arrays) -> float``. Resamples the index axis; each
    replicate calls ``metric_fn`` on the same resample of every array, so the
    CI is on *this* data's variation, not a normal approximation.

    Standard here: 500 resamples, 95% — matching the category's reference
    evaluation protocol (bootstrap CIs on every primary metric).
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must share the first dimension")
    rng = np.random.default_rng(seed)
    vals = np.empty(resamples)
    for b in range(resamples):
        idx = rng.integers(0, n, n)
        vals[b] = metric_fn(*[a[idx] for a in arrays])
    lo = (1 - confidence) / 2
    return {"point": float(metric_fn(*arrays)),
            "lo": float(np.quantile(vals, lo)),
            "hi": float(np.quantile(vals, 1 - lo)),
            "resamples": resamples, "confidence": confidence}


def risk_coverage_curve(confidences, correct) -> List[Dict[str, float]]:
    """Selective-risk vs coverage — sort by confidence, report risk at each
    coverage fraction. The full curve; the points at 50%/80% are the standard
    headline numbers.
    """
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    hit = np.asarray(correct, dtype=np.float64).reshape(-1)
    if conf.shape != hit.shape:
        raise ValueError("confidences and correct must have the same length")
    order = np.argsort(-conf)
    hit_sorted = hit[order]
    out = []
    for i in range(1, len(hit_sorted) + 1):
        out.append({"coverage": i / len(hit_sorted),
                    "risk": float(1.0 - hit_sorted[:i].mean())})
    return out


def aurc(confidences, correct) -> float:
    """Area under the risk-coverage curve — lower is better. The single
    scalar summary of selective-prediction quality."""
    curve = risk_coverage_curve(confidences, correct)
    cov = np.array([p["coverage"] for p in curve])
    risk = np.array([p["risk"] for p in curve])
    return float(np.trapezoid(risk, cov))


def selective_at_coverage(confidences, correct, coverage: float) -> float:
    """Accuracy on the top-``coverage`` fraction by confidence (e.g. 0.5, 0.8)."""
    conf = np.asarray(confidences, dtype=np.float64).reshape(-1)
    hit = np.asarray(correct, dtype=np.float64).reshape(-1)
    n = max(1, int(round(coverage * len(hit))))
    order = np.argsort(-conf)
    return float(hit[order[:n]].mean())
