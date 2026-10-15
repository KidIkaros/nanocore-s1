"""The shared evaluation protocol — one implementation, used by every experiment.

Why this module exists
----------------------
Five one-off kernels each reimplemented `softmax`, `fit_tau`, `knn_class_scores` and
`selective_curve`, and used three different verdict rules. That produced **two wrong
published conclusions**:

1. A temperature grid whose lower bound three baselines fitted, which made cosine
   scoring look catastrophically miscalibrated (log 3.800) when a properly fitted
   temperature gives 0.380. The grid was the bug, not the model.
2. A verdict that compared heads against the *weakest* baseline (`cosine_tau`) and
   therefore reported a Brier win that does not exist against the strongest.

Both are fixed structurally here rather than by being careful:

- ``fit_temperature`` searches a wide range at fine resolution in two stages, so it
  cannot bound-limit the way a single narrow grid did.
- ``compare_to_best`` requires the caller to name the baseline set and compares each
  candidate against the **best** baseline per metric, with an explicit minimum effect.
- ``coverage_at_precision`` refuses to report a number when the requested precision is
  below the task's base accuracy, where the metric saturates and means nothing.

Numpy only — no torch — so this is the one part of the package whose tests are cheap.
See docs/RESEARCH-NOTES.md (R1-R9) and docs/ARCHITECTURE-DECISION-MODEL.md section 0.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from src.decision import metrics


# ── temperature ──────────────────────────────────────────────────────────────

def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with temperature. ``temperature`` must be positive."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    s = np.asarray(scores, dtype=np.float64) / float(temperature)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, targets: np.ndarray,
                    coarse: Optional[Sequence[float]] = None,
                    refine: int = 41) -> float:
    """Fit one temperature by log score, in two stages, without bound-limiting.

    Stage 1 sweeps ``10^-6 … 10^2`` coarsely to locate the basin. Stage 2 re-sweeps a
    decade either side of the winner at fine resolution, so the result is not pinned to a
    grid edge.

    **History.** The first version swept ``10^-0.3 … 10^1.5`` and three baselines fitted
    exactly the lower bound, invalidating their reported log scores — cosine looked
    catastrophically miscalibrated (3.800) when a proper fit gives 0.380. The second
    version widened the coarse grid but refined over ``[t/3, 3t]``, so when the coarse
    winner was the coarse minimum the *refinement's* lower bound could still bind; the
    dispatch run fitted 0.000333 — the refinement edge — for five option-count groups.
    Both stages are now wide enough that an edge hit would require an optimum below
    ``10^-7``, and callers should still inspect the fitted values for clustering at a
    boundary.

    Args:
        scores: ``(n, k)`` raw scores.
        targets: ``(n, k)`` target distributions (one-hot is fine).
        coarse: Optional coarse grid; defaults to 65 log-spaced points over 10^-6 … 10^2.
        refine: Number of points in the refinement sweep.

    Returns:
        The fitted temperature as a float.
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if scores.shape != targets.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {targets.shape}")

    if coarse is None:
        coarse = [10.0 ** e for e in np.linspace(-6.0, 2.0, 65)]

    best_t, best_v = 1.0, float("inf")
    for t in coarse:
        v = metrics.log_score(softmax(scores, t), targets)
        if v < best_v:
            best_t, best_v = float(t), v

    lo, hi = best_t / 10.0, best_t * 10.0
    for t in np.linspace(lo, hi, refine):
        if t <= 0:
            continue
        v = metrics.log_score(softmax(scores, t), targets)
        if v < best_v:
            best_t, best_v = float(t), v
    return best_t


# ── metrics ──────────────────────────────────────────────────────────────────

def metric_block(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """The standard metric block. ECE is reported, never gated (ADR-0004)."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probs.shape != targets.shape:
        raise ValueError(f"shape mismatch: {probs.shape} vs {targets.shape}")
    return {
        "accuracy": float((probs.argmax(axis=1) == targets.argmax(axis=1)).mean()),
        "log_score": float(metrics.log_score(probs, targets)),
        "brier": float(metrics.brier_score(probs, targets)),
        "ece_report_only": float(metrics.expected_calibration_error(probs, targets)),
        "n": int(len(probs)),
    }


def ragged_metric_block(scores_list, targets_list,
                        min_group: int = 2) -> Dict[str, object]:
    """Metric block for decisions with **variable option counts**.

    Dispatch decisions have 2–37 candidate tools, so there is no rectangular ``(n, k)``
    array to hand to :func:`metric_block`. Groups by option count, fits one temperature
    per group by log score (via :func:`fit_temperature`), and aggregates weighted by
    group size. Groups smaller than ``min_group`` still contribute, but their fitted
    temperature is unreliable and that is reported.

    Args:
        scores_list: Sequence of 1-D raw score arrays, one per decision.
        targets_list: Sequence of 1-D target distributions of matching length.
    """
    if len(scores_list) != len(targets_list):
        raise ValueError("scores_list and targets_list must be the same length")
    if not len(scores_list):
        raise ValueError("no decisions supplied")

    groups: Dict[int, list] = {}
    for sc, tg in zip(scores_list, targets_list):
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        tg = np.asarray(tg, dtype=np.float64).reshape(-1)
        if sc.shape != tg.shape:
            raise ValueError(f"per-decision shape mismatch: {sc.shape} vs {tg.shape}")
        groups.setdefault(len(sc), []).append((sc, tg))

    keys = ("accuracy", "log_score", "brier", "ece_report_only")
    totals = {k: 0.0 for k in keys}
    temps, n_total = {}, 0
    for k, rows in sorted(groups.items()):
        sc = np.stack([r[0] for r in rows])
        tg = np.stack([r[1] for r in rows])
        t = fit_temperature(sc, tg)
        temps[k] = float(t)
        block = metric_block(softmax(sc, t), tg)
        for key in keys:
            totals[key] += block[key] * len(rows)
        n_total += len(rows)

    out: Dict[str, object] = {k: totals[k] / n_total for k in keys}
    out["n"] = n_total
    out["temperatures_by_cardinality"] = temps
    out["small_groups"] = sorted(k for k, rows in groups.items() if len(rows) < min_group)
    return out


def selective_curve(probs: np.ndarray, targets: np.ndarray,
                    steps: int = 10) -> list:
    """Selective accuracy as a function of coverage, ranked by confidence."""
    probs = np.asarray(probs, dtype=np.float64)
    correct = (probs.argmax(axis=1) == np.asarray(targets).argmax(axis=1)).astype(float)
    order = np.argsort(-probs.max(axis=1))
    out = []
    for c in np.linspace(0.1, 1.0, steps):
        n = max(1, int(round(c * len(correct))))
        out.append({"coverage": round(float(c), 2),
                    "selective_accuracy": float(correct[order[:n]].mean())})
    return out


def coverage_at_precision(probs: np.ndarray, targets: np.ndarray,
                          precision: float) -> Dict[str, object]:
    """Coverage achievable while keeping ``precision`` on the answers retained.

    **Refuses to report a number when the target is below the base accuracy**, because
    then answering everything satisfies it and the metric carries no information. The
    calibration run reported ``coverage_at_90pct == 1.0`` for every candidate on a task
    with 93% base accuracy, and a verdict was drawn from that artifact.
    """
    probs = np.asarray(probs, dtype=np.float64)
    correct = (probs.argmax(axis=1) == np.asarray(targets).argmax(axis=1)).astype(float)
    base = float(correct.mean())
    if precision <= base:
        return {"precision": precision, "coverage": None, "base_accuracy": base,
                "saturated": True,
                "note": "target precision is at or below base accuracy — metric is uninformative"}
    order = np.argsort(-probs.max(axis=1))
    best = 0.0
    for n in range(1, len(correct) + 1):
        if correct[order[:n]].mean() >= precision:
            best = n / len(correct)
    return {"precision": precision, "coverage": float(best), "base_accuracy": base,
            "saturated": False, "note": ""}


# ── headroom (R1) ────────────────────────────────────────────────────────────

def headroom_check(zero_shot_accuracy: float, min_headroom: float = 0.15) -> Dict[str, object]:
    """Can this benchmark discriminate between heads at all?

    Banking77 with EmbeddingGemma 2 reaches 92.9% zero-shot, leaving 7.1% headroom —
    so every architectural difference we chased lived inside that residual, and a null
    result was uninformative rather than evidence. Measure zero-shot *before* choosing
    a benchmark.
    """
    headroom = 1.0 - float(zero_shot_accuracy)
    return {"zero_shot_accuracy": float(zero_shot_accuracy), "headroom": headroom,
            "min_headroom": min_headroom, "usable": bool(headroom >= min_headroom),
            "note": "" if headroom >= min_headroom else
                    "insufficient headroom — a better head cannot show a better number"}


# ── dispatch metrics (the target task; R9) ───────────────────────────────────

def escalation_summary(confidences: np.ndarray, correct: np.ndarray,
                       quality_target: float,
                       cheap_cost: float = 1.0, strong_cost: float = 100.0) -> Dict[str, float]:
    """Cost saved at fixed quality — the dispatch headline.

    The model auto-handles the most confident fraction and escalates the rest. Reported
    the way the routing literature does: the fraction escalated, the accuracy actually
    achieved among auto-handled items, and the resulting cost per query relative to
    always calling the strong model.

    Args:
        confidences: ``(n,)`` model confidence per request.
        correct: ``(n,)`` 1.0 if the cheap path succeeded for that request.
        quality_target: required accuracy among auto-handled requests.
        cheap_cost: relative cost of the cheap path.
        strong_cost: relative cost of escalation.
    """
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same length")
    if not (0.0 < quality_target <= 1.0):
        raise ValueError(f"quality_target must be in (0, 1], got {quality_target}")

    order = np.argsort(-confidences)
    best = None
    for n in range(1, len(correct) + 1):
        kept = order[:n]
        acc = float(correct[kept].mean())
        if acc >= quality_target:
            best = n
    if best is None:
        return {"quality_target": quality_target, "auto_handled": 0.0, "escalated": 1.0,
                "achieved_accuracy": None, "cost_per_query": float(strong_cost),
                "saved_vs_always_strong": 0.0, "feasible": False}

    coverage = best / len(correct)
    acc = float(correct[order[:best]].mean())
    cost = coverage * cheap_cost + (1.0 - coverage) * strong_cost
    return {"quality_target": quality_target, "auto_handled": float(coverage),
            "escalated": float(1.0 - coverage), "achieved_accuracy": acc,
            "cost_per_query": float(cost),
            "saved_vs_always_strong": float(1.0 - cost / strong_cost),
            "feasible": True}


# ── verdicts ─────────────────────────────────────────────────────────────────

LOWER_IS_BETTER = ("log_score", "brier", "ece_report_only")
HIGHER_IS_BETTER = ("accuracy",)

#: Metrics compared and reported but never used to declare a winner — ADR-0004 gates on
#: strictly proper scores, and ECE is minimised by sharpening rather than by being right.
REPORT_ONLY = ("ece_report_only",)


def compare_to_best(candidates: Dict[str, Dict[str, float]],
                    baselines: Iterable[str],
                    min_effect: float = 0.0) -> Dict[str, object]:
    """Compare candidates against the **best** baseline per metric, not a convenient one.

    An earlier verdict hard-coded ``cosine_tau`` as "the" baseline, which was the weakest
    of four, and reported a Brier win that vanishes against kNN-5. Naming the baseline set
    explicitly, and taking the best member per metric, removes that failure mode.

    Args:
        candidates: ``{name: metric_block}`` for every method, including baselines.
        baselines: names that constitute the baseline set.
        min_effect: a delta must exceed this to count as a win (use the measured seed
            noise, so a difference inside noise cannot be reported as a result).

    Returns:
        ``{"best_baseline": {...}, "per_metric": {...}, "winners": {...}}``
    """
    candidates = {k: dict(v) for k, v in candidates.items()}
    baselines = [b for b in baselines if b in candidates]
    if not baselines:
        raise ValueError("no baseline names matched the candidate set")
    if not min_effect >= 0:
        raise ValueError("min_effect must be >= 0")

    per_metric: Dict[str, Dict[str, object]] = {}
    for metric, direction in [(m, "lower") for m in LOWER_IS_BETTER] + \
                             [(m, "higher") for m in HIGHER_IS_BETTER]:
        present = [b for b in baselines if metric in candidates[b]]
        if not present:
            continue
        pick = min if direction == "lower" else max
        best_name = pick(present, key=lambda b: candidates[b][metric])
        best_value = float(candidates[best_name][metric])
        rows = {}
        for name, block in candidates.items():
            if metric not in block:
                continue
            value = float(block[metric])
            delta = (best_value - value) if direction == "lower" else (value - best_value)
            rows[name] = {"value": value, "delta_vs_best_baseline": float(delta),
                          "beats_best_baseline": bool(delta > min_effect),
                          "is_baseline": name in baselines}
        per_metric[metric] = {"direction": direction, "best_baseline": best_name,
                              "best_baseline_value": best_value, "rows": rows}

    winners = {m: [n for n, r in per_metric[m]["rows"].items()
                   if r["beats_best_baseline"] and not r["is_baseline"]]
               for m in per_metric if m not in REPORT_ONLY}
    return {"baselines": baselines, "min_effect": float(min_effect),
            "per_metric": per_metric, "winners": winners}
