"""Adaptation harness — labeled data in, calibrated bundle out.

One call turns a labeled dataset into a deployable ``DecisionModel``:

    texts, labels  --(validate)-->  split(fit|cal|test, leakage-safe)
        --encode-->  TaskHead.fit(fit)  ->  gate.calibrate(cal)
        ->  evaluate once on test  ->  save bundle + report

The three-way split is load-bearing: head fitting, gate calibration
(T_prob + τ on A, q̂ on B), and evaluation each see disjoint data. Sharing
them biases coverage optimistically — see AGENTS.md "three-way splits".

``time_ordered=True`` switches to contiguous splits for data collected over
time — shuffling time-correlated rows leaks future into the past.

This module is pure orchestration over encoder/head/gate; it encodes only
through the injected encoder, so it runs wherever the encoder runs (Kaggle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from src.decision.scoring import CosineScorer, TaskHead, softmax_rows, aps_members
from src.decision.gate import ConformalGate
from src.decision.model import DecisionModel


@dataclass
class AdaptConfig:
    """Everything needed to reproduce an adaptation run (stateless, R-lineage)."""
    head_kind: str = "linear"
    alpha: float = 0.10
    fit_frac: float = 0.50        # rest splits evenly into cal / test
    cal_frac: float = 0.25        # gate calibration share (gate sub-splits A/B)
    seed: int = 0
    time_ordered: bool = False    # contiguous splits for time-correlated data
    min_cal: int = 200            # gate calibration floor (ConformalGate.min_n)
    head_kwargs: Dict = field(default_factory=dict)  # epochs, lr, hidden, ...
    policy: str = "full"
    answer_precision: float = 0.90


@dataclass
class AdaptResult:
    model: DecisionModel
    report: Dict
    bundle_dir: Optional[Path] = None


def split_indices(n: int, fit_frac: float, cal_frac: float, seed: int,
                  y: Optional[np.ndarray] = None,
                  time_ordered: bool = False) -> tuple:
    """Return (fit, cal, test) index arrays, pairwise disjoint, covering [0, n).

    Stratified shuffle by ``y`` when given and not ``time_ordered``; contiguous
    thirds otherwise.
    """
    if fit_frac + cal_frac >= 1.0:
        raise ValueError("fit_frac + cal_frac must be < 1 (test split is the rest)")

    if time_ordered or y is None:
        order = np.arange(n) if time_ordered else np.random.default_rng(seed).permutation(n)
        n_fit, n_cal = int(n * fit_frac), int(n * cal_frac)
        return order[:n_fit], order[n_fit:n_fit + n_cal], order[n_fit + n_cal:]

    rng = np.random.default_rng(seed)
    fit, cal, test = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        nf, nc = int(len(idx) * fit_frac), int(len(idx) * cal_frac)
        fit += idx[:nf].tolist()
        cal += idx[nf:nf + nc].tolist()
        test += idx[nf + nc:].tolist()
    return np.asarray(fit), np.asarray(cal), np.asarray(test)


def _validate(texts: Sequence[str], labels: Sequence,
              cfg: AdaptConfig) -> tuple:
    """Return (y_idx, classes). Raises on unusable input — fail closed."""
    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) misaligned")
    if len(texts) == 0:
        raise ValueError("empty dataset")
    uniq, y = np.unique(np.asarray(labels), return_inverse=True)
    classes = [str(c) for c in uniq]
    if len(classes) < 2:
        raise ValueError("need at least 2 label classes")
    n_cal = int(len(texts) * cfg.cal_frac)
    if n_cal < cfg.min_cal:
        raise ValueError(
            f"calibration split too small: {n_cal} < min_cal={cfg.min_cal} "
            f"(n={len(texts)}, cal_frac={cfg.cal_frac}). More data or lower min_cal.")
    counts = np.bincount(y)
    if counts.min() < 5:
        raise ValueError(
            f"rarest class has {counts.min()} examples — cannot split safely")
    return y, classes


def _report_metrics(scores: np.ndarray, y: np.ndarray,
                    gate: ConformalGate) -> Dict:
    """One held-out evaluation: accuracy, resolved, coverage, set size, Brier."""
    y = np.asarray(y, dtype=int)
    pred = scores.argmax(axis=1)
    acc = float((pred == y).mean())
    P = softmax_rows(scores, gate.t_prob)
    brier = float(np.mean(np.sum(P ** 2, axis=1)
                          - 2 * P[np.arange(len(y)), y] + 1))
    P_set = softmax_rows(scores, gate.t_set)
    sets = aps_members(P_set, gate.qhat)
    covered = [int(y[i] in s) for i, s in enumerate(sets)]
    sizes = [len(s) for s in sets]
    top = P.max(axis=1)
    resolved = float((top >= gate.tau_answer).mean()) if gate.tau_answer is not None else None
    return {"n": int(len(y)), "accuracy": acc, "brier": brier,
            "resolved_at_tau": resolved,
            "conformal_coverage": float(np.mean(covered)),
            "mean_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes))}


def adapt(texts: Sequence[str], labels: Sequence, encoder,
          cfg: Optional[AdaptConfig] = None,
          out_dir=None, oos_texts: Optional[Sequence[str]] = None,
          embed_kwargs: Optional[Dict] = None) -> AdaptResult:
    """Labeled data → calibrated bundle. Encoder injected (runs on Kaggle).

    texts/labels: aligned examples. ``labels`` may be strings or ints; the
    sorted unique set becomes the question's option labels.
    encoder: a ``StateEncoder``-compatible object (encode + encode_options).
    oos_texts: optional out-of-schema examples for the τ_in_schema leg.
    out_dir: if given, ``model.save(out_dir/'bundle')`` + ``adapt_report.json``.
    """
    cfg = cfg or AdaptConfig()
    embed_kwargs = {"prompt_name": "Classification", "batch_size": 64,
                    **(embed_kwargs or {})}
    y_idx, classes = _validate(texts, labels, cfg)

    fi, ci, ti = split_indices(len(texts), cfg.fit_frac, cfg.cal_frac,
                               cfg.seed, y=y_idx, time_ordered=cfg.time_ordered)

    X = encoder.encode(list(texts), **embed_kwargs)
    X = X.detach().float().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)

    # ── headroom check: what does zero-shot cosine already resolve? (ADR-0011)
    LV = encoder.encode_options(classes)
    LV = LV.detach().float().cpu().numpy() if hasattr(LV, "detach") else np.asarray(LV)
    cos = CosineScorer()
    zs_pred = (X[ti] @ LV.T).argmax(axis=1)
    headroom = {"zeroshot_test_acc": float((zs_pred == y_idx[ti]).mean())}

    # ── fit the head on the fit split only
    head = TaskHead(kind=cfg.head_kind,
                    hidden=cfg.head_kwargs.get("hidden", 256))
    hk = {k: v for k, v in cfg.head_kwargs.items() if k != "hidden"}
    train_rec = head.fit(X[fi], y_idx[fi], labels=classes, seed=cfg.seed, **hk)

    # ── calibrate the gate on the cal split only (gate sub-splits A/B)
    gate = ConformalGate(alpha=cfg.alpha, policy=cfg.policy,
                         answer_precision=cfg.answer_precision,
                         min_n=min(cfg.min_cal, 200))
    cal = gate.calibrate(head.logits(X[ci]), y_idx[ci], seed=cfg.seed)
    head.temperature = gate.t_prob

    # ── optional in-schema threshold (needs real OOS examples)
    tau_in = None
    if oos_texts:
        Xo = encoder.encode(list(oos_texts), **embed_kwargs)
        Xo = Xo.detach().float().cpu().numpy() if hasattr(Xo, "detach") else np.asarray(Xo)
        tau_in = gate.fit_in_schema(head.logits(X[ci]), head.logits(Xo))

    model = DecisionModel(encoder=encoder, scorer=head, gate=gate)

    # ── evaluate once, on the untouched test split
    report = {
        "n_total": len(texts), "n_classes": len(classes),
        "splits": {"fit": len(fi), "cal": len(ci), "test": len(ti),
                   "time_ordered": cfg.time_ordered, "seed": cfg.seed},
        "headroom": headroom,
        "head": {"kind": cfg.head_kind, **train_rec},
        "calibration": cal,
        "tau_in_schema": tau_in,
        "test": _report_metrics(head.logits(X[ti]), y_idx[ti], gate),
        "config": {"head_kind": cfg.head_kind, "alpha": cfg.alpha,
                   "fit_frac": cfg.fit_frac, "cal_frac": cfg.cal_frac,
                   "seed": cfg.seed, "time_ordered": cfg.time_ordered,
                   "policy": cfg.policy, "answer_precision": cfg.answer_precision},
    }

    bundle_dir = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        bundle_dir = out_dir / "bundle"
        model.save(bundle_dir)
        (out_dir / "adapt_report.json").write_text(json.dumps(report, indent=2))
    return AdaptResult(model=model, report=report, bundle_dir=bundle_dir)
