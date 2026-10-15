"""Success-gate measurement for the consequence re-ranker (#18).

The #17 diagnostic named the success signature: does hard-quartile accuracy
rise / depth_slope flatten when the consequence re-ranker is active? This module
runs the served decide() path over a test set with and without the re-ranker and
compares the depth structure via diagnose_frontier.

The re-ranker fires inside DecisionModel.decide (on escalate), so the A/B must
go through decide(), not the gate directly. Both models see the same states;
only the prediction_set ordering on escalations differs. The measurable effect:
on hard cases, does the consequence-ranked first survivor resolve more often?
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.decision.frontier_diag import diagnose_frontier


def _rec_from_decisions(decisions: Sequence[Any], in_mask: Sequence[bool],
                        y_col: Sequence[int]) -> dict:
    """Build a frontier_diag rec dict from a sequence of Prediction objects.
    Mirrors policy_eval's per-example records: top1 (correct), top_prob,
    intent, for in-scope examples only."""
    rec = {"top1": [], "top_prob": [], "intent": [], "action": [],
           "covered": []}
    for i, pred in enumerate(decisions):
        if not in_mask[i]:
            continue
        argmax = int(np.argmax(list(pred.probabilities.values())))
        correct = argmax == int(y_col[i])
        covered = pred.labels[int(y_col[i])] in pred.prediction_set \
            if pred.prediction_set else False
        rec["top1"].append(bool(correct))
        rec["top_prob"].append(float(pred.answer_confidence))
        rec["intent"].append(int(y_col[i]))
        rec["action"].append(pred.action)
        rec["covered"].append(bool(covered))
    return rec


def measure_consequence_gain(model_base: Any, model_cons: Any,
                             states: Sequence[Any], question: Any,
                             in_mask: Sequence[bool], y_col: Sequence[int],
                             n_buckets: int = 4,
                             slope_thresh: float = 0.10) -> Dict[str, Any]:
    """Run both models over ``states`` and compare depth structure.

    ``model_base`` decides without the consequence re-ranker; ``model_cons``
    has it attached. Same states, same question. Returns the base and
    consequence depth_slopes and whether the re-ranker flattened the depth
    bottleneck (the #17 success signature: hard-quartile accuracy rises).
    """
    base_decisions = [model_base.decide(s, question) for s in states]
    cons_decisions = [model_cons.decide(s, question) for s in states]

    rec_base = _rec_from_decisions(base_decisions, in_mask, y_col)
    rec_cons = _rec_from_decisions(cons_decisions, in_mask, y_col)

    # frontier_diag needs a scores matrix for the volume axis; pass the base
    # model's per-example top_prob as a 1-col proxy (volume axis is secondary;
    # the depth slope is the success signal here).
    def _scores(rec):
        return np.array(rec["top_prob"]).reshape(-1, 1)

    diag_base = diagnose_frontier(rec_base, _scores(rec_base),
                                  n_buckets=n_buckets, slope_thresh=slope_thresh)
    diag_cons = diagnose_frontier(rec_cons, _scores(rec_cons),
                                  n_buckets=n_buckets, slope_thresh=slope_thresh)

    base_acc = float(np.mean(rec_base["top1"])) if rec_base["top1"] else 0.0
    cons_acc = float(np.mean(rec_cons["top1"])) if rec_cons["top1"] else 0.0
    # hard-quartile accuracy (the bottom difficulty bucket)
    def _hard_acc(diag):
        bks = diag.buckets.get("difficulty", [])
        return bks[-1]["accuracy"] if bks else float("nan")
    base_hard = _hard_acc(diag_base)
    cons_hard = _hard_acc(diag_cons)

    slope_delta = diag_base.depth_slope - diag_cons.depth_slope  # >0 = flattened
    return {
        "n": len(rec_base["top1"]),
        "base": {"accuracy": round(base_acc, 4),
                 "hard_quartile_acc": round(base_hard, 4),
                 "depth_slope": round(diag_base.depth_slope, 4)},
        "consequence": {"accuracy": round(cons_acc, 4),
                        "hard_quartile_acc": round(cons_hard, 4),
                        "depth_slope": round(diag_cons.depth_slope, 4)},
        "depth_slope_delta": round(slope_delta, 4),
        "hard_quartile_delta": round(cons_hard - base_hard, 4),
        "flattened": bool(slope_delta > 0),
        "earns_slot": bool(cons_hard >= base_hard - 1e-9
                           and slope_delta >= 0),
    }
