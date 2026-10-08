"""Local dry run: cost-calibrated escalation vs the frozen gate.

Real cached CLINC150 arrays, no encoding (allowed by the compute boundary). The
shift needs no construction: CLINC150 ships genuine out-of-scope utterances, so
phase 1 is in-scope traffic and phase 2 is out-of-scope traffic arriving after
it. Every phase-2 item is unanswerable, so the metric that matters is the
unsafe-answer rate.

The comparison is at a **matched escalation rate** — each policy's threshold is
swept and the risk-coverage curve is read at a common handoff budget. Comparing
raw unsafe rates would reward whichever policy abstains most.

Usage:
    python scripts/escalation_dryrun.py [--npz PATH] [--out DIR] [--per-phase N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_ab_local import (calibrate_base, label_mapping, to_columns,
                             verify_alignment)

from src.decision.escalation import (Costs, DecisionFeatures,
                                     calibrated_actions, decision_features,
                                     evaluate_policy, select_error_model,
                                     unsafe_at_escalation)

DEFAULT_NPZ = "reports/runs/s1_verify/v13/verify_scores.npz"
MATCHED_RATES = (0.20, 0.30, 0.40)
COST_RATIOS = (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0)


def load_arrays(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def calibration_rows(arrays: dict, oos: int, columns: dict) -> tuple:
    """Every val row, with out-of-scope rows labelled wrong by definition."""
    cols = to_columns(arrays["y_val"], columns)
    wrong = (cols < 0) | (arrays["scores_val"].argmax(axis=1) != np.maximum(cols, 0))
    return decision_features(arrays["scores_val"], 1.0, 0.0), wrong


def fit_model(arrays: dict, oos: int, columns: dict) -> tuple:
    """Fit and select the risk model on disjoint halves of the val split."""
    features, wrong = calibration_rows(arrays, oos, columns)
    half = len(wrong) // 2
    model, ranking = select_error_model(
        _slice(features, slice(0, half)), wrong[:half],
        _slice(features, slice(half, None)), wrong[half:])
    return model, ranking


def _slice(features, span: slice) -> DecisionFeatures:
    return DecisionFeatures(features.names, features.values[span])


def build_stream(arrays: dict, oos: int, columns: dict, per_phase: int,
                 rng) -> tuple:
    """In-scope traffic, then genuine out-of-scope traffic."""
    y = arrays["y_test"]
    in_scope = np.flatnonzero(y != oos)
    outside = np.flatnonzero(y == oos)
    picked = np.concatenate([rng.choice(in_scope, per_phase, replace=False),
                             rng.choice(outside, per_phase, replace=False)])
    scores = arrays["scores_test"][picked]
    cols = to_columns(y[picked], columns)
    wrong = (cols < 0) | (scores.argmax(axis=1) != np.maximum(cols, 0))
    return scores, wrong


def curve_point(actions: np.ndarray, wrong: np.ndarray) -> tuple:
    """(handoff rate, unsafe rate among answered).

    Handoff is escalate ∪ clarify: both take the decision away from the model,
    so both avoid an unsafe answer. Matching on escalation alone would rig the
    comparison in favour of whichever policy happens to prefer clarify.
    """
    answered = actions == "answer"
    kept = max(answered.mean(), 1e-9)
    return (float(1 - answered.mean()),
            float((answered & wrong).mean() / kept))


def sweep_static(scores: np.ndarray, wrong: np.ndarray, gate) -> list:
    """Trace the frozen gate's curve by moving its in-schema bar."""
    return [curve_point(np.where(scores.max(axis=1) < bar, "escalate", "answer"),
                        wrong)
            for bar in np.linspace(0.50, 0.99, 50)]


def sweep_calibrated(scores: np.ndarray, wrong: np.ndarray, model, gate) -> list:
    """Trace the cost rule by moving the cost ratio."""
    features = decision_features(scores, gate.t_prob, gate.qhat)
    probs = model.predict(features)
    sizes = features.column("set_size")
    return [curve_point(calibrated_actions(probs, sizes, Costs(wrong=ratio),
                                           gate.k_clarify), wrong)
            for ratio in COST_RATIOS]


def sweep_ranked(risk: np.ndarray, wrong: np.ndarray, k_clarify: int) -> list:
    """The curve any risk ranking achieves, by handing off the top-k riskiest.

    This is the ceiling for a rank-then-handoff policy, so comparing it against
    the calibrated curve separates two failure modes: a weak *feature* (the
    ranked curve is better) from a framework that does not help at all (the two
    curves agree).
    """
    order = np.argsort(-np.asarray(risk, dtype=float))
    curve = []
    for k in range(0, len(order) + 1, max(1, len(order) // 40)):
        actions = np.where(np.isin(np.arange(len(order)), order[:k]),
                           "escalate", "answer")
        curve.append(curve_point(actions, wrong))
    return curve


def read_curve(curve: list, target: float) -> float:
    """Unsafe rate at a matched escalation rate, linearly interpolated."""
    curve = sorted(curve)
    rates = [e for e, _ in curve]
    if target < rates[0] or target > rates[-1]:
        return float("nan")
    for (e0, u0), (e1, u1) in zip(curve, curve[1:]):
        if e0 <= target <= e1:
            w = 0.0 if e1 == e0 else (target - e0) / (e1 - e0)
            return u0 + w * (u1 - u0)
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--out", default="reports/runs/escalation/v1")
    ap.add_argument("--per-phase", type=int, default=600)
    args = ap.parse_args()

    arrays = load_arrays(Path(args.npz))
    oos, columns = label_mapping(arrays)
    print(f"label map: oos_id={oos} | cosine test acc "
          f"{verify_alignment(arrays, oos, columns):.4f}")

    gate = calibrate_base(arrays, oos, columns)
    model, ranking = fit_model(arrays, oos, columns)
    print(f"gate: tau_answer={gate.tau_answer:.3f} tau_in_schema={gate.tau_in_schema:.3f} "
          f"k_clarify={gate.k_clarify}")
    print(f"risk model: feature={model.feature} flipped={model.flipped} "
          f"held-out AUROC={model.auroc:.3f}")
    print("  ranking:", {k: round(v, 3) for k, v in
                         sorted(ranking.items(), key=lambda kv: -kv[1])})

    scores, wrong = build_stream(arrays, oos, columns, args.per_phase,
                                 np.random.default_rng(0))
    shift = args.per_phase
    print(f"\nstream: {shift} in-scope then {shift} out-of-scope | "
          f"phase-2 unanswerable rate {wrong[shift:].mean():.3f}")

    static = sweep_static(scores, wrong, gate)
    calibrated = sweep_calibrated(scores, wrong, model, gate)
    # sweep_ranked takes a risk score (higher = riskier), so both are negated:
    # a high margin and a high max score are both *safer*.
    features = decision_features(scores, gate.t_prob, gate.qhat)
    by_max_score = sweep_ranked(-scores.max(axis=1), wrong, gate.k_clarify)
    by_margin = sweep_ranked(-features.column("margin"), wrong, gate.k_clarify)

    rows = {"static": {}, "calibrated": {}, "ranked_max_score": {}, "ranked_margin": {}}
    print(f"\nunsafe-answer rate at a matched handoff rate (escalate ∪ clarify)")
    print(f"  {'handoff':>8} {'frozen':>8} {'cost-cal':>9} "
          f"{'rank max_score':>15} {'rank margin':>12}")
    for target in MATCHED_RATES:
        vals = {"static": read_curve(static, target),
                "calibrated": read_curve(calibrated, target),
                "ranked_max_score": read_curve(by_max_score, target),
                "ranked_margin": read_curve(by_margin, target)}
        for key, value in vals.items():
            rows[key][f"{target:.2f}"] = value
        print(f"  {target:>8.2f} {vals['static']:>8.3f} {vals['calibrated']:>9.3f} "
              f"{vals['ranked_max_score']:>15.3f} {vals['ranked_margin']:>12.3f}")

    print("\noperating points (each policy as it would ship)")
    features = decision_features(scores, gate.t_prob, gate.qhat)
    probs = model.predict(features)
    print(f"  {'policy':>12} {'handoff':>8} {'answered':>9} "
          f"{'claimed P(wrong)':>17} {'realised':>9}")
    for ratio in (10.0, 20.0):
        costs = Costs(wrong=ratio)
        actions = calibrated_actions(probs, features.column("set_size"),
                                     costs, gate.k_clarify)
        m = evaluate_policy(actions, wrong)
        answered = actions == "answer"
        realised = (answered & wrong).sum() / max(answered.sum(), 1)
        print(f"  {f'cost {ratio:.0f}:1':>12} {1 - m['answered_rate']:>8.3f} "
              f"{m['answered_rate']:>9.3f} {costs.threshold:>17.3f} {realised:>9.3f}")
    frozen_actions = np.where(scores.max(axis=1) < gate.tau_in_schema,
                              "escalate", "answer")
    m = evaluate_policy(frozen_actions, wrong)
    print(f"  frozen gate : esc {m['escalation_rate']:.3f} unsafe {m['unsafe_rate']:.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dryrun.json").write_text(json.dumps(
        {"oos_id": oos, "per_phase": args.per_phase, "matched_rates": MATCHED_RATES,
         "risk_model": {"feature": model.feature, "flipped": model.flipped,
                        "held_out_auroc": model.auroc, "ranking": ranking},
         "gate": {"tau_answer": gate.tau_answer,
                  "tau_in_schema": gate.tau_in_schema},
         "matched": rows,
         "curves": {"static": static, "calibrated": calibrated}}, indent=1))
    print(f"\nwrote {out_dir / 'dryrun.json'}")


if __name__ == "__main__":
    main()
