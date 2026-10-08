"""Local A/B of the slow-state policy arms on cached CLINC150 arrays.

Real data, no encoding: the v10 kernel cached train embeddings, val/test score
matrices and label vectors, so the stream and every arm are pure array math —
allowed by the compute boundary (``AGENTS.md``). The point is to find out
whether the glial arm is worth a GPU session before spending one.

Two streams, both real items with real labels; only the *ordering and option
set* are constructed, and that is stated in the output:

    novel_intents   phase-2 items whose class is absent from the deployed
                    option set — the correct action is escalate
    prior_shift     phase-2 items from classes phase 1 never saw, option set
                    unchanged — a healthy stream, where the policy must NOT move

Arms differ only in what they may observe (see ``src/decision/policy.py``).

The cached arrays hold CLINC's raw dataset ids, where ``oos`` sits in the
middle of the label list (id 80), not at the end. ``label_mapping`` recovers the
kernel's id→column map from the data and is verified against the kernel's own
published cosine accuracy before any arm runs.

Usage:
    python scripts/policy_ab_local.py [--npz PATH] [--out DIR] [--per-phase N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision.gate import ConformalGate, fit_in_schema_threshold
from src.decision.policy import (GlialPolicy, PolicyThresholds,
                                 RecalibrateConfig, RecalibratePolicy,
                                 SlowStateConfig, StaticPolicy, Stream,
                                 StreamConfig, run_stream)

DEFAULT_NPZ = "reports/runs/s1_verify/v10/verify_scores.npz"
EXPECTED_COSINE_TEST_ACC = 0.7562    # v10's published cosine accuracy
SPLIT = 75                           # columns below this are "deployed"
SEED = 0


def load_arrays(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def label_mapping(arrays: dict) -> tuple:
    """Recover (oos_id, id→column) from the cached arrays.

    The kernel's label vectors drop ``oos`` and keep the remaining ids in
    ascending order, so column *c* is the *c*-th smallest non-OOS id. ``oos`` is
    the id holding the out-of-scope split's rows — an order of magnitude more
    than any single intent.
    """
    counts = np.unique(arrays["y_test"], return_counts=True)
    ids, freq = counts[0], counts[1]
    oos = int(ids[np.argmax(freq)])
    if freq.max() < 10 * np.median(freq):
        raise ValueError("no dominant out-of-scope id — cached arrays are unexpected")
    columns = {int(i): c for c, i in enumerate(sorted(int(i) for i in ids if int(i) != oos))}
    return oos, columns


def to_columns(labels: np.ndarray, columns: dict) -> np.ndarray:
    """Column ids for in-scope labels; out-of-scope rows map to -1 (as the kernel did)."""
    return np.array([columns.get(int(v), -1) for v in labels])


def verify_alignment(arrays: dict, oos: int, columns: dict) -> float:
    """Guard: the derived map must reproduce the kernel's own cosine accuracy."""
    keep = arrays["y_test"] != oos
    pred = arrays["scores_test"].argmax(axis=1)
    acc = float((pred[keep] == to_columns(arrays["y_test"], columns)[keep]).mean())
    if abs(acc - EXPECTED_COSINE_TEST_ACC) > 0.01:
        raise ValueError(f"label map does not reproduce v10 cosine acc: {acc:.4f}")
    return acc


def calibrate_base(arrays: dict, oos: int, columns: dict,
                   alpha: float = 0.10) -> ConformalGate:
    """The frozen incumbent: thresholds fitted on val, including the in-schema bar."""
    keep = arrays["y_val"] != oos
    gate = ConformalGate(alpha=alpha, min_n=200)
    gate.calibrate(arrays["scores_val"][keep],
                   to_columns(arrays["y_val"], columns)[keep], seed=SEED)
    gate.fit_in_schema(arrays["scores_val"][keep], arrays["scores_val"][~keep])
    return gate


def reference_scale(arrays: dict, oos: int) -> float:
    """Mean raw top score on the calibration distribution — the drift baseline."""
    keep = arrays["y_val"] != oos
    return float(arrays["scores_val"][keep].max(axis=1).mean())


def train_pool(arrays: dict, oos: int, columns: dict) -> tuple:
    """Recompute train scores from cached embeddings; drop out-of-scope rows."""
    keep = arrays["y_train"] != oos
    scores = arrays["emb_train"][keep] @ arrays["label_vecs"].T
    return scores, to_columns(arrays["y_train"], columns)[keep]


def sample_stream(scores: np.ndarray, labels: np.ndarray, columns: np.ndarray,
                  per_phase: int, rng) -> tuple:
    """Two equal phases of real items; ``columns`` is the deployed option set."""
    low = np.flatnonzero(labels < SPLIT)
    high = np.flatnonzero(labels >= SPLIT)
    picked = np.concatenate([rng.choice(low, per_phase, replace=False),
                             rng.choice(high, per_phase, replace=False)])
    return scores[picked][:, columns], labels[picked]


def arms(base: PolicyThresholds, reference: float) -> list:
    """Static incumbent, the glial arm, and the two label-driven baselines."""
    return [StaticPolicy(base),
            GlialPolicy(base, SlowStateConfig(reference_max_score=reference)),
            RecalibratePolicy(base, RecalibrateConfig(delay=200, refit_every=50,
                                                      n_options=SPLIT)),
            RecalibratePolicy(base, RecalibrateConfig(delay=0, refit_every=50,
                                                      n_options=SPLIT))]


def run_streams(arrays: dict, gate: ConformalGate, per_phase: int,
                seed: int) -> dict:
    oos, columns = label_mapping(arrays)
    scores, labels = train_pool(arrays, oos, columns)
    deployed, full = np.arange(SPLIT), np.arange(scores.shape[1])
    base = PolicyThresholds(gate.tau_answer, gate.k_clarify, gate.tau_in_schema)
    reference = reference_scale(arrays, oos)
    rng = np.random.default_rng(seed)
    streams = {"novel_intents": sample_stream(scores, labels, deployed, per_phase, rng),
               "prior_shift": sample_stream(scores, labels, full, per_phase, rng)}
    out = {}
    for name, (S, y) in streams.items():
        stream = Stream(S, y, per_phase)
        cfg = StreamConfig(t_prob=gate.t_prob, qhat=gate.qhat,
                           k_clarify=gate.k_clarify)
        out[name] = {p.name: run_stream(p, stream, cfg)
                     for p in arms(base, reference)}
    return out


AGGREGATED_METRICS = ("accuracy", "escalation_rate", "wrong_answer_rate",
                      "selective_acc")


def aggregate(runs: list) -> dict:
    """Mean and spread of the deciding metrics across seeds.

    One seed is luck; the A/B verdict is read off these means.
    """
    streams, policies = runs[0].keys(), runs[0][list(runs[0])[0]].keys()
    out = {}
    for stream in streams:
        out[stream] = {}
        for policy in policies:
            out[stream][policy] = {
                phase: {k: {"mean": float(np.mean([r[stream][policy][phase][k]
                                                   for r in runs])),
                            "sd": float(np.std([r[stream][policy][phase][k]
                                                for r in runs]))}
                        for k in AGGREGATED_METRICS}
                for phase in ("phase1", "phase2")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--out", default="reports/runs/policy_ab/v1")
    ap.add_argument("--per-phase", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    arrays = load_arrays(Path(args.npz))
    oos, columns = label_mapping(arrays)
    print(f"label map: oos_id={oos} | cosine test acc "
          f"{verify_alignment(arrays, oos, columns):.4f} (matches v10)")
    gate = calibrate_base(arrays, oos, columns)
    print(f"base: tau={gate.tau_answer:.3f} qhat={gate.qhat:.3f} "
          f"k_clarify={gate.k_clarify} t_prob={gate.t_prob:.3f} "
          f"tau_in_schema={gate.tau_in_schema:.3f} "
          f"ref_max_score={reference_scale(arrays, oos):.3f}")

    runs = [run_streams(arrays, gate, args.per_phase, s) for s in range(args.seeds)]
    agg = aggregate(runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ab_results.json").write_text(json.dumps(
        {"base": gate.calibration, "per_phase": args.per_phase, "seeds": args.seeds,
         "split_column": SPLIT, "aggregate": agg, "per_seed": runs}, indent=1))

    for stream, by_policy in agg.items():
        print(f"\n== {stream}  (mean over {args.seeds} seeds)")
        for policy, phases in by_policy.items():
            p1, p2 = phases["phase1"], phases["phase2"]
            print(f"  {policy:22} "
                  f"acc {p1['accuracy']['mean']:.3f}->{p2['accuracy']['mean']:.3f} "
                  f"esc {p1['escalation_rate']['mean']:.3f}->{p2['escalation_rate']['mean']:.3f} "
                  f"wrong {p2['wrong_answer_rate']['mean']:.3f}"
                  f"±{p2['wrong_answer_rate']['sd']:.3f} "
                  f"sel@0.8={p2['selective_acc']['mean']:.3f}")
    print(f"\nwrote {out_dir / 'ab_results.json'}")


if __name__ == "__main__":
    main()
