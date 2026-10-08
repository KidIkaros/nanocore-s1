"""ConformalGate — calibrated sets and the action layer (ADR-0009, ADR-0013).

The gate owns everything the scorer must not decide itself: reported
probabilities (T_prob, floored — log-loss fitting otherwise sharpens to
degeneracy), prediction-set construction (T_set, APS), the answer threshold
(τ_answer at a precision target), the in-schema boundary (τ_in_schema on raw
max score), and the action ∈ {answer, clarify, escalate, abstain}.

Three measured traps are encoded as defaults:

- **Dual temperature** (s1_clinc150_oos): the T that minimizes log loss is not
  the T that builds usable sets. ``t_prob`` is floored at ``t_prob_floor``;
  ``t_set`` is fixed unless explicitly overridden.
- **Three-way calibration** (s1_policy_v2): A split fits T_prob and τ_answer,
  B split calibrates q̂ — sharing data biases coverage optimistically.
- **Per-scorer calibration**: ``calibrate`` is called per scorer+schema; a
  cross-entropy head needs its own T_prob/q̂ or its sets degenerate (q̂≈1.0).
  Coverage is verified to land in a *band*, not merely ≥ target.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.decision.scoring import aps_members, fit_temperature, mass_needed, softmax_rows

ACTIONS = ("answer", "clarify", "escalate", "abstain")


@dataclass
class GateResult:
    """One gated decision over a label set."""
    probabilities: Dict[str, float]          # reported probs at T_prob
    prediction_set: List[str]                # APS members at T_set
    action: str                              # answer / clarify / escalate / abstain
    top_prob: float
    max_score: float                         # raw top score — the OOS signal
    alpha: float


class ConformalGate:
    """Split-conformal gate with dual temperatures and action routing.

    Args:
        alpha: Miscoverage target (set coverage ≈ 1 − alpha, verified in a band).
        min_n: Fail-closed minimum calibration size — refuse below it
            (the MIN_BUCKET_N lesson: small-n calibration silently uncalibrates).
        t_prob_floor: Floor on the log-loss-fitted probability temperature.
        t_set: Temperature for APS set construction (1.0 measured sane).
        k_clarify: Max set size that routes to ``clarify`` instead of ``escalate``.
        answer_precision: Target precision among answered items for τ_answer.
        policy: ``"full"`` enables clarify; ``"escalate"`` maps clarify→escalate
            (headless router); ``"answer"`` bypasses gating entirely.
    """

    def __init__(self, alpha: float = 0.10, min_n: int = 200,
                 t_prob_floor: float = 0.25, t_set: float = 1.0,
                 k_clarify: int = 3, answer_precision: float = 0.90,
                 policy: str = "full"):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if policy not in ("full", "escalate", "answer"):
            raise ValueError(f"policy must be full/escalate/answer, got {policy!r}")
        self.alpha, self.min_n = float(alpha), int(min_n)
        self.t_prob_floor, self.t_set = float(t_prob_floor), float(t_set)
        self.k_clarify, self.answer_precision = int(k_clarify), float(answer_precision)
        self.policy = policy
        self.t_prob: Optional[float] = None
        self.tau_answer: Optional[float] = None
        self.qhat: Optional[float] = None
        self.tau_in_schema: Optional[float] = None
        self.calibration: Dict = {}

    # ── calibration ──────────────────────────────────────────────────────

    def calibrate(self, scores_A: np.ndarray, targets_A: np.ndarray,
                  scores_B: Optional[np.ndarray] = None,
                  targets_B: Optional[np.ndarray] = None,
                  seed: int = 0) -> Dict:
        """Fit T_prob + τ_answer on A, q̂ on B. Splits A if B is not given.

        ``scores_*`` are ``(n, k)`` raw scorer outputs; ``targets_*`` are the
        true class indices. Raises below ``min_n`` — an underfed gate must
        fail closed, not silently miscalibrate.
        """
        scores_A = np.asarray(scores_A, dtype=np.float64)
        targets_A = np.asarray(targets_A, dtype=int).reshape(-1)
        if scores_B is None:
            if len(scores_A) < self.min_n:
                raise ValueError(
                    f"min_n guard: {len(scores_A)} calibration examples < {self.min_n}")
            perm = np.random.default_rng(seed).permutation(len(scores_A))
            half = len(perm) // 2
            A_idx, B_idx = perm[:half], perm[half:]
            scores_B, targets_B = scores_A[B_idx], targets_A[B_idx]
            scores_A, targets_A = scores_A[A_idx], targets_A[A_idx]
        else:
            scores_B = np.asarray(scores_B, dtype=np.float64)
            targets_B = np.asarray(targets_B, dtype=int).reshape(-1)

        for name, s, t in (("A", scores_A, targets_A), ("B", scores_B, targets_B)):
            if len(s) < self.min_n // 2:
                raise ValueError(
                    f"min_n guard: split {name} has {len(s)} < {self.min_n // 2}")

        self.t_prob = fit_temperature(scores_A, targets_A, floor=self.t_prob_floor)

        P_A = softmax_rows(scores_A, self.t_prob)
        top = P_A.max(axis=1)
        correct = scores_A.argmax(axis=1) == targets_A
        self.tau_answer = self._fit_tau_answer(top, correct)

        m = mass_needed(softmax_rows(scores_B, self.t_set), targets_B)
        n = len(m)
        self.qhat = float(np.quantile(
            m, min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)))

        self.calibration = {"t_prob": self.t_prob, "tau_answer": self.tau_answer,
                            "qhat": self.qhat, "t_set": self.t_set,
                            "n_A": int(len(scores_A)), "n_B": int(len(scores_B))}
        return self.calibration

    def _fit_tau_answer(self, top_prob: np.ndarray, correct: np.ndarray,
                        min_answered: int = 30) -> float:
        """Smallest top-prob threshold reaching answer_precision; fail closed."""
        cands = np.quantile(top_prob, np.linspace(0, 0.95, 40))
        for t in np.sort(cands):
            mask = top_prob >= t
            if mask.sum() < min_answered:
                continue
            if correct[mask].mean() >= self.answer_precision:
                return float(t)
        return float(np.quantile(top_prob, 0.9))   # answer the least, not the most

    def fit_in_schema(self, in_scores: np.ndarray, oos_scores: np.ndarray) -> float:
        """τ_in_schema on raw max score via the Youden point.

        ``in_scores``/``oos_scores`` are ``(n, k)`` score matrices; the
        in-schema signal is the raw max similarity, not a probability —
        measured AUROC 0.934 on CLINC150 (s1_clinc150_oos).
        """
        ins = np.asarray(in_scores, dtype=np.float64).max(axis=1)
        oos = np.asarray(oos_scores, dtype=np.float64).max(axis=1)
        y = np.concatenate([np.ones(len(ins)), np.zeros(len(oos))])
        x = np.concatenate([ins, oos])
        best_tau, best_j = float(np.median(ins)), -1.0
        for t in np.unique(x):
            j = float((x[y == 1] >= t).mean() + (x[y == 0] < t).mean() - 1)
            if j > best_j:
                best_j, best_tau = j, float(t)
        self.tau_in_schema = best_tau
        self.calibration["tau_in_schema"] = best_tau
        self.calibration["in_schema_j"] = best_j
        return best_tau

    # ── inference ────────────────────────────────────────────────────────

    def decide(self, scores: np.ndarray, labels: Sequence[str]) -> GateResult:
        """Score vector → probabilities, prediction set, action.

        Order of checks (ADR-0013, corrected triggers): out-of-schema first,
        then calibrated-confidence answer, then small-set clarify, else escalate.
        """
        labels = list(labels)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(scores) != len(labels):
            raise ValueError("scores and labels must be the same length")
        if self.qhat is None or self.t_prob is None:
            raise RuntimeError("gate is not calibrated")

        P = softmax_rows(scores, self.t_prob)[0]
        top_prob = float(P.max())
        max_score = float(scores.max())
        P_set = softmax_rows(scores, self.t_set)[0]
        members = aps_members(P_set[None, :], self.qhat)[0]
        pred_set = [labels[i] for i in members]
        probs = {l: float(p) for l, p in zip(labels, P)}

        if self.policy == "answer":
            action = "answer"
        elif self.tau_in_schema is not None and max_score < self.tau_in_schema:
            action = "escalate"
        elif top_prob >= self.tau_answer:
            action = "answer"
        elif len(pred_set) <= self.k_clarify and self.policy == "full":
            action = "clarify"
        else:
            action = "escalate"

        return GateResult(probabilities=probs, prediction_set=pred_set,
                          action=action, top_prob=top_prob,
                          max_score=max_score, alpha=self.alpha)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path) -> None:
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "alpha": self.alpha, "min_n": self.min_n,
            "t_prob_floor": self.t_prob_floor, "t_set": self.t_set,
            "k_clarify": self.k_clarify, "answer_precision": self.answer_precision,
            "policy": self.policy, "t_prob": self.t_prob,
            "tau_answer": self.tau_answer, "qhat": self.qhat,
            "tau_in_schema": self.tau_in_schema, "calibration": self.calibration,
        }, indent=1))

    @classmethod
    def load(cls, path) -> "ConformalGate":
        import json
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        g = cls(alpha=d["alpha"], min_n=d["min_n"], t_prob_floor=d["t_prob_floor"],
                t_set=d["t_set"], k_clarify=d["k_clarify"],
                answer_precision=d["answer_precision"], policy=d["policy"])
        g.t_prob, g.tau_answer, g.qhat = d["t_prob"], d["tau_answer"], d["qhat"]
        g.tau_in_schema, g.calibration = d["tau_in_schema"], d["calibration"]
        return g
