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

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.decision.scoring import (aps_members, fit_set_temperature,
                                  fit_temperature, mass_needed,
                                  qhat_from_scores, softmax_rows)
from src.decision.slow import (Observation, PolicyThresholds, SlowState,
                               SlowStateConfig, fit_answer_threshold,
                               fit_in_schema_threshold, observe_row)

ACTIONS = ("answer", "clarify", "escalate", "abstain")


def action_for(top_prob: float, max_score: float, set_size: int,
               thresholds: PolicyThresholds, policy: str = "full") -> str:
    """The action rule, in its documented order (ADR-0013).

    Out-of-schema first, then calibrated confidence, then set size. This is the
    single implementation: ``ConformalGate._gate_action`` and the experiment's
    ``policy.action_for`` both delegate here, so the contract cannot drift into
    two copies of itself.
    """
    if policy == "answer":
        return "answer"
    if thresholds.tau_in_schema is not None and max_score < thresholds.tau_in_schema:
        return "escalate"
    if top_prob >= thresholds.tau_answer:
        return "answer"
    if set_size <= thresholds.k_clarify and policy == "full":
        return "clarify"
    return "escalate"


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
                 policy: str = "full", fit_t_set: bool = True):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if policy not in ("full", "escalate", "answer"):
            raise ValueError(f"policy must be full/escalate/answer, got {policy!r}")
        self.alpha, self.min_n = float(alpha), int(min_n)
        self.t_prob_floor, self.t_set = float(t_prob_floor), float(t_set)
        #: Fit t_set from the calibration data instead of trusting the 1.0
        #: default. Off only to reproduce pre-v19 behaviour.
        self.fit_t_set = bool(fit_t_set)
        self.k_clarify, self.answer_precision = int(k_clarify), float(answer_precision)
        self.policy = policy
        self.t_prob: Optional[float] = None
        self.tau_answer: Optional[float] = None
        self.qhat: Optional[float] = None
        self.tau_in_schema: Optional[float] = None
        self.calibration: Dict = {}
        self.slow: Optional[SlowState] = None

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
        self.tau_answer = fit_answer_threshold(top, correct, self.answer_precision)

        # Select t_set on B, but keep q̂ fitted on ALL of B: the *choice* of
        # temperature is what needs out-of-sample validation (fit_set_temperature
        # halves its input internally to do that), while the quantile is the
        # thing carrying the coverage guarantee and wants every row. Halving B
        # here instead moved q̂ 0.0239 → 0.0254 on the policy_v2 fixture and
        # pushed mean set size 3.80 → 3.99, which is enough to lose every
        # clarify decision at k_clarify=3.
        if self.fit_t_set:
            self.t_set = fit_set_temperature(scores_B, targets_B, self.alpha,
                                             seed=seed)

        self.qhat = qhat_from_scores(scores_B, targets_B, self.t_set, self.alpha)

        self.calibration = {"t_prob": self.t_prob, "tau_answer": self.tau_answer,
                            "qhat": self.qhat, "t_set": self.t_set,
                            "n_A": int(len(scores_A)), "n_B": int(len(scores_B))}
        return self.calibration

    def fit_in_schema(self, in_scores: np.ndarray, oos_scores: np.ndarray) -> float:
        """τ_in_schema on raw max score via the Youden point.

        ``in_scores``/``oos_scores`` are ``(n, k)`` score matrices; the
        in-schema signal is the raw max similarity, not a probability —
        measured AUROC 0.934 on CLINC150 (s1_clinc150_oos).
        """
        self.tau_in_schema = fit_in_schema_threshold(in_scores, oos_scores)
        self.calibration["tau_in_schema"] = self.tau_in_schema
        return self.tau_in_schema

    # ── slow state (optional) ────────────────────────────────────────────

    def attach_slow_state(self, cfg: Optional[SlowStateConfig] = None) -> SlowState:
        """Modulate this gate's thresholds from label-free decision statistics.

        The fitted values stay the base, so detaching restores the frozen
        policy exactly. Call ``calibrate_slow`` with a healthy reference stream
        before serving — the activation bar is fitted, not guessed.
        """
        self.slow = SlowState(self.base_thresholds(), cfg or SlowStateConfig())
        return self.slow

    def calibrate_slow(self, reference: Sequence[Observation]) -> int:
        """Fit the slow state's activation bar on a healthy reference stream."""
        if self.slow is None:
            raise RuntimeError("no slow state attached")
        return self.slow.calibrate(reference)

    def base_thresholds(self) -> PolicyThresholds:
        """The calibrated thresholds, before any slow-state movement."""
        return PolicyThresholds(self.tau_answer, self.k_clarify, self.tau_in_schema)

    def active_thresholds(self) -> PolicyThresholds:
        """The thresholds in force *now* — what a decision was actually chosen
        under. Public because the prediction log records them for off-policy
        replay (component 42): replaying a logged score row under logged
        thresholds must reproduce the logged action."""
        return self._active_thresholds()

    def _active_thresholds(self) -> PolicyThresholds:
        return self.slow.thresholds() if self.slow is not None else self.base_thresholds()

    # ── inference ────────────────────────────────────────────────────────

    def decide(self, scores: np.ndarray, labels: Sequence[str]) -> GateResult:
        """Score vector → probabilities, prediction set, action.

        Order of checks (ADR-0013, corrected triggers): out-of-schema first,
        then calibrated-confidence answer, then small-set clarify, else escalate.
        A slow state, when attached, observes this decision and moves the
        thresholds the checks read — the gate's own fitted values stay the base.
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

        thresholds = self._active_thresholds()
        if self.slow is not None:
            # observe_row recomputes what this method already has. That is
            # deliberate: the state must see exactly the statistics a caller
            # builds the healthy reference from, and two code paths for the
            # same statistics is how the calibration drifted out of agreement
            # twice already.
            obs, _ = observe_row(scores, self.t_prob, self.qhat, self.t_set)
            self.slow.observe(obs)

        action = self._gate_action(top_prob, max_score, len(pred_set), thresholds)
        return GateResult(probabilities=probs, prediction_set=pred_set,
                          action=action, top_prob=top_prob,
                          max_score=max_score, alpha=self.alpha)

    def _gate_action(self, top_prob: float, max_score: float, set_size: int,
                     thresholds: PolicyThresholds) -> str:
        """This gate's action, from the one shared implementation."""
        return action_for(top_prob, max_score, set_size, thresholds, self.policy)

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
            "slow_state": None if self.slow is None else asdict(self.slow.fitted_config()),
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
        if d.get("slow_state") is not None:
            # The activation bar is fitted from a reference stream, which is a
            # run-time input — a reloaded gate serves the config bar until the
            # caller calls calibrate_slow().
            g.attach_slow_state(SlowStateConfig(**d["slow_state"]))
        return g
