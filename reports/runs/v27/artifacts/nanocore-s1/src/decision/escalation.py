"""Cost-calibrated escalation — the expected-loss action rule.

The gate's thresholds are fitted for a *precision* target. This module fits the
decision that precision is a proxy for: the action with the lowest expected
cost under an explicit cost matrix.

    answer iff  P(wrong) · C_wrong  <  C_escalate

so the escalation threshold is ``C_escalate / C_wrong`` — interpretable, and
movable by an operator who knows what a wrong answer costs relative to a human
review. ``P(wrong)`` is calibrated by isotonic regression from a *label-free*
feature (margin, set size, entropy, raw top score), which is what makes the
rule deployable: at run time nothing but the score vector is needed.

Two design notes that come from measurement:

- **Fit on in-distribution *and* out-of-distribution calibration rows.** Our
  label-driven arm moved the confidence bar while the in-schema bar was the
  binding constraint, so the feature range must cover the novel regime. Isotonic
  regression is a step function and clips outside its fitted range, so a model
  fitted on in-scope rows only would answer novel input with whatever the
  lowest in-scope error rate happened to be.
- **Compare policies at matched escalation rate.** Unsafe answers fall
  monotonically as a policy escalates more, so an unmatched comparison rewards
  the policy that abstains most. ``unsafe_at_escalation`` reads the risk-
  coverage curve at a common operating point.

**Authority.** ``gate.action_for`` remains the shipped action rule. This module
is the expected-loss *alternative*: measured on cached CLINC150 it did not beat
the frozen gate at a matched handoff rate, because the feature that won
selection on the calibration target is not the one that wins on the shifted
stream. Adopt it when it beats the gate on the same comparison, not before.

Reference points: UCCI (2026) maps margin uncertainty to a per-query error
probability by isotonic regression and picks the threshold by constrained cost
minimisation, proving threshold policies on the calibrated score are
cost-optimal under three stated assumptions. See
``docs/research/learning-from-failure.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.decision.slow import observe_row

FEATURE_NAMES = ("margin", "top_prob", "set_size", "entropy", "max_score")


@dataclass(frozen=True)
class Costs:
    """What each outcome costs. The ratio is the whole policy.

    ``wrong`` is the cost of answering and being wrong (a silent failure the
    caller acts on); ``escalate`` is the cost of sending the item to a human or
    a stronger model. With the 10:1 default, an item is answered only while its
    calibrated error probability stays under 10%.
    """
    wrong: float = 10.0
    escalate: float = 1.0

    def __post_init__(self) -> None:
        if self.wrong <= 0 or self.escalate <= 0:
            raise ValueError("costs must be positive")

    @property
    def threshold(self) -> float:
        return self.escalate / self.wrong


@dataclass(frozen=True)
class DecisionFeatures:
    """A matrix of label-free per-decision signals, with their names."""
    names: Tuple[str, ...]
    values: np.ndarray

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]


def decision_features(scores: np.ndarray, t_prob: float,
                      qhat: float, t_set: float) -> DecisionFeatures:
    """Every signal the escalation decision may use — none needs a label.

    Built from ``slow.observe_row`` so each statistic has one definition. The
    per-row loop costs little at these sizes and buys the guarantee that the
    risk features and the slow state's inputs can never disagree.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2-D (n, n_options) matrix")
    rows = [observe_row(row, t_prob, qhat, t_set)[0] for row in scores]
    return DecisionFeatures(FEATURE_NAMES, np.column_stack([
        [obs.margin for obs in rows], [obs.top_prob for obs in rows],
        [obs.set_size for obs in rows], [obs.entropy for obs in rows],
        [obs.max_score for obs in rows]]))


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """AUROC with sklearn kept at the boundary (lazy, as evaluate.py does).

    A single-class target has no ranking to score, and 0.5 is the honest answer
    — chance — rather than an error: feature selection must be able to run on a
    split that happens to contain no errors.
    """
    from sklearn.metrics import roc_auc_score
    positive = np.asarray(positive, dtype=bool)
    if positive.all() or not positive.any():
        return 0.5
    return float(roc_auc_score(positive, score))


def feature_auroc(features: DecisionFeatures, wrong: np.ndarray) -> Dict[str, float]:
    """How well each signal ranks wrong decisions, orientation-corrected.

    A feature is used as a *risk* score, so AUROC below 0.5 means the signal
    runs the other way (higher margin is safer, for instance) and is flipped.
    """
    return {name: max(_auroc(features.column(name), wrong),
                      1 - _auroc(features.column(name), wrong))
            for name in features.names}


@dataclass(frozen=True)
class ErrorModel:
    """Isotonic P(wrong) from one observable feature.

    ``auroc`` is the *oriented* ranking quality — the feature as a risk score.
    Storing the raw AUROC would make an informative signal that runs the other
    way (a high margin is safe) look useless, and selection would discard it.
    """
    feature: str
    flipped: bool
    isotonic: object
    auroc: float

    def predict(self, features: DecisionFeatures) -> np.ndarray:
        raw = features.column(self.feature)
        return np.clip(self.isotonic.predict(-raw if self.flipped else raw), 0.0, 1.0)


def fit_error_model(features: DecisionFeatures, wrong: np.ndarray,
                    feature: str) -> ErrorModel:
    """Fit P(wrong) on one feature, oriented so it is a risk score.

    ``ErrorModel.auroc`` reports the in-sample ranking quality. Scoring a model
    on data it was fitted to is meaningless, so selection re-scores it — see
    ``select_error_model``.
    """
    from sklearn.isotonic import IsotonicRegression

    wrong = np.asarray(wrong, dtype=float)
    raw = features.column(feature)
    raw_auroc = _auroc(raw, wrong)
    flipped = raw_auroc < 0.5
    x = -raw if flipped else raw
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, wrong)
    return ErrorModel(feature, flipped, iso, max(raw_auroc, 1 - raw_auroc))


def select_error_model(fit: DecisionFeatures, fit_wrong: np.ndarray,
                       select: DecisionFeatures,
                       select_wrong: np.ndarray) -> Tuple[ErrorModel, Dict[str, float]]:
    """Pick the feature whose calibrated risk ranks held-out errors best.

    Selection happens on data the isotonic fit never saw, so the winner is not
    the feature that overfits its own calibration.
    """
    candidates = [(name, fit_error_model(fit, fit_wrong, name)) for name in fit.names]
    ranking = {name: _auroc(model.predict(select), select_wrong)
               for name, model in candidates}
    best_name = max(ranking, key=ranking.get)
    best = next(model for name, model in candidates if name == best_name)
    return ErrorModel(best.feature, best.flipped, best.isotonic,
                      ranking[best_name]), ranking


def calibrated_actions(error_probs: np.ndarray, set_sizes: np.ndarray,
                       costs: Costs, k_clarify: int) -> np.ndarray:
    """Expected-loss action per item, keeping the clarify semantics.

    Escalating is chosen when the calibrated risk exceeds the cost ratio; the
    item is *clarified* instead when the conformal set is small enough to ask
    which option was meant, which is cheaper than escalating a well-posed
    question.
    """
    error_probs = np.asarray(error_probs, dtype=float)
    set_sizes = np.asarray(set_sizes)
    action = np.where(error_probs >= costs.threshold, "escalate", "answer")
    narrow = set_sizes <= k_clarify
    return np.where((action == "escalate") & narrow, "clarify", action)


def unsafe_at_escalation(risk: np.ndarray, wrong: np.ndarray,
                         target: float) -> Optional[float]:
    """Unsafe-answer rate at a matched escalation rate.

    Decisions are ranked by risk and the riskiest ``target`` fraction is handed
    off; the returned rate is the share of the remainder that is wrong. This is
    the risk-coverage curve read at a common operating point, which is what
    stops "escalate everything" from looking good: every policy is charged the
    same escalation budget.
    """
    risk = np.asarray(risk, dtype=float)
    wrong = np.asarray(wrong, dtype=bool)
    if len(risk) != len(wrong):
        raise ValueError("risk and wrong must be the same length")
    if not 0 <= target <= 1:
        raise ValueError("target must be a rate in [0, 1]")
    keep = int(round(len(risk) * (1 - target)))
    if keep <= 0:
        return 0.0
    if keep >= len(risk):
        return float(wrong.mean())
    safest = np.argsort(risk)[:keep]             # lowest risk kept
    return float(wrong[safest].mean())


def curve_at(curve: Sequence[Tuple[float, float]], target: float) -> float:
    """Unsafe rate from a (handoff, unsafe) curve at a matched handoff rate.

    Policies have different operating knobs — a fixed bar, a cost ratio, a
    tolerated error rate — so comparing them at whatever point each happens to
    choose says nothing. Interpolating to a common handoff rate is the
    comparison. Returns ``nan`` when the curve never reaches the rate, which is
    itself informative: the policy cannot operate there.
    """
    ordered = sorted(curve)
    rates = [handoff for handoff, _ in ordered]
    if not ordered or target < rates[0] or target > rates[-1]:
        return float("nan")
    for (h0, u0), (h1, u1) in zip(ordered, ordered[1:]):
        if h0 <= target <= h1:
            weight = 0.0 if h1 == h0 else (target - h0) / (h1 - h0)
            return float(u0 + weight * (u1 - u0))
    return float(ordered[-1][1])


def evaluate_policy(actions: Sequence[str], wrong: np.ndarray) -> Dict:
    """Escalation rate and unsafe-answer rate for a policy's decisions."""
    actions = np.asarray(actions)
    wrong = np.asarray(wrong, dtype=bool)
    answered = actions == "answer"
    return {"escalation_rate": float((actions == "escalate").mean()),
            "clarify_rate": float((actions == "clarify").mean()),
            "answered_rate": float(answered.mean()),
            "unsafe_rate": float((answered & wrong).mean())}
