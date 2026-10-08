"""Slow state — label-free modulation of the gate's thresholds.

The shipped mechanism. ``policy.py`` holds the A/B experiment that measures it;
this module holds the part the gate actually runs, so a bundle can carry the
mode without carrying the experiment.

The mechanism is kindred's astrocyte rule (``angn.rs``: n-of-m hysteresis over
a window) applied to policy rather than to drives:

- a decision is *anomalous* when the frozen policy would already escalate it,
  or when its raw top score sits under the in-schema boundary;
- ``activate_at`` anomalous decisions in the last ``window`` raise the answer
  bar, and the bar is fitted from a healthy reference stream rather than
  guessed — on cached CLINC150 the median healthy decision sits exactly on
  ``tau_answer``, so half of healthy decisions look anomalous and a constant
  bar false-activates;
- a Schmitt trigger holds the bar through ambiguity and releases only on
  sustained calm, because a shift near the drift margin otherwise flickers.

Direction is fixed a priori: drift makes the policy more conservative, never
less. No parameters are added anywhere — the state only changes what the gate
reads, which is the capacity confound the ANGN literature cannot rule out.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence

import numpy as np

from src.decision.scoring import aps_members, softmax_rows

TAU_CEILING = 0.999   # a probability threshold never reaches certainty


def fit_answer_threshold(top_prob: np.ndarray, correct: np.ndarray,
                         precision: float, min_answered: int = 30) -> float:
    """Smallest top-prob threshold reaching ``precision`` among answered items.

    Fails closed: below ``min_answered`` candidates the threshold answers the
    least, never the most.
    """
    top_prob = np.asarray(top_prob, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    for t in np.sort(np.quantile(top_prob, np.linspace(0, 0.95, 40))):
        mask = top_prob >= t
        if mask.sum() >= min_answered and correct[mask].mean() >= precision:
            return float(t)
    return float(np.quantile(top_prob, 0.9))


def fit_in_schema_threshold(in_scores: np.ndarray, oos_scores: np.ndarray) -> float:
    """Youden point on raw max score — the in-schema / out-of-schema boundary.

    Operates on max scores, not probabilities: the in-schema signal is the raw
    similarity (measured AUROC 0.934 on CLINC150).
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
    return best_tau


@dataclass(frozen=True)
class Observation:
    """One decision's statistics. Everything here is computable without labels."""
    top_prob: float
    set_size: int
    entropy: float
    pred: int
    max_score: float


@dataclass(frozen=True)
class PolicyThresholds:
    """The knobs ``ConformalGate.decide`` reads, in its own order.

    ``tau_in_schema`` is the escalation lever that actually gates novel input —
    measured on the frozen policy as the knob that decides whether an item is
    answered at all. A gate that carries no in-schema boundary leaves it None.
    """
    tau_answer: float
    k_clarify: int
    tau_in_schema: Optional[float] = None


@dataclass(frozen=True)
class SlowStateConfig:
    """Astrocyte n-of-m rule, plus a drift check on the score distribution.

    ``reference_max_score`` is the mean raw top score seen at calibration time.
    Supplying it lets the state notice a population shift that never crosses a
    threshold — the case where every decision looks individually fine while the
    distribution has moved underneath the frozen policy.
    """
    window: int = 20
    activate_at: int = 14
    release_at: int = 6
    tau_step: float = 0.04
    max_shift: float = 0.30
    reference_max_score: Optional[float] = None
    drift_margin: float = 0.10


def observe_row(row: np.ndarray, t_prob: float, qhat: float) -> tuple:
    """The label-free statistics of one decision row — the state's only input.

    Also how a caller builds the healthy reference stream the activation bar is
    fitted on, using exactly the statistics the state will see at run time.
    """
    P = softmax_rows(row, t_prob)[0]
    members = aps_members(P[None, :], qhat)[0]
    obs = Observation(top_prob=float(P.max()), set_size=len(members),
                      entropy=float(-(P * np.log(P + 1e-12)).sum()),
                      pred=int(P.argmax()), max_score=float(row.max()))
    return obs, members


def _windows(items: Sequence, window: int) -> List[List]:
    return [list(items[i:i + window]) for i in range(len(items) - window + 1)]


class SlowState:
    """Windowed slow state over label-free anomalies.

    No hidden time constant: the window *is* the memory, and the streak counts
    consecutive windows at or above the bar. That makes the mechanism
    inspectable — every threshold move traces to an observable decision.
    """

    def __init__(self, base: PolicyThresholds, cfg: SlowStateConfig):
        if not 0 < cfg.activate_at <= cfg.window:
            raise ValueError("activate_at must be in (0, window]")
        if not 0 <= cfg.release_at < cfg.activate_at:
            raise ValueError("release_at must be in [0, activate_at)")
        self._base, self._cfg = base, cfg
        self._window: deque = deque(maxlen=cfg.window)
        self._streak = 0
        self._activate_at = cfg.activate_at
        self._release_at = cfg.release_at
        # Attaching this state makes the gate stateful, and the gate is served
        # by a ThreadingHTTPServer: the lock lives here, with the state, so
        # every caller is serialized by construction rather than by contract.
        self._lock = threading.Lock()

    def calibrate(self, reference: Sequence[Observation]) -> int:
        """Set the activation bar from a healthy reference stream.

        The bar is the most anomalous window seen while healthy, plus one: the
        anomaly criterion fires at the scorer's base rate, so a constant bar
        cannot separate a shift from ordinary variation.
        """
        counts = [sum(self._anomalous(o) for o in w)
                  for w in _windows(reference, self._cfg.window)]
        if not counts:
            raise ValueError(
                f"reference stream of {len(reference)} is shorter than the "
                f"{self._cfg.window}-window it must calibrate")
        with self._lock:
            self._activate_at = min(self._cfg.window, max(counts) + 1)
            self._release_at = min(self._cfg.release_at, self._activate_at - 1)
            return self._activate_at

    @property
    def config(self) -> SlowStateConfig:
        """The configuration this state was built from."""
        return self._cfg

    def fitted_config(self) -> SlowStateConfig:
        """Config carrying the *fitted* bar — what a bundle must persist.

        The bar is a calibrated parameter like ``qhat``: reloading a bundle
        should not silently fall back to the config default and re-introduce
        the false-activation the calibration removed.
        """
        return replace(self._cfg, activate_at=self._activate_at,
                       release_at=self._release_at)

    def observe(self, obs: Observation) -> None:
        with self._lock:
            self._window.append(obs)
            self._streak = self._next_streak()

    def _next_streak(self) -> int:
        """Schmitt trigger: activate on strong evidence, hold through ambiguity.

        A single quiet window is not grounds to release — measured on real data,
        a shift that sits near ``drift_margin`` otherwise flickers, and the
        threshold snaps back to base while the shift is still in force.
        """
        if len(self._window) < self._cfg.window:
            return 0
        anomalies = sum(self._anomalous(o) for o in self._window)
        if anomalies >= self._activate_at or self._drifted():
            return self._streak + 1
        if self._streak and anomalies <= self._release_at:
            return 0
        return self._streak

    def _anomalous(self, obs: Observation) -> bool:
        below_schema = (self._base.tau_in_schema is not None
                        and obs.max_score < self._base.tau_in_schema)
        return (obs.top_prob < self._base.tau_answer
                or obs.set_size > self._base.k_clarify
                or below_schema)

    def _drifted(self) -> bool:
        reference = self._cfg.reference_max_score
        if reference is None or reference <= 0:
            return False
        running = float(np.mean([o.max_score for o in self._window]))
        return running < reference * (1 - self._cfg.drift_margin)

    def thresholds(self) -> PolicyThresholds:
        """One step of movement per activated window.

        τ_answer climbs with the streak (bounded by ``max_shift``) and k_clarify
        drops by exactly one, so drift routes the marginal set to escalate
        without collapsing clarify entirely.

        τ_in_schema moves only when a reference score scale is known, and then
        in proportion to it — the boundary lives in raw-score units, so an
        absolute step would mean something different on every dataset.
        """
        with self._lock:
            streak = self._streak
        shift = min(self._cfg.max_shift, streak * self._cfg.tau_step)
        reference = self._cfg.reference_max_score
        if self._base.tau_in_schema is None or reference is None:
            schema_bar = self._base.tau_in_schema
        else:
            schema_bar = min(TAU_CEILING, self._base.tau_in_schema + shift * reference)
        return PolicyThresholds(
            tau_answer=min(TAU_CEILING, self._base.tau_answer + shift),
            k_clarify=max(1, self._base.k_clarify - int(streak > 0)),
            tau_in_schema=schema_bar)
