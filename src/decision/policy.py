"""Slow-state policy control — the glial seam, measured against its baselines.

The gate fits ``tau_answer`` and ``k_clarify`` once on a calibration split and
then freezes them. Under distribution shift the confidence distribution moves,
conformal sets grow, and a frozen policy keeps answering items it should have
escalated. This module asks whether a slow state over *label-free* decision
statistics recovers what the frozen policy loses — and it asks against arms
that have strictly more information, so a win is a real win.

Four arms, differing only in what they may observe:

    StaticPolicy                    nothing                       (incumbent)
    GlialPolicy                     per-decision statistics       (label-free)
    RecalibratePolicy(delay=k)      labels k steps late           (fair baseline)
    RecalibratePolicy(delay=0)      labels immediately            (ceiling)

The glial mechanism is kindred's astrocyte rule (``angn.rs``: n-of-m hysteresis
over a window) applied to policy rather than to drives. A decision is
*anomalous* when the frozen policy would already escalate it; ``activate_at``
anomalous decisions in the last ``window`` raise the answer bar. Direction is
fixed a priori — drift makes the policy more conservative, never less.

``run_stream`` is the only component that knows about ordering. Policies are
pure functions of what they have observed, so the arms differ in information,
not in code path.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np

from src.decision import metrics as M
from src.decision.gate import fit_answer_threshold, fit_in_schema_threshold
from src.decision.scoring import aps_members, softmax_rows

TAU_CEILING = 0.999   # a probability threshold never reaches certainty


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
    answered at all. Arms that carry no in-schema boundary leave it ``None``.
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


class SlowState:
    """Windowed slow state over label-free anomalies.

    No hidden time constant: the window *is* the memory, and the streak counts
    consecutive windows at or above ``activate_at``. That makes the mechanism
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

    def calibrate(self, reference: Sequence[Observation]) -> int:
        """Set the activation bar from a healthy reference stream.

        The anomaly criterion fires at the *scorer's* base rate, not at a
        rare-event rate: on cached CLINC150 the median healthy decision sits
        exactly on ``tau_answer``, so half of healthy decisions look anomalous
        and a constant bar false-activates. The bar is therefore the most
        anomalous window seen while healthy, plus one.
        """
        counts = [sum(self._anomalous(o) for o in w) for w in _windows(reference, self._cfg.window)]
        if not counts:
            raise ValueError(
                f"reference stream of {len(reference)} is shorter than the "
                f"{self._cfg.window}-window it must calibrate")
        self._activate_at = min(self._cfg.window, max(counts) + 1)
        self._release_at = min(self._cfg.release_at, self._activate_at - 1)
        return self._activate_at

    def observe(self, obs: Observation) -> None:
        self._window.append(obs)
        self._streak = self._next_streak()

    def _next_streak(self) -> int:
        """Schmitt trigger: activate on strong evidence, hold through ambiguity.

        A single quiet window is not grounds to release — measured on real
        data, a shift that sits near ``drift_margin`` otherwise flickers, and
        the threshold snaps back to base while the shift is still in force.
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
        shift = min(self._cfg.max_shift, self._streak * self._cfg.tau_step)
        reference = self._cfg.reference_max_score
        if self._base.tau_in_schema is None or reference is None:
            schema_bar = self._base.tau_in_schema
        else:
            schema_bar = min(TAU_CEILING, self._base.tau_in_schema + shift * reference)
        return PolicyThresholds(
            tau_answer=min(TAU_CEILING, self._base.tau_answer + shift),
            k_clarify=max(1, self._base.k_clarify - int(self._streak > 0)),
            tau_in_schema=schema_bar)


class Policy(Protocol):
    """An information budget plus a threshold rule. No IO, no globals.

    ``label_delay`` is the contract the stream runner reads: ``None`` means the
    arm never sees a label, an int means labels arrive that many steps late.
    """

    name: str
    label_delay: Optional[int]

    def thresholds(self) -> PolicyThresholds: ...
    def observe(self, obs: Observation) -> None: ...
    def observe_labeled(self, obs: Observation, label: int) -> None: ...


class StaticPolicy:
    """Frozen thresholds — the incumbent the others must beat."""

    name = "static"
    label_delay: Optional[int] = None

    def __init__(self, base: PolicyThresholds):
        self._base = base

    def thresholds(self) -> PolicyThresholds:
        return self._base

    def observe(self, obs: Observation) -> None:
        return None

    def observe_labeled(self, obs: Observation, label: int) -> None:
        return None


class GlialPolicy:
    """Label-free slow state modulating the frozen thresholds."""

    name = "glial"
    label_delay: Optional[int] = None

    def __init__(self, base: PolicyThresholds,
                 cfg: Optional[SlowStateConfig] = None):
        self._state = SlowState(base, cfg or SlowStateConfig())

    def calibrate_reference(self, reference: Sequence[Observation]) -> int:
        """Feed the healthy reference stream the activation bar is fitted on."""
        return self._state.calibrate(reference)

    def thresholds(self) -> PolicyThresholds:
        return self._state.thresholds()

    def observe(self, obs: Observation) -> None:
        self._state.observe(obs)

    def observe_labeled(self, obs: Observation, label: int) -> None:
        return None


@dataclass(frozen=True)
class Labeled:
    """A decision plus the label that eventually arrived for it."""
    top_prob: float
    correct: bool
    max_score: float
    label: int


@dataclass(frozen=True)
class RecalibrateConfig:
    """What the label-driven arm may use.

    ``delay=0`` is the oracle ceiling, not deployable. ``n_options`` is the
    deployed label space size: with it, the arm can also refit the in-schema
    boundary, because a delayed label outside the option set marks that item as
    out of schema — the same evidence the frozen boundary was fitted on.
    """
    delay: int = 200
    refit_every: int = 50
    n_options: Optional[int] = None
    precision: float = 0.90
    min_answered: int = 30
    min_group: int = 20


class RecalibratePolicy:
    """Refits thresholds from labels that arrive ``delay`` steps late.

    The fair baseline: same machinery as the incumbent, strictly more
    information. A glial win has to survive this comparison.
    """

    def __init__(self, base: PolicyThresholds, cfg: RecalibrateConfig):
        if cfg.delay < 0 or cfg.refit_every < 2:
            raise ValueError("delay must be >= 0 and refit_every >= 2")
        self._base, self._cfg = base, cfg
        self._labeled: List[tuple] = []
        self._tau, self._tau_in = base.tau_answer, base.tau_in_schema
        self.label_delay = cfg.delay
        self.name = f"recalibrate(delay={cfg.delay})"

    def thresholds(self) -> PolicyThresholds:
        return PolicyThresholds(self._tau, self._base.k_clarify, self._tau_in)

    def observe(self, obs: Observation) -> None:
        return None

    def observe_labeled(self, obs: Observation, label: int) -> None:
        self._labeled.append(Labeled(obs.top_prob, obs.pred == label,
                                     obs.max_score, label))
        if len(self._labeled) >= self._cfg.refit_every:
            self._refit()

    def _refit(self) -> None:
        recent = self._labeled[-self._cfg.refit_every:]
        self._tau = fit_answer_threshold(
            np.array([r.top_prob for r in recent]),
            np.array([r.correct for r in recent]),
            self._cfg.precision, self._cfg.min_answered)
        if self._cfg.n_options is not None:
            self._tau_in = self._refit_in_schema(recent)

    def _refit_in_schema(self, recent: List[Labeled]) -> Optional[float]:
        inside = np.array([r.max_score for r in recent if r.label < self._cfg.n_options])
        outside = np.array([r.max_score for r in recent if r.label >= self._cfg.n_options])
        if min(len(inside), len(outside)) < self._cfg.min_group:
            return self._tau_in
        return fit_in_schema_threshold(inside[:, None], outside[:, None])


@dataclass(frozen=True)
class Stream:
    """A labelled score stream and the point where its distribution changes.

    ``shift_at`` lives here, not in ``StreamConfig``: it describes the data
    under test, not how the runner reports on it.
    """
    scores: np.ndarray
    labels: np.ndarray
    shift_at: int


@dataclass(frozen=True)
class StreamConfig:
    """The frozen calibration the runner reports against."""
    t_prob: float
    qhat: float
    k_clarify: int
    coverage_target: float = 0.8
    recovery_window: int = 100


def run_stream(policy: Policy, stream: Stream, cfg: StreamConfig) -> Dict:
    """Replay a stream in order; return per-phase metrics and recovery latency.

    Every action mirrors ``ConformalGate.decide``, including the in-schema
    check, so the arms are compared on the policy the gate actually ships.
    """
    scores, labels = _as_stream_arrays(stream)
    records: List[Dict] = []
    observations: List[Observation] = []
    for i, (row, label) in enumerate(zip(scores, labels)):
        obs, record = _step(policy, row, label, cfg)
        observations.append(obs)
        records.append(record)
        delay = policy.label_delay
        if delay is not None and i >= delay:
            policy.observe_labeled(observations[i - delay], labels[i - delay])
    return _summarize(policy, records, stream.shift_at, cfg)


def _as_stream_arrays(stream: Stream) -> tuple:
    scores = np.asarray(stream.scores, dtype=np.float64)
    labels = np.asarray(stream.labels, dtype=int).reshape(-1)
    if len(scores) != len(labels):
        raise ValueError("stream scores and labels must be the same length")
    if not 0 < stream.shift_at < len(scores):
        raise ValueError("shift_at must split the stream into two non-empty parts")
    return scores, labels


def observe_row(row: np.ndarray, cfg: StreamConfig) -> tuple:
    """The label-free statistics of one decision row — the state's only input.

    Public so a caller can build the healthy reference stream the activation
    bar is calibrated on, using exactly the statistics the state will see.
    """
    P = softmax_rows(row, cfg.t_prob)[0]
    members = aps_members(P[None, :], cfg.qhat)[0]
    obs = Observation(top_prob=float(P.max()), set_size=len(members),
                      entropy=float(-(P * np.log(P + 1e-12)).sum()),
                      pred=int(P.argmax()), max_score=float(row.max()))
    return obs, members


def _windows(items: Sequence, window: int) -> List[List]:
    return [list(items[i:i + window]) for i in range(len(items) - window + 1)]


def _step(policy: Policy, row: np.ndarray, label: int,
          cfg: StreamConfig) -> tuple:
    """One decision: probabilities, conformal set, action, observation."""
    obs, members = observe_row(row, cfg)
    policy.observe(obs)
    thresholds = policy.thresholds()
    action = _action(obs, thresholds)
    record = {"correct": float(obs.pred == label),
              "covered": float(label in members),
              "escalated": float(action == "escalate"),
              "wrong_answer": float(action == "answer" and obs.pred != label),
              "top_prob": obs.top_prob,
              "tau_answer": thresholds.tau_answer,
              "tau_in_schema": (np.nan if thresholds.tau_in_schema is None
                                else thresholds.tau_in_schema)}
    return obs, record


def _action(obs: Observation, thresholds: PolicyThresholds) -> str:
    """``ConformalGate.decide`` order: in-schema, confidence, set size."""
    if thresholds.tau_in_schema is not None and obs.max_score < thresholds.tau_in_schema:
        return "escalate"
    if obs.top_prob >= thresholds.tau_answer:
        return "answer"
    return "clarify" if obs.set_size <= thresholds.k_clarify else "escalate"


def _phase_block(records: Sequence[Dict], coverage_target: float) -> Dict:
    """Metrics plus the policy's own state during this phase.

    The threshold statistics matter: an end-of-stream snapshot can read "base"
    while the phase was governed by a raised bar, which makes the metrics look
    impossible. These are the numbers that explain them.
    """
    conf = np.array([r["top_prob"] for r in records])
    hit = np.array([r["correct"] for r in records])
    schema = np.array([r["tau_in_schema"] for r in records])
    return {"n": len(records),
            "accuracy": float(hit.mean()),
            "selective_acc": float(M.selective_at_coverage(conf, hit, coverage_target)),
            "coverage": float(np.mean([r["covered"] for r in records])),
            "escalation_rate": float(np.mean([r["escalated"] for r in records])),
            "wrong_answer_rate": float(np.mean([r["wrong_answer"] for r in records])),
            "aurc": float(M.aurc(conf, hit)),
            "tau_answer_mean": float(np.mean([r["tau_answer"] for r in records])),
            "tau_in_schema_mean": float(np.nanmean(schema)),
            "tau_in_schema_max": float(np.nanmax(schema))}


def _recovery_latency(records: Sequence[Dict], shift_at: int,
                      window: int, tol: float = 0.02) -> Optional[int]:
    """Items after the shift until rolling accuracy returns to phase-1 level."""
    hits = np.array([r["correct"] for r in records])
    baseline = float(hits[:shift_at].mean())
    for start in range(shift_at, len(hits) - window + 1):
        if hits[start:start + window].mean() >= baseline - tol:
            return start - shift_at
    return None


def _summarize(policy: Policy, records: Sequence[Dict], shift_at: int,
               cfg: StreamConfig) -> Dict:
    phase2 = records[shift_at:]
    hit2 = np.array([r["correct"] for r in phase2])
    conf2 = np.array([r["top_prob"] for r in phase2])
    return {"policy": policy.name,
            "phase1": _phase_block(records[:shift_at], cfg.coverage_target),
            "phase2": _phase_block(phase2, cfg.coverage_target),
            "phase2_selective_ci": M.bootstrap_ci(
                lambda h, c: M.selective_at_coverage(c, h, cfg.coverage_target),
                hit2, conf2, resamples=500),
            "recovery_latency": _recovery_latency(records, shift_at,
                                                  cfg.recovery_window)}
