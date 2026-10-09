"""When to retrain — the other half of the retraining pipeline.

``adapt()`` is the stateless train: give it data and a config, get a bundle.
This module decides *when* to call it. Two cadences, either or both enabled:

- **scheduled** — every N new labeled samples, whatever the monitor sees.
  Predictable cost, and it retrains through a slow drift that never crosses an
  alert threshold.
- **drift-triggered** — the monitor alerted. Faster, but it needs a floor:
  retraining on a shifted window can *hurt* (measured: the delayed-label arm
  reached 0.305 unsafe answers against static's 0.143), so a trigger that fires
  on the first alert thrashes. ``drift_min_samples`` is that floor.

The decision is a pure function of (new labeled samples, drift report, config),
so the caller owns the counters and this stays testable without a clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.decision.monitor import DriftReport


@dataclass(frozen=True)
class CadenceConfig:
    """Retrain thresholds. ``scheduled_every=None`` disables the scheduled path."""

    scheduled_every: Optional[int] = 1000
    drift_min_samples: int = 500


@dataclass(frozen=True)
class Trigger:
    """Whether to retrain, which cadence fired, and why — the reason is the log line."""

    retrain: bool
    cadence: str
    reason: str


def should_retrain(new_samples: int, drift: Optional[DriftReport],
                   cfg: CadenceConfig) -> Trigger:
    """First matching cadence wins; drift outranks scheduled (faster reaction).

    A drift alert below the floor holds rather than retrains, and says so —
    the alternative is a retrain loop that fires on every shifted window.
    """
    if new_samples < 0:
        raise ValueError(f"new_samples cannot be negative, got {new_samples}")
    drifted = bool(drift is not None and drift.alerts)

    if drifted and new_samples >= cfg.drift_min_samples:
        return Trigger(True, "drift", f"drift: {'; '.join(drift.alerts[:2])}")
    if cfg.scheduled_every and new_samples >= cfg.scheduled_every:
        return Trigger(True, "scheduled",
                       f"scheduled: {new_samples} new labeled samples "
                       f">= {cfg.scheduled_every}")
    if drifted:
        return Trigger(False, "none",
                       f"drift alert but {new_samples} new labeled samples "
                       f"< {cfg.drift_min_samples} floor")
    return Trigger(False, "none",
                   f"no cadence reached ({new_samples} new labeled samples)")
