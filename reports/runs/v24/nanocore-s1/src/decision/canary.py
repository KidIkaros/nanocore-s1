"""Canary release: which arm a request goes to, and whether to keep it.

The model owns the *decision*, not the transport. Splitting live traffic is the
deployer's job — a mesh, a load balancer, a feature-flag system — so this module
provides only the two things that are decisions:

- ``arm_for`` — deterministic bucketing. The same request key always lands in
  the same arm, so a user does not flip versions mid-session and the two arms
  stay comparable.
- ``canary_verdict`` — the guardrail readout. Compare the canary arm against
  control over the same window and say promote / hold / rollback, naming the
  metric that decided it.

Metrics absent from either arm are reported ``unchecked`` rather than treated
as passing — a guard that fails open is not a guard.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass(frozen=True)
class GuardrailConfig:
    """Tolerances for the canary arm. ``None`` disables that check."""

    max_unsafe_delta: Optional[float] = 0.02
    max_escalate_delta: Optional[float] = 0.05
    max_latency_ratio: Optional[float] = 1.5
    min_n: int = 100


@dataclass(frozen=True)
class Verdict:
    """``decision`` is one of promote / hold / rollback; ``reason`` names the metric."""

    decision: str
    reason: str
    deltas: Dict[str, float] = field(default_factory=dict)
    checked: List[str] = field(default_factory=list)
    unchecked: List[str] = field(default_factory=list)


def arm_for(key: str, fraction: float, salt: str = "") -> str:
    """``"canary"`` for a ``fraction`` of keys, deterministically.

    Hash-based rather than counter-based: a counter is unstable across
    processes and restarts, so the same input would land in different arms and
    the comparison would be between two arbitrary slices.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    digest = hashlib.sha256(f"{salt}\x00{key}".encode()).digest()
    return "canary" if int.from_bytes(digest[:8], "big") < fraction * 2 ** 64 \
        else "control"


def _run_checks(control: Dict[str, float], canary: Dict[str, float],
                cfg: GuardrailConfig) -> tuple:
    """(deltas, checked, unchecked, breaches) for every enabled guardrail.

    Lower is better for the delta-checked metrics, so a breach is a *positive*
    delta past tolerance. A metric absent from either arm lands in ``unchecked``
    rather than silently passing.
    """
    deltas: Dict[str, float] = {}
    checked: List[str] = []
    unchecked: List[str] = []
    breaches: List[str] = []

    delta_checks = (("unsafe_rate", cfg.max_unsafe_delta),
                    ("escalate_rate", cfg.max_escalate_delta))
    for metric, tol in delta_checks:
        if tol is None:
            continue
        if metric not in control or metric not in canary:
            unchecked.append(metric)
            continue
        delta = canary[metric] - control[metric]
        deltas[metric] = delta
        checked.append(metric)
        if delta > tol:
            breaches.append(f"{metric} {control[metric]:.3f} -> "
                            f"{canary[metric]:.3f} (+{delta:.3f} > {tol})")

    if cfg.max_latency_ratio is not None:
        if "latency_p95" in control and "latency_p95" in canary:
            ratio = canary["latency_p95"] / max(control["latency_p95"], 1e-9)
            deltas["latency_ratio"] = ratio
            checked.append("latency_p95")
            if ratio > cfg.max_latency_ratio:
                breaches.append(f"latency_p95 x{ratio:.2f} > {cfg.max_latency_ratio}")
        else:
            unchecked.append("latency_p95")

    return deltas, checked, unchecked, breaches


def canary_verdict(control: Dict[str, float], canary: Dict[str, float],
                   cfg: GuardrailConfig) -> Verdict:
    """Compare the two arms and decide. Insufficient evidence holds, not promotes.

    ``control`` and ``canary`` carry ``n`` plus any of ``unsafe_rate``,
    ``escalate_rate``, ``latency_p95``.
    """
    deltas, checked, unchecked, breaches = _run_checks(control, canary, cfg)
    n = min(control.get("n", 0), canary.get("n", 0))
    if n < cfg.min_n:
        return Verdict("hold", f"insufficient evidence: {n} < {cfg.min_n} per arm",
                       deltas, checked, unchecked)
    if breaches:
        return Verdict("rollback", "; ".join(breaches), deltas, checked, unchecked)
    return Verdict("promote", "no guardrail breached", deltas, checked, unchecked)
