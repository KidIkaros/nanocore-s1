"""Monitoring — two layers over the prediction log.

Phase 5 (roadmap): a deployed decision layer must report whether it is still
working. This module reads the JSONL ``PredictionLogger`` output and computes:

- **Operational**: latency percentiles (never means alone), throughput,
  request volume.
- **ML**: action distribution, top-prob distribution, set-size distribution —
  **per slice** (e.g. per qtype, per label, per caller tag).
- **Drift**: reference-vs-recent comparison naming the shift type —
  score/top-prob shift (covariate), label/action shift (label), accuracy shift
  when labels arrive (concept).

Simple statistics first (KS test, rate deltas) — no detector framework needed
to know when "the distribution moved" is true. Alert = named check crossed a
named threshold; a monitor that never says *which* signal moved is useless.

Everything is numpy + the log file. No model loading, no weights.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ── log reading ──────────────────────────────────────────────────────────────

def read_log(path) -> List[dict]:
    """Parse a PredictionLogger JSONL file. Tolerates a trailing partial line."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # torn write at EOF — skip, don't die
    return out


# ── operational layer ────────────────────────────────────────────────────────

def operational_metrics(records: List[dict]) -> Dict:
    """Latency percentiles, throughput, volume — from a window of records."""
    if not records:
        return {"requests": 0}
    lat = np.asarray([r.get("latency_ms", 0.0) for r in records])
    span = max(r["ts"] for r in records) - min(r["ts"] for r in records)
    return {
        "requests": len(records),
        "window_s": round(span, 1),
        "throughput_rps": round(len(records) / max(span, 1e-9), 3),
        "latency_ms": {"p50": float(np.percentile(lat, 50)),
                       "p95": float(np.percentile(lat, 95)),
                       "p99": float(np.percentile(lat, 99)),
                       "mean": float(lat.mean())},
    }


# ── ML layer ─────────────────────────────────────────────────────────────────

def ml_metrics(records: List[dict],
               slice_key: Optional[str] = None) -> Dict:
    """Action/top-prob/set-size distributions, optionally per slice.

    ``slice_key`` names a record field to group by (e.g. ``"qtype"`` or a
    caller-injected tag). Returns both the aggregate and ``per_slice``.
    """
    def block(recs):
        if not recs:
            return {}
        acts = {}
        for r in recs:
            acts[r.get("action", "?")] = acts.get(r.get("action", "?"), 0) + 1
        top = np.asarray([r.get("top_prob", np.nan) for r in recs])
        sets = np.asarray([len(r.get("prediction_set") or []) for r in recs])
        return {"n": len(recs),
                "action_dist": {k: v / len(recs) for k, v in sorted(acts.items())},
                "top_prob_mean": float(np.nanmean(top)),
                "top_prob_p05": float(np.nanpercentile(top, 5)),
                "set_size_mean": float(sets.mean()),
                "escalate_rate": acts.get("escalate", 0) / len(recs)}

    out = {"aggregate": block(records)}
    if slice_key:
        groups: Dict[str, list] = {}
        for r in records:
            groups.setdefault(str(r.get(slice_key, "?")), []).append(r)
        out["per_slice"] = {k: block(v) for k, v in sorted(groups.items())}
    return out


# ── drift ────────────────────────────────────────────────────────────────────

def ks_2samp(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (numpy, no scipy)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.abs(ca - cb).max())


@dataclass
class DriftReport:
    covariate_ks: float      # top_prob distribution shift
    action_shift: Dict       # per-action rate delta (recent - reference)
    set_size_ks: float
    shift_types: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


def detect_drift(reference: List[dict], recent: List[dict],
                 ks_threshold: float = 0.15,
                 action_delta: float = 0.20) -> DriftReport:
    """Reference window vs recent window. Names the shift types it finds.

    - covariate shift: the top-prob distribution moved (inputs look different)
    - label/action shift: action rates moved (e.g. escalate rate doubled)
    - concept drift needs labels — see ``detect_drift`` callers adding labeled
      windows; the detector flags the *signal* it can see here.
    """
    ref_top = [r.get("top_prob", 0.0) for r in reference]
    rec_top = [r.get("top_prob", 0.0) for r in recent]
    ref_set = [len(r.get("prediction_set") or []) for r in reference]
    rec_set = [len(r.get("prediction_set") or []) for r in recent]

    k_top = ks_2samp(ref_top, rec_top)
    k_set = ks_2samp(ref_set, rec_set)

    def rates(recs):
        d = {}
        for r in recs:
            d[r.get("action", "?")] = d.get(r.get("action", "?"), 0) + 1
        return {k: v / max(len(recs), 1) for k, v in d.items()}

    ra, rb = rates(reference), rates(recent)
    delta = {a: rb.get(a, 0.0) - ra.get(a, 0.0)
             for a in set(ra) | set(rb)}

    rep = DriftReport(covariate_ks=k_top, action_shift=delta, set_size_ks=k_set)
    if k_top > ks_threshold:
        rep.shift_types.append("covariate")
        rep.alerts.append(f"top_prob KS={k_top:.3f} > {ks_threshold}")
    if k_set > ks_threshold:
        rep.alerts.append(f"set_size KS={k_set:.3f} > {ks_threshold}")
    for a, d in delta.items():
        if abs(d) > action_delta:
            rep.shift_types.append("label" if a in ("answer", "escalate") else a)
            rep.alerts.append(f"action '{a}' rate moved {d:+.2f}")
    rep.shift_types = sorted(set(rep.shift_types))
    return rep


# ── the monitor ──────────────────────────────────────────────────────────────

class Monitor:
    """Windowed monitor over a prediction log. ``check()`` returns a status
    dict; ``alert`` entries are named, thresholded findings — the thing an
    operator (or Phase 6's retraining trigger) consumes."""

    def __init__(self, log_path, reference_frac: float = 0.5,
                 latency_p99_ms: float = 500.0,
                 escalate_rate_max: float = 0.6,
                 ks_threshold: float = 0.15):
        self.log_path = Path(log_path)
        self.reference_frac = reference_frac
        self.latency_p99_ms = latency_p99_ms
        self.escalate_rate_max = escalate_rate_max
        self.ks_threshold = ks_threshold

    def check(self, slice_key: Optional[str] = None) -> Dict:
        records = read_log(self.log_path)
        status = {"n": len(records), "alerts": []}
        if not records:
            status["alerts"].append("empty prediction log")
            return status

        ops = operational_metrics(records)
        ml = ml_metrics(records, slice_key=slice_key)
        status["operational"] = ops
        status["ml"] = ml

        if ops["latency_ms"]["p99"] > self.latency_p99_ms:
            status["alerts"].append(
                f"p99 latency {ops['latency_ms']['p99']:.0f}ms > {self.latency_p99_ms}ms")
        if ml["aggregate"]["escalate_rate"] > self.escalate_rate_max:
            status["alerts"].append(
                f"escalate_rate {ml['aggregate']['escalate_rate']:.2f} > "
                f"{self.escalate_rate_max}")

        n_ref = max(10, int(len(records) * self.reference_frac))
        if len(records) >= 2 * n_ref:
            drift = detect_drift(records[:n_ref], records[-n_ref:],
                                 ks_threshold=self.ks_threshold)
            status["drift"] = {"covariate_ks": drift.covariate_ks,
                               "set_size_ks": drift.set_size_ks,
                               "shift_types": drift.shift_types,
                               "action_shift": drift.action_shift}
            status["alerts"] += drift.alerts
        return status
