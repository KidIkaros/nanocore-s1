"""Monitoring — drift detection, slicing, thresholds. Pure logic, no weights."""
import json

import numpy as np
import pytest

from src.decision.monitor import (Monitor, detect_drift, ks_2samp, ml_metrics,
                                  operational_metrics, read_log)


def _rec(ts, action="answer", top=0.9, setn=1, lat=10.0, tag="a"):
    return {"ts": ts, "action": action, "top_prob": top,
            "prediction_set": ["x"] * setn, "latency_ms": lat, "tag": tag}


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_read_log_tolerates_trailing_partial(tmp_path):
    p = tmp_path / "l.jsonl"
    _write(p, [_rec(1), _rec(2)])
    with p.open("a") as f:
        f.write('{"ts": 3, "act')  # torn write
    assert len(read_log(p)) == 2


def test_operational_metrics_percentiles(tmp_path):
    recs = [_rec(i, lat=float(i)) for i in range(1, 101)]
    m = operational_metrics(recs)
    assert m["requests"] == 100
    assert m["latency_ms"]["p50"] == pytest.approx(50.5, abs=1)
    assert m["latency_ms"]["p99"] > m["latency_ms"]["p50"]


def test_ml_metrics_slices(tmp_path):
    recs = [_rec(i, action="answer", tag="alpha") for i in range(10)]
    recs += [_rec(i, action="escalate", tag="beta") for i in range(10)]
    m = ml_metrics(recs, slice_key="tag")
    assert m["per_slice"]["alpha"]["escalate_rate"] == 0.0
    assert m["per_slice"]["beta"]["escalate_rate"] == 1.0
    assert m["aggregate"]["action_dist"]["answer"] == 0.5


def test_ks_2samp_sanity():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 500), rng.normal(0, 1, 500)
    assert ks_2samp(a, b) < 0.1
    c = rng.normal(3, 1, 500)
    assert ks_2samp(a, c) > 0.8


def test_detect_drift_names_covariate():
    rng = np.random.default_rng(0)
    ref = [_rec(i, top=float(rng.normal(0.9, 0.05))) for i in range(200)]
    recent = [_rec(i, top=float(rng.normal(0.5, 0.1))) for i in range(200)]
    rep = detect_drift(ref, recent)
    assert "covariate" in rep.shift_types
    assert rep.alerts


def test_detect_drift_names_action_shift():
    ref = [_rec(i, action="answer") for i in range(200)]
    recent = [_rec(i, action="escalate") for i in range(200)]
    rep = detect_drift(ref, recent)
    assert "label" in rep.shift_types or "escalate" in rep.shift_types


def test_monitor_alerts_on_latency_and_drift(tmp_path):
    p = tmp_path / "l.jsonl"
    rng = np.random.default_rng(1)
    recs = [_rec(i, top=0.95) for i in range(200)]          # reference
    recs += [_rec(200 + i, top=0.4, lat=900.0)              # drifted + slow
             for i in range(200)]
    _write(p, recs)
    status = Monitor(p).check()
    assert any("p99" in a for a in status["alerts"])
    assert "covariate" in status["drift"]["shift_types"]


def test_monitor_clean_log_no_alerts(tmp_path):
    p = tmp_path / "l.jsonl"
    recs = [_rec(i, top=0.9 + 0.01 * (i % 3)) for i in range(400)]
    _write(p, recs)
    status = Monitor(p, latency_p99_ms=100.0).check()
    assert status["alerts"] == []
