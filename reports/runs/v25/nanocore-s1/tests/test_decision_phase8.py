"""Phase 8 — parity metrics, the memory gate, and the privacy claim.

The acceptance criteria are specific: parity within tolerance, measured latency,
memory gate passed, inputs never leave the device. The first three are
measurements; the last is a *claim*, and a claim that is only asserted in prose
is not evidence — so it is tested by removing the ability to make a call.
"""
import json
import socket

import numpy as np
import pytest

from src.decision import guard
from src.decision.cli import main
from src.decision.parity import (ParityTolerance, cosine_agreement,
                                 decision_agreement, latency_summary,
                                 parity_report)
from src.decision.schema import Prediction

# ── parity metrics ───────────────────────────────────────────────────────────


def _unit_rows(rng, n, d):
    a = rng.normal(0, 1, (n, d))
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_identical_embeddings_are_perfect_parity():
    rng = np.random.default_rng(0)
    A = _unit_rows(rng, 50, 32)
    out = cosine_agreement(A, A.copy())
    assert out["mean"] == pytest.approx(1.0)
    assert out["min"] == pytest.approx(1.0)


def test_the_mean_does_not_hide_one_bad_vector():
    """A mean of 0.999 hiding a single vector at 0.4 is a real incompatibility,
    and the mean is exactly the statistic that would hide it."""
    rng = np.random.default_rng(1)
    A = _unit_rows(rng, 40, 32)
    B = A.copy()
    B[7] = rng.normal(0, 1, 32)
    B[7] /= np.linalg.norm(B[7])
    out = cosine_agreement(A, B)
    assert out["mean"] > 0.95          # looks fine
    assert out["min"] < 0.8            # is not
    assert out["worst_index"] == 7


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="must align"):
        cosine_agreement(np.zeros((3, 8)), np.zeros((4, 8)))


def _pred(action, members):
    return Prediction(qtype="choice", labels=members,
                      probabilities={m: 1 / len(members) for m in members},
                      answer_confidence=0.9, entropy_confidence=0.1,
                      max_score=1.0, prediction_set=members, action=action,
                      alpha=0.1)


def test_decision_agreement_counts_actions_and_set_overlap():
    a = [_pred("answer", ["x"]), _pred("escalate", ["x", "y"])]
    b = [_pred("answer", ["x"]), _pred("escalate", ["y", "z"])]
    out = decision_agreement(a, b)
    assert out["action_agreement"] == 1.0
    assert out["mean_set_jaccard"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_latency_reports_percentiles_not_just_a_mean():
    out = latency_summary([10, 10, 10, 10, 200])
    assert out["p50"] == 10 and out["p99"] > out["p50"] and out["max"] == 200


def test_parity_report_fails_when_cosine_is_short():
    cos = {"mean": 0.97, "min": 0.9, "p05": 0.95}
    dec = {"action_agreement": 0.99, "mean_set_jaccard": 0.9}
    rep = parity_report(cos, dec, tol=ParityTolerance())
    assert not rep["cosine_ok"] and rep["actions_ok"] and not rep["parity_ok"]


def test_parity_report_passes_within_tolerance():
    cos = {"mean": 0.995, "min": 0.98, "p05": 0.99}
    dec = {"action_agreement": 0.98, "mean_set_jaccard": 0.95}
    rep = parity_report(cos, dec, {"p50": 90.0}, {"p50": 40.0})
    assert rep["parity_ok"] and rep["latency"]["p50_ratio"] == pytest.approx(40 / 90)


# ── the memory gate ──────────────────────────────────────────────────────────


def test_gate_refuses_when_memory_is_short(monkeypatch):
    monkeypatch.setattr(guard, "available_gb", lambda: 1.6)
    check = guard.check_memory(2.6)
    assert not check.ok
    assert "1.6 GB free" in check.reason and "refusing" in check.reason


def test_gate_admits_when_there_is_headroom(monkeypatch):
    monkeypatch.setattr(guard, "available_gb", lambda: 32.0)
    assert guard.check_memory(1.1).ok


def test_unknown_memory_skips_visibly(monkeypatch):
    """A guard that guesses is worse than no guard — but it must say so."""
    monkeypatch.setattr(guard, "available_gb", lambda: None)
    check = guard.check_memory(1.1)
    assert check.ok and check.available_gb is None
    assert "unknown" in check.reason


def test_headroom_is_applied(monkeypatch):
    monkeypatch.setattr(guard, "available_gb", lambda: 2.0)
    assert guard.check_memory(1.0, headroom=1.5).ok        # needs 1.5, has 2.0
    assert not guard.check_memory(1.5, headroom=1.5).ok    # needs 2.25, has 2.0


def test_negative_requirement_rejected():
    with pytest.raises(ValueError, match="negative"):
        guard.check_memory(-1.0)


def test_headroom_is_applied_exactly_once(tmp_path, monkeypatch):
    """A 40 GB file needs 69 GB free (40 × 1.15 activations, then × 1.5
    headroom), not 90. Applying the headroom in both places is how a guard
    reports a number an operator stops trusting — caught in the first dry run."""
    big = tmp_path / "big.gguf"
    with big.open("wb") as fh:
        fh.truncate(40 * 2 ** 30)                   # sparse: no disk used
    est = guard.model_gb(big)
    assert est == pytest.approx(46.0, abs=0.5)      # 40 x 1.15, headroom NOT here
    monkeypatch.setattr(guard, "available_gb", lambda: 100.0)
    assert guard.check_memory(est).ok
    assert est * guard.DEFAULT_HEADROOM == pytest.approx(69.0, abs=0.5)


def test_available_gb_reads_a_real_number_locally():
    """Not a mock: this host can answer, so the gate is exercised for real."""
    avail = guard.available_gb()
    assert avail is None or avail > 0


# ── privacy: the claim, tested ───────────────────────────────────────────────


def test_a_decision_makes_no_network_call(tmp_path, capsys):
    """"Inputs never leave the device" is a claim, and an untested claim is
    prose. Here the ability to open a socket is removed; if any part of the
    decision path tried to call out, this fails instead of quietly succeeding."""
    data = tmp_path / "t.csv"
    rows = ["text,label"]
    for c in ("cancel", "book"):
        rows += [f"{c} request {j},{c}" for j in range(40)]
    data.write_text("\n".join(rows) + "\n")
    main(["adapt", "--data", str(data), "--out", str(tmp_path / "run"),
          "--backend", "stub", "--min-cal", "10", "--epochs", "2"])
    capsys.readouterr()

    def _blocked(*a, **kw):
        raise AssertionError("the decision path tried to open a socket")

    original = socket.socket
    socket.socket = _blocked
    try:
        main(["decide", "cancel request 3", "--options", "cancel,book",
              "--backend", "stub", "--bundle", str(tmp_path / "run" / "bundle"),
              "--json"])
    finally:
        socket.socket = original
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] in ("answer", "clarify", "escalate", "abstain")


def test_no_module_in_the_decision_path_imports_a_network_client():
    """The structural half of the same claim: nothing in the decision layer
    even has the ability to reach out, `serve` excluded — it *is* the server."""
    import src.decision.model as model_mod
    for mod in (model_mod,):
        src = mod.__file__
        text = open(src).read()
        for client in ("import requests", "import urllib", "import httpx"):
            assert client not in text, f"{src} imports {client}"
