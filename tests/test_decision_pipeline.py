"""Pipeline tests — gate/scorer/cache/model on mocks AND the cached real
CLINC150 score matrices (reports/runs/s1_policy_v2/policy_v2_scores.npz).

The real-data regression asserts the implemented gate reproduces the kernel's
measured v2 numbers (dual temps, τ_answer, qhat, action distribution) — the
code under test is the same math that produced them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.decision.cache import DecisionCache, decision_key
from src.decision.gate import ConformalGate
from src.decision.model import DecisionModel
from src.decision.schema import Question, State
from src.decision.scoring import (CosineScorer, TaskHead, aps_members,
                                  fit_temperature, mass_needed, softmax_rows)

NPZ = Path("reports/runs/s1_policy_v2/policy_v2_scores.npz")
OOS_LABEL = 80                      # "oos" sits at index 80 in the CLINC label list
REAL = pytest.mark.skipif(not NPZ.exists(), reason="cached score matrix missing")


def _clinc():
    d = np.load(NPZ)
    # label index -> score-column index (OOS -> -1), per the kernel's to_cols
    col_of = {v: c for c, v in enumerate(i for i in range(151) if i != OOS_LABEL)}
    to_cols = lambda y: np.array([col_of.get(int(v), -1) for v in y])
    rng = np.random.default_rng(13)                 # kernel's split seed
    perm = rng.permutation(len(d["y_val"]))
    return d, to_cols, perm[: len(perm) // 2], perm[len(perm) // 2:]


# ── pure math ────────────────────────────────────────────────────────────

def test_softmax_rows_and_aps():
    P = softmax_rows(np.array([[3.0, 1.0, 0.0]]), 1.0)
    assert P.shape == (1, 3) and abs(P.sum() - 1) < 1e-9
    members = aps_members(P, qhat=0.5)
    assert members[0][0] == 0                     # argmax first
    assert len(members[0]) <= 2                   # 0.93 mass on argmax → set {0}
    # mass_needed: true class at rank r needs its cumulative mass
    m = mass_needed(P, np.array([1]))
    assert abs(m[0] - (P[0, 0] + P[0, 1])) < 1e-9


def test_fit_temperature_floor():
    # a nearly-perfect scorer makes log-loss prefer extreme sharpening;
    # the floor must stop it (the T=0.02 CLINC pathology)
    rng = np.random.default_rng(0)
    sc = np.zeros((400, 5)); sc[np.arange(400), 0] = 10.0
    sc += rng.normal(0, 0.1, sc.shape)
    t = fit_temperature(sc, np.zeros(400, int), floor=0.25)
    assert t >= 0.25


def test_cosine_scorer_ranks():
    s = CosineScorer()
    state = np.array([1.0, 0.0])
    opts = np.array([[1.0, 0.0], [0.0, 1.0]])
    sc = s.scores(state, opts)
    assert sc[0] > sc[1] and abs(sc[0] - 1.0) < 1e-9


# ── gate mechanics ───────────────────────────────────────────────────────

def test_gate_min_n_fails_closed():
    g = ConformalGate(min_n=200)
    with pytest.raises(ValueError, match="min_n"):
        g.calibrate(np.zeros((50, 4)), np.zeros(50, int))


def test_gate_uncalibrated_refuses():
    g = ConformalGate(min_n=2)
    with pytest.raises(RuntimeError, match="not calibrated"):
        g.decide(np.ones(3), ["a", "b", "c"])


def test_gate_actions_on_toy_scores():
    rng = np.random.default_rng(1)
    k, n = 6, 400
    sc = rng.normal(0, 0.5, (n, k))
    sc[np.arange(n), np.arange(n) % k] += 6.0    # mostly confident, correct
    y = np.arange(n) % k
    g = ConformalGate(alpha=0.10, min_n=200)
    g.calibrate(sc, y)
    # confident + in-schema → answer
    r = g.decide(sc[0], list("abcdef"))
    assert r.action == "answer" and r.prediction_set
    # out-of-schema (no tau fit → can't trigger) but diffuse → escalate path
    g2 = ConformalGate(alpha=0.10, min_n=200)
    g2.calibrate(sc, y)
    g2.tau_in_schema = 99.0                        # everything is OOS
    assert g2.decide(sc[0], list("abcdef")).action == "escalate"


# ── cache ────────────────────────────────────────────────────────────────

def test_cache_round_trip_and_key(tmp_path):
    c = DecisionCache(tmp_path)
    q = Question(qtype="choice", options=["a", "b"])
    k1 = decision_key(State.text("hello"), q, "enc", {"alpha": 0.1})
    assert c.get(k1) is None
    c.put(k1, {"action": "answer"})
    assert c.get(k1) == {"action": "answer"}
    # key changes with knobs, not with State wrapper vs bare text
    assert k1 == decision_key("hello", q, "enc", {"alpha": 0.1})
    assert k1 != decision_key("hello", q, "enc", {"alpha": 0.2})


# ── model with stub encoder ──────────────────────────────────────────────

class _StubEncoder:
    model_name = "stub"
    def __init__(self):
        self.calls = 0
    def encode_state(self, items):
        self.calls += 1
        import torch
        return torch.ones(len(items), 4)
    def encode_options(self, texts):
        import torch
        out = torch.zeros(len(texts), 4)
        for i in range(len(texts)):
            out[i, i % 4] = 1.0
        return out


def test_decision_model_gated_and_cached(tmp_path):
    rng = np.random.default_rng(2)
    k, n = 4, 400
    sc = rng.normal(0, 0.5, (n, k)); sc[np.arange(n), np.arange(n) % k] += 6.0
    g = ConformalGate(alpha=0.10, min_n=200)
    g.calibrate(sc, np.arange(n) % k)
    enc = _StubEncoder()
    m = DecisionModel(encoder=enc, scorer=CosineScorer(), gate=g,
                      cache=DecisionCache(tmp_path))
    q = Question(qtype="choice", options=list("abcd"))
    p1 = m.decide("hello", q)
    assert p1.action in ("answer", "clarify", "escalate", "abstain")
    assert p1.prediction_set and p1.alpha == 0.10
    assert set(p1.probabilities) == set("abcd")
    calls_after_first = enc.calls
    p2 = m.decide("hello", q)                      # cache hit
    assert enc.calls == calls_after_first
    assert p2.action == p1.action


# ── the real regression: gate reproduces the kernel's v2 numbers ─────────

@REAL
def test_gate_reproduces_policy_v2_on_clinc():
    d, to_cols, A_idx, B_idx = _clinc()
    SC_val, SC_te = d["scores_val"], d["scores_test"]
    yv, yt = d["y_val"], d["y_test"]
    val_in = yv != OOS_LABEL
    te_in = yt != OOS_LABEL
    yv_c, yt_c = to_cols(yv), to_cols(yt)
    labels = [f"intent_{i}" for i in range(150)]

    g = ConformalGate(alpha=0.10, min_n=200, t_prob_floor=0.25, t_set=1.0,
                      k_clarify=3, answer_precision=0.90)
    # kernel protocol: A fits T_prob+τ on in-scope, B calibrates qhat in-scope
    g.calibrate(SC_val[A_idx][val_in[A_idx]], yv_c[A_idx][val_in[A_idx]],
                SC_val[B_idx][val_in[B_idx]], yv_c[B_idx][val_in[B_idx]])
    assert abs(g.t_prob - 0.25) < 1e-9                       # floored
    assert abs(g.tau_answer - 0.0153) < 3e-3                 # kernel: 0.0152795
    assert abs(g.qhat - 0.0239) < 3e-3                       # kernel: 0.0239393

    # in-schema τ on val_A: in-scope vs OOS max-score Youden — kernel: 0.6991
    g.fit_in_schema(SC_val[A_idx][val_in[A_idx]], SC_val[A_idx][~val_in[A_idx]])
    assert abs(g.tau_in_schema - 0.6991) < 0.02

    actions = {"answer": 0, "clarify": 0, "escalate": 0, "abstain": 0}
    cov_hits = n_in = 0
    for i in range(len(yt)):
        r = g.decide(SC_te[i], labels)
        actions[r.action] += 1
        if te_in[i]:
            n_in += 1
            cov_hits += int(f"intent_{yt_c[i]}" in r.prediction_set)
    # kernel counts: answer 1797, clarify 55, escalate 3648 — allow ±5% drift
    assert abs(actions["answer"] - 1797) < 120
    assert abs(actions["clarify"] - 55) < 40
    assert abs(actions["escalate"] - 3648) < 150
    coverage = cov_hits / n_in
    assert abs(coverage - 0.914) < 0.03                       # non-degenerate band


@REAL
def test_taskhead_learns_on_real_embeddings():
    """Linear head on cached CLINC embeddings — real data, ~40MB in RAM."""
    d, to_cols, _, _ = _clinc()
    Xtr, ytr = d["emb_train"], d["y_train"]
    Xv, yv = d["emb_val"], d["y_val"]
    col = to_cols(yv)
    in_scope = yv != OOS_LABEL
    h = TaskHead(kind="linear")
    # emb_train is class-sorted — slicing the first N rows drops most classes
    rec = h.fit(Xtr, to_cols(ytr), epochs=15, batch_size=256,
                labels=[f"intent_{i}" for i in range(150)])
    assert rec["n_classes"] == 150
    probs = h.logits(Xv[in_scope]).argmax(1)
    acc = float((probs == col[in_scope]).mean())
    assert acc > 0.80                      # kernel: 0.970 at full data/epochs
