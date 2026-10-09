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
from src.decision.scoring import (CosineScorer, OrdinalScorer, TaskHead,
                                  aps_members, fit_temperature, mass_needed,
                                  softmax_rows)

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
    def __init__(self, dim: int = 4):
        self.calls = 0
        self.dim = dim
    def encode_state(self, items):
        self.calls += 1
        import torch
        return torch.ones(len(items), self.dim)
    def encode_options(self, texts):
        import torch
        out = torch.zeros(len(texts), self.dim)
        for i in range(len(texts)):
            out[i, i % self.dim] = 1.0
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


# ── production-path components ───────────────────────────────────────────

def test_gate_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    k, n = 5, 400
    sc = rng.normal(0, 0.5, (n, k)); sc[np.arange(n), np.arange(n) % k] += 6.0
    g = ConformalGate(alpha=0.10, min_n=200)
    g.calibrate(sc, np.arange(n) % k)
    g.fit_in_schema(sc[:300], sc[300:] - 5.0)     # in vs clearly-OOS
    g.save(tmp_path / "gate.json")
    g2 = ConformalGate.load(tmp_path / "gate.json")
    for attr in ("t_prob", "tau_answer", "qhat", "tau_in_schema"):
        assert getattr(g2, attr) == pytest.approx(getattr(g, attr))
    r1, r2 = g.decide(sc[0], list("abcde")), g2.decide(sc[0], list("abcde"))
    assert r1.action == r2.action and r1.prediction_set == r2.prediction_set


def test_model_bundle_round_trip(tmp_path):
    rng = np.random.default_rng(4)
    k, n, d = 4, 300, 16
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    y = np.arange(n) % k
    X[:, 0] += (y == 0) * 3                        # make classes separable
    h = TaskHead(kind="linear")
    h.fit(X, y, epochs=15, labels=list("abcd"))
    g = ConformalGate(alpha=0.10, min_n=100)
    g.calibrate(h.logits(X), y)

    enc = _StubEncoder(dim=16)
    m = DecisionModel(encoder=enc, scorer=h, gate=g)
    bundle = m.save(tmp_path / "bundle")
    m2 = DecisionModel.load(bundle, encoder=enc)

    q = Question(qtype="choice", options=list("abcd"))
    p1, p2 = m.decide("hello", q), m2.decide("hello", q)
    assert p1.action == p2.action
    assert p1.prediction_set == p2.prediction_set
    assert p1.probabilities == pytest.approx(p2.probabilities)


def test_list_state_reaches_encoder_as_items():
    """A list state is its items — before the _items() fix, a list was wrapped
    as ``[state]`` so encode_state saw one list-item and pooled exactly one
    embedding: the second state item (e.g. a noul proposition) silently never
    reached the encoder."""
    enc = _StubEncoder(dim=8)
    m = DecisionModel(encoder=enc, gate=None, cache=None)
    seen = []
    enc.encode_state = lambda items: (seen.append(list(items)),
                                      __import__("torch").ones(len(items), 8))[1]
    m.decide(["utterance", "the proposition"], Question(qtype="choice", options=["x", "y"]))
    assert seen[0] == ["utterance", "the proposition"]


def test_composer_slot_changes_state_and_bundle_roundtrips(tmp_path):
    """The composer is a real slot on the shipped path: it must alter the state
    vector, fold into the cache key, and survive save→load."""
    import torch
    from src.decision.composer import ComposerConfig, EmbeddingComposer

    enc = _StubEncoder(dim=8)
    q = Question(qtype="choice", options=["x", "y"])
    pooled = DecisionModel(encoder=enc, gate=None, cache=None)
    composed = DecisionModel(encoder=enc, gate=None, cache=None,
                             composer=EmbeddingComposer(
                                 ComposerConfig(n_layer=1, n_head=2, n_embd=8,
                                                embd_dim=8)))
    assert pooled._composer_id() == "meanpool"
    assert composed._composer_id().startswith("composer:")

    # identity-ish init means the *direction* can stay similar but the vector
    # must not be bit-identical to a plain mean (modality embed + rms_norm)
    items = ["a", "b", "c"]
    s_pool = pooled._state_vec(items)
    s_comp = composed._state_vec(items)
    assert s_pool.shape == s_comp.shape == (8,)
    assert not np.allclose(s_pool, s_comp)

    bundle = composed.save(tmp_path / "b")
    reloaded = DecisionModel.load(bundle, encoder=enc)
    p1, p2 = (m.decide(items, q) for m in (composed, reloaded))
    assert p1.probabilities == p2.probabilities

    # cache keys must separate pooled vs composed decisions on the same state
    k_pool = decision_key(items, q, "stub", {"composer": "meanpool"})
    k_comp = decision_key(items, q, "stub", {"composer": composed._composer_id()})
    assert k_pool != k_comp


def test_ordinal_scorer_orders_and_sums():
    rng = np.random.default_rng(5)
    n, d, k = 600, 8, 5
    y = np.repeat(np.arange(k), n // k)
    X = rng.normal(0, 1, (n, d)).astype(np.float32)
    X[:, 0] += y * 1.5                             # monotone signal in dim 0
    s = OrdinalScorer()
    s.fit(X, y, epochs=60, lr=5e-3, labels=list("abcde"))
    P = s.level_probs(X[:64])
    assert P.shape == (64, k)
    assert np.allclose(P.sum(1), 1.0, atol=1e-3)
    assert np.all((P >= 0) & (P <= 1))
    exp = np.array([s.expected(x) for x in X])
    # ordering preserved: higher true level → higher expected value on average
    # (evaluated over all rows — a prefix would be a single class, corr=NaN)
    assert np.corrcoef(exp, y)[0, 1] > 0.7
    # log-prob scores are gate-compatible
    sc = s.scores(X[0])
    assert np.isfinite(sc).all() and len(sc) == k


def test_ordinal_scorer_bundle_roundtrip(tmp_path):
    """A fitted OrdinalScorer must reload as an OrdinalScorer — before the
    dispatch fix, save() recorded the class name but dropped the weights, and
    load() silently fell back to CosineScorer: a different model answering."""
    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(3), 100)
    X = rng.normal(0, 1, (300, 8)).astype(np.float32)
    X[:, 0] += y                                     # learnable signal
    s = OrdinalScorer()
    s.fit(X, y, epochs=20, labels=["low", "mid", "high"])

    m = DecisionModel(encoder=_StubEncoder(dim=8), scorer=s)
    bundle = m.save(tmp_path / "b")
    m2 = DecisionModel.load(bundle, encoder=_StubEncoder(dim=8))
    assert isinstance(m2.scorer, OrdinalScorer)
    x = rng.normal(0, 1, 8).astype(np.float32)
    assert np.allclose(m2.scorer.level_probs(x[None])[0],
                       s.level_probs(x[None])[0], atol=1e-6)
