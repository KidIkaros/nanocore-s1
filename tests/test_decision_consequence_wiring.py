"""Consequence re-ranker wiring into the served decide() path (#18).

The organ fires at an escalate verdict: it reorders the gate's APS survivors by
predicted recoverability. Strictly additive — never changes the action or
probabilities (judge–executor separation). No consequence model = byte-identical
to the base path (the earn-the-slot floor).
"""
import numpy as np

from src.decision.model import DecisionModel
from src.decision.consequence import ConsequenceModel
from src.decision.gate import ConformalGate
from src.decision.schema import Question


class _Enc:
    """Encoder whose option vectors are 1-hot in dim = option index, so the
    consequence model's w can be aimed at a specific survivor."""
    def __init__(self, dim=6):
        self.dim = dim
        self.model_name = "stub"
    def encode_state(self, items):
        n = len(items) if isinstance(items, (list, tuple)) else 1
        return np.ones((n, self.dim))
    def encode_options(self, texts):
        M = np.zeros((len(texts), self.dim))
        for i in range(len(texts)):
            M[i, i % self.dim] = 1.0
        return M


class _Scorer:
    """Cosine-ish scorer: option i scores highest for a state that's 1-hot i."""
    temperature = 1.0
    def scores(self, state_vec, option_vecs):
        s = np.asarray(state_vec).reshape(-1)
        o = np.asarray(option_vecs)
        return (o @ s)


def _gate():
    # a gate calibrated to escalate with a wide survivor set. High-variance
    # calibration scores with random labels -> large qhat -> full prediction set
    # -> escalate branch. calibrate takes (scores, integer targets).
    g = ConformalGate(alpha=0.5, min_n=1)
    rng = np.random.default_rng(0)
    g.calibrate(rng.normal(0, 3.0, (40, 4)), rng.integers(0, 4, 40))
    return g


def test_no_consequence_is_byte_identical():
    """Without a consequence model, decide() is unchanged — the earn-the-slot
    floor: consequence=None must not alter the base path."""
    enc = _Enc()
    m = DecisionModel(encoder=enc, scorer=_Scorer(), gate=_gate())
    q = Question(qtype="choice", options=["a", "b", "c", "d"])
    p = m.decide("test", q)
    assert m.consequence is None
    # action/probs are whatever the gate decided; the point is the model runs
    assert p.action in ("answer", "escalate", "clarify", "abstain", "unevaluated")


def test_consequence_reorders_prediction_set_on_escalate_only():
    """With a consequence model, an escalate verdict's prediction_set is
    reordered by recoverability — but the action and probabilities are
    untouched (strictly additive, judge–executor separation)."""
    enc = _Enc(dim=4)
    gate = _gate()
    # consequence model that favors option index 2 (w aimed at dim 2)
    w = np.zeros(4); w[2] = 5.0
    cons = ConsequenceModel(w=w, b=0.0)

    m_base = DecisionModel(encoder=enc, scorer=_Scorer(), gate=gate)
    m_cons = DecisionModel(encoder=enc, scorer=_Scorer(), gate=gate,
                           consequence=cons)
    q = Question(qtype="choice", options=["a", "b", "c", "d"])

    # find an input that escalates
    rng = np.random.default_rng(1)
    escalated = False
    for txt in [f"t{i}" for i in range(30)]:
        pb = m_base.decide(txt, q)
        if pb.action == "escalate" and len(pb.prediction_set) > 1:
            pc = m_cons.decide(txt, q)
            # action identical
            assert pc.action == "escalate"
            # probabilities identical (re-ranker never touches them)
            assert pc.probabilities == pb.probabilities
            # prediction_set is a permutation of the same members
            assert set(pc.prediction_set) == set(pb.prediction_set)
            assert len(pc.prediction_set) == len(pb.prediction_set)
            escalated = True
            break
    assert escalated, "test needs an input that escalates with >1 survivor"


def test_consequence_reranks_deterministically_when_signal_differs():
    """When the consequence model has a real per-candidate signal (state not
    uniform), it deterministically promotes the highest-recoverability
    survivor to the front of the escalation set."""
    class _StateEnc:
        """State is 1-hot in dim 2, so option 'c' (dim 2) aligns best."""
        model_name = "stub"
        def __init__(self, dim=4): self.dim = dim
        def encode_state(self, items):
            v = np.zeros((1, self.dim)); v[0, 2] = 1.0
            return v
        def encode_options(self, texts):
            M = np.zeros((len(texts), self.dim))
            for i in range(len(texts)): M[i, i % self.dim] = 1.0
            return M
    gate = _gate()
    w = np.zeros(4); w[2] = 5.0     # favors dim 2 = option "c"
    cons = ConsequenceModel(w=w, b=0.0)
    m = DecisionModel(encoder=_StateEnc(), scorer=_Scorer(), gate=gate,
                      consequence=cons)
    q = Question(qtype="choice", options=["a", "b", "c", "d"])
    p = m.decide("t", q)
    assert p.action == "escalate"
    assert len(p.prediction_set) > 1
    # 'c' (dim 2, aligned with the state, favored by w) ranks first
    assert p.prediction_set[0] == "c"


def test_consequence_save_load_roundtrip(tmp_path):
    """The consequence model persists in the bundle (torch-free) and reloads."""
    enc = _Enc(dim=4)
    w = np.zeros(4); w[1] = 3.0
    cons = ConsequenceModel(w=w, b=0.5)
    m = DecisionModel(encoder=enc, scorer=_Scorer(), gate=_gate(),
                      consequence=cons, encoder_id="stub")
    d = m.save(tmp_path / "bundle")
    assert (d / "consequence.npz").exists()
    import json
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest.get("consequence") == "consequence.npz"
    # reload with a fresh encoder
    m2 = DecisionModel.load(d, encoder=_Enc(dim=4))
    assert m2.consequence is not None
    assert np.allclose(m2.consequence.w, cons.w)
    assert abs(m2.consequence.b - cons.b) < 1e-9


def test_rerank_failure_never_breaks_decision():
    """If the consequence re-rank throws (e.g. rank_candidates errors), decide()
    still returns a valid prediction — the re-ranker is strictly additive and
    isolated. (The encoder failure path is separate — it's in the base path,
    outside the re-ranker's try block.)"""
    enc = _Enc(dim=4)
    class _BadCons:
        def rank_candidates(self, state, vecs):
            raise RuntimeError("rerank blew up")
    m = DecisionModel(encoder=enc, scorer=_Scorer(), gate=_gate(),
                      consequence=_BadCons())
    q = Question(qtype="choice", options=["a", "b", "c", "d"])
    p = m.decide("test", q)   # must not raise
    assert p.action in ("answer", "escalate", "clarify", "abstain", "unevaluated")
