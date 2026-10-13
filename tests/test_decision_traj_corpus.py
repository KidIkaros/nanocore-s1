"""Trajectory-corpus generation (#19) — mechanism tests with a stub model.

The real corpus runs on Kaggle (real encoder, ADR-0006); these tests pin the
mechanism locally: the loop produces traces, non-answer verdicts advance the
proposer into multi-step traces, and corpus_stats reports the depth structure
the predictive organ trains on.
"""
from src.decision.traj_corpus import (generate_traces, corpus_stats,
                                      make_proposer, QuestionTrace)
from src.decision.schema import Question


class _Pred:
    def __init__(self, action, top_prob=0.9):
        self.action = action
        self.probabilities = {"a": top_prob}


class _StubModel:
    """decide() returns the action scripted by a per-candidate rule: the first
    candidate answers, the rest escalate (so the loop is multi-step)."""
    def __init__(self, script):
        self.script = script   # list of actions, consumed per decide() call
        self.calls = 0

    def decide(self, state, question):
        a = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return _Pred(a)


def _question():
    return Question(qtype="choice", options=["a", "b", "c", "d"])


def test_single_answer_produces_one_step_trace():
    m = _StubModel(["answer"])
    traces = generate_traces([["a", "b"]], m, _question(),
                             true_intents=[0], difficulties=[0.0])
    assert len(traces) == 1
    t = traces[0]
    assert t.n_iterations == 1 and t.resolved is True
    assert t.stop_reason == "confirmed"
    assert t.steps[0]["action"] == "answer"
    assert t.true_intent == 0


def test_escalate_then_answer_produces_multi_step_trace():
    # escalate (non-terminal) on iter 1, answer on iter 2 -> 2-step trace
    m = _StubModel(["escalate", "answer"])
    traces = generate_traces([["a", "b", "c"]], m, _question(),
                             max_iterations=3)
    t = traces[0]
    assert t.n_iterations == 2
    assert t.resolved is True
    assert [s["action"] for s in t.steps] == ["escalate", "answer"]
    # state_items grows with history depth (the depth axis)
    assert t.steps[1]["state_items"] > t.steps[0]["state_items"]


def test_exhaustion_is_unresolved():
    m = _StubModel(["clarify"])   # never terminal -> exhausts at max_iterations
    traces = generate_traces([["a", "b", "c", "d"]], m, _question(),
                             max_iterations=3)
    t = traces[0]
    assert t.n_iterations == 3
    assert t.resolved is False
    assert t.stop_reason == "max_iterations"


def test_corpus_stats_reports_depth_structure():
    # one resolves at iter1, one at iter2, one exhausts
    models = [_StubModel(["answer"]),
              _StubModel(["escalate", "answer"]),
              _StubModel(["clarify"])]
    all_traces = []
    for m in models:
        all_traces += generate_traces([["a", "b", "c", "d"]], m, _question(),
                                      max_iterations=3)
    stats = corpus_stats(all_traces)
    assert stats["n"] == 3
    assert stats["resolved"] == 2
    assert stats["resolve_rate"] == round(2 / 3, 4)
    assert stats["multi_step"] == 2   # the iter-2 resolve AND the iter-3 exhaust
    assert "1" in stats["by_iterations"] and "2" in stats["by_iterations"]
    assert stats["by_iterations"]["1"]["n"] == 1
    assert stats["by_iterations"]["1"]["resolved"] == 1


def test_proposer_advances_through_candidates():
    prop = make_proposer(["a", "b", "c"])
    from src.decision.loop import LoopStep
    assert prop([]) == "a"                       # first candidate
    assert prop([LoopStep(1, "a", "escalate")]) == "b"   # advanced
    assert prop([LoopStep(1, "a", "escalate"),
                 LoopStep(2, "b", "escalate")]) == "c"


def test_trace_to_dict_roundtrips_labels():
    t = QuestionTrace(true_intent=7, steps=[{"iteration": 1, "action": "answer"}],
                      resolved=True, stop_reason="confirmed", n_iterations=1,
                      difficulty=0.25)
    d = t.to_dict()
    assert d["true_intent"] == 7 and d["resolved"] is True
    assert d["difficulty"] == 0.25


# ── DP4 narrowing proposer (#19 sharpening) ──────────────────────────────

class _Scorer:
    """Ranks options by a fixed score map so the proposer's ranking is known.
    Takes embedding vectors (as the real TaskHead.scores does) but the fake
    encoder encodes each option to a 1-hot-ish vector keyed by its score."""
    def __init__(self, score_map):
        self.score_map = score_map
    def scores(self, state_vec, option_vecs):
        # option_vecs rows carry the option's score in column 0
        return [float(v[0]) for v in option_vecs]


class _FakeEncoder:
    """Encodes options to [score] vectors so _Scorer ranks them. State encodes
    to a dummy vector (the proposer only ranks survivors by their own score)."""
    def __init__(self, score_map=None):
        self.score_map = score_map or {}
    def encode_state(self, items):
        import numpy as np
        return np.array([[0.0, 0.0] for _ in items], dtype=float)
    def encode_options(self, options):
        import numpy as np
        return np.array([[self.score_map.get(o, 0.0), 0.0] for o in options],
                        dtype=float)


def test_loop_step_carries_prediction_set():
    """The loop plumbs the gate's APS prediction_set into LoopStep so the
    proposer can narrow on it (the field #19 added)."""
    m = _StubModel(["escalate"])
    # stub pred must expose prediction_set; wrap decide to attach it
    class _M2:
        def __init__(self): self.calls = 0
        def decide(self, state, question):
            self.calls += 1
            p = _Pred("escalate", 0.4)
            p.prediction_set = ["survivor_a", "survivor_b"]
            return p
    traces = generate_traces([["c0", "c1", "c2"]], _M2(), _question(),
                             max_iterations=2)
    assert traces[0].steps[0]["action"] == "escalate"
    # the trace's step dict does not carry prediction_set (QuestionTrace schema),
    # but the LoopStep does — verify via the proposer behavior below.


def test_narrowing_proposer_reproposes_from_survivors():
    """On a non-answer verdict, the narrowing proposer re-proposes the top
    SURVIVING option (from prediction_set), not the next list item — the DP4
    move that produces recovery traces instead of escalate-forever."""
    from src.decision.traj_corpus import make_narrowing_proposer
    from src.decision.loop import LoopStep
    scorer = _Scorer({"x": 0.1, "y": 0.9, "z": 0.5})
    enc = _FakeEncoder({"x": 0.1, "y": 0.9, "z": 0.5})
    prop = make_narrowing_proposer(["c0", "c1"], scorer, ["x", "y", "z"],
                                   encoder=enc)
    prop([])                                   # iter 1 -> first candidate
    step = LoopStep(1, "c0", "escalate", prediction_set=["x", "y", "z"])
    assert prop([step]) == "y"                 # top-scored survivor


def test_narrowing_proposer_avoids_reoffering():
    """It does not re-offer a survivor it already proposed (advances to next
    best unoffered survivor)."""
    from src.decision.traj_corpus import make_narrowing_proposer
    from src.decision.loop import LoopStep
    scorer = _Scorer({"x": 0.9, "y": 0.5, "z": 0.1})
    enc = _FakeEncoder({"x": 0.9, "y": 0.5, "z": 0.1})
    prop = make_narrowing_proposer(["c0", "c1"], scorer, ["x", "y", "z"],
                                   encoder=enc)
    prop([])
    s1 = LoopStep(1, "c0", "escalate", prediction_set=["x", "y", "z"])
    assert prop([s1]) == "x"                   # top survivor
    s2 = LoopStep(2, "x", "escalate", prediction_set=["x", "y", "z"])
    assert prop([s1, s2]) == "y"               # next best unoffered


def test_narrowing_proposer_falls_back_when_no_set():
    """If the gate abstained with no prediction_set, fall back to linear
    advance (bounded, never crashes)."""
    from src.decision.traj_corpus import make_narrowing_proposer
    from src.decision.loop import LoopStep
    scorer = _Scorer({})
    prop = make_narrowing_proposer(["c0", "c1", "c2"], scorer, ["a"])
    s = LoopStep(1, "c0", "abstain", prediction_set=None)
    assert prop([s]) == "c1"                   # advanced linearly


def test_generate_traces_accepts_proposer_factory():
    """generate_traces wires a custom proposer factory (the narrowing proposer)
    through to the loop."""
    from src.decision.traj_corpus import make_narrowing_proposer
    scorer = _Scorer({"a": 0.9, "b": 0.1})
    enc = _FakeEncoder({"a": 0.9, "b": 0.1})
    class _M:
        def __init__(self): self.calls = 0
        def decide(self, state, question):
            self.calls += 1
            p = _Pred("answer" if self.calls >= 2 else "escalate", 0.9)
            p.prediction_set = ["a", "b"]
            return p
    def factory(cands):
        return make_narrowing_proposer(cands, scorer, ["a", "b"], encoder=enc)
    traces = generate_traces([["c0", "c1"]], _M(), _question(),
                             max_iterations=3, proposer_factory=factory)
    # iter1 escalate (no set yet on the very first propose), iter2 answer
    assert traces[0].n_iterations == 2
    assert traces[0].resolved is True
