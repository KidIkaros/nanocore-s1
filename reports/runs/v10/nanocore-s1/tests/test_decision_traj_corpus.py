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
