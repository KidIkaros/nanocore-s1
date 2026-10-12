"""Decision loop — System One as a component inside an iteration.

The DP4 confirmation-loop pattern (arXiv:2609.28940): a generative proposer
offers a candidate; the typed decision head adjudicates it; a non-answer
verdict feeds back into the next proposal. The decision head serves the
loop — it is the calibrated adjudicator each step consults, never the
thing being looped. The gate's action is the loop's control signal with no
new vocabulary. Each iteration is a trajectory row (component 42).

No weights run — a stub model supplies verdicts, proving the control flow.
"""
import pytest

from src.decision.loop import (ABSTAIN, NON_TERMINAL, TERMINAL_ACCEPT,
                               LoopResult, LoopStep, run_loop)


class _StubModel:
    """decide() returns a scripted sequence of actions, then the last one
    repeats. Records the state it was shown each iteration."""

    def __init__(self, script):
        self.script = list(script)
        self.seen_states = []

    class _P:
        def __init__(self, action, probs=None):
            self.action = action
            self.probabilities = probs or {"a": 0.9}

    def decide(self, state, question):
        self.seen_states.append(state)
        action = self.script.pop(0) if self.script else self._last
        self._last = action
        return self._P(action)


def _proposer_factory(candidates):
    """A proposer that yields candidates in order, then repeats the last
    indefinitely (so max_iterations can't exhaust the iterator)."""
    def propose(steps):
        i = len(steps)
        return candidates[i] if i < len(candidates) else candidates[-1]
    return propose


# ── the loop confirms on an answer and stops ──────────────────────────────

def test_loop_stops_on_first_answer():
    m = _StubModel(["answer"])
    r = run_loop(m, _proposer_factory(["x"]), None)
    assert r.confirmed and r.accepted == "x"
    assert r.stop_reason == "confirmed"
    assert r.n_iterations == 1


def test_loop_re_proposes_on_non_answer_then_confirms():
    """clarify → re-propose → answer: the DP4 alternation."""
    m = _StubModel(["clarify", "escalate", "answer"])
    r = run_loop(m, _proposer_factory(["a", "b", "c"]), None)
    assert r.confirmed and r.accepted == "c"
    assert r.n_iterations == 3
    assert [s.action for s in r.steps] == ["clarify", "escalate", "answer"]


# ── the verdict feeds back into the state ─────────────────────────────────

def test_prior_verdicts_reach_the_next_decide():
    """The loop's history must be visible to decide() — the state grows
    with each prior verdict (the composer later composes it; the loop is
    correct without it, just pooled)."""
    m = _StubModel(["clarify", "answer"])
    run_loop(m, _proposer_factory(["a", "b"]), None)
    # iteration 1: state is the bare candidate
    assert m.seen_states[0] == "a"
    # iteration 2: candidate + the prior verdict
    assert m.seen_states[1] == ["b", "previous verdict: clarify"]


def test_proposer_sees_prior_steps():
    """DP4's 'informed by observed behavior': the proposer receives the
    steps so far so it can narrow on a non-answer verdict."""
    seen = {}

    def propose(steps):
        seen["n"] = len(steps)
        seen["actions"] = [s.action for s in steps]
        return "cand"

    m = _StubModel(["escalate", "answer"])
    run_loop(m, propose, None)
    # propose runs BEFORE the step is appended: 1st call sees 0 steps,
    # 2nd call (after the escalate) sees 1.
    assert seen["n"] == 1
    assert seen["actions"] == ["escalate"]


# ── termination ───────────────────────────────────────────────────────────

def test_loop_stops_at_max_iterations():
    m = _StubModel(["clarify"] * 20)  # never confirms
    r = run_loop(m, _proposer_factory(["x"]), None, max_iterations=5)
    assert not r.confirmed
    assert r.stop_reason == "max_iterations"
    assert r.n_iterations == 5


def test_loop_stops_on_abstain():
    m = _StubModel(["clarify", "abstain"])
    r = run_loop(m, _proposer_factory(["a", "b"]), None)
    assert not r.confirmed
    assert r.stop_reason == "abstained"
    assert r.n_iterations == 2


def test_abstain_is_terminal_not_a_retry():
    """abstain ends the loop — it is not another non-terminal nudge."""
    m = _StubModel(["abstain", "answer"])
    r = run_loop(m, _proposer_factory(["a", "b"]), None)
    assert not r.confirmed
    assert r.n_iterations == 1  # never reached the second script entry


# ── trajectory: every step is a component-42 row ──────────────────────────

def test_steps_are_trajectory_units():
    m = _StubModel(["escalate", "answer"])
    r = run_loop(m, _proposer_factory(["a", "b"]), None)
    for s in r.steps:
        assert isinstance(s, LoopStep)
        assert s.iteration >= 1
        assert s.action in (TERMINAL_ACCEPT,) + NON_TERMINAL + (ABSTAIN,)
    assert [s.iteration for s in r.steps] == [1, 2]


def test_on_step_callback_fires_per_iteration():
    seen = []
    m = _StubModel(["clarify", "answer"])
    run_loop(m, _proposer_factory(["a", "b"]), None,
             on_step=lambda s: seen.append(s.iteration))
    assert seen == [1, 2]


def test_top_prob_recorded_per_step():
    m = _StubModel(["answer"])
    m._P = lambda action, probs=None: _StubModel._P(action, {"a": 0.77})
    r = run_loop(m, _proposer_factory(["x"]), None)
    assert r.steps[0].top_prob == pytest.approx(0.77)


# ── the constants name the loop's vocabulary ──────────────────────────────

def test_terminal_and_nonterminal_partition_the_actions():
    assert TERMINAL_ACCEPT == "answer"
    assert set(NON_TERMINAL) == {"clarify", "escalate"}
    assert ABSTAIN == "abstain"


def test_empty_proposer_history_on_first_step():
    m = _StubModel(["answer"])
    run_loop(m, _proposer_factory(["x"]), None)
    assert m.seen_states[0] == "x"  # bare candidate, no verdict prefix


# ── unevaluated (no gate) is a committed answer, not a retry ──────────────

def test_unevaluated_action_confirms():
    """An ungated model returns 'unevaluated' — a committed prediction with
    no abstention signal. The loop must treat it as confirmed, or it runs
    to max_iterations on a verdict that never varies."""
    m = _StubModel(["unevaluated"] * 10)
    r = run_loop(m, _proposer_factory(["x"]), None, max_iterations=5)
    assert r.confirmed
    assert r.n_iterations == 1
    assert r.stop_reason == "confirmed"


def test_loop_against_real_decide_path():
    """End-to-end on the real decide() with a stub encoder: an ungated
    model commits on iteration 1 — the gate is what makes it iterate."""
    from src.decision.backends import StubEncoder
    from src.decision.model import DecisionModel
    from src.decision.scoring import CosineScorer
    from src.decision.schema import Question
    m = DecisionModel(encoder=StubEncoder(dim=64), scorer=CosineScorer(),
                      gate=None, cache=None)
    q = Question(qtype="choice", options=["alpha", "beta"])
    r = run_loop(m, lambda steps: "alpha", q)
    assert r.confirmed and r.n_iterations == 1
