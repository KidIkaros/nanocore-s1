"""jev-benchmarking adapter — answer-format contract, weight-free.

The harness's ``evaluate.compute`` consumes Jev-format answers; the adapter
must emit exactly those shapes or the same-protocol comparison silently
measures nothing. A stub model supplies probabilities; no encoder runs.
"""
import numpy as np
import pytest

from src.decision.jevbench import (NOUL_NEGATIVE, confidence,
                                   nanocore_answers, option_label)


class _StubModel:
    """decide() over a fixed uniform-plus-bump distribution, no encoder."""
    class _P:
        def __init__(self, probs, action="answer"):
            self.probabilities = probs
            self.action = action

    def decide(self, state, question):
        k = len(question.options)
        p = np.full(k, 1.0 / k)
        p[0] += 0.2
        p /= p.sum()
        return self._P({o: float(x) for o, x in zip(question.options, p)})


class _AbstainingModel(_StubModel):
    """Every head abstains — the gated arm's extreme case."""

    def decide(self, state, question):
        p = super().decide(state, question)
        p.action = "escalate"
        return p


# ── option rendering mirrors the harness's prompts.render ────────────────────

def test_anonymous_key_scores_the_description():
    assert option_label("A", "sports news") == "sports news"
    assert option_label("AA", None) == "AA"


def test_named_key_scores_key_with_description():
    assert option_label("billing", "invoices and payments") == \
        "billing (invoices and payments)"
    assert option_label("billing", None) == "billing"


# ── answer shapes match what evaluate.compute reads ─────────────────────────

def test_choice_answer_carries_probabilities_and_a_key():
    q = {"answer": {"type": "choice",
                    "instructions": "topic?",
                    "criteria": {"A": "sports", "B": "finance"}}}
    out = nanocore_answers(_StubModel(), {"text": "about sports"}, q)
    a = out["answer"]
    assert a["type"] == "choice"
    assert a["choice"] == "A"
    assert a["probabilities"].keys() == {"A", "B"}
    assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-9
    assert 0.0 <= a["confidence"] <= 1.0


def test_noul_answer_uses_prop_in_options_contract():
    """The proposition is an option; P(prop) is the noul value — the measured
    contract, not a yes/no head."""
    q = {"judge": {"type": "noul",
                   "instructions": "Is `text` spam?"}}
    out = nanocore_answers(_StubModel(), {"text": "buy now"}, q)
    a = out["judge"]
    assert a["type"] == "noul"
    assert isinstance(a["noul"], float)
    # prop is the instructions text; it got the bumped first-option mass
    assert a["noul"] > 0.5


def test_score_answer_reports_expected_index_and_legend():
    q = {"rate": {"type": "score",
                  "instructions": "quality?",
                  "criteria": ["bad", "ok", "good"]}}
    out = nanocore_answers(_StubModel(), "the thing", q)
    a = out["rate"]
    assert a["type"] == "score"
    assert 0.0 <= a["score"] <= 2.0
    assert a["legend"] == {"0": "bad", "1": "ok", "2": "good"}
    assert a["probabilities"].keys() == {"0", "1", "2"}


def test_state_dict_is_split_and_instructions_fold_in():
    """Same-protocol honesty: instructions reach the model via the state
    (Question has no instructions field — component 34). The state dict is
    split into per-field items (the v27 pair fix), and the instruction folds
    in as its own item."""
    seen = {}

    class Spy(_StubModel):
        def decide(self, state, question):
            seen["state"] = state
            return super().decide(state, question)

    q = {"answer": {"type": "choice", "instructions": "topic?",
                    "criteria": {"A": "x", "B": "y"}}}
    nanocore_answers(Spy(), {"query": "hello"}, q)
    # Single-field state: the instruction folds into the state string — the
    # measured prompt-ablation shape, preserved so the single-field path
    # the anchor runs through is byte-identical to what was measured.
    items = list(seen["state"].items)
    assert len(items) == 1
    assert "topic?" in items[0] and "hello" in items[0]


def test_noul_proposition_stays_out_of_the_state():
    """Regression: instructions-as-proposition must not leak into the state —
    verbatim overlap would inflate P(prop) through surface matching, which is
    the failure G4's withheld-state probe exists to catch."""
    seen = {}

    class Spy(_StubModel):
        def decide(self, state, question):
            seen["state"] = state
            return super().decide(state, question)

    q = {"judge": {"type": "noul", "instructions": "Is `text` spam?"}}
    nanocore_answers(Spy(), "buy now", q)
    assert "spam" not in " ".join(seen["state"].items)


def test_abstention_withholds_the_whole_example():
    """A single non-answer head withholds the request — the harness scores
    complete answers only, and n_answered/n_examples is where coverage shows."""
    q = {"judge": {"type": "noul", "instructions": "spam?"}}
    assert nanocore_answers(_AbstainingModel(), "buy now", q) is None


def test_unevaluated_action_still_answers():
    """No gate attached (action='unevaluated') is answered, not abstained —
    abstention is a gate decision; a gate that isn't there cannot make one."""
    class Ungated(_StubModel):
        def decide(self, state, question):
            p = super().decide(state, question)
            p.action = "unevaluated"
            return p

    q = {"judge": {"type": "noul", "instructions": "spam?"}}
    assert nanocore_answers(Ungated(), "buy now", q) is not None


def test_run_task_reports_coverage():
    """The gated arm's signal: n_answered < n_examples, metrics on the
    answered subset only."""
    from src.decision.jevbench import run_task

    ex = type("E", (), {"state": "s", "gold": {"j": True},
                        "questions": {"j": {"type": "noul",
                                            "instructions": "spam?"}}})()

    class FakeTask:
        name = "fake"
        def examples(self, split, limit=None):
            return [ex, ex, ex]

    r = run_task(_AbstainingModel(), FakeTask())
    assert r["n_examples"] == 3 and r["n_answered"] == 0
    assert r["coverage"] == 0.0


def test_unknown_question_type_raises():
    with pytest.raises(ValueError, match="unknown question type"):
        nanocore_answers(_StubModel(), "s", {"x": {"type": "mystery"}})


def test_confidence_is_the_harness_formula():
    assert confidence([1.0, 0.0]) == pytest.approx(1.0)
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert confidence([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
