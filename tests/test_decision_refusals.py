"""The refusal battery — behaviour, and the guard that keeps it meaningful.

A battery satisfiable by a model that refuses everything is not a safety test,
so the positive control is tested as hard as the refusals. The battery is also
fed the model's *own* label space: a schema-bound scorer asked about a label it
was never fitted on must raise, and a battery whose every case raised would
report a refusal pass for entirely the wrong reason.
"""
import pytest

from src.decision.refusals import (CASES, RefusalCase, battery_is_meaningful,
                                   run_refusal_battery)
from src.decision.schema import Prediction

LABELS = ["cancel transfer", "card arrival", "balance", "exchange rate",
          "lost card", "top up"]


def _pred(action):
    return Prediction(qtype="choice", labels=["a", "b"],
                      probabilities={"a": 0.6, "b": 0.4},
                      answer_confidence=0.6, entropy_confidence=0.3,
                      max_score=1.0, prediction_set=["a"], action=action,
                      alpha=0.1)


class Always:
    def __init__(self, action):
        self.action = action

    def decide(self, state, question):
        return _pred(self.action)


class WellBehaved:
    """Declines everything except the verbatim control, and caps long input."""

    def decide(self, state, question):
        return _pred("answer" if state == LABELS[0] else "escalate")


class Explodes:
    def decide(self, state, question):
        raise RuntimeError("boom")


def test_a_well_behaved_model_passes_every_case():
    summary = run_refusal_battery(WellBehaved(), LABELS)
    assert summary["passed"] == summary["n"], [r for r in summary["results"]
                                               if not r["pass"]]
    assert summary["unsafe_answers"] == 0
    assert battery_is_meaningful(summary) is True


def test_a_model_that_answers_everything_is_caught():
    summary = run_refusal_battery(Always("answer"), LABELS)
    assert summary["unsafe_answers"] == 4              # the four not_answer cases
    assert summary["unsafe_answer_rate"] == 1.0
    assert "empty" in summary["unsafe_cases"]


def test_a_model_that_refuses_everything_is_not_meaningful():
    """The failure mode a naive battery rewards: declining everything scores
    well on the refusals and proves nothing."""
    summary = run_refusal_battery(Always("escalate"), LABELS)
    assert summary["unsafe_answers"] == 0              # looks perfect
    assert battery_is_meaningful(summary) is False     # but the control failed


def test_the_positive_control_requires_an_answer():
    control = [c for c in CASES if c.must == "answer"]
    assert control and control[0].name == "verbatim_option"
    assert control[0].state_from_options is True


def test_raising_fails_a_not_crash_case():
    summary = run_refusal_battery(Explodes(), LABELS)
    not_crash = [r for r in summary["results"] if r["must"] == "not_crash"]
    assert not_crash and not any(r["pass"] for r in not_crash)
    assert "boom" in not_crash[0]["error"]


def test_raising_still_counts_as_declining_for_not_answer_cases():
    """A crash is not an answer. It is a bad way to decline, but it is not the
    failure this battery is looking for — that is a confident wrong answer."""
    summary = run_refusal_battery(Explodes(), LABELS)
    assert summary["unsafe_answers"] == 0


def test_cases_use_the_models_own_label_space():
    """The bug this signature exists to prevent: hardcoded labels raise on every
    case for a model fitted on a different schema, and the battery then reports
    a pass for the wrong reason."""
    summary = run_refusal_battery(WellBehaved(), LABELS)
    assert all(r["n_options"] == len(LABELS) for r in summary["results"]
               if r["case"] != "single_option")
    single = next(r for r in summary["results"] if r["case"] == "single_option")
    assert single["n_options"] == 1


def test_the_verbatim_control_uses_a_real_label():
    seen = {}

    class Recorder:
        def decide(self, state, question):
            seen[state] = True
            return _pred("answer")

    run_refusal_battery(Recorder(), LABELS)
    assert LABELS[0] in seen


def test_an_empty_label_space_is_rejected():
    with pytest.raises(ValueError, match="label space"):
        run_refusal_battery(Always("answer"), [])


def test_custom_cases_are_honoured():
    cases = (RefusalCase("t", "not_answer", "why"),)
    assert run_refusal_battery(Always("clarify"), LABELS, cases)["n"] == 1


def test_every_case_explains_itself():
    """A case without a reason is a rule nobody can argue with or delete."""
    assert all(c.why for c in CASES)
    assert all(c.must in ("not_answer", "answer", "not_crash") for c in CASES)
