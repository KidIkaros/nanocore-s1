"""The causal readout — paired outcomes, a CI, and an honest null."""
import numpy as np
import pytest

from src.decision.causal import escalations, paired_readout, unsafe_answers
from src.decision.schema import answer_labels
from src.decision.schema import Prediction


def _pred(action="answer", top="a", probs=None):
    p = probs or {top: 0.9, "b": 0.1}
    return Prediction(qtype="choice", labels=list(p), probabilities=p,
                      answer_confidence=max(p.values()), entropy_confidence=0.1,
                      max_score=1.0, prediction_set=[top], action=action,
                      alpha=0.1)


def test_answer_labels_picks_the_top_probability_label():
    assert answer_labels([_pred(top="b", probs={"a": 0.2, "b": 0.8})]) == ["b"]


def test_only_a_wrong_answer_is_unsafe():
    preds = [_pred("answer", "a"), _pred("answer", "a"),
             _pred("escalate", "a"), _pred("clarify", "a")]
    gold = ["a", "b", "b", "b"]
    assert list(unsafe_answers(preds, gold)) == [0.0, 1.0, 0.0, 0.0]


def test_answering_an_out_of_schema_item_is_unsafe_however_confident():
    preds = [_pred("answer", "a"), _pred("escalate", "a")]
    assert list(unsafe_answers(preds, ["oos", "oos"], oos_label="oos")) == [1.0, 0.0]


def test_misaligned_arrays_rejected():
    with pytest.raises(ValueError, match="gold labels"):
        unsafe_answers([_pred()], ["a", "b"])


def test_escalation_flags():
    assert list(escalations([_pred("escalate"), _pred("answer")])) == [1.0, 0.0]


def test_paired_ci_is_degenerate_on_a_constant_shift():
    """The proof that the resample is shared: every item improves by exactly
    0.1, so every bootstrap replicate must return exactly -0.1. An unpaired
    resample would spread."""
    incumbent = np.tile([0.0, 1.0, 0.0, 0.0, 1.0], 40)
    candidate = incumbent - 0.1
    out = paired_readout("unsafe_rate", incumbent, candidate)
    assert out["delta"]["point"] == pytest.approx(-0.1)
    assert out["delta"]["hi"] == pytest.approx(-0.1)
    assert out["verdict"] == "improved"


def test_identical_models_are_the_honest_null():
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.2).astype(float)
    out = paired_readout("unsafe_rate", a, a.copy())
    assert out["delta"]["point"] == 0.0
    assert out["verdict"] == "indistinguishable"
    assert out["direction"] == "lower_is_better"


def test_a_real_regression_is_named():
    rng = np.random.default_rng(1)
    incumbent = (rng.random(600) < 0.05).astype(float)
    candidate = (rng.random(600) < 0.20).astype(float)
    out = paired_readout("unsafe_rate", incumbent, candidate)
    assert out["verdict"] == "regressed" and out["delta"]["lo"] > 0


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="item-for-item"):
        paired_readout("unsafe_rate", np.zeros(3), np.zeros(4))
