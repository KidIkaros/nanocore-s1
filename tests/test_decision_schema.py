"""Contract tests for the typed decision schema — no model weights."""
import numpy as np
import pytest

from src.decision.schema import (
    DecisionExample, Prediction, Question, State, normalized_entropy,
)


class TestQuestion:
    def test_choice_requires_options(self):
        with pytest.raises(ValueError):
            Question(qtype="choice", options=[])

    def test_noul_defaults_to_yes_no(self):
        q = Question(qtype="noul")
        assert q.options == ["yes", "no"]

    def test_noul_rejects_three_options(self):
        with pytest.raises(ValueError):
            Question(qtype="noul", options=["a", "b", "c"])

    def test_bad_qtype_rejected(self):
        with pytest.raises(ValueError):
            Question(qtype="generate", options=["a"])

    def test_score_scale_length(self):
        with pytest.raises(ValueError):
            Question(qtype="score", options=["low", "high"], scale=[1.0])


class TestState:
    def test_text_shortcut(self):
        assert State.text("hi").items == ["hi"]

    def test_media_dict(self):
        s = State.media(image="x.png")
        assert s.items == [{"image": "x.png"}]

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            State(items=[])


class TestPrediction:
    def _pred(self, qtype="choice"):
        return Prediction(qtype=qtype, labels=["a", "b"],
                          probabilities={"a": 0.7, "b": 0.3},
                          answer_confidence=0.7, entropy_confidence=0.6)

    def test_choice_argmax(self):
        assert self._pred().choice == "a"

    def test_noul_reads_yes(self):
        p = Prediction(qtype="noul", labels=["yes", "no"],
                       probabilities={"yes": 0.8, "no": 0.2},
                       answer_confidence=0.8, entropy_confidence=0.5)
        assert p.noul == 0.8
        assert p.choice is None

    def test_score_is_expected_value(self):
        p = Prediction(qtype="score", labels=["l0", "l1", "l2"],
                       probabilities={"l0": 0.0, "l1": 0.0, "l2": 1.0},
                       answer_confidence=1.0, entropy_confidence=0.0)
        assert p.score == 2.0


class TestDecisionExample:
    def test_target_must_sum_to_one(self):
        with pytest.raises(ValueError):
            DecisionExample(state_embedding=np.zeros(8), qtype="choice",
                            labels=["a", "b"], target=[0.9, 0.9])

    def test_label_target_length(self):
        with pytest.raises(ValueError):
            DecisionExample(state_embedding=np.zeros(8), qtype="choice",
                            labels=["a", "b"], target=[1.0])

    def test_option_embeddings_must_match_labels(self):
        with pytest.raises(ValueError):
            DecisionExample(state_embedding=np.zeros(8), qtype="choice",
                            labels=["a", "b"], target=[1.0, 0.0],
                            option_embeddings=[np.zeros(8)])


def test_normalized_entropy_bounds():
    assert normalized_entropy([1.0, 0.0]) == pytest.approx(0.0, abs=1e-6)
    assert normalized_entropy([0.5, 0.5]) == pytest.approx(1.0, abs=1e-6)
