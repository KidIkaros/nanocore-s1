"""Success-gate measurement for the consequence re-ranker (#18) — mechanism
tests with stub models. The real A/B runs on Kaggle through DecisionModel.decide.
"""
import numpy as np

from src.decision.consequence_eval import measure_consequence_gain, _rec_from_decisions
from src.decision.schema import Question


class _Pred:
    def __init__(self, action, answer_confidence, probabilities, pred_set,
                 labels):
        self.action = action
        self.answer_confidence = answer_confidence
        self.probabilities = probabilities
        self.labels = labels
        self.prediction_set = pred_set


class _StubModel:
    """decide() returns scripted per-state results. Each result's correctness
    is relative to y_col[i]: correct means argmax == y_col[i]."""
    def __init__(self, results, y_col):
        self.results = results   # list of (answer_confidence, hard, correct)
        self.y_col = y_col
        self.i = 0
        self.labels = ["a", "b", "c", "d"]
    def decide(self, state, question):
        conf, hard, correct = self.results[self.i]
        yi = int(self.y_col[self.i])
        self.i += 1
        # argmax at the true label if correct, else a wrong label
        probs = {l: 0.0 for l in self.labels}
        if correct:
            probs[self.labels[yi]] = 1.0
        else:
            wrong = (yi + 1) % len(self.labels)
            probs[self.labels[wrong]] = 1.0
        return _Pred("escalate" if hard else "answer", conf, probs,
                     self.labels, self.labels)


def _make_results(n, hard_correct_frac, seed=0):
    """n (conf, hard, correct) tuples: even indices easy+correct+confident,
    odd indices hard, correct at hard_correct_frac with low confidence."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        hard = i % 2 == 1
        conf = 0.2 if hard else 0.95
        correct = (not hard) or (rng.random() < hard_correct_frac)
        out.append((conf, hard, correct))
    return out


def _question():
    return Question(qtype="choice", options=["a", "b", "c", "d"])


def test_rec_from_decisions_in_scope_only():
    labels = ["a", "b", "c", "d"]
    probs_correct = {l: (1.0 if l == "a" else 0.0) for l in labels}
    probs_wrong = {l: (1.0 if l == "b" else 0.0) for l in labels}
    preds = [_Pred("answer", 0.9, probs_correct, ["a"], labels),
             _Pred("escalate", 0.2, probs_wrong, labels, labels)]
    rec = _rec_from_decisions(preds, [True, False], [0, 1])
    assert len(rec["top1"]) == 1          # only in-scope
    assert rec["top1"][0] is True         # argmax 'a' == y_col 0
    assert rec["top_prob"][0] == 0.9


def test_measure_detects_gain_when_consequence_improves_hard():
    """When the consequence model makes hard cases correct (that the base gets
    wrong), the measurement reports a positive hard-quartile delta and
    earns_slot=True."""
    n = 40
    labels = ["a", "b", "c", "d"]
    in_mask = [True] * n
    y_col = list(range(4)) * (n // 4)
    base = _StubModel(_make_results(n, 0.0, seed=1), y_col)   # hard all wrong
    cons = _StubModel(_make_results(n, 1.0, seed=1), y_col)   # hard all right
    r = measure_consequence_gain(base, cons, [f"s{i}" for i in range(n)],
                                 _question(), in_mask, y_col)
    assert r["consequence"]["accuracy"] > r["base"]["accuracy"]
    assert r["hard_quartile_delta"] > 0
    assert r["earns_slot"] is True


def test_measure_no_gain_when_identical():
    """Identical models -> zero delta, earns_slot still True (non-inferior)."""
    n = 40
    labels = ["a", "b", "c", "d"]
    in_mask = [True] * n
    y_col = list(range(4)) * (n // 4)
    res = _make_results(n, 0.5, seed=2)
    base = _StubModel(list(res), y_col)
    cons = _StubModel(list(res), y_col)
    r = measure_consequence_gain(base, cons, [f"s{i}" for i in range(n)],
                                 _question(), in_mask, y_col)
    assert r["hard_quartile_delta"] == 0.0
    assert r["depth_slope_delta"] == 0.0
    assert r["earns_slot"] is True


def test_measure_reports_regression():
    """When the consequence model makes hard cases WORSE, earns_slot=False."""
    n = 40
    labels = ["a", "b", "c", "d"]
    in_mask = [True] * n
    y_col = list(range(4)) * (n // 4)
    base = _StubModel(_make_results(n, 1.0, seed=3), y_col)   # hard all right
    cons = _StubModel(_make_results(n, 0.0, seed=3), y_col)   # hard all wrong
    r = measure_consequence_gain(base, cons, [f"s{i}" for i in range(n)],
                                 _question(), in_mask, y_col)
    assert r["hard_quartile_delta"] < 0
    assert r["earns_slot"] is False
