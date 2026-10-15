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


def test_action_filter_restricts_to_target_population():
    """action_filter='escalate' keeps only states where the base model
    escalates — the population the re-ranker targets. Answered states are
    dropped from the measurement."""
    n = 40
    labels = ["a", "b", "c", "d"]
    in_mask = [True] * n
    y_col = list(range(4)) * (n // 4)
    # base: even=answer, odd=escalate. consequence improves the odd (escalate) ones.
    base_res = [(0.95, False, True) if i % 2 == 0 else (0.2, True, False)
                for i in range(n)]
    cons_res = [(0.95, False, True) if i % 2 == 0 else (0.2, True, True)
                for i in range(n)]
    base = _StubModel(base_res, y_col)
    cons = _StubModel(cons_res, y_col)
    # unfiltered: consequence improves the hard (odd) states
    r_all = measure_consequence_gain(base, cons, [f"s{i}" for i in range(n)],
                                     _question(), in_mask, y_col)
    # filtered to escalate: only the odd states, where consequence flips wrong->right
    base2 = _StubModel(base_res, y_col)
    cons2 = _StubModel(cons_res, y_col)
    r_esc = measure_consequence_gain(base2, cons2, [f"s{i}" for i in range(n)],
                                     _question(), in_mask, y_col,
                                     action_filter="escalate")
    assert r_esc["n"] < r_all["n"]            # fewer states (escalate-only)
    assert r_esc["n"] == n // 2               # exactly the odd states
    # on the escalate population, consequence strictly improves accuracy
    assert r_esc["consequence"]["accuracy"] > r_esc["base"]["accuracy"]
    assert r_esc["hard_quartile_delta"] > 0
    assert r_esc["earns_slot"] is True


def test_first_survivor_metric_detects_rerank_gain():
    """The organ's real signal: first_survivor_hit. When the consequence model
    reorders prediction_set so the true intent is first (base put it second),
    first_survivor_delta > 0 even though argmax accuracy is unchanged."""
    class _PredSet:
        def __init__(self, conf, pred_set, y_i, labels):
            self.action = "escalate"
            self.answer_confidence = conf
            self.labels = labels
            # argmax is ALWAYS the true label (single-shot correct) — the
            # re-ranker never changes this; only pred_set order differs.
            self.probabilities = {l: (1.0 if l == labels[y_i] else 0.0)
                                  for l in labels}
            self.prediction_set = pred_set
    class _ReorderModel:
        def __init__(self, first_is_true, y_col, labels):
            self.first_is_true = first_is_true
            self.y_col = y_col
            self.labels = labels
            self.i = 0
        def decide(self, state, question):
            yi = int(self.y_col[self.i]); self.i += 1
            true_lbl = self.labels[yi]
            others = [l for l in self.labels if l != true_lbl]
            if self.first_is_true:
                pset = [true_lbl] + others       # true intent first
            else:
                pset = [others[0], true_lbl] + others[1:]  # true intent second
            return _PredSet(0.2, pset, yi, self.labels)
    n = 20
    labels = ["a", "b", "c", "d"]
    y_col = list(range(4)) * (n // 4)
    in_mask = [True] * n
    base = _ReorderModel(False, y_col, labels)    # true intent NOT first
    cons = _ReorderModel(True, y_col, labels)     # true intent first
    r = measure_consequence_gain(base, cons, [f"s{i}" for i in range(n)],
                                 _question(), in_mask, y_col,
                                 action_filter="escalate")
    # argmax accuracy identical (re-ranker never touches probabilities)
    assert r["base"]["accuracy"] == r["consequence"]["accuracy"]
    # but the first-survivor hit rate jumps — the organ's real value
    assert r["base"]["first_survivor_hit"] == 0.0
    assert r["consequence"]["first_survivor_hit"] == 1.0
    assert r["first_survivor_delta"] == 1.0
    assert r["earns_slot"] is True
