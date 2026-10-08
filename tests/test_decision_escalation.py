"""Cost-calibrated escalation — features, isotonic risk, and the cost rule."""
import numpy as np
import pytest

from src.decision.escalation import (Costs, calibrated_actions,
                                     decision_features, evaluate_policy,
                                     feature_auroc, fit_error_model,
                                     select_error_model, unsafe_at_escalation)


def _features(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, 4)) * 0.5
    return decision_features(scores, t_prob=1.0, qhat=0.9)


def test_features_are_named_and_shaped():
    f = _features(50)
    assert f.names == ("margin", "top_prob", "set_size", "entropy", "max_score")
    assert f.values.shape == (50, len(f.names))
    assert np.isfinite(f.values).all()


def test_features_come_from_the_one_source_of_decision_statistics():
    """A second code path for these numbers is how calibration drifted twice."""
    from src.decision.slow import observe_row

    rng = np.random.default_rng(4)
    scores = rng.standard_normal((25, 5))
    features = decision_features(scores, t_prob=0.7, qhat=0.9)
    for i, row in enumerate(scores):
        obs, members = observe_row(row, t_prob=0.7, qhat=0.9)
        assert features.column("top_prob")[i] == pytest.approx(obs.top_prob)
        assert features.column("set_size")[i] == len(members)
        assert features.column("entropy")[i] == pytest.approx(obs.entropy)
        assert features.column("margin")[i] == pytest.approx(obs.margin)
        assert features.column("max_score")[i] == pytest.approx(obs.max_score)


def test_features_reject_a_score_vector():
    with pytest.raises(ValueError):
        decision_features(np.array([1.0, 2.0]), t_prob=1.0, qhat=0.9)


def test_costs_ratio_is_the_threshold():
    assert Costs(wrong=10.0, escalate=1.0).threshold == pytest.approx(0.1)
    assert Costs(wrong=4.0, escalate=2.0).threshold == pytest.approx(0.5)
    with pytest.raises(ValueError):
        Costs(wrong=0.0)


def test_auroc_flips_a_feature_that_runs_the_other_way():
    """A higher margin is *safer*, so the signal must be oriented as risk."""
    f = _features(300, seed=1)
    wrong = f.column("margin") < np.median(f.column("margin"))   # low margin → wrong
    auroc = feature_auroc(f, wrong)
    assert auroc["margin"] > 0.9            # oriented, not raw


def test_isotonic_risk_is_monotone_in_the_feature():
    f = _features(300, seed=2)
    wrong = f.column("margin") < np.median(f.column("margin"))
    model = fit_error_model(f, wrong, "margin")
    assert model.flipped                    # low margin ⇒ high risk, so flip it
    assert model.auroc > 0.9                # the *oriented* ranking quality
    ordered = np.sort(f.column("margin"))
    probe = type(f)(f.names, np.column_stack([
        ordered,
        np.zeros_like(ordered), np.zeros_like(ordered),
        np.zeros_like(ordered), np.zeros_like(ordered)]))
    risk = model.predict(probe)
    assert np.all(np.diff(risk) <= 1e-9)    # non-increasing as margin rises


def test_selection_prefers_the_feature_that_generalises():
    f = _features(400, seed=3)
    wrong = f.column("margin") < np.median(f.column("margin"))
    model, ranking = select_error_model(f, wrong, f, wrong)
    assert model.feature == "margin"
    assert ranking["margin"] >= max(ranking.values()) - 1e-9


def test_cost_rule_escalates_above_the_ratio():
    probs = np.array([0.01, 0.05, 0.10, 0.40])
    sizes = np.array([1, 1, 5, 5])
    actions = calibrated_actions(probs, sizes, Costs(wrong=10.0), k_clarify=3)
    assert list(actions) == ["answer", "answer", "escalate", "escalate"]
    # a narrow set becomes clarify instead of escalate — cheaper than a handoff
    narrow = calibrated_actions(probs, np.array([1, 1, 1, 1]),
                                Costs(wrong=10.0), k_clarify=3)
    assert list(narrow) == ["answer", "answer", "clarify", "clarify"]


def test_higher_wrong_cost_escalates_more():
    probs = np.linspace(0, 1, 100)
    sizes = np.full(100, 9)
    cheap = calibrated_actions(probs, sizes, Costs(wrong=2.0), k_clarify=3)
    dear = calibrated_actions(probs, sizes, Costs(wrong=50.0), k_clarify=3)
    assert (dear == "escalate").sum() > (cheap == "escalate").sum()


def test_unsafe_at_escalation_reads_the_curve_at_a_matched_rate():
    """Hand off the riskiest half; the safest half is what gets answered."""
    risk = np.array([0.9, 0.8, 0.1, 0.05])
    wrong = np.array([True, True, False, False])
    assert unsafe_at_escalation(risk, wrong, 0.5) == pytest.approx(0.0)
    assert unsafe_at_escalation(risk, wrong, 0.0) == pytest.approx(0.5)
    assert unsafe_at_escalation(risk, wrong, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        unsafe_at_escalation(risk, wrong[:2], 0.5)
    with pytest.raises(ValueError):
        unsafe_at_escalation(risk, wrong, 1.5)


def test_evaluate_policy_reports_the_operating_point():
    actions = ["answer", "answer", "clarify", "escalate"]
    out = evaluate_policy(actions, np.array([False, True, True, True]))
    assert out["escalation_rate"] == pytest.approx(0.25)
    assert out["clarify_rate"] == pytest.approx(0.25)
    assert out["unsafe_rate"] == pytest.approx(0.25)
