"""Refusals — the cost-derived boundary, the taxonomy, and the invariants.

The design is grounded in Barbosa (arXiv:2609.28940): "review" is the refusal and
its threshold follows from the declared cost matrix. So the tests that matter
check that the boundary is *derived* rather than asserted, and that a battery
cannot be passed by a layer that asserts nothing.
"""
import json

import numpy as np
import pytest

from src.decision.refusals import (ACTION_TO_LOSS, BatterySpec, CANNOT_DO, PROBES, REASONS,
                                   SHOULD_NOT_DO, TAXONOMY, Invariant, LossMatrix,
                                   PolicyCase, Probe, battery_is_meaningful,
                                   battery_report, check_invariants,
                                   load_policy_cases, order_invariance,
                                   run_policy_cases, run_probes,
                                   threshold_monotonicity)
from src.decision.schema import Prediction

LABELS = ["cancel transfer", "card arrival", "balance", "exchange rate",
          "lost card", "top up"]


def _pred(action, top=None):
    top = top or "a"
    return Prediction(qtype="choice", labels=["a", "b"],
                      probabilities={top: 0.6, "b": 0.4},
                      answer_confidence=0.6, entropy_confidence=0.3,
                      max_score=1.0, prediction_set=[top], action=action, alpha=0.1)


class Always:
    def __init__(self, action, top=None):
        self.action, self.top = action, top

    def decide(self, state, question):
        return _pred(self.action, self.top)


class WellBehaved:
    """Declines everything except the verbatim control."""

    def decide(self, state, question):
        if state == LABELS[0]:
            return _pred("answer", LABELS[0])
        return _pred("escalate")


# ── the cost matrix: where the boundary comes from ───────────────────────────


def test_defaults_follow_the_papers_typical_engagement():
    """cfp >> cr and cfn >> cr: asserting a false finding and discarding a real
    one both cost more than reviewing."""
    L = LossMatrix()
    assert L.cfp > L.cr and L.cfn > L.cr


def test_thresholds_are_derived_not_chosen():
    L = LossMatrix(cfp=10.0, cfn=20.0, cr=1.0)
    assert L.thresholds() == {"discard_below": pytest.approx(0.05),
                              "assert_above": pytest.approx(0.90)}


def test_the_three_regions_map_to_the_three_actions():
    L = LossMatrix()
    assert L.optimal_action(0.01) == "discard"
    assert L.optimal_action(0.50) == "review"
    assert L.optimal_action(0.99) == "assert"


def test_a_recall_posture_widens_the_assert_zone():
    """A deployment that fears missing a real finding more than asserting a false
    one must assert more — and that must fall out of the costs, not a config knob."""
    strict = LossMatrix(cfp=10.0, cfn=20.0, cr=1.0)
    recall = LossMatrix(cfp=2.0, cfn=50.0, cr=1.0)
    assert recall.thresholds()["assert_above"] < strict.thresholds()["assert_above"]


def test_ties_resolve_toward_the_safer_action():
    """At p where assert and review cost the same, review wins — the layer must
    never break a tie in favour of asserting."""
    L = LossMatrix(cfp=10.0, cfn=20.0, cr=1.0)
    p = 1.0 - L.cr / L.cfp                       # exactly on the boundary
    assert L.expected_loss(p)["assert"] == pytest.approx(L.expected_loss(p)["review"])
    assert L.optimal_action(p) == "review"


def test_expected_loss_is_monotone_in_p():
    L = LossMatrix()
    a = [L.expected_loss(p)["assert"] for p in (0.1, 0.5, 0.9)]
    assert a[0] > a[1] > a[2]                    # asserting gets cheaper as p rises


def test_invalid_costs_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        LossMatrix(cfp=-1.0)
    with pytest.raises(ValueError, match="review cost"):
        LossMatrix(cr=0.0)


def test_action_mapping_covers_the_schema():
    assert set(ACTION_TO_LOSS) == {"answer", "clarify", "escalate", "abstain"}
    assert ACTION_TO_LOSS["clarify"] == ACTION_TO_LOSS["escalate"] == "review"


# ── the taxonomy ─────────────────────────────────────────────────────────────


def test_taxonomy_splits_capability_from_policy():
    """Brahman et al.'s finding: taxonomies cover policy refusals and ignore
    capability ones, while a decision layer is almost entirely the latter."""
    fams = {r.family for r in TAXONOMY}
    assert fams == {CANNOT_DO, SHOULD_NOT_DO}
    assert sum(r.family == CANNOT_DO for r in TAXONOMY) > sum(
        r.family == SHOULD_NOT_DO for r in TAXONOMY)


def test_the_policy_reason_has_no_built_in_verification():
    """This layer has no content policy; inventing one would be the model author
    making a deployer's decision for them."""
    assert REASONS["policy"].verified_by == "none"
    assert REASONS["policy"].family == SHOULD_NOT_DO


def test_every_reason_says_how_it_is_verified():
    assert all(r.verified_by for r in TAXONOMY)
    assert all(r.definition for r in TAXONOMY)


def test_reasons_reference_real_invariants_or_probes():
    from src.decision.refusals import INVARIANTS
    ids = {i.id for i in INVARIANTS} | {"I3", "I6"}
    probe_reasons = {p.reason for p in PROBES}
    for r in TAXONOMY:
        if r.verified_by.startswith("invariant:"):
            assert r.verified_by.split(":")[1] in ids, r.id
        elif r.verified_by.startswith("probe:"):
            assert r.verified_by.split(":")[1] in probe_reasons, r.id


# ── invariants ───────────────────────────────────────────────────────────────


class _Gate:
    def __init__(self, tau_answer=None, tau_in_schema=None, t_prob=1.0):
        self.tau_answer, self.tau_in_schema, self.t_prob = tau_answer, tau_in_schema, t_prob


def _rows(*specs):
    return [{"top_prob": p, "max_score": s, "action": a} for p, s, a in specs]


def test_i1_catches_asserting_below_the_bar():
    rows = _rows((0.10, 1.0, "answer"), (0.80, 1.0, "answer"))
    out = {i["id"]: i for i in check_invariants(rows, {"gate": _Gate(tau_answer=0.5)})}
    assert out["I1"]["status"] == "fail" and "1 asserted" in out["I1"]["detail"]


def test_i1_passes_when_the_bar_is_respected():
    rows = _rows((0.10, 1.0, "escalate"), (0.80, 1.0, "answer"))
    out = {i["id"]: i for i in check_invariants(rows, {"gate": _Gate(tau_answer=0.5)})}
    assert out["I1"]["status"] == "pass"


def test_an_unfitted_bar_is_deferred_never_passed():
    """The fail-open lesson: 'we did not fit it' must not read as 'fine'."""
    out = {i["id"]: i for i in check_invariants(_rows((0.1, 1.0, "answer")),
                                                {"gate": _Gate()})}
    assert out["I1"]["status"] == "deferred" and out["I2"]["status"] == "deferred"
    assert "not evaluable" in out["I1"]["detail"]


def test_i4_flags_an_ungated_action():
    out = {i["id"]: i for i in check_invariants(_rows((0.9, 1.0, "unevaluated")),
                                                {"gate": _Gate()})}
    assert out["I4"]["status"] == "fail"


def test_i5_checks_the_action_is_cost_optimal():
    """The paper's decision rule read backwards: if a deployment declares its
    costs, the thresholds in force must follow from them."""
    L = LossMatrix(cfp=10.0, cfn=20.0, cr=1.0)
    ok = _rows((0.99, 1.0, "answer"), (0.5, 1.0, "escalate"), (0.01, 1.0, "abstain"))
    bad = _rows((0.5, 1.0, "answer"))            # should be review at p=0.5
    ctx = {"gate": _Gate(), "loss": L}
    assert {i["id"]: i for i in check_invariants(ok, ctx)}["I5"]["status"] == "pass"
    assert {i["id"]: i for i in check_invariants(bad, ctx)}["I5"]["status"] == "fail"


def test_a_raising_invariant_is_an_error_not_a_pass():
    def boom(rows, ctx):
        raise RuntimeError("nope")

    inv = Invariant("X", "policy", "s", boom)
    from src.decision import refusals
    original = refusals.INVARIANTS
    refusals.INVARIANTS = (inv,)
    try:
        out = check_invariants(_rows((0.9, 1.0, "answer")), {"gate": _Gate()})
        assert out[0]["status"] == "error" and "nope" in out[0]["detail"]
    finally:
        refusals.INVARIANTS = original


def test_order_invariance_detects_position_bias():
    class PositionBiased:
        def decide(self, state, question):
            return _pred("answer", question.options[0])   # always the first option

    assert order_invariance(PositionBiased(), LABELS, ["a text"], cap=5)[0] is False


def test_order_invariance_passes_for_a_stable_model():
    class Stable:
        def decide(self, state, question):
            return _pred("escalate")

    assert order_invariance(Stable(), LABELS, ["a text"], cap=5)[0] is True


def test_threshold_monotonicity_holds_for_a_threshold_rule():
    """The additive principle: the layer may only lower assertions, so raising
    the bar cannot increase them."""
    scores = np.random.default_rng(0).normal(0, 1, (300, 5))
    ok, n, detail = threshold_monotonicity(scores, 1.0, LossMatrix())
    assert ok and n == 7


def test_threshold_monotonicity_is_deferred_without_scores():
    assert threshold_monotonicity(None, 1.0, LossMatrix()) is None


# ── probes ───────────────────────────────────────────────────────────────────


def test_probes_are_generated_from_the_label_space():
    rows = run_probes(Always("escalate"), LABELS)
    assert all(r["action"] == "escalate" for r in rows)
    assert "empty" in {r["probe"] for r in rows}


def test_a_layer_that_asserts_nothing_fails_the_control():
    rows = run_probes(Always("escalate"), LABELS)
    report = {"positive_control_passed": all(r["pass"] for r in rows
                                             if r["must"] == "answer")}
    assert battery_is_meaningful(report) is False


def test_answering_the_wrong_label_is_not_a_pass():
    probe = Probe("t", "policy", "answer", "why",
                  lambda o, t: ("x", list(o)), expect_option=True)
    opts = ["block ip", "close ticket"]
    assert run_probes(Always("answer", "block ip"), opts, (), (probe,))[0]["pass"]
    assert not run_probes(Always("answer", "close ticket"), opts, (), (probe,))[0]["pass"]


def test_probes_reject_an_empty_label_space():
    with pytest.raises(ValueError, match="label space"):
        run_probes(Always("answer"), [])


# ── the report ───────────────────────────────────────────────────────────────


def test_report_is_keyed_by_reason_with_statuses():
    rows = _rows((0.99, 1.0, "answer"), (0.5, 1.0, "escalate"), (0.01, 1.0, "abstain"))
    rep = battery_report(WellBehaved(), LABELS, rows, BatterySpec(
        sample_texts=["some text"], gate=_Gate(tau_answer=0.5)))
    assert set(rep["reasons"]) == set(REASONS)
    assert rep["loss"]["assert_above"] == pytest.approx(0.9)
    assert all(v["status"] in ("pass", "fail", "deferred", "unverified")
               for v in rep["reasons"].values())


def test_the_policy_reason_is_reported_unverified_not_passed():
    rows = _rows((0.99, 1.0, "answer"))
    rep = battery_report(WellBehaved(), LABELS, rows,
                         BatterySpec(gate=_Gate(tau_answer=0.5)))
    assert rep["reasons"]["policy"]["status"] == "unverified"
    assert "policy" in rep["unverified_reasons"]


def test_over_refusal_is_reported_separately_from_unsafe_answers():
    rows = _rows((0.99, 1.0, "answer"))
    rep = battery_report(Always("escalate"), LABELS, rows,
                         BatterySpec(gate=_Gate(tau_answer=0.5)))
    assert rep["unsafe_answers"] == 0
    assert "verbatim_option" in rep["over_refusals"]


# ── deployment-supplied policy cases ─────────────────────────────────────────


def test_policy_cases_load_and_validate(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"id": "a", "state": "s", "why": "w"}]))
    assert load_policy_cases(p)[0].id == "a"


def test_policy_case_typos_are_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"id": "a", "state": "s", "why": "w", "mst": "x"}]))
    with pytest.raises(ValueError, match="unknown field"):
        load_policy_cases(p)


def test_policy_cases_must_decline():
    cases = (PolicyCase("a", "anything", "w"),)
    assert run_policy_cases(Always("escalate"), cases, LABELS)[0]["pass"]
    assert not run_policy_cases(Always("answer"), cases, LABELS)[0]["pass"]
