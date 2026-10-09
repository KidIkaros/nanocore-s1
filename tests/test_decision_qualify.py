"""The qualification gate — severities, and the two ways it must refuse to lie.

A gate is only worth having if it can say no, so the tests that matter are the
ones asserting it does: a failed blocker yields ``not_qualified``, absent
evidence yields ``deferred`` rather than a pass, and a check that *raises* is
not silently treated as satisfied.
"""
import json
from pathlib import Path

import pytest

from src.decision.qualify import MUST_FIX, MUST_PASS, evaluate

V20 = Path("reports/runs/s1_verify/v20/results.json")


def _good() -> dict:
    """Minimal evidence that satisfies every measurable criterion."""
    return {
        "provenance": {"bundle_sha256": "abc123"},
        "verdict": {"suite_green_on_kaggle": True, "bundle_roundtrip_identical": True},
        "breadth": {"datasets": {
            "a": {"status": "ran", "blocks": {"taskhead": {"log": {"point": 0.1},
                                                           "brier": 0.05},
                                              "tfidf_lr": {"log": {"point": 0.6},
                                                           "brier": 0.2}}},
            "mnli": {"status": "ran", "blocks": {"taskhead": {"log": {"point": 0.9},
                                                              "brier": 0.4,
                                                              "acc": {"point": 0.65}},
                                                 "tfidf_lr": {"log": {"point": 1.2},
                                                              "brier": 0.6}}},
        }},
        "cosine_leg": {"resolved": 0.48, "set_coverage_in_scope": 0.914,
                       "mean_set_size": 3.8, "alpha": 0.10},
        "taskhead_leg": {"set_coverage_in_scope": 0.988, "mean_set_size": 1.94,
                         "alpha": 0.10},
        "operate": {"monitor": {"escalate_rate": 0.63},
                    "handlers": {"counts": {"answer": 109}},
                    "registry": {"rolled_back_to": "v1", "current_after_rollback": "v1",
                                 "shadow": {"n": 300}}},
        "refusals": {"unsafe_answers": 0, "positive_control_passed": True,
                     "over_refusals": []},
        "multilingual": {"languages": {
            "en": {"status": "ran", "in_language_head_acc": {"point": 0.876},
                   "cross_lingual_en_head_acc": 0.876},
            "de": {"status": "ran", "in_language_head_acc": {"point": 0.797},
                   "cross_lingual_en_head_acc": 0.772}}},
        "multimodal": {"n_options": 4, "arms": {"image_only": {"block": {"acc": {"lo": 0.398}}}},
                       "vs_text_only": {"verdict": "improved"},
                       "vs_image_only": {"verdict": "improved"}},
        "readiness": {
            "cosine_leg": {
                "coverage_by_band": {"target": 0.90, "min_group_n": 200,
                                     "groups": {"low": {"n": 300, "coverage": 0.88},
                                                "mid": {"n": 300, "coverage": 0.95},
                                                "high": {"n": 300, "coverage": 0.97}},
                                     "undercovered": [], "small": []},
                "deferral": {"well_aimed": True},
                "rejection_by_intent": {"concentrated": []},
            },
            "taskhead_leg": {
                "coverage_by_band": {"target": 0.90, "min_group_n": 200,
                                     "groups": {"low": {"n": 300, "coverage": 0.91},
                                                "mid": {"n": 300, "coverage": 0.97},
                                                "high": {"n": 300, "coverage": 0.99}},
                                     "undercovered": [], "small": []},
                "deferral": {"well_aimed": True},
                "rejection_by_intent": {"concentrated": []},
            },
            "memorization": {"passed": True},
        },
    }


def test_a_complete_evidence_set_qualifies():
    rep = evaluate(_good())
    assert rep["verdict"] == "qualified", rep["must_pass_failed"]
    assert rep["must_pass_failed"] == [] and rep["must_fix_open"] == []


def test_a_failed_blocker_means_not_qualified():
    ev = _good()
    ev["verdict"]["bundle_roundtrip_identical"] = False
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert rep["must_pass_failed"] == ["A2"]


def test_a_failed_must_fix_does_not_block_but_is_reported():
    """The severity distinction: an open capability gap blocks the *claim* of
    pre-production, not the shipped behaviour."""
    ev = _good()
    ev["breadth"]["datasets"]["mnli"]["blocks"]["taskhead"]["acc"]["point"] = 0.449
    rep = evaluate(ev)
    assert rep["verdict"] == "qualified"
    assert rep["must_fix_open"] == ["E5"]


def test_absent_evidence_is_deferred_not_passed():
    """The fail-open lesson, applied to the gate itself: 'we did not look' must
    never read as 'fine'."""
    ev = _good()
    del ev["refusals"]
    rep = evaluate(ev)
    assert "D1" in rep["deferred"] and "D2" in rep["deferred"]
    assert "D1" not in rep["passed"]
    row = next(r for r in rep["criteria"] if r["id"] == "D1")
    assert row["status"] == "deferred" and row["pass"] is None
    assert "evidence absent" in row["note"]


def test_the_device_latency_criterion_is_deferred_by_construction():
    """Phase 8 has not run, so this can never pass by accident."""
    rep = evaluate(_good())
    assert "F2" in rep["deferred"]


def test_a_raising_check_is_not_a_pass(monkeypatch):
    from src.decision import qualify

    def boom(_):
        raise RuntimeError("check exploded")

    monkeypatch.setattr(qualify, "CRITERIA",
                        (qualify.Criterion("X1", "X", "explodes", MUST_PASS, "-", boom),))
    rep = qualify.evaluate(_good())
    assert rep["verdict"] == "not_qualified"
    assert rep["check_errors"] == ["X1"]
    # NOT deferred: "the check broke" means the gate cannot vouch for this
    # criterion, which is different from "we have not measured it yet".
    assert rep["criteria"][0]["status"] == "error"
    assert "check raised" in rep["criteria"][0]["note"]


def test_refusing_everything_fails_the_meaningfulness_criterion():
    ev = _good()
    ev["refusals"] = {"unsafe_answers": 0, "positive_control_passed": False,
                      "over_refusals": ["verbatim_option"]}
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert rep["must_pass_failed"] == ["D2", "D3"]


def test_over_refusal_blocks_on_its_own():
    """Declining an answerable input is a behaviour failure — and in a security
    deployment it is the expensive one, since legitimate defensive work shares
    its vocabulary with offensive work."""
    ev = _good()
    ev["refusals"]["over_refusals"] = ["control_block"]
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert rep["must_pass_failed"] == ["D3"]


def test_an_empty_escalate_rate_is_degenerate():
    ev = _good()
    ev["operate"]["monitor"]["escalate_rate"] = 1.0
    assert evaluate(ev)["must_pass_failed"] == ["C5"]


def test_every_criterion_states_its_claim_and_threshold():
    from src.decision.qualify import CRITERIA
    assert all(c.statement and c.threshold and c.id for c in CRITERIA)
    assert {c.severity for c in CRITERIA} <= {MUST_PASS, MUST_FIX, "report"}


def test_readiness_criteria_defer_when_the_leg_did_not_run():
    """A build whose kernel skipped the readiness cell must not pass G by
    omission — the checks defer, and deferred is listed, never a pass."""
    ev = _good()
    del ev["readiness"]
    rep = evaluate(ev)
    assert {"G1", "G2", "G4"} <= set(rep["deferred"])
    assert not ({"G1", "G2", "G4"} & set(rep["passed"]))


def test_the_qtype_criteria_cover_the_whole_typed_contract():
    """The contract declares choice/score/noul — score and noul must have real
    evidence, and a schema-bound bundle must refuse a noul question loudly."""
    ev = _good()  # no ordinal/noul legs -> both criteria defer, not pass
    rows = {r["id"]: r for r in evaluate(ev)["criteria"]}
    assert rows["B4"]["status"] == "deferred"
    assert rows["D4"]["status"] == "deferred"

    ev["ordinal"] = {"status": "ran"}
    ev["noul"] = {"status": "ran", "bundle_refuses": True}
    rows = {r["id"]: r for r in evaluate(ev)["criteria"]}
    assert rows["B4"]["status"] == "pass" and rows["D4"]["status"] == "pass"

    ev["noul"]["bundle_refuses"] = False
    rep = evaluate(ev)
    assert "D4" in rep["must_pass_failed"]  # silent out-of-schema scoring


def test_the_anchor_criterion_defers_without_a_shared_harness_run():
    """Self-reported numbers alone are not comparable claims — but the anchor
    is REPORT severity, so its absence is listed, not blocking."""
    ev = _good()
    rep = evaluate(ev)
    row = next(r for r in rep["criteria"] if r["id"] == "B3")
    assert row["status"] == "deferred" and row["severity"] == "report"

    ev["mteb_anchor"] = {"status": "ran", "tasks": {}}
    row = next(r for r in evaluate(ev)["criteria"] if r["id"] == "B3")
    assert row["status"] == "pass"


def test_an_undercovered_slice_fails_g1_even_with_a_good_marginal():
    """The whole point of G1: the marginal can be over target while a populated
    slice is catastrophically under-covered."""
    ev = _good()
    ev["readiness"]["taskhead_leg"]["coverage_by_band"]["groups"]["low"] = {
        "n": 400, "coverage": 0.60}  # 0.90 target - 0.15 tol = 0.75 floor
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert "G1" in rep["must_pass_failed"]


def test_misaimed_deferral_fails_g2():
    """Escalating items the model would have answered right is a misroute."""
    ev = _good()
    ev["readiness"]["cosine_leg"]["deferral"]["well_aimed"] = False
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert "G2" in rep["must_pass_failed"]


def test_a_failed_memorization_probe_fails_g4():
    ev = _good()
    ev["readiness"]["memorization"] = {"passed": False}
    rep = evaluate(ev)
    assert rep["verdict"] == "not_qualified"
    assert "G4" in rep["must_pass_failed"]


def test_over_rejection_is_reported_not_blocking():
    """Concentrated rejection surfaces as a REPORT row — it cannot change the
    verdict, but it must not silently pass."""
    ev = _good()
    ev["readiness"]["taskhead_leg"]["rejection_by_intent"] = {
        "concentrated": ["intent_x"]}
    rep = evaluate(ev)
    assert rep["verdict"] == "qualified"
    row = next(r for r in rep["criteria"] if r["id"] == "G3")
    assert row["status"] == "fail" and row["severity"] == "report"


@pytest.mark.skipif(not V20.exists(), reason="v20 artifacts not pulled")
def test_against_the_real_v20_artifact():
    """A regression anchor: the gate must read our own evidence and say no.

    v20 has no bundle digest and mnli at .449, so it cannot be qualified — and
    the reason must be those two, not a parsing accident.
    """
    rep = evaluate(json.loads(V20.read_text()),
                   provenance={"git_commit": "fa7e6f7", "bundle_sha256": None})
    assert rep["verdict"] == "not_qualified"
    assert rep["must_pass_failed"] == ["A3"]
    assert rep["must_fix_open"] == ["E5"]
    assert set(rep["deferred"]) >= {"D1", "D2", "F2"}
