"""Delivery artifacts — the bundle digest, the model card, the external rubric.

These are what a recipient of the model actually receives. The tests that matter
are the ones asserting they refuse to overstate: a card that renders an
unmeasured criterion as a pass, or a digest that cannot distinguish two bundles,
would each be worse than shipping nothing.
"""
import json

import pytest

from src.decision import rubric
from src.decision.model import bundle_digest
from src.decision.modelcard import REQUIRED_SECTIONS, build_card, render_markdown


# ── the bundle digest ────────────────────────────────────────────────────────


def _bundle(tmp_path, name="b", scorer=b"weights-1", gate='{"t_set": 1.0}'):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "scorer.pt").write_bytes(scorer)
    (d / "gate.json").write_text(gate)
    (d / "manifest.json").write_text("{}")
    return d


def test_identical_content_gives_the_same_digest(tmp_path):
    assert bundle_digest(_bundle(tmp_path, "a")) == bundle_digest(_bundle(tmp_path, "b"))


def test_a_changed_scorer_changes_the_digest(tmp_path):
    a = bundle_digest(_bundle(tmp_path, "a"))
    b = bundle_digest(_bundle(tmp_path, "b", scorer=b"weights-2"))
    assert a != b


def test_a_changed_calibration_changes_the_digest(tmp_path):
    """The gate calibration is part of the artifact — a refitted qhat is a
    different model even when the weights are identical."""
    a = bundle_digest(_bundle(tmp_path, "a"))
    b = bundle_digest(_bundle(tmp_path, "b", gate='{"t_set": 4.0}'))
    assert a != b


def test_a_moved_bundle_keeps_its_digest(tmp_path):
    """Content-addressed: the path is not part of the identity, so a bundle
    copied between hosts can still be proven to be the one that was qualified."""
    a = _bundle(tmp_path, "here")
    first = bundle_digest(a)
    b = _bundle(tmp_path / "elsewhere", "there")
    assert bundle_digest(b) == first


def test_missing_and_empty_bundles_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="no bundle at"):
        bundle_digest(tmp_path / "nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        bundle_digest(empty)


# ── the model card ───────────────────────────────────────────────────────────

def _qualification(status="deferred", verdict="not_qualified"):
    return {
        "verdict": verdict,
        "provenance": {"git_commit": "abc1234", "bundle_sha256": "deadbeef"},
        "criteria": [
            {"id": "A1", "severity": "must_pass", "status": "pass", "statement": "s"},
            {"id": "D1", "severity": "must_pass", "status": status, "statement": "s"},
            {"id": "E5", "severity": "must_fix", "status": "fail", "statement": "s"},
        ],
    }


def test_the_card_has_every_required_section():
    built = build_card(_qualification())
    assert set(REQUIRED_SECTIONS) <= set(built["card"])
    assert all(built["card"][s] for s in REQUIRED_SECTIONS)


def test_an_empty_section_is_refused_not_shipped():
    """A card with an empty Limitations is the standard failure of this artifact."""
    import src.decision.modelcard as mc

    original = mc.REQUIRED_SECTIONS
    mc.REQUIRED_SECTIONS = original + ("a_section_that_does_not_exist",)
    try:
        with pytest.raises(ValueError, match="missing required sections"):
            build_card(_qualification())
    finally:
        mc.REQUIRED_SECTIONS = original


def test_unmeasured_criteria_are_not_rendered_as_passes():
    built = build_card(_qualification(status="deferred"))
    analyses = built["card"]["quantitative_analyses"]
    assert analyses["not_measured"] == ["D1"]
    assert "D1" not in analyses["failed"]


def test_out_of_scope_names_the_modalities_never_run():
    """The failure mode of a card is a reader assuming coverage that was never
    measured, so the untested modalities are stated as limits."""
    oos = " ".join(build_card(_qualification())["card"]["intended_use"]["out_of_scope"])
    assert "audio and video" in oos and "generation" in oos


def test_the_card_names_the_artifact_it_describes():
    built = build_card(_qualification())
    assert built["card"]["model_details"]["artifact_digest"] == "deadbeef"
    assert built["card"]["model_details"]["version"] == "abc1234"


def test_a_missing_digest_is_shouted_not_blank():
    q = _qualification()
    q["provenance"] = {"git_commit": "abc1234"}
    assert build_card(q)["card"]["model_details"]["artifact_digest"] == "NOT RECORDED"


def test_the_independence_gap_is_stated_in_the_card():
    """SR 11-7's requirement is the largest outstanding gap; a card that omits it
    is describing a different system."""
    caveats = build_card(_qualification())["card"]["caveats"]
    assert "SR 11-7" in caveats["independence"]


def test_the_card_reads_headroom_from_both_real_shapes():
    """The v23 failure: `adapt()` writes `{"zeroshot_test_acc": x}` while a summary
    carries the value directly, and reading only the first crashed the generator —
    leaving a run that looked complete with no delivery artifact at all."""
    report_shape = build_card(_qualification(), {"headroom": {"zeroshot_test_acc": 0.709}})
    summary_shape = build_card(_qualification(), {"headroom": 0.709})
    assert report_shape["card"]["training_data"]["headroom"] == 0.709
    assert summary_shape["card"]["training_data"]["headroom"] == 0.709
    assert build_card(_qualification(), {})["card"]["training_data"]["headroom"] is None


def test_model_index_is_hub_shaped():
    built = build_card(_qualification(), {"test": {"accuracy": 0.9, "brier": 0.1}},
                       dataset="clinc150")
    mi = built["model_index"]
    assert mi["name"] and mi["results"][0]["dataset"]["name"] == "clinc150"
    assert mi["results"][0]["metrics"][0]["type"] == "accuracy"


def test_render_markdown_is_front_matter_plus_sections():
    md = render_markdown(build_card(_qualification(),
                                    {"test": {"accuracy": 0.9}}, dataset="d"))
    assert md.startswith("---") and "model-index:" in md
    assert "# nanocore-s1" in md and "## Caveats" in md and "Evaluation results" in md


# ── the external rubric ──────────────────────────────────────────────────────


def test_rubric_counts_are_consistent():
    rep = rubric.coverage()
    o = rep["overall"]
    assert o["covered"] + o["partial"] + o["missing"] == o["n"] == len(rubric.ITEMS)
    assert 0.0 <= o["fraction_covered"] <= 1.0
    assert sum(c["n"] for c in rep["by_category"].values()) == o["n"]


def test_the_unclaimable_item_is_marked_missing():
    """Offline/online correlation needs a deployment. Claiming it would be the
    exact failure this whole exercise exists to prevent."""
    rep = rubric.coverage()
    online = [i for i in rubric.ITEMS if "online impact" in i.practice]
    assert online and online[0].status == rubric.MISSING


def test_every_covered_item_names_where_it_is_implemented():
    assert all(i.where and i.where != "-" for i in rubric.ITEMS if i.status != rubric.MISSING)


def test_gaps_list_only_non_covered_items():
    rep = rubric.coverage()
    assert all(g["status"] != rubric.COVERED for g in rep["gaps"])
    assert len(rep["gaps"]) == rep["overall"]["partial"] + rep["overall"]["missing"]


def test_render_is_readable():
    out = rubric.render()
    assert "ML Test Score coverage" in out and "gaps:" in out


# ── the package's front door ─────────────────────────────────────────────────


def test_the_package_exports_the_shipped_model():
    """The front door must describe the model we ship.

    It previously exported only ``NanoCoreS1`` — the abandoned decoder-era path —
    and omitted ``DecisionModel``, the gate, the scorers and ``adapt``. Nothing in
    the repo then said what the model *was*, which is how the CLI ended up
    functioning as the de facto interface.
    """
    import src.decision as d

    for name in ("DecisionModel", "Prediction", "Question", "State", "adapt",
                 "AdaptConfig", "bundle_digest", "ConformalGate", "TaskHead",
                 "CosineScorer", "StateEncoder"):
        assert name in d.__all__, f"{name} missing from the public API"
        assert hasattr(d, name), f"{name} declared but not importable"


def test_the_deprecated_path_is_not_the_front_door():
    """Still importable from its own module for the record it holds; not offered
    as the model."""
    import src.decision as d
    from src.decision.model import NanoCoreS1       # module import still works

    assert NanoCoreS1 is not None
    assert "NanoCoreS1" not in d.__all__
    assert not hasattr(d, "NanoCoreS1")


def test_the_contract_is_one_call():
    """``decide(state, question) -> Prediction`` is the whole interface; if that
    signature moves, every downstream consumer moves with it."""
    import inspect

    from src.decision import DecisionModel, Prediction
    sig = inspect.signature(DecisionModel.decide)
    assert list(sig.parameters)[:2] == ["self", "state"]
    assert "question" in sig.parameters
    assert sig.return_annotation in ("Prediction", Prediction)
