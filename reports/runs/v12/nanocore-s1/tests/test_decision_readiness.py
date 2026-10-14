"""The readiness checks — each guards a failure mode a marginal number hides.

These are the four human-interaction readiness checks from
``docs/research/benchmarks-and-evaluation-readiness.md``: conditional coverage,
deferral quality, over-rejection, and the memorization probes. All pure array
logic, so the same code that runs on Kaggle runs here on synthetic inputs.
"""
import numpy as np

from src.decision import readiness


# ── confidence bands ─────────────────────────────────────────────────────────

def test_confidence_bands_split_on_tertiles_and_label_low_first():
    p = np.linspace(0.0, 1.0, 300)
    bands = readiness.confidence_bands(p)
    assert set(bands) == {"low", "mid", "high"}
    # the lowest-confidence items land in "low"
    assert bands[0] == "low" and bands[-1] == "high"


# ── conditional coverage ─────────────────────────────────────────────────────

def test_a_marginal_number_can_hide_an_undercovered_slice():
    # A dominant group at 95% + a populated hard group at 60% -> marginal ~0.91,
    # comfortably over the 0.90 target, while the hard group is undercovered.
    covered = np.array([True] * 1900 + [False] * 100 +      # easy: 2000 @ .95
                       [True] * 150 + [False] * 100)        # hard: 250 @ .60
    groups = np.array(["easy"] * 2000 + ["hard"] * 250)
    rep = readiness.group_coverage(covered, groups, alpha=0.10)
    assert rep["n"] == 2250
    # marginal coverage (2050/2250 = .911) is fine; the slice is not
    assert "hard" in rep["undercovered"] and "easy" not in rep["undercovered"]


def test_a_tiny_group_is_small_not_undercovered():
    """The Mondrian caveat: coverage on a group too small to trust is reported
    as small, never flagged as a coverage failure."""
    covered = np.array([True] * 500 + [False] * 5)
    groups = np.array(["big"] * 500 + ["tiny"] * 5)
    rep = readiness.group_coverage(covered, groups, alpha=0.10)
    assert "tiny" in rep["small"] and "tiny" not in rep["undercovered"]


# ── deferral quality ─────────────────────────────────────────────────────────

def test_deferral_is_well_aimed_when_it_targets_errors():
    # asserted items are mostly right; deferred items are the ones it'd miss
    actions = ["answer"] * 90 + ["escalate"] * 10
    correct = [True] * 80 + [False] * 10 + [False] * 10  # assert .89, defer 0.0
    rep = readiness.deferral_quality(actions, correct)
    assert rep["well_aimed"] is True
    assert rep["acc_asserted"] > rep["acc_deferred"]
    assert rep["error_enrichment"] and rep["error_enrichment"] > 1
    assert rep["breakeven_human_acc"] == rep["acc_deferred"]


def test_deferral_is_misaimed_when_it_escalates_easy_items():
    """Escalating items the model would have answered correctly is a misroute."""
    actions = ["answer"] * 50 + ["escalate"] * 50
    correct = [False] * 10 + [True] * 40 + [True] * 45 + [False] * 5  # defer .9
    rep = readiness.deferral_quality(actions, correct)
    assert rep["well_aimed"] is False


def test_deferral_reports_the_break_even_human_accuracy():
    """A human must beat the model's own accuracy on the deferred set for the
    handoff to lower system error (Mozannar et al. 2023)."""
    actions = ["answer"] * 50 + ["clarify"] * 50
    correct = [True] * 50 + [True] * 20 + [False] * 30  # deferred acc .4
    rep = readiness.deferral_quality(actions, correct)
    assert rep["breakeven_human_acc"] == 0.4


# ── over-rejection ───────────────────────────────────────────────────────────

def test_a_concentrated_rejection_is_surfaced():
    """Over-rejection bites a minority slice — a group can only outrun the
    overall rate by concentrating on a minority, since its own rejections pull
    the overall up. That is exactly the Pugnana & Ruggieri failure mode."""
    # A dominant low-rejection group + a small high-rejection group.
    actions = (["answer"] * 900 + ["escalate"] * 100 +   # A: 1000 items, 10%
               ["escalate"] * 60 + ["answer"] * 40)      # B: 100 items, 60%
    groups = ["A"] * 1000 + ["B"] * 100
    rep = readiness.rejection_by_group(actions, groups, min_group_n=50,
                                       concentration=3.0)
    # overall ~0.145; B at 0.60 is ~4.1x -> concentrated
    assert "B" in rep["concentrated"] and rep["max_ratio"] > 3.0


def test_uniform_rejection_is_not_concentrated():
    actions = ["answer"] * 60 + ["escalate"] * 40
    groups = ["A"] * 50 + ["B"] * 50
    rep = readiness.rejection_by_group(actions, groups, min_group_n=10)
    assert rep["concentrated"] == []


# ── memorization probes ──────────────────────────────────────────────────────

def test_memorization_passes_on_invariant_labels_and_no_answer_on_nothing():
    rep = readiness.memorization_verdict(
        rotation_consistency=1.0,
        withheld={"top_prob": 0.01, "answered": 0}, n_options=150)
    assert rep["passed"] is True


def test_a_withheld_state_that_gets_answered_fails_the_probe():
    rep = readiness.memorization_verdict(
        rotation_consistency=1.0,
        withheld={"top_prob": 0.9, "answered": 2}, n_options=150)
    assert rep["passed"] is False and rep["withholding_collapses"] is False


def test_a_positional_model_fails_rotation_invariance():
    rep = readiness.memorization_verdict(
        rotation_consistency=0.4,
        withheld={"top_prob": 0.01, "answered": 0}, n_options=150)
    assert rep["passed"] is False and rep["rotation_invariant"] is False


def test_absent_probe_evidence_does_not_pass():
    """Fail-closed: 'we did not run the withheld probe' is not a pass."""
    rep = readiness.memorization_verdict(rotation_consistency=1.0,
                                         withheld=None, n_options=150)
    assert rep["passed"] is False
