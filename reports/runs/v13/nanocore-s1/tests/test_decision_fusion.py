"""Late fusion — decision-level aggregation of per-modality probabilities.

The v27 multimodal leg measured mean-pooled image+text (.428) BELOW
image-only (.433): early fusion at the embedding layer destroys signal
when the modalities are not commensurable. Late fusion scores each
modality separately and aggregates the predictions (Baltrusaitis et al.,
*Multimodal Deep Learning* §4.2.4). Every piece exists — decide() per
modality — so these tests exercise the aggregation rule and the routing
with a stub model; no encoder runs.
"""
import numpy as np
import pytest

from src.decision.fusion import RULES, aggregate, late_fusion_decide


class _StubModel:
    """decide() returns a fixed distribution keyed off the state's first
    character, so different modality groups produce different vectors."""

    class _P:
        def __init__(self, probs):
            self.probabilities = probs
            self.action = "unevaluated"

    def __init__(self, table):
        self.table = table  # state-key -> {option: prob}
        self.calls = []

    def decide(self, state, question):
        self.calls.append(state)
        key = state[0] if isinstance(state, (list, tuple)) else state
        key = key[0] if isinstance(key, (list, tuple)) else key
        key = str(key)[:1]
        return self._P(dict(self.table[key]))


_OPTS = ["a", "b", "c"]


def _vec(**kw):
    v = {o: kw.get(o, 0.0) for o in _OPTS}
    t = sum(v.values())
    return {o: x / t for o, x in v.items()}


# ── aggregation rules ─────────────────────────────────────────────────────

def test_mean_rule_averages():
    p1 = _vec(a=1.0, b=0.0, c=0.0)
    p2 = _vec(a=0.0, b=1.0, c=0.0)
    out = aggregate([p1, p2], rule="mean")
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] == pytest.approx(0.0)


def test_product_rule_sharpens_agreement():
    p1 = _vec(a=0.8, b=0.1, c=0.1)
    p2 = _vec(a=0.7, b=0.2, c=0.1)
    mean = aggregate([p1, p2], rule="mean")
    prod = aggregate([p1, p2], rule="product")
    # both favor 'a'; the geometric mean concentrates more mass there
    assert prod["a"] > mean["a"]


def test_product_rule_concentrates_on_supported_options():
    """Geometric mean of disjoint supports: the option NEITHER modality
    supports collapses to ~0, the two supported options split the mass —
    the 'experts must concur' semantics, not uniform-over-all."""
    p1 = _vec(a=1.0, b=0.0, c=0.0)
    p2 = _vec(a=0.0, b=1.0, c=0.0)
    out = aggregate([p1, p2], rule="product")
    assert out["a"] == pytest.approx(0.5)
    assert out["b"] == pytest.approx(0.5)
    assert out["c"] < 1e-3


def test_max_rule_keeps_any_modalitys_support():
    p1 = _vec(a=0.9, b=0.05, c=0.05)
    p2 = _vec(a=0.1, b=0.8, c=0.1)
    out = aggregate([p1, p2], rule="max")
    assert out["a"] > out["c"] and out["b"] > out["c"]


def test_weights_shift_the_aggregation():
    p1 = _vec(a=1.0, b=0.0, c=0.0)
    p2 = _vec(a=0.0, b=1.0, c=0.0)
    light = aggregate([p1, p2], weights=[0.9, 0.1], rule="mean")
    heavy = aggregate([p1, p2], weights=[0.1, 0.9], rule="mean")
    assert light["a"] > light["b"]
    assert heavy["b"] > heavy["a"]


def test_weights_are_normalized():
    p1 = _vec(a=1.0, b=0.0, c=0.0)
    p2 = _vec(a=0.0, b=1.0, c=0.0)
    out = aggregate([p1, p2], weights=[3.0, 1.0], rule="mean")
    assert out["a"] == pytest.approx(0.75)


def test_output_is_a_distribution():
    out = aggregate([_vec(a=0.5, b=0.3, c=0.2), _vec(a=0.2, b=0.5, c=0.3)],
                    rule="product")
    assert sum(out.values()) == pytest.approx(1.0)


# ── contract errors ───────────────────────────────────────────────────────

def test_mismatched_option_sets_raise():
    with pytest.raises(ValueError, match="same option set"):
        aggregate([{"a": 1.0}, {"b": 1.0}])


def test_bad_rule_raises():
    with pytest.raises(ValueError, match="rule must be one of"):
        aggregate([_vec(a=1.0)], rule="median")


def test_negative_weights_raise():
    with pytest.raises(ValueError, match="non-negative"):
        aggregate([_vec(a=1.0), _vec(b=1.0)], weights=[-0.5, 1.5])


def test_zero_total_weights_raise():
    with pytest.raises(ValueError, match="positive"):
        aggregate([_vec(a=1.0), _vec(b=1.0)], weights=[0.0, 0.0])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one"):
        aggregate([])


def test_wrong_weight_count_raises():
    with pytest.raises(ValueError, match="weights for"):
        aggregate([_vec(a=1.0), _vec(b=1.0)], weights=[1.0])


# ── the routing: decide per modality, then aggregate ──────────────────────

def test_late_fusion_decide_scores_each_modality_separately():
    """The whole point: each modality is decided on its own state, and
    the aggregation combines the predictions — not the embeddings."""
    model = _StubModel({
        "i": _vec(a=1.0, b=0.0, c=0.0),   # image favors a
        "t": _vec(a=0.0, b=1.0, c=0.0),   # text favors b
    })
    probs, per_modality = late_fusion_decide(
        model, [["image item"], ["text item"]], None)
    assert len(model.calls) == 2
    assert len(per_modality) == 2
    assert probs["a"] == pytest.approx(0.5)
    assert probs["b"] == pytest.approx(0.5)


def test_late_fusion_returns_auditable_per_modality_vectors():
    """The fusion must be inspectable: which modality drove the answer
    is a property of the decision, not a hidden average."""
    model = _StubModel({"i": _vec(a=1.0), "t": _vec(b=1.0)})
    _, per_modality = late_fusion_decide(model, [["i"], ["t"]], None)
    assert per_modality[0]["a"] == pytest.approx(1.0)
    assert per_modality[1]["b"] == pytest.approx(1.0)


def test_late_fusion_respects_rule_and_weights():
    model = _StubModel({"i": _vec(a=1.0), "t": _vec(b=1.0)})
    probs, _ = late_fusion_decide(model, [["i"], ["t"]], None,
                                  weights=[0.75, 0.25], rule="mean")
    assert probs["a"] == pytest.approx(0.75)


def test_rules_constant_matches_aggregate():
    assert set(RULES) == {"mean", "product", "max"}


def test_single_modality_is_identity():
    """With one modality the fusion must be a passthrough — late fusion
    with nothing to fuse is the modality's own decision."""
    model = _StubModel({"t": _vec(a=0.7, b=0.2, c=0.1)})
    probs, per_modality = late_fusion_decide(model, [["t"]], None)
    assert probs == per_modality[0]
