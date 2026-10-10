"""PairScorer — cosine(u,v) with the SentenceSimilarity prompt.

The v27 jev anchor root cause: pair tasks (paws .440, stsb .215,
boolq .583) scored a pooled state against options, destroying the
comparison signal. Item splitting gave per-item vectors; this scorer
reads them. EG2's native similarity primitive, never before used.
No model weights run — a stub encoder supplies vectors.
"""
import numpy as np
import pytest

from src.decision.pair import (SIMILARITY_PROMPT, PairScorer,
                               noul_probability, pair_cosine)


class _StubEncoder:
    """Deterministic vectors keyed off the text, so identical texts
    cosine to 1.0 and different texts to something < 1."""

    def __init__(self, dim=32):
        self.dim = dim
        self.prompts_seen = []

    def encode(self, texts, prompt_name=None):
        self.prompts_seen.append(prompt_name)
        rows = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


# ── the primitive ─────────────────────────────────────────────────────────

def test_identical_vectors_cosine_to_one():
    v = np.array([1.0, 2.0, 3.0])
    assert pair_cosine(v, v) == pytest.approx(1.0)


def test_orthogonal_vectors_cosine_to_zero():
    assert pair_cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_opposite_vectors_cosine_to_minus_one():
    assert pair_cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == \
        pytest.approx(-1.0)


def test_dim_mismatch_raises():
    with pytest.raises(ValueError, match="dim mismatch"):
        pair_cosine(np.array([1.0, 0.0]), np.array([1.0]))


def test_zero_vector_is_zero_not_nan():
    assert pair_cosine(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


# ── the logistic map ──────────────────────────────────────────────────────

def test_noul_probability_is_monotone_in_cosine():
    ps = [noul_probability(c) for c in (-0.5, 0.0, 0.25, 0.5, 1.0)]
    assert ps == sorted(ps)


def test_noul_probability_in_unit_interval():
    for c in (-1.0, -0.5, 0.0, 0.5, 1.0):
        assert 0.0 <= noul_probability(c) <= 1.0


def test_default_boundary_near_quarter_cosine():
    # zero-knowledge default: decision boundary at cosine = 0.25
    assert noul_probability(0.25) == pytest.approx(0.5)


def test_extreme_sigmoid_is_stable():
    # no overflow/nan at extreme logits; 1.0/0.0 are valid probabilities
    hi = noul_probability(1.0, slope=50.0, intercept=0.0)
    lo = noul_probability(-1.0, slope=50.0, intercept=0.0)
    assert np.isfinite(hi) and np.isfinite(lo)
    assert hi > 0.999 and lo < 0.001


# ── the scorer ────────────────────────────────────────────────────────────

def test_identical_texts_score_as_a_match():
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    assert ps.noul("the cat sat", "the cat sat") > 0.9


def test_different_texts_score_below_the_boundary():
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    # different texts -> cosine well below 0.25 -> P(match) < 0.5
    assert ps.noul("the cat sat on the mat",
                   "quantum field theory lecture") < 0.5


def test_similarity_uses_the_similarity_prompt_on_both_sides():
    """The measured-best config: symmetric SentenceSimilarity pairing."""
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    ps.similarity("a", "b")
    assert enc.prompts_seen == [SIMILARITY_PROMPT, SIMILARITY_PROMPT]


def test_similarity_accepts_pre_encoded_vectors():
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    v = enc.encode(["hello"])[0]
    enc.prompts_seen.clear()          # the setup call is not the assertion
    assert ps.similarity(v, v) == pytest.approx(1.0)
    # no encode call happened for vector inputs
    assert enc.prompts_seen == []


def test_pair_signal_depends_on_both_items():
    """The property the JSON blob lost: change either side, the score
    moves. A pooled-pair scorer cannot express this."""
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    base = ps.similarity("alpha", "beta")
    moved = ps.similarity("alpha", "gamma")
    assert base != moved


def test_no_encoder_raises_on_text_input():
    ps = PairScorer(encoder=None)
    with pytest.raises(ValueError, match="needs an encoder"):
        ps.similarity("a", "b")


# ── the map fit ───────────────────────────────────────────────────────────

def test_fit_map_recovers_a_separable_boundary():
    """When cosines cleanly separate the labels, the fitted map should
    place P near 0/1 on the two groups — the placement the diagnostic
    says to fit explicitly rather than assume at 0.5."""
    enc = _StubEncoder()
    ps = PairScorer(encoder=enc)
    cosines = [0.9, 0.8, 0.85, 0.1, 0.05, 0.15]   # high = match
    gold = [True, True, True, False, False, False]
    ps.fit_map(cosines, gold)
    assert ps.noul_probability_map(0.85) > 0.9
    assert ps.noul_probability_map(0.1) < 0.1


def test_fit_map_chaining_returns_self():
    ps = PairScorer(encoder=_StubEncoder())
    assert ps.fit_map([0.9, 0.1], [True, False]) is ps


def test_fit_map_rejects_mismatched_lengths():
    ps = PairScorer(encoder=_StubEncoder())
    with pytest.raises(ValueError, match="matched"):
        ps.fit_map([0.9, 0.1], [True])


# ── the adapter pair path ─────────────────────────────────────────────────
# When a PairScorer is supplied and the state is a two-text-item pair,
# noul/score route through cosine(u,v) instead of pooled-state scoring.

from src.decision.jevbench import nanocore_answers


class _StubPairScorer:
    """Fixed cosine so the pair path is deterministic without weights."""

    def __init__(self, cos=0.8):
        self.cos = cos
        self.calls = []

    def similarity(self, u, v):
        self.calls.append((u, v))
        return self.cos

    def noul(self, u, v):
        self.similarity(u, v)   # the real scorer routes noul through cosine
        return 0.9


class _NeverCalledModel:
    """The pair path must NOT touch the pooled-state model."""

    def decide(self, state, question):
        raise AssertionError("pair path should bypass model.decide")


def test_pair_noul_uses_cosine_not_pooled_state():
    ps = _StubPairScorer()
    q = {"answer": {"type": "noul", "instructions": "same meaning?"}}
    out = nanocore_answers(_NeverCalledModel(),
                           {"s1": "alpha", "s2": "beta"}, q, pair=ps)
    assert out["answer"]["noul"] == pytest.approx(0.9)
    assert ps.calls == [("alpha", "beta")]


def test_pair_score_varies_with_the_cosine():
    """The property the pooled path lost: the expected level tracks the
    pair's similarity instead of sitting at the middle."""
    q = {"answer": {"type": "score", "instructions": "how similar?",
                    "criteria": ["dissimilar", "somewhat", "equivalent"]}}
    hi = nanocore_answers(_NeverCalledModel(),
                          {"s1": "a", "s2": "b"}, q,
                          pair=_StubPairScorer(cos=0.9))["answer"]
    lo = nanocore_answers(_NeverCalledModel(),
                          {"s1": "a", "s2": "b"}, q,
                          pair=_StubPairScorer(cos=-0.5))["answer"]
    assert hi["score"] > lo["score"]


def test_no_pair_scorer_falls_back_to_pooled_state():
    """Default (pair=None) keeps the pooled-state path — the adapter
    stays weight-free and the single-field behavior is unchanged."""
    class M:
        class _P:
            def __init__(self, probs):
                self.probabilities = probs
                self.action = "unevaluated"
        def decide(self, state, question):
            # key on the option the adapter asked for (the proposition)
            return self._P({question.options[0]: 0.7,
                            question.options[1]: 0.3})
    q = {"answer": {"type": "noul", "instructions": "same meaning?"}}
    out = nanocore_answers(M(), {"s1": "a", "s2": "b"}, q)
    assert out["answer"]["noul"] == pytest.approx(0.7)


def test_single_field_state_ignores_pair_scorer():
    """A one-item state is not a pair — the pair path must not fire."""
    ps = _StubPairScorer()
    class M:
        class _P:
            def __init__(self, probs):
                self.probabilities = probs
                self.action = "unevaluated"
        def decide(self, state, question):
            return self._P({question.options[0]: 0.6,
                            question.options[1]: 0.4})
    q = {"answer": {"type": "noul", "instructions": "spam?"}}
    out = nanocore_answers(M(), "buy now", q, pair=ps)
    assert out["answer"]["noul"] == pytest.approx(0.6)
    assert ps.calls == []
