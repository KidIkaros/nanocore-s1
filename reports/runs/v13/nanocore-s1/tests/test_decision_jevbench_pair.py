"""jev-benchmarking adapter — pair-state and level-description contracts.

v27 jev anchor root causes this file pins:

1. **Pair states were JSON blobs.** A state dict ``{"sentence_1": A,
   "sentence_2": B}`` was rendered as one JSON string and encoded as ONE
   item, so the composer saw T=1 and mean-pooling averaged the two
   sentences into one vector. cosine(option, pooled_pair) cannot express
   "are these two similar?" — paws .440, stsb .215, boolq .583 are all
   pair tasks. The fix: split the state dict into one item per field.

2. **Terse score levels are unrankable.** sst5's levels
   ``["Very negative.", "Negative.", ...]`` are near-identical in
   embedding space, so cosine cannot rank them (.177 = chance .20).
   The adapter expands terse levels into full descriptions.

The shipped decide() path is untouched — these are adapter-level changes.
No encoder runs; a stub model supplies probabilities.
"""
import numpy as np

from src.decision.jevbench import (NOUL_NEGATIVE, _level_text, confidence,
                                   nanocore_answers, option_label)


class _StubModel:
    """decide() over a fixed uniform-plus-bump distribution, no encoder."""

    class _P:
        def __init__(self, probs, action="answer"):
            self.probabilities = probs
            self.action = action

    def decide(self, state, question):
        k = len(question.options)
        p = np.full(k, 1.0 / k)
        p[0] += 0.2
        p /= p.sum()
        return self._P({o: float(x) for o, x in zip(question.options, p)})


class _AbstainingModel(_StubModel):
    """Every head abstains — the gated arm's extreme case."""

    def decide(self, state, question):
        p = super().decide(state, question)
        p.action = "escalate"
        return p


class _SpyState(_StubModel):
    """Records the state the adapter passes to decide()."""

    def decide(self, state, question):
        self.seen_state = state
        return super().decide(state, question)


def _items_of(state):
    return (list(state.items) if hasattr(state, "items") else list(state))


# ── pair states: the JSON-blob encoding destroyed the pair signal ───────────

def test_pair_state_splits_into_two_items():
    """A two-field state dict must reach decide() as two separate items —
    one vector per sentence, so the pair signal survives."""
    spy = _SpyState()
    q = {"answer": {"type": "noul", "instructions": "same meaning?"}}
    nanocore_answers(spy, {"sentence_1": "alpha beta",
                           "sentence_2": "alpha beta"}, q)
    items = _items_of(spy.seen_state)
    assert len(items) == 2
    assert "alpha beta" in items


def test_single_field_state_stays_one_item():
    """A one-field state is still one item — no spurious splitting."""
    spy = _SpyState()
    q = {"answer": {"type": "choice", "instructions": "topic?",
                    "criteria": {"A": "x", "B": "y"}}}
    nanocore_answers(spy, {"query": "hello"}, q)
    assert len(_items_of(spy.seen_state)) == 1


def test_string_state_stays_one_item():
    spy = _SpyState()
    q = {"answer": {"type": "noul", "instructions": "spam?"}}
    nanocore_answers(spy, "buy now", q)
    assert _items_of(spy.seen_state) == ["buy now"]


def test_instructions_still_fold_into_choice_state():
    """Item splitting must not lose the instruction text. For a single-field
    state the instruction folds into the state string (the measured
    prompt-ablation shape, one item); for a pair it becomes its own item.
    Either way the prompt content reaches the model — the same-prompt
    honesty the harness comparison depends on."""
    spy = _SpyState()
    q = {"answer": {"type": "choice", "instructions": "topic?",
                    "criteria": {"A": "x", "B": "y"}}}
    nanocore_answers(spy, {"article": "news text"}, q)
    items = _items_of(spy.seen_state)
    assert len(items) == 1
    assert "topic?" in items[0] and "news text" in items[0]


def test_instructions_fold_as_own_item_for_pair_states():
    """The pair case is the exception: the instruction is part of the
    comparison prompt and gets its own item, alongside the split fields."""
    spy = _SpyState()
    q = {"answer": {"type": "choice", "instructions": "topic?",
                    "criteria": {"A": "x", "B": "y"}}}
    nanocore_answers(spy, {"sentence_1": "a", "sentence_2": "b"}, q)
    items = _items_of(spy.seen_state)
    assert "topic?" in items
    assert "a" in items and "b" in items
    assert len(items) == 3


# ── richer level descriptions for terse score levels ───────────────────────

def test_terse_score_levels_are_expanded():
    """A bare level like 'Very negative.' must be expanded to a full
    description the encoder can actually rank."""
    assert _level_text("Very negative.") != "Very negative."
    assert "negative" in _level_text("Very negative.").lower()
    assert len(_level_text("Very negative.")) > len("Very negative.")


def test_rich_score_levels_pass_through_unchanged():
    """stsb's levels are already full sentences — expansion is idempotent."""
    rich = "The two sentences are completely dissimilar."
    assert _level_text(rich) == rich


def test_score_answer_legend_uses_original_levels():
    """The harness scores against ORIGINAL level indices — the expansion is
    internal to the scorer; the legend/answer shape stays harness-native."""
    q = {"rate": {"type": "score", "instructions": "sentiment?",
                  "criteria": ["Very negative.", "Neutral.",
                               "Very positive."]}}
    out = nanocore_answers(_StubModel(), "the movie was great", q)
    assert out["rate"]["legend"] == {"0": "Very negative.", "1": "Neutral.",
                                     "2": "Very positive."}


def test_pair_choice_routes_through_pair_scorer():
    """Increment 4a: a 2-option choice on a pair state reads cosine(u,v) as
    P(first option) — the native similarity primitive, not pooled cosine."""
    from src.decision.pair import PairScorer
    from src.decision.backends import StubEncoder
    from src.decision.jevbench import nanocore_answers

    enc = StubEncoder(dim=16)
    pair = PairScorer(encoder=enc)
    state = {"sentence1": "the cat sat on the mat",
             "sentence2": "a feline rested on the rug"}
    q = {"type": "choice", "instructions": "are these paraphrases?",
         "criteria": {"yes": "yes", "no": "no"}}

    class FakeModel:
        def decide(self, *a, **k):
            raise AssertionError("pooled path must not run for a pair choice")

    ans = nanocore_answers(FakeModel(), state, {"q1": q}, pair=pair)
    assert ans is not None and "q1" in ans
    a = ans["q1"]
    assert a["type"] == "choice"
    assert a["choice"] in ("yes", "no")
    assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-9


def test_non_pair_choice_still_uses_pooled_path():
    """A single-field state keeps the pooled-state cosine path (no regression)."""
    from src.decision.pair import PairScorer
    from src.decision.backends import StubEncoder
    from src.decision.jevbench import nanocore_answers
    from src.decision.model import DecisionModel
    from src.decision.scoring import CosineScorer

    enc = StubEncoder(dim=16)
    pair = PairScorer(encoder=enc)
    model = DecisionModel(encoder=enc, scorer=CosineScorer(), gate=None, cache=None)
    state = "a single sentence"  # not a pair
    q = {"type": "choice", "instructions": "classify",
         "criteria": {"a": "class a", "b": "class b"}}
    ans = nanocore_answers(model, state, {"q1": q}, pair=pair)
    assert ans is not None and "q1" in ans  # went through pooled decide()
