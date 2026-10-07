# ADR-0003: Bidirectional transformer composer over encoder item vectors for multi-item states

**Date**: 2026-10-07
**Status**: **proposed — unvalidated**
**Deciders**: project owner, agent

## Context

The encoder maps each state item — a text chunk, an image, an audio clip — to one
768-d vector in a shared space. A multi-item state is therefore a *sequence* of
embedding vectors. Nothing in the current stack composes that sequence: both head
modes consume a single pre-pooled vector, so a three-item state is reduced to a mean
before the model ever sees it.

This is the only component of the design that is genuinely novel relative to the
existing stack, and it is the reason the "karpathy framework" pillar is a component
rather than a code style.

## Decision

Add a small **bidirectional** transformer over the item-embedding sequence
(`EmbeddingComposer`, 28,315,392 parameters at 4 layers, 768-d, 6 heads), producing
one composed state vector. Bidirectional, not causal: the state is composed, not
generated, which matches the encoder-only lineage of the category. Mean pooling
remains available as the baseline path (`composer=None`).

## Alternatives Considered

### Alternative 1: Masked mean pooling (no composer)
- **Pros**: free, no parameters, order-invariant, already implemented.
- **Cons**: cannot represent order, negation, or which item matters; every item
  contributes equally to every decision.
- **Why not**: it is the baseline, not a rejected option. **It has not yet been
  beaten**, which is precisely why this ADR is `proposed`.

### Alternative 2: Concatenate items and score the concatenation
- **Pros**: trivially simple; keeps one vector.
- **Cons**: fixed item count; vector grows with item count; no notion of item
  boundaries.
- **Why not**: does not generalise across variable-length states.

### Alternative 3: Cross-encoder — re-encode (state ⊕ option) per option
- **Pros**: strongest function class; the option is seen by the encoder.
- **Cons**: re-encodes the *entire state*, including any image or audio, once per
  option. With a 300M audio encoder that multiplies the dominant cost by the option
  count.
- **Why not**: it hollows out the multimodal capability that motivates the design.

### Alternative 4: Causal (decoder-style) composer
- **Pros**: reuses the existing nanochat block design unchanged.
- **Cons**: item *n* cannot see item *n+1*; for a set-like state this is an
  arbitrary handicap with no benefit, since nothing is generated.
- **Why not**: the state is composed, not predicted token by token.

## Consequences

### Positive
- Multi-item states (text + screenshot + audio note) become a first-class input.
- Attention over items is where composition, ordering, and relevance weighting can
  actually be learned.

### Negative
- Adds 28.3M trainable parameters and a GPU training step, breaking the
  "train on cached embeddings on CPU" property that made the head cheap.
- Two configurations to evaluate and version (composer vs pooling).

### Risks
- **This is unproven and may not survive.** If the composer cannot beat masked mean
  pooling *at equal information* on a real multi-item task, the correct action is to
  delete it and let the pillar revert to code style. The composer has so far passed
  only a shape-and-finiteness smoke test.
- **The obvious trap**: building a synthetic multi-item task that flatters
  attention. The ablation must use a real benchmark (ScienceQA, multimodal `Choice`)
  with the same head and split for both arms.
- Note the zero-init convention: attention and MLP output projections start at zero,
  so the composer is the identity at initialization. A test that does not de-zero
  those projections measures pooling, not composition.
