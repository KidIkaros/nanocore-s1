# ADR-0001: Frozen EmbeddingGemma 2 encoder with a trained typed head, not a from-scratch decoder

**Date**: 2026-10-07
**Status**: **partially superseded by [ADR-0008](0008-cosine-scoring.md)** — the trained-head half
is rejected; the frozen-encoder half stands and is strengthened by [ADR-0007](0007-llamacpp-runtime.md)
**Deciders**: project owner, agent

> **Amendment (2026-10-07).** This ADR made two claims. The **encoder** half — freeze
> EmbeddingGemma 2 rather than train a decoder — is validated and now central to the
> architecture; it is served through llama.cpp per ADR-0007. The **trained head** half is
> rejected: three experiments found no win over a one-parameter temperature-scaled cosine on any
> proper metric, and the head was *worse* than zero-shot on dispatch routing (90.84% vs 93.07%).
> Scoring is now cosine-based (ADR-0008), with ordinal `Score` the only remaining trained
> component (ADR-0010). The risk section below is retained as the record of how this was found.

## Context

NanoCore-S1 began as a decoder-only, autoregressive GPT-style model (12 layers,
768-d, 135,266,328 parameters at the intended vocabulary) trained from scratch to
emit typed decisions as *text* that a caller then parsed. The independent evaluation
of the category's flagship product criticises exactly that interface: the decision
has to be phrased as a prompt, the answer has to be parsed out of generated text,
the model's probability for the answer is usually unavailable or unreliable, and
every generated token is paid for.

The repository's own evidence agrees. The only stored training report is 20 steps at
loss 6.7661 on an 867-token vocabulary, where `ln(867) = 6.7650` — exactly the
uniform-random baseline. Separately, a frozen off-the-shelf encoder already reaches
92.92% zero-shot on Banking77 with no training at all.

## Decision

Use a **frozen** EmbeddingGemma 2 encoder plus a small **trained** typed head. Do not
train a decoder from scratch for bounded decision tasks. Retain the decoder path only
under the Route-4 condition: no suitable pretrained model has seen the target data,
or data-sovereignty rules forbid unknown pretrained weights.

## Alternatives Considered

### Alternative 1: Continue the from-scratch decoder
- **Pros**: full control of the tokenizer and output format; no dependency on a
  pretrained checkpoint; the existing `src/model.py` already exists.
- **Cons**: needs a pretraining corpus, a multi-stage pipeline (base → midtrain →
  SFT → RLCD), and GPU-months to reach parity with a frozen encoder's zero-shot
  score; its probabilities remain unreliable.
- **Why not**: the cost is real and the measured baseline is uniform-random. The
  justification for Route 4 was never established.

### Alternative 2: Frozen encoder plus a *linear probe* only
- **Pros**: cheapest possible; measured accuracy is competitive (see ADR-0004 note).
- **Cons**: no calibrated distribution over options, no abstention path, no
  order-invariance guarantee, no typed Score/Noul.
- **Why not**: it answers the accuracy question but not the decision-model question.
  Kept as a baseline rather than a solution.

### Alternative 3: Fine-tune the encoder end-to-end
- **Pros**: highest ceiling; the encoder could adapt to the decision distribution.
- **Cons**: destroys the cheap-training property (head is `options x dim` and fits
  on cached vectors); needs GPU for every change; risks catastrophic forgetting of
  the multimodal space.
- **Why not**: deferred, not rejected. Revisit if the frozen representation proves
  to be the binding constraint.

## Consequences

### Positive
- Head training is small enough to run on CPU over cached embeddings.
- Multimodal states become reachable without training a fusion model.
- The encoder is swappable, so encoder choice is a config, not a rewrite.

### Negative
- The system inherits the encoder's weaknesses, including its prompt sensitivity
  (an unablated assumption — see the architecture map §4).
- Two artifacts to version (encoder choice + head weights).

### Risks
- **The head did not earn its place — settled by evidence (2026-10-07).** Three
  experiments, no win over the best baseline on any proper metric:
  - *Banking77*: a 1-parameter temperature on zero-shot cosine reaches log 0.380 /
    Brier 0.117 / ECE 0.045 against the 59,136-parameter head's 0.377 / 0.116 / 0.046 —
    statistically indistinguishable. kNN-5 is more accurate than both.
  - *Prompt ablation*: the head improves to 93.60% under the corrected
    `SearchQuery`/`Document` pairing, but kNN-5 reaches 94.58%.
  - *BFCL dispatch routing*: the head is **worse** than zero-shot, 90.84% vs 93.07%.
  - Initializing from option embeddings has no effect (93.12% vs 93.11%), and
    normalizing fingerprints plus training is actively harmful (93.11% → 79.66%).
- **The reason is partly saturation, not head quality.** Both benchmarks sit at ~93%
  zero-shot, leaving ~7% headroom, so a null result there is uninformative rather than
  damning. See `docs/STATE-OF-THE-PROJECT.md` §4.
- **What survives of this ADR is the encoder half.** Frozen EmbeddingGemma 2 plus a
  typed interface and temperature-scaled scoring is supported and needs no training.
  The trained head remains justified only for ordinal `Score`, which
  temperature-scaled cosine cannot express at all.

## Sources

- `docs/ARCHITECTURE-REASSESSMENT.md` — why the decoder was set aside
- `reports/runs/audit/` — the decoder's only stored run: 20 steps at loss 6.7661 against
  `ln(867) = 6.7650`, i.e. the uniform-random baseline
- `reports/runs/s1_calibration/`, `reports/runs/s1_dispatch/`, `reports/runs/s1_sharpening/` —
  the three experiments that retired the trained-head half
- `reports/runs/s1_prompt_ablation/` — the corrected prompt pairing
