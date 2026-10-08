# ADR-0002: Typed non-autoregressive output (Choice/Score/Noul); never parse generated text

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

The original design produced decisions as generated text — a `[CHOICE]` or
`[SCORE]` block that the caller parsed back into a value. That makes the model's
probability for its answer either unavailable or a re-derived approximation, makes
every decision cost a full autoregressive decode, and introduces parse-failure modes
as a category of production incident. Measured CPU latency for the decoder path was
~150 ms/token with ~3.6 s for a short generation, against a documented claim of
15–35 ms.

The category's converged interface is the opposite: the model returns a typed value
directly, with probabilities as first-class outputs.

## Decision

The model exposes exactly three typed primitives and returns them as structured data:

- `Choice` — a distribution over a supplied option set.
- `Score` — a distribution over an ordered rubric.
- `Noul` — `P(yes)` for a binary judgment.

Options are supplied at request time and scored **individually**, so the argmax is
invariant to option order by construction. No free-form generation occurs anywhere in
the decision path.

## Alternatives Considered

### Alternative 1: Keep generated text plus a parser
- **Pros**: flexible; one model handles many shapes; human-readable output.
- **Cons**: unreliable probabilities; decode cost per decision; parsing is a new
  failure surface; option order changes the answer.
- **Why not**: it is the interface the category moved away from, for documented
  reasons. Retained only as a wire format if a caller needs one.

### Alternative 2: Constrained decoding over a typed grammar
- **Pros**: keeps a single generative model while guaranteeing parseable output.
- **Cons**: still autoregressive, still pays per token, and probabilities remain a
  function of the decoding procedure rather than a model output.
- **Why not**: solves the parsing symptom, not the probability or latency problem.

### Alternative 3: Independent binary heads per option (one-vs-rest)
- **Pros**: trivially parallel; each option independent.
- **Cons**: no normalised distribution across options without an extra step;
  probabilities not comparable between options.
- **Why not**: the option set must be scored jointly for a distribution to mean
  anything. Subsumed by isolated scoring plus a softmax.

## Consequences

### Positive
- Probabilities are the model's actual output, so proper scoring rules apply
  directly (ADR-0004).
- Order invariance holds by construction rather than by prompt engineering.
- A new option needs no retraining in `interaction` mode — it is just another
  embedding. **Corrected 2026-10-07:** this does *not* hold in `fingerprint` mode,
  where the score is `states @ fingerprints.T` and `option_embeddings` is never read.
  A new option in fingerprint mode has no fingerprint and must be registered and
  learned. Measured: two different option prompts produced byte-identical fingerprint
  head results.
- Latency is one forward pass, not a decode loop.

### Negative
- The model cannot answer a question whose options were not anticipated as a set.
- Callers must express their problem as one of three primitives, which is a real
  modelling constraint.

### Risks
- **Only `Choice` has ever been exercised on real data.** `Score` and `Noul` have
  unit-tested schemas and no measured behaviour. Treat them as implemented, not
  validated.
- **Order invariance was asserted by construction and is now measured — resolved
  (2026-10-07).** Banking77, fixed 77 labels: max |Δp| = **2.38e-07**, 0 argmax flips. BFCL
  dispatch, **2–37 options varying per request**: max |Δp| = **1.19e-07**, 0 flips. The
  property holds on real embeddings at both fixed and variable cardinality.
- **`Score` and `Noul` have never been exercised on real data.** `Score` is addressed by
  [ADR-0010](0010-ordinal-score.md); `Noul` has only served as a relevance signal, and its
  measured separation is weak (0.097), addressed by [ADR-0009](0009-conformal-abstention.md).

## Sources

- `reports/runs/s1_calibration/` — order invariance on a fixed 77-label option set
- `reports/runs/s1_dispatch/` — order invariance on variable 2–37 option sets
- `reports/runs/audit/` — the decoder's ~150 ms/token against a documented 15–35 ms claim
- Santos, *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses* (2026) —
  the critique of generative decision interfaces and the non-autoregressive robustness argument
