# ADR-0004: Gate on proper scoring rules (log score, Brier); report ECE only

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

Calibration metrics are easy to game and easy to misread. Expected Calibration Error
is minimised by *sharpening* — raising `max(p)` closes the gap to accuracy without
improving the underlying distribution. In this project's audit, a broken model with
zeroed weights produced an ECE of 0.001953 (exactly 1/vocabulary) while a real
initialised model produced ~0.00202 — the broken model scored *better* on the metric.

The calibration experiment then produced a second demonstration: `softmax(cosine/τ)`
has an ECE of **0.907** with a fitted temperature, while kNN-5 with a fitted
temperature reaches **0.038**. The same metric family, on the same embeddings, spans
two orders of magnitude depending on the aggregation — it does not by itself tell you
which model is usable.

## Decision

Gate on **strictly proper** scoring rules: the log score and the Brier score. Their
unique optimum is the true conditional distribution, so they cannot be improved by
sharpening. Train the head on the log score. Report ECE alongside, clearly labelled
as reported rather than gated.

## Alternatives Considered

### Alternative 1: Gate on ECE
- **Pros**: the metric the category publishes; directly interpretable as a
  confidence-accuracy gap.
- **Cons**: not strictly proper; minimised by sharpening; insensitive to a model
  being entirely broken (demonstrated above).
- **Why not**: it can be optimised into looking good while the distribution gets
  worse. It remains useful as a diagnostic.

### Alternative 2: Gate on accuracy
- **Pros**: simple; matches how the category's benchmarks are reported.
- **Cons**: ignores the entire point of a typed decision model. Measured here: the
  trained head is *less accurate* than kNN-5 while being better on the log score —
  accuracy alone would pick the wrong artifact.
- **Why not**: it cannot distinguish a calibrated distribution from a hard label.

### Alternative 3: Gate on Brier only
- **Pros**: proper; bounded in [0, 2]; less sensitive to a single confident error.
- **Cons**: less discriminative than the log score for the confident-wrong case,
  which is the failure mode that matters most here.
- **Why not**: kept as a secondary gate; the log score leads because a confidently
  wrong answer is the intolerable outcome.

## Consequences

### Positive
- The objective cannot be gamed by sharpening.
- The failure that matters — a confident wrong answer — is the one the log score
  punishes hardest.
- Temperature fitting and abstention thresholds can be fitted on the same proper
  objective, consistently.

### Negative
- Log score is unbounded and hard to explain to a non-technical reader; business
  reporting will need accuracy and coverage alongside it.
- Soft teacher distributions are required for the log score to express its full
  value; current heads train on one-hot labels, which reduces the objective to
  ordinary cross-entropy on a linear probe.

### Risks
- **Proper scores are necessary but not sufficient.** The current evidence is that
  kNN-5 beats both heads on Brier and ECE and loses only on log score. A head that
  wins one proper score while losing another is not a clear win — see ADR-0001's
  risk note.
- **Mitigation**: report all three (log score, Brier, ECE) plus accuracy and the
  abstention curve, and do not declare a winner on a single metric.
