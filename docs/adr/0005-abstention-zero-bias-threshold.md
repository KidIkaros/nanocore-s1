# ADR-0005: Abstention via a zero-bias max-score threshold at a target precision

**Date**: 2026-10-07
**Status**: **superseded by [ADR-0009](0009-conformal-abstention.md)**
**Deciders**: project owner, agent

> **Amendment (2026-10-07).** Superseded. Measurement showed the fitted threshold is not usable
> as a product: on BFCL irrelevance the raw score separates relevant from irrelevant pairs by
> only **0.097**, so holding 90% precision requires escalating **96.5%** of traffic and saves
> 3.4% of cost. On Banking77 the 90%-precision operating point saturates entirely, because base
> accuracy already exceeds 90%. Abstention moves to conformal prediction sets (APS/RAPS) with a
> domain-asymmetric policy. The zero-bias final layer itself is retained — it is what makes
> `max_score` an open-set signal — but it no longer supplies the decision rule. The analysis
> below is kept as the record.

## Context

A decision model in a production path must be able to decline. The requirement
priority puts reliability first, and the tolerable failure mode is an abstention,
not a confident wrong answer — especially on high-cardinality option sets, the
documented weak spot of the category.

Two things are needed: a signal that says "no option actually matches", and a
threshold policy that turns that signal into an accept/decline decision.

## Decision

Give the final layer **no bias term**, so `score_j = <x, f_j>` measures match quality
alone and cannot be inflated by a learned class prior. Threshold the *raw* maximum
score: below the threshold the decision is reported as **unseen** and the caller
abstains. Fit the threshold on held-out data to the highest coverage that still meets
a target precision on the items kept, with a floor on coverage so that "abstain on
everything" cannot satisfy the criterion.

## Alternatives Considered

### Alternative 1: Threshold on softmax confidence (`max p`)
- **Pros**: no extra machinery; one number.
- **Cons**: `max p` is a function of the score *scale*, not of match quality, and is
  miscalibrated exactly where it matters. Measured: the same 77-class problem spans
  ECE 0.907 (cosine) to 0.038 (kNN-5) depending on aggregation.
- **Why not**: confidence is not evidence of a match.

### Alternative 2: A learned "none of the above" option
- **Pros**: the model decides abstention itself; one interface.
- **Cons**: requires abstention examples in training data; the reject class competes
  with real options in the softmax and absorbs probability mass from them.
- **Why not**: it changes the meaning of the option distribution. May be revisited
  once real escalation data exists.

### Alternative 3: Entropy threshold
- **Pros**: captures the whole distribution, not just the top.
- **Cons**: high entropy can mean "genuinely ambiguous between two good options",
  which is a different condition from "no option matches".
- **Why not**: conflates ambiguity with out-of-distribution. `entropy_confidence` is
  returned for diagnostics, not used as the gate.

## Consequences

### Positive
- Abstention is a first-class output, so the caller gets an escalation path instead
  of a guess.
- The threshold is fitted to a *precision target*, which is the number a product
  owner actually cares about.
- A low max score is a genuine open-set signal, not a confidence artefact.

### Negative
- Thresholds are per-question-type and per-encoder; a change of encoder invalidates
  them.
- Raw score scales differ between head modes (measured: fitted thresholds of 10.6
  for fingerprint and 56.4 for interaction on the same data), so thresholds are not
  portable across modes.

### Risks
- **The relevance signal is weak on a real open-set task (2026-10-07).** On BFCL
  irrelevance — 1,252 relevant pairs against 2,300 irrelevant — the raw cosine score
  separates them by only **0.097** (0.7401 vs 0.6428). To hold 90% precision the gate
  must escalate **96.5%** of traffic, saving 3.4% of cost. That is not a product.
  Note the task is deliberately hard (BFCL constructs tempting-but-wrong tools), so this
  is an unsaturated problem rather than a broken method — which is exactly why it is the
  right place to work next.
- **Measured on Banking77 too, and it was close rather than decisive.** At 50% coverage
  the fingerprint head reaches 99.61% selective accuracy (best of any candidate) and both
  heads hit 100.00% at 10% coverage, but at 95% precision kNN-5 achieves slightly more
  coverage (0.9679 vs 0.9659).
- **The 90%-precision operating point is uninformative when base accuracy exceeds 90%.**
  Every candidate reported coverage 1.0 on Banking77. `protocol.coverage_at_precision`
  now refuses to report a saturated number instead of returning a misleading one.
- **Baseline temperatures were grid-bound** in the run that produced the coverage
  figures, so that comparison is provisional. Fixed in `protocol.fit_temperature`.
- **Thresholds are not portable across head modes** — fitted values were 10.6
  (fingerprint) vs 56.4 (interaction) on the same data, because the raw score scales
  differ.

## Sources

- `reports/runs/s1_calibration/` — selective curves; fingerprint head at 99.61% selective
  accuracy at 50% coverage, kNN-5 at 0.9679 coverage at 95% precision against the head's 0.9659
- `reports/runs/s1_dispatch/` — relevance separation 0.097; 3.46% auto-handled at 90% precision
- Liu et al., *Zero-Bias Deep Learning for Accurate Identification of IoT Devices* (2021) —
  the zero-bias layer and the threshold-selection approach
- `src/decision/protocol.py` — `coverage_at_precision`, which now refuses to report a
  saturated number
