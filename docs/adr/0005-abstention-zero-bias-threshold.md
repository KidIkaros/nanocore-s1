# ADR-0005: Abstention via a zero-bias max-score threshold at a target precision

**Date**: 2026-10-07
**Status**: accepted (measured; competitive at moderate coverage, not decisive)
**Deciders**: project owner, agent

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
