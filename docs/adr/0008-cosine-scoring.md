# ADR-0008: Score `Choice` with a temperature-scaled cosine, not a trained head

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent
**Partially supersedes**: [ADR-0001](0001-frozen-encoder-trained-head.md) (the trained-head half; the frozen-encoder half stands)

## Context

ADR-0001 chose a frozen encoder plus a *trained* typed head, and justified the head by the
assumption that training would beat the encoder's zero-shot similarity. Three experiments
tested that assumption and it failed.

**Experiment 1 — Banking77 (calibration v3).** A 59,136-parameter fingerprint head trained for
120 epochs, against a **one-parameter** temperature on the zero-shot cosine score:

| method | params | accuracy | log score | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| cosine + fitted τ | **1** | 92.92% | 0.380 | 0.117 | **0.045** |
| kNN-5 + fitted τ | 0 | **93.64%** | 0.508 | 0.117 | 0.052 |
| fingerprint head | 59,136 | 93.11% | **0.377** | **0.116** | 0.046 |

Statistically indistinguishable. The head also lost to kNN-5 on accuracy, Brier and ECE.

**Experiment 2 — prompt ablation.** Under the corrected `SearchQuery`/`Document` pairing the
head improved to 93.60%, but kNN-5 reached 94.58%.

**Experiment 3 — BFCL dispatch routing (1,252 queries, 2–37 options).** The head was *worse*
than zero-shot: **90.84% vs 93.07%**, with `compare_to_best` returning **no winners on any
metric**. Initializing the head from the option embeddings had no effect (93.12% vs 93.11%,
identical log score), and normalizing fingerprints plus training was actively harmful
(93.11% → **79.66%**).

## Decision

**Retire the trained head as the scorer for `Choice` and `Noul`.** Score with a
temperature-scaled cosine over option embeddings, where the only fitted parameter is a single
temperature (per question type). Keep `DecisionHead` in the codebase, marked deprecated for
these types, because it is the ablation that justifies this decision and must remain runnable.

`Score` is exempt: see [ADR-0010](0010-ordinal-score.md).

## Alternatives Considered

### Alternative 1: Keep the head and add soft targets (RLCD)
- **Pros**: the design's documented objective; the reference paper trains against strictly
  proper scoring rules; soft targets are the one route not yet tried.
- **Cons**: requires harvesting teacher distributions; the bottleneck is verifiable labels, not
  head capacity; and on two benchmarks there is no headroom for *any* head to show a gain.
- **Why not**: deferred, not rejected. It is the only untested route back to a useful head, and
  it is blocked on data rather than on architecture. Recorded as an open question.

### Alternative 2: kNN everywhere
- **Pros**: wins on classification-shaped tasks (93.64% on Banking77) and needs no training.
- **Cons**: **collapses on routing** — 71.5–73.1% on BFCL against 93.07% for cosine — because
  the query→tool mapping is not locally smooth in embedding space. It also stores the corpus,
  which is the memory cost cosine avoids.
- **Why not**: task-dependent. Retained as a `Scorer` implementation for classification-shaped
  tasks rather than as the default.

### Alternative 3: Interaction mode only (the dual-encoder head)
- **Pros**: scores `(state, option)` pairs from option *embeddings*, so a new option needs no
  retraining; 92.48% on Banking77.
- **Cons**: it is still a trained head, and it was worse than the fingerprint head on every
  metric and worse than zero-shot on routing.
- **Why not**: no evidence it helps, and it adds parameters where a single temperature suffices.

### Alternative 4: Enlarge the head
- **Pros**: intuition that capacity is the constraint.
- **Cons**: the literature attributes linear-probe generalisation to the *spectral structure of
  the features*, not classifier capacity, and the gap here is headroom, not capacity.
- **Why not**: there is no headroom to win on either benchmark (~7%).

## Consequences

### Positive
- The scoring path becomes **one fitted parameter** instead of 59,136, and it is
  indistinguishable on the proper metrics.
- Latency accounting simplifies: scoring is 0.2% of a decision, so nothing in the hot path
  needs training.
- The head remains as a reproducible ablation, so this decision can be re-tested cheaply.

### Negative
- **The measured win belongs to the encoder, not to this project.** Our 92.92% Banking77
  zero-shot beats Jev's *independently measured* 79.7% by 13.2 points and Laya's 42.5% by 50
  points — but that is EmbeddingGemma 2's similarity structure, not anything we built. Stated
  plainly because it changes what the project can claim.
- We lose the "a trained model improves with use" story for `Choice`. It survives only for
  `Score`, and only if Stage 4 succeeds.

### Risks
- **Both benchmarks are saturated at ~93% zero-shot**, so this decision is partly a statement
  about the benchmarks. [ADR-0011](0011-headroom-check.md) makes the headroom check mandatory
  precisely so this cannot recur unnoticed.
- If a headroom-positive task is found and a trained head *does* win there, this ADR is wrong
  for that task class. The revisit condition is explicit: headroom ≥ 15% and a head that beats
  the best baseline on the proper scores.

## Sources

- `reports/runs/s1_calibration/` — the 1-parameter vs 59,136-parameter comparison
- `reports/runs/s1_prompt_ablation/` — the corrected pairing
- `reports/runs/s1_dispatch/` — the head losing to zero-shot, with no winners on any metric
- `reports/runs/s1_sharpening/` — initialization and normalization outcomes
- `docs/RESEARCH-NOTES.md` R12 — the competitive comparison, including the encoder's role
