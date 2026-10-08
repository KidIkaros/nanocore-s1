# ADR-0011: Benchmark admission requires a headroom check

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

Two benchmarks were chosen for architectural comparisons. Both turned out to be **saturated by
the frozen encoder alone**:

| benchmark | EG2 zero-shot | headroom | usable |
|---|---:|---:|---|
| Banking77 intent classification | 92.92% | 6.9% | **no** |
| BFCL tool routing (2–37 options) | 93.07% | 6.9% | **no** |

The consequence was not a wrong number but an **uninterpretable one**. Every difference we
chased — head 93.11% vs cosine 92.92% vs kNN-5 93.64% — lives inside a 7% residual. On a
saturated task a better head *cannot* produce a better headline number, so a null result means
"no head can help here" rather than "our head is bad." Two experiments were run and reported
before this was recognised, and one conclusion ("the head loses to kNN-5") was true but
weaker than it appeared: the honest framing is that the task had no room to discriminate.

The mechanism is documented. LP++ (CVPR 2024) finds a linear probe scoring ~20% *below* zero-shot
in the 1-shot regime, so probing is not inherently weak — its advantage is regime-dependent. And
"Why Linear Probing Works" (ECCV 2026) attributes linear-probe generalisation to the spectral
structure of the features, not classifier capacity. Headroom is the binding experimental
constraint, and it is measurable in advance for free.

## Decision

**No architectural comparison may be drawn on a benchmark whose zero-shot baseline leaves less
than 15% headroom.** `protocol.headroom_check(zero_shot_accuracy, min_headroom=0.15)` is
mandatory before any head, scorer, composer, or calibration comparison, and its result is
recorded in the run's `results.json` alongside the metrics.

Where a benchmark fails the check, the honest outcomes are:

1. **Report the saturation itself as the finding** — "a frozen encoder already solves this;
   no trained component can be justified here" is a real result, and it is the one we obtained.
2. **Move to a task with headroom** before making architecture claims.
3. **Never** report a marginal delta from a saturated task as evidence for or against a
   component.

## Alternatives Considered

### Alternative 1: Simply use harder benchmarks
- **Pros**: addresses the symptom directly; no new process.
- **Cons**: "harder" is unquantified, so the same mistake recurs silently the next time a task
  is chosen — which is exactly what happened between Banking77 and BFCL.
- **Why not**: a criterion is enforceable; an adjective is not.

### Alternative 2: Report accuracy with confidence intervals instead of a criterion
- **Pros**: statistically honest; no threshold to justify.
- **Cons**: confidence intervals tell you whether a *delta* is real, not whether the task could
  ever have shown one. On a 7%-headroom task a tight interval around a small delta is still
  uninformative about the architecture.
- **Why not**: intervals are necessary and insufficient. They are already reported; the headroom
  check is the missing piece.

### Alternative 3: Accept saturation and stop making architecture claims at all
- **Pros**: no benchmark search needed; honest by construction.
- **Cons**: forecloses the one remaining question — whether ordinal `Score` or composition can
  beat a cosine — because both would be unmeasurable by decree.
- **Why not**: the criterion routes around saturation rather than surrendering to it.

## Consequences

### Positive
- A free, mechanical check that would have saved two experiments and one withdrawn conclusion.
- Makes "this benchmark cannot discriminate" a **recorded finding** rather than an excuse.
- Forces the search for headroom-positive tasks, which is where the market's own benchmarks
  live (OCR, documents, visual QA — tasks requiring generation and grounding, not similarity).

### Negative
- Constrains benchmark choice, and a genuinely suitable task is not yet identified — it is
  scoped as a follow-up, not solved here.
- Adds a required step to every experiment's report.

### Risks
- **15% is a chosen threshold, not a derived one.** It is deliberately conservative; the
  measured failures were at 6.9%. If it proves too strict it can be relaxed *with evidence*, and
  the value is a parameter rather than a constant.
- **A headroom-positive task may not exist for this architecture.** That is a possible outcome
  and it would mean the encoder already solves the class of problem we can address — which is
  itself the most important finding available, and better surfaced than hidden.

## Sources

- `protocol.headroom_check` in `src/decision/protocol.py`
- `reports/runs/s1_dispatch/` and `reports/runs/s1_baseline/` — the two saturation measurements
- `docs/RESEARCH-NOTES.md` R1 — the headroom finding and the LP++ evidence
- Huang et al., *LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP*, CVPR 2024
- *Why Linear Probing Works: Non-Vacuous Generalization Bounds via Effective Dimension*, 2026
