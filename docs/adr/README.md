# Architecture Decision Records — NanoCore-S1

Decisions for the decision-model line of work. Each record states the context, the decision, the
alternatives that were rejected, and the consequences — including the unflattering ones.

Background: `../ARCHITECTURE-DECISION-MODEL.md` (architecture map and validation status),
`../STATE-OF-THE-PROJECT.md` (consolidated findings), `../ARCHITECTURE-REASSESSMENT.md` (why the
decoder was set aside).

## Current decisions

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0002](0002-typed-non-autoregressive-output.md) | Typed non-autoregressive output (Choice/Score/Noul); never parse generated text | accepted | 2026-10-07 |
| [0004](0004-proper-scoring-rules.md) | Gate on proper scoring rules (log score, Brier); report ECE only | accepted | 2026-10-07 |
| [0006](0006-kaggle-only-execution.md) | Anything touching model weights or real arrays executes on Kaggle, never locally | accepted | 2026-10-07 |
| [0007](0007-llamacpp-runtime.md) | Serve the encoder through llama.cpp GGUFs, not PyTorch | accepted | 2026-10-07 |
| [0008](0008-cosine-scoring.md) | Score `Choice` with a temperature-scaled cosine, not a trained head | accepted | 2026-10-07 |
| [0009](0009-conformal-abstention.md) | Use conformal prediction sets (APS/RAPS) for abstention, asymmetric by domain | accepted | 2026-10-07 |
| [0010](0010-ordinal-score.md) | Ordinal `Score` is the only trained component; CORN/CORAL with RPS | accepted | 2026-10-07 |
| [0011](0011-headroom-check.md) | Benchmark admission requires a headroom check | accepted | 2026-10-07 |
| [0012](0012-input-length-cap.md) | The encoder enforces a hard input-length cap | accepted | 2026-10-07 |

## Superseded, amended, or deprecated

| ADR | Title | Status | Superseded by |
|-----|-------|--------|---------------|
| [0001](0001-frozen-encoder-trained-head.md) | Frozen encoder with a trained typed head | partially superseded — **encoder half stands**, head half rejected | [0008](0008-cosine-scoring.md) |
| [0003](0003-bidirectional-composer.md) | Bidirectional composer for multi-item states | **deprecated — parked**, revisit condition recorded | — |
| [0005](0005-abstention-zero-bias-threshold.md) | Abstention via a fitted max-score threshold | **superseded** | [0009](0009-conformal-abstention.md) |

## Reading the revision

The original architecture was `frozen EG2 → trained head → threshold abstention`. Six
measurements changed it:

1. **Both benchmarks are saturated** at ~93% zero-shot (0007, 0011) — so no head could show a
   difference.
2. **A one-parameter cosine matches the 59,136-parameter head** on every proper metric, and
   beats it on routing (0008).
3. **The runtime was the bottleneck, not the model** — llama.cpp closes 4.08× with no quality
   loss (0007).
4. **The threshold gate is not a product** — 0.097 separation (0009).
5. **`Score` was using the wrong loss** — cross-entropy discards ordering (0010).
6. **Long inputs cost 19.7× short ones** (0012).

The revised architecture is a llama.cpp-served frozen encoder with temperature-scaled cosine
scoring, a typed interface, and a conformal gate — with ordinal `Score` as the only trained
component.

See [template.md](template.md) for the blank format.
