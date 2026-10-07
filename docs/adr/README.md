# Architecture Decision Records — NanoCore-S1

Decisions for the decision-model line of work. Each record states the context, the
decision, the alternatives that were rejected, and the consequences — including the
unflattering ones.

Background: `../ARCHITECTURE-DECISION-MODEL.md` (architecture map and validation
status), `../ARCHITECTURE-REASSESSMENT.md` (why the decoder was set aside).

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-frozen-encoder-trained-head.md) | Frozen EmbeddingGemma 2 encoder with a trained typed head, not a from-scratch decoder | accepted | 2026-10-07 |
| [0002](0002-typed-non-autoregressive-output.md) | Typed non-autoregressive output (Choice/Score/Noul); never parse generated text | accepted | 2026-10-07 |
| [0003](0003-bidirectional-composer.md) | Bidirectional transformer composer over encoder item vectors for multi-item states | proposed | 2026-10-07 |
| [0004](0004-proper-scoring-rules.md) | Gate on proper scoring rules (log score, Brier); report ECE only | accepted | 2026-10-07 |
| [0005](0005-abstention-zero-bias-threshold.md) | Abstention via a zero-bias max-score threshold at a target precision | accepted | 2026-10-07 |
| [0006](0006-kaggle-only-execution.md) | Anything touching model weights or real arrays executes on Kaggle, never locally | accepted | 2026-10-07 |

See [template.md](template.md) for the blank format.
