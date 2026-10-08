# ADR-0012: The encoder enforces a hard input-length cap

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

The latency run measured state length as the single largest remaining cost, on both runtimes:

| runtime | short state (~20 tok) | long state (~500 tok) | ratio |
|---|---:|---:|---:|
| PyTorch fp32, CPU | 363.5 ms | **6,595 ms** | 18.1× |
| llama.cpp Q8_0, CPU | 89.2 ms | **1,754.8 ms** | 19.7× |

**~19.7× the latency for ~8× the tokens** — super-linear, and it survives the 4.08× runtime
improvement, so it is a property of the computation rather than of the framework.

The cause is now understood. EG2 uses **alternating local/global attention at 5:1 with a
1024-token sliding window**. Below 1024 tokens every token still attends to every preceding
token up to the window bound, so **attention is effectively quadratic for our inputs** — the
sliding window only pays off *beyond* 1024 tokens, which our states never reach.

Nothing in the architecture currently bounds input length. A state is whatever the caller
supplies, and a long one costs 1.75 s per decision on CPU.

## Decision

**The encoder declares and enforces `max_tokens` as part of its contract.** Concretely:

- `Encoder` exposes `max_tokens: int`, and every `encode_*` call truncates deterministically to
  it.
- Truncation is **explicit and reported**: the encoder emits a warning (or a flag on the result)
  when it truncates, so a silently degraded decision is impossible.
- The default cap is chosen from a **published latency/quality curve** measured at 128, 256 and
  512 tokens, not guessed.
- The truncation policy is documented per modality: text is truncated by tokens from the end;
  media items are truncated by their own modality limits (a frame budget for video, a duration
  budget for audio).

## Alternatives Considered

### Alternative 1: No cap — let the caller decide
- **Pros**: no loss of information; the caller knows its own budget.
- **Cons**: the caller does not know that 500 tokens costs 19.7× a short state, so the default
  behaviour is the expensive one; and the failure is silent, producing a slow decision rather
  than an error.
- **Why not**: it makes the worst case the default case, and the design's claim is latency.

### Alternative 2: A dynamic cap based on content
- **Pros**: preserves long inputs when they matter; more faithful.
- **Cons**: makes latency unpredictable, which is the property a router most needs; and it needs
  a relevance model to decide what to drop — the thing we do not have.
- **Why not**: unpredictable latency is worse than bounded latency for a dispatch component.

### Alternative 3: Cap in the tokenizer only
- **Pros**: cheapest change; one place.
- **Cons**: the tokenizer is shared with the legacy decoder path (8k context) and knows nothing
  about the encoder's attention window or the latency budget; and it does not cover media items.
- **Why not**: it puts a latency constraint in the wrong layer.

### Alternative 4: Fix the quadratic behaviour instead
- **Pros**: addresses the root cause; would make long states cheap.
- **Cons**: the sliding window is already 1024 tokens and is a property of the pretrained model —
  we cannot change it without retraining, and the encoder is frozen by ADR-0001.
- **Why not**: out of reach for a frozen encoder. A cap is the available lever.

## Consequences

### Positive
- Latency becomes **bounded and predictable**, which is the property a router needs.
- Removes the largest remaining cost from the design without touching the model.
- Makes the information loss explicit instead of silent.

### Negative
- Long states are truncated, so some information is discarded. The decision quality cost must be
  measured and published, not assumed to be small.
- Callers with genuinely long states must chunk them into multiple decisions or accept the cap.

### Risks
- **The cap may cost accuracy.** Stage 5's acceptance criterion is a *published latency/quality
  curve*, not a chosen cap, so the trade-off is visible rather than hidden.
- **Truncation may be a silent correctness hazard** if the warning is ignored. The flag on the
  result exists so a caller can assert on it.
- The 1024-token window is read from the model card and `config.json`; the *effective* quadratic
  behaviour below it is inferred from our own measurements, and the curve in Stage 5 is what
  would confirm it directly.

## Sources

- `reports/runs/s1_latency/` — 6,595 ms long vs 363.5 ms short on CPU (18.1×)
- `reports/runs/s1_llamacpp/` — 1,754.8 ms long vs 89.2 ms short (19.7×) after the 4.08× win
- `google/embeddinggemma-2` `config.json` and model card — 5:1 local/global, 1024-token window
- `docs/ENCODER-EFFICIENCY.md` §0 — the corrected architecture explanation
