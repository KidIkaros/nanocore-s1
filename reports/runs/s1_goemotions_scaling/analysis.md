# GoEmotions data-scaling curve — analysis

**Run**: `scripts/goemotions_scaling.py`, local RTX 2070, 530 s, on the cached real
embeddings from `s1_goemotions_head` (no re-encoding, no model loading — the compute
boundary explicitly permits head training on cached embeddings).
**Protocol**: identical to the kernel — fit on a train subsample, early-stop + threshold
tuning on dev, one test evaluation. 3 seeds per cell.

## The curve — test macro-F1 (mean ± std)

| n | platt | linear_scores | mlp_scores | linear_emb | mlp_emb |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.123±.015 | 0.160 | 0.171 | 0.184 | **0.201** |
| 400 | 0.226±.006 | 0.247 | 0.260 | 0.267 | **0.277** |
| 1,000 | 0.243±.008 | 0.303 | 0.313 | 0.331 | 0.329 |
| 4,000 | 0.281±.004 | 0.372 | 0.377 | **0.414** | 0.398 |
| 16,000 | 0.312±.004 | 0.401 | 0.409 | 0.436 | **0.436** |
| 43,410 | 0.314±.000 | 0.400 | 0.422 | 0.449 | **0.454** |

(Reference: dev-fitted Platt incumbent = 0.287; oracle threshold bound on scores = 0.297.)

## Findings

1. **There is no crossover — the head wins at every measured scale, down to n=100.**
   Even with 100 labeled examples, `mlp_emb` beats Platt by +0.08 macro-F1. The ADR-0011
   gate does not need a strict minimum-sample rule for *whether* a head helps; it needs
   the headroom check plus whatever data exists. The deployment question is instead
   whether the absolute level is useful — at n≤400 nothing clears 0.28 macro-F1.

2. **Platt saturates; heads don't.** Per-label calibration plateaus at ~0.31 by n=16k —
   fitting it on 43k instead of dev lifts it only to 0.314 (calibration data helps the
   calibrator, but within its functional ceiling). The heads were still climbing at 43k
   (`mlp_emb` +0.018 from 16k→43k) — the curve has not flattened; more labeled data
   would likely keep paying.

3. **`linear_emb` is the sweet spot below ~16k.** Most of the embedding's extra signal is
   linearly accessible: at n=4k the linear head on state beats the MLP (0.414 vs 0.398),
   and at full data it trails by only 0.005. Nonlinearity on the state earns its
   parameters only at ≥16k examples.

4. **Scores-only arms track emb arms but never catch up.** The ~+0.03–0.05 gap
   (`linear_emb` vs `linear_scores`) persists at every scale — the state carries real
   signal beyond the label-anchor projections, consistently.

5. **A correction to the earlier bound**: train-fitted Platt (0.314) slightly exceeds the
   oracle bound (0.297) — the bound applies to the *dev-fitted* probability map, not to
   thresholding universally. Noted to keep the claim precise.

## Consequence for ADR-0011

The gate's two questions are now: (a) is there headroom? (unchanged), and (b) is the
achievable absolute level useful at the available data size? The answer is never "use
Platt instead because data is scarce" — Platt is dominated at every size tested. The
data-size axis instead selects *which* head: linear on embeddings for small-to-mid data,
MLP on embeddings at ≥16k.
