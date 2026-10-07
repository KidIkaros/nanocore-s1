# s1_sharpening — S3, S5, S7, S9

**Kaggle:** `mauricew/nanocore-s1-sharpening`, version 1, **CPU-only** (zero GPU quota).
**Status: ERRORED at the S5 diagnostic step.** Incremental persistence saved the baselines
and all four head arms; the diagnostic, Matryoshka and verdict cells did not run.

## Headline: the wide temperature grid overturned an earlier conclusion

S9 widened the temperature search from `10^-0.3 … 10^1.5` to `10^-3 … 10^2`. Three of the
four baselines had previously fitted the *lower bound* of the narrow grid, so their
reported log scores were not their best. Corrected:

| baseline | log score, narrow grid | log score, wide grid | |
|---|---:|---:|---|
| cosine + τ | 3.800 | **0.380** | was bound-limited |
| kNN-1 + τ | 2.659 | **0.570** | was bound-limited |
| kNN-5 + τ | 0.429 | 0.508 | grid now coarser |
| kNN-20 + τ | 0.443 | 0.480 | grid now coarser |

**The claim that "naive retrieval calibration fails badly" was an artifact of my own grid
bound, not a property of cosine scoring.** With a properly fitted temperature
(τ = 0.0126), a plain cosine classifier reaches **log 0.380, Brier 0.117, ECE 0.045**.

## The head buys essentially nothing over one fitted parameter

| method | parameters | accuracy | log | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| cosine + τ | **1** | 92.92% | 0.380 | 0.117 | **0.045** |
| kNN-5 + τ | 0 (corpus) | **93.64%** | 0.508 | 0.117 | 0.052 |
| fingerprint head, random init | 59,136 | 93.11% | **0.377** | **0.116** | 0.046 |

A 59,136-parameter head trained for 120 epochs is **statistically indistinguishable** from a
one-parameter temperature on the zero-shot cosine score. This is the honest verdict for the
`Choice` task, and it is stronger than the earlier "the head loses to kNN-5" framing.

## S3 rejected: initialization from option embeddings has no effect

| arm | accuracy | log | Brier |
|---|---:|---:|---:|
| random init, trained | 93.11% ±0.0002 | 0.377 | 0.116 |
| **init from option embeddings, unnormalized, trained** | 93.12% ±0.0000 | 0.377 | 0.116 |

Identical to three decimals. The hypothesis was that random initialization throws away the
92.92% zero-shot prior — but with 9,003 training examples and 120 epochs the optimizer
converges to the same solution from either start, so the initialization washes out.
**S3 was my highest-ranked proposal and the evidence kills it.**

## S5 rejected as implemented: normalizing fingerprints plus training is harmful

| arm | accuracy | log | Brier | ECE |
|---|---:|---:|---:|---:|
| init + normalized, **untrained** | 92.86% | 3.800 | 0.968 | 0.906 |
| init + normalized, **trained** | **79.66%** ±0.0018 | 3.441 | 0.947 | 0.761 |

Training the normalized head **dropped accuracy from 93.11% to 79.66%**. Hypothesis (not
established): normalizing inside the forward pass while optimizing the unnormalized
parameter produces a `1/|v|` gradient factor, and Adam at lr = 0.05 rotates the fingerprints
away from the prior into a badly conditioned solution. The Zero-Bias paper's second remedy
assumes a parameterization that keeps the norm controlled; we applied it as a forward-pass
transform instead.

## Two protocol defects this run exposed

1. **The head's internal temperature grid is narrow and inconsistent with the baselines'.**
   `DecisionHead.fit_temperature` defaults to `10^-0.3 … 10^0.7`, so for cosine-scale scores
   it bound-limits (the normalized arms report log 3.800 with a fitted τ of 0.5012). The
   notebook fitted the *baselines* on the wide grid and the *heads* on the narrow one — an
   unfair comparison in the heads' disfavour. Must be unified.
2. **The wide grid is coarser, not finer.** 51 points over five decades is ~0.1 decade per
   step, against 0.05 before, which is why kNN-5/20 fitted slightly worse. The fix is a
   fine grid over a wide range, or two-stage refinement — not merely a wider one.

## Why the kernel died

`AttributeError: 'DecisionHead' object has no attribute 'fingerprint_distance_matrix'`.

The S5 proposal said "run the separability diagnostic we built and never used." That is true
of **jev-stack**, not of this repository — when the head was ported into `nanocore-s1`, the
diagnostics methods were not carried across. The claim in
`docs/ARCHITECTURE-SHARPENING.md` was wrong for this repo and has been corrected.

## What survives

- **S9 works and matters**: it changed a headline conclusion. Adopt it, with a finer grid.
- **S7 (Matryoshka) untested** — the cell never ran.
- **S3 and S5 as written: rejected.**

## Artifacts

- `results.json` — baselines and all four head arms (incremental snapshot)
- `nanocore-s1-sharpening.log` — kernel log with the traceback
- `kernel_run_status.json`
