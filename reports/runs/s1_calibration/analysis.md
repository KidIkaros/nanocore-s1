# s1_calibration — does the typed head earn its place?

**Kaggle:** `mauricew/nanocore-s1-calibration`, version 3, **CPU-only** (no accelerator; zero GPU quota spent).
**Status: COMPLETE.** This supersedes the `INCOMPLETE` v2 record, which died at the
abstention step. Full structured output: `results_complete.json`. Kernel log: `kernel_v3.log`.

## What was asked

The v13 baseline measured **accuracy**, where the head cannot win. The typed-decision
claim is not accuracy — it is calibrated probabilities and an abstention gate. This run
measured those axes against *calibrated* retrieval baselines rather than raw similarity
scores.

## Results (Banking77, 3,080-item test split)

| Method | Accuracy | Log score | Brier | ECE | Coverage @95% precision |
|---|---:|---:|---:|---:|---:|
| cosine + fitted τ | 92.92% | 3.800 | 0.968 | 0.907 | 0.9393 |
| kNN-1 + fitted τ | 93.60% | 2.659 | 0.868 | 0.857 | 0.9649 |
| **kNN-5 + fitted τ** | **93.64%** | 0.429 | **0.1131** | **0.0380** | **0.9679** |
| kNN-20 + fitted τ | 93.54% | 0.443 | 0.1154 | 0.0425 | 0.9675 |
| Fingerprint head (3 seeds) | 93.11% ±0.0002 | **0.3766** | 0.1155 | 0.0461 | 0.9659 |
| Interaction head (2 seeds) | 92.48% ±0.0002 | 0.3834 | 0.1233 | 0.0475 | 0.9510 |

Selective accuracy at operating points (the product-relevant view):

| Method | @10% coverage | @50% coverage |
|---|---:|---:|
| Fingerprint head | **100.00%** | **99.61%** |
| Interaction head | **100.00%** | 99.48% |
| kNN-20 | **100.00%** | 99.29% |
| kNN-5 | 99.68% | 99.16% |
| cosine | 99.68% | 99.03% |
| kNN-1 | 99.35% | 98.96% |

## What is settled

**1. Order invariance holds on real data.** 40 states, options randomly permuted,
maximum absolute probability difference **2.38e-07** (float32 epsilon) and **0 argmax
flips**. This is the first design property confirmed empirically rather than asserted,
and it is a genuine advantage over a generative interface. See ADR-0002.

**2. The head does not win on accuracy, and this is temperature-independent.**
kNN-5 93.64% > fingerprint 93.11% > interaction 92.48%. Seed variance is ±0.0002, so
the 0.5-point gap is real on this split (split-level variance remains unmeasured).

**3. The head is best at moderate coverage.** At 50% coverage the fingerprint head
reaches 99.61% selective accuracy — the highest of any candidate — and both heads hit
100.00% at 10% coverage. So there *is* a usable operating point where the head leads.

## Three flaws in this analysis (mine)

**Flaw 1 — the temperature grid bound the baselines.** `cosine_tau`, `knn1_tau` and
`knn5_tau` all fitted τ = **0.5012**, which is the *lower bound* of the search grid
(10^-0.3). Three different methods landing on the same boundary value means the bound
binds and the true optimum lies below it. Their reported log scores are therefore not
their best, which inflates the head's apparent log-score win. **The head-vs-kNN-5 log
score comparison is not settled.** Fix: widen the grid (e.g. 10^-3 … 10^2) and re-fit.

**Flaw 2 — the verdict compares against the wrong baseline.** The verdict block used
`cosine_tau` as "the" calibrated retrieval baseline — the weakest of the four. It
reports `head_beats_calibrated_retrieval_on_brier: true` because 0.1155 < 0.968. Against
the *strongest* baseline, kNN-5, the head **loses** on Brier (0.1155 vs 0.1131), ECE
(0.0461 vs 0.0380), and accuracy. The verdict's `true` values are misleading and should
be recomputed against the best baseline, not the most convenient one.

**Flaw 3 — the 90%-precision metric saturates and carries no information.** Every
candidate reports `coverage_at_90pct_precision = 1.0`, because the task's base accuracy
(93%) already exceeds 90%, so keeping everything still satisfies the criterion. The
verdict line `head_better_abstention_at_90pct_precision: false` is an artefact of
saturation, not a finding. 95% is the discriminating threshold; 90% is useless here.

## Honest conclusion

- **Validated:** order invariance (empirically, on real embeddings).
- **Robust:** the head does not win on accuracy. kNN-5 is more accurate.
- **Competitive but not decisive:** abstention. The head leads at 50% coverage and is
  marginally behind kNN-5 at 95% precision — with the baselines handicapped by Flaw 1,
  so this is provisional.
- **Not yet justified:** the typed head as a whole. Its clear wins are order invariance
  and the typed interface; `Score` and `Noul` remain unexercised.

The next measurement that could change this is the prompt ablation (running), which
tests whether the state/option prompt asymmetry was suppressing head quality.

## Process notes

- Ran entirely on Kaggle CPU: **no GPU quota spent** for a full calibration study.
- The kernel wrote accumulated results to disk after every stage; the v2 failure is why.
- v2's failure was a notebook-generator cell-pruning bug that deleted two helper
  definitions. The fix is versioned as `notebooks/s1_calibration/fix_missing_defs.py`.

## Artifacts

- `results_complete.json` — full v3 output
- `results.json` — the superseded v2 partial record (kept for provenance)
- `kernel_v3.log`, `nanocore-s1-calibration.log` — kernel logs
- `kernel_run_status.json` — Kaggle's status record
