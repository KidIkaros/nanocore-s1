# v1 Baseline — trained decision head on frozen EmbeddingGemma2, promoted + gated

**Status:** shipped and verified on the real encoder (Kaggle kernel v8, COMPLETE).
**Spec:** #9. **Build tickets:** #10–#15 (all closed). **Glossary:** CONTEXT.md.
**Date recorded:** v1 complete — this file is the canonical "where we are" record.

## What v1 is

A decision core that takes the frozen EmbeddingGemma2 encoder and turns it into
something that **decides**: a trained TaskHead sits over the encoder's pooled
embedding, the head is promoted over the cosine baseline by a three-part gate,
the glial regulator's shift gains are learned, calibration is measured with a
debiased estimator, and pair/ordinal tasks route to their native primitives.

The shipped `decide()` path is torch-free (ADR-0007) via numpy TaskHead
inference; the optional composer is train-time/torch-only with a torch-free
mean-pool default.

## The measured result (clinc150, real EG2, Kaggle T4, kernel v8)

| metric | **head (shipped)** | cosine (baseline) |
|---|---|---|
| in-scope accuracy | **0.969** | — |
| resolved (decides vs defers) | **0.931** | 0.482 |
| answer / escalate / clarify | 4381 / 1119 / 0 | 1797 / 3648 / 55 |
| gate t_prob (confidence) | 0.855 | 0.25 |
| mean set size | 1.94 | 3.80 |

The head resolves **93%** of traffic with clean answers at **97%** accuracy and
escalates 20%; cosine resolves 48% by escalating two-thirds of everything. The
head's gate is calibrated — its high-confidence answers land; cosine's gate
cannot produce confidence (t_prob 0.25) so it defers constantly.

Full protocol suite on Kaggle: **581 passed, 3 skipped**, no errors.

## The three-part gate (why the head is promoted)

Per-anchor, on the held-out cal split, the head ships only if it wins on
**proper score (Brier) by a margin**; a near-tie keeps cosine (ADR-0008
non-inferiority). Measured across anchors (cached-embedding gate, #10/#11):

- **decision win:** clinc150 +0.494, emotion +0.127, banking77 Brier +0.81
  (head 0.124 vs cosine 0.937 — a 7.6x better proper score).
- **calibration (debiased equal-mass ECE + asymptotic CI):** head ECE 0.020,
  CI upper 0.021 ≤ 0.05 → passes. (The biased equal-width estimator reported
  0.052 "narrowly missed" — that miss was the estimator's upward bias, not the
  head's calibration.)
- **risk-coverage:** selective accuracy 0.978 @ 0.93 coverage.

`adapt()` now ships the head per-anchor where it wins (`select_scorer_gated`),
proven on the real encoder: `scorer_choice=taskhead` on both retrain slices.

## What each ticket landed

| ticket | artifact |
|---|---|
| #10 three-part gate across anchors | head wins all three (decision, Brier, risk-coverage) |
| #11 debiased ECE + CI | `equal_mass_ece`, `equal_mass_ece_ci` (arXiv:2012.08668, 2408.08998) |
| #14 gated wiring | `select_scorer_gated` — head ships on Brier margin, near-tie keeps cosine |
| #13 learned regulator | `fit_regulator_gains` — glial shift gains fitted from trajectory log |
| #12 composer | `EmbeddingComposer` built; npz deferred (optional, torch-only; mean-pool is the torch-free default) |
| #15 pair/ordinal routing | PairScorer / OrdinalScorer as selectable arms; ordinal ships torch-free npz |

## Process fix earned this cycle

`dry_run.py --full` is now the pre-push gate. The default dry-run runs only the
torch-free tier (full collection OOM'd the host once), which structurally hid
stale assertions in torch-tier files until they crashed Kaggle — **v4**
(float-tolerance, fidelity guard shelved) and **v7** (argmax-vs-Brier stale test)
were the same recurrence. `--full` runs the whole suite in-process (~2 min,
584 tests) and catches that class locally. Full suite 584 green, torch-free 317
green.

## Honest boundaries (what v1 is not)

- **Composer:** optional, torch-only at serve. A numpy port of the transformer
  block stack is deferred — mean-pool is the torch-free default (ADR-0007).
- **M-style latent dynamics:** deferred (v1-lean) until a recurrent/generative
  organ exists. The learned piece in v1 is the regulator *gains*, not a latent
  predictor.
- **No from-scratch model, no generation.** v1 keeps EG2 frozen (the correction
  that resolved the fog) and adds a trained, promoted, calibrated decision head.

## Next frontier (per the wayfinder map #1)

All gated on "does the typed-decision core need it":
looped depth · hybrid (attention + SSM) backbone · M-style latent dynamics ·
the memory plane. Not started — v1 is the baseline to build from.
