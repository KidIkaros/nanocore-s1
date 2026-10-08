# GoEmotions Noul battery — the first headroom-positive benchmark

**Result: headroom confirmed (0.713, ADR-0011's gate passes); Noul exercised for the first
time; tuned micro-F1 ties Jev (0.386 vs 0.387); macro-F1 sits between Gemma and Qwen. And
one self-correction: the kernel's own "not a head opportunity" read is too strong.**

Kernel `mauricew/nanocore-s1-goemotions` v1 — complete in **141 s** on a T4. 10,853 texts
encoded, 1.27 GiB peak.

## The numbers

Two label-text templates scored as `s·e` per (comment, emotion) pair; per-label Platt
sigmoids fitted on validation (n=5,426); evaluated on test (n=5,427).

| template | macro-F1 @0.5 | macro-F1 @tuned | micro-F1 @0.5 | micro-F1 @tuned | macro-AUROC |
|---|---:|---:|---:|---:|---:|
| **bare name** (`anger`) | 0.075 | **0.287** | 0.099 | **0.386** | **0.824** |
| sentence (`a comment expressing anger`) | 0.047 | 0.241 | 0.073 | 0.352 | 0.789 |

**Bare emotion names beat the descriptive template** — the extra words dilute the signal.
Useful negative result for label-text design; names it is.

## Against the Jev paper's Table 4 (arXiv:2609.37647)

| model | macro @0.5 | macro @tuned | micro @0.5 | micro @tuned |
|---|---:|---:|---:|---:|
| **ours (EG2 + Platt)** | 0.075 | 0.287 | 0.099 | **0.386** |
| Jev | 0.243 | **0.353** | 0.239 | **0.387** |
| Qwen3.8-27B | 0.255 | 0.323 | 0.236 | 0.317 |
| Gemma-4-E4B | 0.211 | 0.267 | 0.207 | 0.268 |

- **Tuned micro-F1 is a statistical tie with Jev** — 0.386 vs 0.387, the best number on the
  table. We beat Qwen (+0.069) and Gemma (+0.118) there.
- **Tuned macro-F1 sits between Gemma and Qwen** (0.287), behind Jev (0.353) — the gap is
  concentrated in rare labels (grief has 6 test positives, relief 11, pride 16).
- **Fixed-0.5 is a collapse** (0.075) because our tuned thresholds are all low
  (0.02–0.28) — Platt probabilities are conservative, and F1-optimal operating points sit
  far below 0.5. The same paper finding applies to us that applied to Jev: fixed-threshold
  F1 understates multilabel systems.

## The interpretation — and the correction

**What the AUROC-vs-F1 gap (0.824 vs 0.287) actually says.** The ordering of
scores-per-label is genuinely strong — cosine against bare emotion names ranks relevant
emotions well. The F1 gap is *partly* a calibration artifact, and the kernel's canned read
("a scoring-rule fix, not a head opportunity") is **too strong in our favour**:

- **Calibration fixes a chunk** — fixed→tuned moved macro-F1 0.075 → 0.287 (3.8×) with no
  model change. Better calibration machinery (conformal, isotonic, per-label thresholds) is
  worth real points.
- **But per-label Platt sigmoids are structurally limited**: they score each emotion
  independently and cannot exploit label co-occurrence (joy/love, annoyance/disapproval),
  which is exactly what trained multilabel classifiers do. Fine-tuned-BERT-era systems
  reach macro-F1 ~0.46 here — ~0.17 above our tuned score. **That gap is the headroom a
  learned component could legitimately chase**, and 0.824 AUROC leaves room above it.

The honest statement is two-sided: calibration/threshold machinery captures the cheap part
of the gap; a learned multilabel head is a **live hypothesis** for the rest — testable now
for the first time, because this benchmark passes ADR-0011's headroom gate (0.713).

## What this run established

1. **A headroom-positive benchmark exists.** GoEmotions admits trained-component
   experiments — the first task we've found that does. It unblocks fair evaluation of
   learned heads, customer-fitted heads, and any future `Score` variant, where Banking77
   and BFCL could not.
2. **`Noul` works as a scored primitive.** 28 binary questions per comment is a natural
   multilabel formulation — no forcing through single-label `Choice`.
3. **Per-example scores now exist.** `goemotions_scores.npz` (dev+test score matrices,
   gold, label names) is the first saved-score artifact — exactly what Stage 3's conformal
   gate needs for calibration, and what any head experiment needs as input.
4. **Label text matters and is cheap to ablate**: bare names > sentence templates.
5. **Run cost: 141 s.** The pattern (encode → score → calibrate → save scores) is now
   proven cheap enough to rerun freely.

## What it did not establish

- **Whether a trained head beats Platt thresholds** — the experiment is designed now, not
  run. The score arrays make it a CPU-cheap next kernel.
- **Conformal coverage** — the calibration data exists; the gate does not.
- **Rare-label behaviour** — 3 labels have <20 test positives; macro-F1 on them is noise.
  Report micro-F1 as the more stable headline.
