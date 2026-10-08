# GoEmotions head ladder — analysis

**Run**: `mauricew/nanocore-s1-goemotions-head` v1 — Kaggle T4, ~9 min, completed clean.
**Artifacts**: `results.json`, `REPORT.md`, `goemotions_embeddings.npz` (83MB, kept local,
gitignored — it is a *working* artifact for future head experiments, not a claim artifact),
kernel log, `run_status.json`.

## Result

| arm | input | macro-F1 (test) | micro-F1 (test) | macro-AUROC | vs Platt |
|---|---|---:|---:|---:|---:|
| platt | 28 scores | 0.287 | 0.385 | 0.824 | — |
| linear_scores | 28 scores | 0.398 ±0.001 | 0.463 | 0.901 | **+0.111** |
| mlp_scores | 28 scores | 0.419 ±0.004 | 0.489 | 0.908 | +0.133 |
| linear_emb | 768-dim state | 0.447 ±0.003 | 0.526 | 0.914 | +0.160 |
| **mlp_emb** | 768-dim state | **0.455 ±0.003** | **0.525** | **0.922** | **+0.168** |
| oracle (bound) | 28 scores | 0.297 | 0.395 | — | threshold ceiling |

Seed variance is ±0.003 or better everywhere — the ordering is stable, not noise.

The Platt arm reproduces `s1_goemotions` to within 0.001 (0.287/0.385 vs 0.287/0.386) —
the pipeline is consistent.

## Reading the ladder

**Every learned arm beats the oracle bound.** Platt at 0.287 macro sits essentially *at*
the ceiling of what any thresholding of the score features can achieve (0.297). Even
`linear_scores` clears that ceiling by +0.10 — so learning is not better thresholding; it
extracts signal the Platt structure cannot reach. This closes the "calibration might fix
it" hypothesis from `s1_goemotions`: calibration was already near its own ceiling.

Decomposition of the +0.168 total gain:

| step | Δ macro-F1 | what it credits |
|---|---:|---|
| platt → linear_scores | +0.111 | joint fitting on 43k examples + cross-label mixing — learnable **from scores alone** |
| linear_scores → mlp_scores | +0.021 | nonlinearity within score space |
| mlp_scores → linear_emb | +0.028 | raw state beyond the label-anchor projection |
| linear_emb → mlp_emb | +0.008 | nonlinearity on the state |

The AUROC track (0.824 → 0.901 → 0.908 → 0.914 → 0.922) confirms each step improves the
*ranking*, not just threshold placement — a real signal gain, not a calibration artifact.

**The dominant signal is label co-occurrence** (+0.13 of +0.17 available from 28-dim
scores alone). The raw embedding adds a modest but real +0.036 (linear_emb vs mlp_scores)
— there is signal in the state that the name-anchor projection discards, but less than
the co-occurrence structure.

## What this changes

1. **A trained head is now evidence-backed — as a task-fitted component.** ADR-0011's gate
   worked as designed: headroom was found, the fight was fair, the head won by +0.17
   macro-F1. `mlp_emb` at 0.455 beats every published comparator on this dataset (Jev tuned
   0.353, Qwen3.8-27B tuned 0.323) and reaches parity with fine-tuned-BERT-era results
   (~0.46). The fair comparator for a task-fitted head is a task-fitted model — and we
   are at that frontier with a 740M frozen encoder plus a ~200K-parameter head.

2. **The winning head is label-schema-bound.** It knows GoEmotions' 28 emotions and was
   trained on 43k labeled examples. It does *not* generalize to new label sets — the
   zero-shot property dies the moment a head is fitted. This is exactly the
   "customer-fitted head" upgrade path (the NotDiamond pattern): fit a small head per
   deployment label schema when labeled data and headroom both exist.

3. **The minimal-ML posture survives, correctly scoped.** On saturated tasks
   (Banking77/BFCL ~93% ceilings) trained components demonstrably add nothing — cosine
   scoring is the right default there. On headroom-positive tasks with labeled data, a
   fitted head wins decisively. The architecture needs both modes, gated by the ADR-0011
   check.

4. **Noul gains a third scorer mode.** The scorer layer is now: zero-shot cosine
   (universal, no data), Platt-calibrated cosine (few hundred examples), task-fitted head
   (thousands of examples, headroom-positive task). The data requirement is the missing
   dimension in the ADR-0011 gate — with 43k examples the head wins; with 400 the answer
   is probably Platt. A future experiment should map the data-scaling curve.

## Caveats

- **Data appetite**: the head saw 43k training examples. Whether it still wins at 1k or
  4k examples is unknown — the data-scaling curve is the natural next run (cheap: heads
  train on the saved embeddings, no re-encoding).
- **Comparators are different protocols**: Jev/Qwen/Gemma are generative zero/few-shot
  numbers; a task-fitted head is a different regime. At-parity with fine-tuned BERT is
  the honest claim, not "beats everything."
- **Early stopping uses dev**, and thresholds are tuned on dev — mild selection coupling,
  standard practice; test was touched once.
- **pos_weight sharpening**: heads trained with per-label pos_weight (clamped 1–30) are
  deliberately uncalibrated; thresholds re-tune on dev. For deployment the head's outputs
  would feed the conformal gate, which re-establishes coverage guarantees.

## Consequences for the plan

- ADR-0011 status: gate functioned — **task-fitted heads are in scope** for Noul-family
  tasks with labeled data.
- Architecture doc: the scorer layer needs the three-mode description above; the
  "customer-fitted head" upgrade path is no longer speculative — it has a measured win.
- Next cheap experiments (all CPU on saved embeddings): data-scaling curve for the head
  (subsample train to 1k/4k/16k), and head → conformal gate composition.
