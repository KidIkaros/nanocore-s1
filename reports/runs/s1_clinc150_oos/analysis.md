# CLINC150 OOS — knowing-when benchmark, analysis

**Run**: `mauricew/nanocore-s1-clinc150-oos` v2 — Kaggle T4, completed clean.
( v1 errored: `clinc_oos` is a legacy no-namespace HF dataset id; newer
`huggingface_hub` requires `namespace/name`. Fixed with the official `clinc/oos-eval`
GitHub JSON fallback — verified: 151 labels, 5,500 test rows / 1,000 OOS. )
**Artifacts**: `results.json`, `REPORT.md`, `clinc_scores.npz` (score matrices for both
splits — Stage-3 conformal work reuses these), kernel log, `run_status.json`.
**Post-hoc**: temperature sweep recomputed locally on the cached scores (the kernel's
verdict was confounded by a degenerate temperature; the sweep is the corrected result).

## Headline

| measurement | value |
|---|---:|
| In-scope 150-way accuracy | **0.756** (val 0.770) — headroom-positive |
| Best OOS signal | `max_sim` raw cosine — **AUROC 0.934** |
| Conformal coverage @α=0.10, T=1.0 | **0.913** — guarantee holds |
| Clarify resolution: P(true intent ∈ set \| \|set\|≤3) | **0.975** |
| In-scope items producing a ≤3-option set | **27.9%** |

## The three verdict questions — all answered yes, with one correction

**1. Is OOS separable? Yes — but by score magnitude, not set structure.** `max_sim`
(raw top-1 cosine) achieves AUROC 0.934 with no fitting at all. Every signal clears
0.89. The conformal set size *appeared* to separate (0.912) only under a degenerate
temperature — at T=1.0 the separation collapses (mean set 3.7 in-scope vs 4.0 OOS).
The honest OOS detector is the score magnitude: OOS inputs simply match no intent
well.

**2. Does coverage hold? Yes — at the unsharpened temperature.** With raw softmax
(T=1.0) coverage lands at 0.913 against the 0.90 target. The kernel's reported
coverage of 0.997 at q̂=0.9975 was a pathology: the log-loss fit chose T=0.02 —
extreme sharpening that saturates every distribution and produces bloated sets
(mean 56.7 in-scope). Same failure family as Laya's invalid shipped temperature:
**a scoring-rule optimum that destroys the distribution shape conformal machinery
needs.**

**3. Does `clarify` resolve? Yes — strongly.** At T=1.0, 27.9% of in-scope inputs
yield a conformal set of ≤3 options, and the true intent is inside that set 97.5% of
the time. A one-question follow-up on the ambiguous quarter of traffic resolves it
at ~98% precision. This is the curve-softening mechanism from ADR-0013 with measured
teeth.

## The temperature pathology — the real finding of this run

Log-loss temperature fitting picked T=0.02 because on a dev set where argmax is
mostly right, sharpening reduces loss on correct predictions — the ECE-gaming trap in
a new guise, applied to the log score itself. The consequence: **the temperature that
is optimal for reporting probabilities is wrong for constructing prediction sets.**

Consequence for the design: the scorer and the gate need *separate* treatments —
- a **calibrated probability** path (temperature for reported probabilities, floored
  to prevent degenerate sharpening — the MIN-like guard idea applied to T), and
- a **set-construction** path (unsharpened or mildly-tempered scores — conformal only
  needs relative mass, and T=1.0 empirically held coverage).

This belongs in ADR-0009's implementation notes and Stage 3's acceptance: *coverage
must hold at a non-degenerate temperature* — the test that would have caught this.

## Other measurements

- **In-scope accuracy 75.6%** on 150 intents — headroom-positive (vs 0.93 saturation
  on Banking77/BFCL). If a task-fitted head is ever wanted for intent classification,
  CLINC is a second fair arena alongside GoEmotions.
- **Margin is the weakest signal** (AUROC 0.807) — "two close candidates" is not how
  OOS manifests; OOS inputs match *nothing* well, which is what max_sim captures.
- First real test of the dataset-fallback pattern under load: the GitHub JSON path
  was needed immediately (v1 crashed on the legacy HF id). Pattern works.

## Status of ADR-0013

- `clarify`: **evidence-backed** — mechanism validated with a resolution rate (0.975)
  and a measurable trigger fraction (27.9%).
- Meta-routing in-schema check: **evidence-backed** — OOS separability at 0.934 AUROC
  is exactly the in-schema test a TaskHead needs before firing.
- Gate implementation: unblocked with one design note — separate temperatures for
  probabilities vs set construction (above).
