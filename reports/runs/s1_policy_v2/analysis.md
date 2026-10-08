# Ambiguity-aware policy v2 — confidence-triggered, dual-temperature

**Run**: `mauricew/nanocore-s1-policy-v2` (consolidated: `notebooks/s1_policy/` v2) —
Kaggle T4, ~3.2 min, clean.
**Artifacts**: `results.json`, `REPORT.md`, `policy_v2_scores.npz`, kernel log.
**Protocol**: 3-way split — train (fit head) / val_A (fit T_prob, τ_answer, τ_in-schema)
/ val_B (conformal q̂) / test (evaluate once). Dual temperature: `T_prob` floored at 0.25
for reported probs, `T_set=1.0` for APS construction.

## What changed vs v1

v1 keyed `answer` on `|APS set| == 1` — unreachable on a flat 150-way softmax
(top-prob ~0.008 < q̂=0.0235), so it answered zero items and lost −0.26 to a plain
threshold gate. v2 keys `answer` on calibrated top-prob ≥ τ_answer (smallest τ reaching
≥0.90 precision on val_A) and demotes the set to a clarification payload (`|set| ≤ 3`).

## Results

| policy | resolved | enc/item | actions |
|---|---:|---:|---|
| always_answer | 0.619 | 1.00 | answer 5500 |
| threshold (OOS abstain) | 0.719 | 1.00 | answer 4694, abstain 806 |
| **ambiguity_v2 (cosine)** | 0.482 | 1.01 | answer 1797, **clarify 55**, escalate 3648 |
| **ambiguity_v2 + TaskHead** | **0.918** | 1.00 | answer 4342, escalate 1158, clarify 0 |

`v2_beats_threshold: false` (cosine leg), `coverage_holds: true` (0.914 vs 0.90 target —
non-degenerate, first time the gate's coverage guarantee held at sane parameters).

## The real finding: the policy is scorer-dependent, and that's the architecture

**On the flat cosine scorer, honest uncertainty *is* escalation.** The 90%-precision
τ_answer (0.0153) can only admit the easy third of in-scope items — the top-prob
distribution has no mass to offer. The remaining traffic is diffuse by definition, so
`|set| ≤ 3` fires only 55 times (1%). On a wide schema, clarify occupies a narrow band
between "confident" and "totally diffuse" — and CLINC150's in-scope items are mostly
either confidently classifiable or hopeless, so the band is nearly empty.

**On the task-fitted head, the same policy resolves 91.8%** — +0.20 over the threshold
gate, +0.30 over always-answer. The head's top-probs are real confidences (in-scope
accuracy 0.970, τ_answer=0.252 covering 97% of val_A), the answer branch does its job,
and escalation is reserved for genuinely out-of-schema input. The head also improves
the in-schema check itself: 86.9% OOS caught at 6.4% false-reject, vs cosine
max_sim's 63.0% at 3.9%.

This is the meta-pipeline validated end-to-end: headroom check admits the task →
fitted head produces meaningful confidence → calibrated policy beats thresholding.
The ambiguity-aware wrapper pays *on top of* a confident scorer; it cannot manufacture
confidence a flat scorer doesn't have.

## Fourth sighting of the sharpening pathology

The head leg's conformal sets are degenerate: q̂ ≈ 0.9999, mean coverage 0.9996 vs the
0.90 target, `clarify` never fires. A cross-entropy head is trained toward confidence;
its T_set=1.0 softmax spreads so little mass that nearly every nonconformity score is
~1. Same pathology as Laya's shipped T=0.1006, our log-loss T=0.02, and the v1 head —
now confirmed as a *systematic* failure mode, not an edge case. Stage 3 acceptance
hardened: set coverage must hold within a band (e.g. 0.88–0.95), not merely ≥ target.

## Honest verdict on `clarify`

The mechanism survives; the CLINC150 contribution does not. Clarify resolved correctly
when it fired, but 55/5500 items (1%) means it is nearly vestigial *on this dataset* —
which is bimodal (clearly in-scope vs clearly OOS) with a thin ambiguous band. The
right claim: clarify is the correct action *where ambiguity exists*; datasets with
genuine in-schema ambiguity (e.g. real user phrasing, not templated intents) are where
it should be evaluated next.

## Consequences for ADR-0013

- `answer` trigger: calibrated top-prob ≥ τ (validated — works when the scorer has
  mass to offer).
- `clarify` trigger: `|APS set| ≤ k` below τ — mechanism correct, dataset-limited.
- New hard requirement: **the policy requires a confidence-meaningful scorer**. On
  schemas where cosine is flat (wide label sets), the ambiguity-aware policy is a
  strict downgrade from threshold gating; the task-fitted head is a prerequisite,
  not an option, for the full three-action policy on wide schemas.
- Stage 3 gate must implement per-scorer calibration (separate T_prob, τ_answer, q̂) —
  v2 confirmed both branches need independent calibration.
