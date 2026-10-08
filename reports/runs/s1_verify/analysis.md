# s1_verify — end-to-end verification of the shipped package

**Run**: `mauricew/nanocore-s1-verify` v2 — Kaggle T4, ~4.7 min.
**Question**: does the *implemented* pipeline (`src/decision/` classes) reproduce the
measured v2 policy numbers, or was the earlier result an artifact of kernel-local math?
**Artifacts**: `results.json`, `verify_scores.npz` (local only — redundant with
`s1_policy_v2/policy_v2_scores.npz`), kernel log, `run_status.json`.

## Verdict — all green

| check | result |
|---|---|
| test suite on Kaggle image | 149 passed |
| cosine leg resolved | **0.482** — identical to `s1_policy_v2` (same splits/data) |
| head leg resolved | **0.931** — vs 0.918 in v2 (slightly better; gate's in-schema τ on raw logits vs prob-max, monotone-equivalent) |
| in-scope coverage | **0.914** @ α=0.10 — non-degenerate band |
| head in-scope accuracy | **0.969** |

## What this establishes

- `ConformalGate.calibrate` reproduces the kernel protocol: T_prob floored at 0.25,
  τ_answer ≈ 0.0153, qhat ≈ 0.0239, τ_in_schema ≈ 0.70 — through the shipped class.
- `DecisionModel.decide` produces the typed contract on real inputs: live calls show
  `answer` on a clear in-scope intent and `escalate` with an interpretable candidate
  set on both a borderline ambiguous input and an unrelated one.
- `DecisionCache` exercised live (repeat `decide()` calls hit cache, zero re-encode).
- `verify_scores.npz` kept local (redundant with `s1_policy_v2`'s matrix; regenerable
  in ~5 min on T4).

## The workable-model claim, now artifact-backed

A caller can today: `DecisionModel(encoder, CosineScorer(), gate, cache)` +
`gate.calibrate(val_scores, y)` + `gate.fit_in_schema(in, oos)` →
`model.decide(text, Question("choice", options=...))` → calibrated probabilities +
conformal set + action. On headroom-positive schemas, `TaskHead().fit(embeddings)`
plugs in as the scorer and lifts resolved fraction from 0.48 → 0.93 on CLINC150.
