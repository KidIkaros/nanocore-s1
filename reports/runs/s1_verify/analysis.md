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

## v4 — production-surface checks added

| check | result |
|---|---|
| bundle round-trip | **200/200 identical** decisions after save→load |
| ordinal `score` (SST-5, real) | CORN acc **0.432** / MAE 0.732 vs zero-shot 0.318 / 0.875 — head wins |
| llama.cpp backend | ran; argmax agreement **0.815** on the 150-way probe (QAT Q8_0 GGUF) |

The llama.cpp number is honest but below the Stage-1 acceptance bar (≥0.999 argmax
agreement): the **QAT** quant degrades fine-grained ranking on wide schemas. The
non-QAT `embeddinggemma-300M-Q8_0.gguf` build is the follow-up to try before calling
the backend accepted — cosine-agreement, not argmax parity, is the real gate anyway
(argmax hides near-ties).

## The workable-model claim, now artifact-backed

A caller can today: `DecisionModel(encoder, CosineScorer(), gate, cache)` +
`gate.calibrate(val_scores, y)` + `gate.fit_in_schema(in, oos)` →
`model.decide(text, Question("choice", options=...))` → calibrated probabilities +
conformal set + action. On headroom-positive schemas, `TaskHead().fit(embeddings)`
plugs in as the scorer and lifts resolved fraction from 0.48 → 0.93 on CLINC150.
