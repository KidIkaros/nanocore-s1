# NanoCore-S1 — Roadmap

**Created**: 2026-10-08 · **Status**: active · **Owner**: project owner

This is the single source of truth for what we are building and what "done" means.
It exists because the project has a documented tendency to stray: to measure without
building, to open a new kernel per revision, to re-litigate settled decisions.

**Rule for every session: read this file first. If a proposed action is not in this
roadmap, either add it here with acceptance criteria, or don't do it.**

---

## 0. Governing rules

| # | Rule |
|---|---|
| R1 | **Local machine = write code, read files, static checks. Kaggle = every execution.** No local model loading, ever. The host froze once already. |
| R2 | **Acceptance criteria are written before a phase starts.** No "we'll figure out what done means." |
| R3 | **No claim without an artifact** in `reports/runs/`. |
| R4 | **One notebook per concern, iterated by version.** No new kernel per revision. |
| R5 | **Nothing ships without a rollback path.** |
| R6 | **Stop-doing list (§6) is binding.** |

### Status legend

`DONE` verified with artifact · `PARTIAL` exists but unverified · `TODO` not started ·
`BLOCKED` waiting on another phase · `OPT` optional / scope-dependent

---

## 1. Current state — what exists and is verified

All verified in `reports/runs/s1_verify/` (Kaggle T4, v4 green):

| Result | Value |
|---|---|
| Test suite | 149 pass |
| CLINC150 resolved — cosine scorer | 0.482 |
| CLINC150 resolved — fitted TaskHead | 0.931 |
| In-scope conformal coverage @ α=0.10 | 0.914 (in-band, non-degenerate) |
| TaskHead in-scope accuracy | 0.969 |
| Bundle save→load round-trip | 200/200 identical decisions |
| SST-5 ordinal (CORN) vs zero-shot | 0.432 / 0.732 MAE vs 0.318 / 0.875 |
| GoEmotions macro-F1 (head ladder) | 0.455 (`mlp_emb`) vs 0.287 Platt |
| llama.cpp GGUF argmax agreement | 0.815 (QAT Q8_0) — below the ≥0.999 bar |

**Blunt summary:** the decision layer (scorer + gate + cache + bundle) is implemented and
verified. It is *not* yet a usable model — there is no adaptation path, no runnable entry
point, no escalation handling, no serving, no monitoring.

---

## 2. Component inventory

The full stack. Everything not marked `DONE` is roadmap work.

| # | Component | Role | Status | Phase |
|---|---|---|---|---|
| 1 | Encoder (`StateEncoder`, `LlamaCppEncoder`) | text/media → embeddings | DONE (code) | — |
| 2 | `CosineScorer` | zero-shot scoring | DONE | — |
| 3 | `OrdinalScorer` (CORN) | ordered `Score` | DONE | — |
| 4 | `TaskHead` | per-deployment fitted scorer | DONE | — |
| 5 | `ConformalGate` | calibrated sets + actions | DONE | — |
| 6 | `DecisionCache` | no repeat encodes | DONE | — |
| 7 | `DecisionModel` | orchestrator | DONE | — |
| 8 | Bundle save/load | shippable artifact | DONE | — |
| 9 | CLI | entry point | PARTIAL (never executed) | 1 |
| 10 | **Adaptation harness** | labeled data → head + gate → bundle | TODO | 2 |
| 11 | **Escalation handler** | what `escalate` does | TODO | 3 |
| 12 | **Clarify presenter** | set → question payload | TODO | 3 |
| 13 | **Serving layer** | model server + prediction endpoint | TODO | 4 |
| 14 | **Prediction log** | inputs/outputs/version/latency | TODO | 4 |
| 15 | **Cost accounting** | tokens, USD, throughput | TODO | 4 |
| 16 | **Operational monitoring** | latency percentiles, errors | TODO | 5 |
| 17 | **ML monitoring** | prediction/feature distribution, per-slice | TODO | 5 |
| 18 | **Drift detection** | covariate/label/concept, SPC | TODO | 5 |
| 19 | **Uncertainty decomposition** | aleatoric vs epistemic | TODO | 5 |
| 20 | **Retraining pipeline** | stateless, scheduled/triggered | TODO | 6 |
| 21 | **Model registry + lineage** | which data+config made this | TODO | 6 |
| 22 | **Release strategy** | shadow → canary → rollback | TODO | 6 |
| 23 | **Causal readout** | did the model cause the gain | TODO | 6 |
| 24 | Our own baselines | TF-IDF/LR, fine-tuned small model | TODO | 7 |
| 25 | Benchmark breadth | ~15 datasets / 7 families | TODO | 7 |
| 26 | Multimodal real-data proof | vision (+audio) | TODO | 7 |
| 27 | Multilingual slice | language-gradient risk | TODO | 7 |
| 28 | Continuous numeric output | requested-precision values | OPT | 8 |
| 29 | llama.cpp parity + device latency | on-device runtime | TODO (you trigger) | 8 |
| 30 | Harness integration contract | where it sits in an agent loop | TODO | 9 |
| 31 | Agentic memory control | System-One memory plane | OPT | 10 |
| 32 | README + ops runbook | how anyone uses/runs it | TODO | 11 |
| 33 | Responsible design | bias per slice, privacy, compliance | TODO | 11 |

---

## 3. Phases

Each phase: goal → where it runs → deliverables → **acceptance** → dependencies.

### Phase 0 — Workspace rule + repo hygiene
**Where:** local (docs only). **Depends:** —

- Write R1 into `AGENTS.md` with a preflight guard.
- Remove local model artifacts; confirm no local execution path exists.
- **Acceptance:** rule documented; no script in the repo loads weights on this host.

### Phase 1 — Runnable slice
**Where:** Kaggle, 1 kernel. **Depends:** 0.

- Produce a **real calibrated bundle** (encoder id + scorer + gate calibration).
- Execute the CLI on real inputs inside the kernel; record the transcript.
- **Acceptance:** `bundle/` artifact + transcript showing `action`, prediction set,
  and probabilities on ≥10 real inputs.

### Phase 2 — Adaptation harness
**Where:** Kaggle, 1 kernel. **Depends:** 1.

- Pipeline: labeled data → encoder → `TaskHead.fit` → `gate.calibrate` (3-way splits)
  → bundle. One command.
- Leakage safety: time-based splits where data is time-correlated.
- Schema validation at ingestion.
- **One frozen template per task, no per-dataset tuning** (the benchmark rule).
- Prove on a task we have *not* already calibrated.
- **Acceptance:** one command turns a labeled dataset into a bundle; the bundle
  reproduces held-out metrics within tolerance.

### Phase 3 — Action handlers
**Where:** local code, Kaggle test. **Depends:** 1.

- **Escalation handler:** pluggable — System Two call / queue / human handoff.
- **Clarify presenter:** prediction set → question payload.
- **Acceptance:** every action has a concrete tested handler; no action is a dead end.

### Phase 4 — Serving layer + prediction logging + cost
**Where:** local code, Kaggle test. **Depends:** 3.

- Record the **prediction-mode decision** (online vs batch) explicitly — don't default into it.
- **Model server** (loaded models, predict endpoint) decoupled from business logic.
- **Prediction log:** inputs, outputs, model version, latency.
- **Cost accounting:** tokens, USD, throughput per decision.
- **Acceptance:** a running service answers a request; every decision is logged with
  version + latency; cost per decision is reported.

### Phase 5 — Monitoring, two layers
**Where:** local code, Kaggle test. **Depends:** 4 (needs the log).

- **Operational:** latency percentiles (not means), throughput, error rate.
- **ML:** prediction distribution, feature distribution, accuracy when labels arrive —
  **per slice**, thresholds tied to business tolerance.
- **Drift:** simple stats first → KS/SPC; name the shift type.
- **Uncertainty decomposition:** aleatoric vs epistemic (epistemic drives "collect more").
- **Acceptance:** both layers alert with thresholds; a synthetic shift is detected.

### Phase 6 — Retraining, registry, release
**Where:** Kaggle, 1 kernel. **Depends:** 5.

- **Stateless retrain pipeline** (recreate from data + config) with cadence decision:
  scheduled vs drift-triggered.
- **Model registry with lineage:** which data + config produced this bundle.
- **Release strategy:** shadow → canary → full, with rollback.
- **Causal readout:** did the change cause the improvement (not just correlate).
- **Acceptance:** a retrain produces a versioned bundle; a shadow run compares it on
  live-logged traffic.

### Phase 7 — Evidence: evaluation rigor + breadth
**Where:** Kaggle, 2–3 kernels. **Depends:** 2 (adaptation) for breadth.

Must meet the category's evaluation standard (§4). Deliverables:

- **Our own baselines:** TF-IDF+LR and/or a fine-tuned small model on identical splits.
- **Benchmark breadth:** ~15 datasets across the 7 families (classification, routing,
  NLI/grounding, reading comprehension, commonsense, moderation, rubric scoring).
- **Multimodal proof:** one real vision task (audio if it fits).
- **Multilingual slice:** measure the language gradient.
- **Acceptance:** a head-to-head table we ran ourselves; all primary metrics carry
  bootstrap CIs; AURC + ECE/Brier reported per dataset; memorization probes pass.

### Phase 8 — On-device deployment *(you trigger this)*
**Where:** your PC, memory-gated. **Depends:** 7.

- llama.cpp parity (correct metric: **cosine agreement**, not argmax) + **real latency
  on your hardware**.
- **Continuous numeric output** (recursive range refinement) — optional.
- Privacy: inputs never leave the device.
- **Acceptance:** parity within tolerance; measured latency; memory gate passed.

### Phase 9 — Harness integration contract
**Where:** local code, Kaggle test. **Depends:** 4.

- Define where NanoCore sits in an agent loop: context-management hook, safety-controls
  hook, orchestration surface, extension surface.
- Frame **tool/model selection** as a first-class routing use case.
- **Acceptance:** a documented integration contract + a working reference integration.

### Phase 10 — Agentic memory control *(optional)*
**Where:** Kaggle. **Depends:** 9.

- System-One memory plane: memory typing, query routing, retrieval-budget allocation,
  adaptive stopping.
- **Acceptance:** demonstrated on a long-horizon retrieval task, or explicitly dropped.

### Phase 11 — Docs + responsible design
**Where:** local, continuous. **Depends:** all.

- **README:** install, adapt, run, serve. **Ops runbook:** alerts, retrain, rollback.
- **Responsible design:** bias per slice, privacy posture, compliance posture for
  regulated domains — requirements, not a pre-ship checklist.
- **Acceptance:** a new person can install, adapt on their data, and run it from the README.

---

## 4. Evaluation standard (binding for Phase 7+)

Derived from the category's reference evaluation (*Evaluating and Benchmarking the
System One Model Jev*, 37 datasets / 346k requests). Every primary metric must carry:

| Requirement | Detail |
|---|---|
| **Bootstrap CIs** | 95% percentile bootstrap, 500 resamples, on every primary metric |
| **Selective prediction** | accuracy on top-80% and top-50% confidence; **AURC** (risk–coverage) |
| **Calibration** | **ECE over 15 bins** + **Brier**, per dataset |
| **Threshold-free** | AUROC + AUPRC for binary tasks |
| **Cost** | tokens, USD, latency, throughput per decision |
| **Memorization probes** | option rotation + question withholding |
| **Thresholds** | chosen on train/val, applied unchanged to test |
| **Templates** | one frozen template per dataset, no tuning |
| **Cache** | raw responses content-addressed so every table recomputes offline |

**Our differentiator:** conformal prediction sets — the reference evaluation has none.
Our gate is ahead of the category standard on uncertainty; keep it that way.

---

## 5. Definition of done (project level)

The project is complete when:

1. A person can install it, point it at their labeled data, and get a calibrated bundle (Phase 2).
2. That bundle answers, clarifies, or escalates — and escalation goes somewhere real (Phase 3).
3. Something else can call it (Phase 4).
4. It reports whether it is still working (Phase 5).
5. It can be updated and rolled back safely (Phase 6).
6. Its numbers are ours and carry uncertainty (Phase 7).
7. It runs on the target device with measured latency (Phase 8).

---

## 6. Stop-doing list (binding)

| Don't | Because |
|---|---|
| Run anything that loads weights locally | R1; the host froze |
| Open a new kernel per revision | R4; burns quota, creates untraceable history |
| Start a phase without written acceptance criteria | R2 |
| Claim a number without an artifact | R3 |
| Compare our numbers to published numbers from other papers | different splits/protocols; run our own baseline |
| Add a component because a paper does it | needs a measured deficiency first |
| Write docs without a code change behind them | the doc/code ratio is already 2:1 |
| Report a point estimate as if it were a result | Phase 7 requires CIs |
| Ship without a rollback path | R5 |

---

## 7. Progress log (append-only)

| Date | Phase | Event |
|---|---|---|
| 2026-10-08 | — | Roadmap created. Phases 0–11 defined, component inventory complete (33 items). |
