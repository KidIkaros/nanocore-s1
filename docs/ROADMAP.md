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
| llama.cpp GGUF parity | **UNMEASURED** — see the correction below |

**Blunt summary:** the decision layer (scorer + gate + cache + bundle) is implemented and
verified. It is *not* yet a usable model — there is no adaptation path, no runnable entry
point, no escalation handling, no serving, no monitoring.

> **Correction (2026-10-08, v23).** Two things in the table above are wrong, and both were
> found while preparing Phase 8.
>
> 1. **The GGUF parity row was unsourced.** It read *"argmax agreement 0.815 (QAT Q8_0) —
>    below the ≥0.999 bar"*, but **no artifact contains that number** and the `s1_llamacpp`
>    kernel never implemented an agreement measurement — it measured sizes, thread scaling,
>    load time, encode latency and Banking77 quality. The figure violated the stop-doing
>    rule against claiming a number without an artifact, and it was the most consequential
>    number in the table, because it asserted that the on-device path fails parity.
>    **`src/decision/parity.py` (built for Phase 8) is the first implementation of that
>    measurement.** Phase 8 will answer a question this table had been answering by
>    assertion.
> 2. **The "not yet a usable model" summary is stale by ~19 versions.** Adaptation
>    (`adapt`), a runnable entry point (the CLI), escalation handling (`handlers`),
>    serving (`serve`) and monitoring (`monitor`) all exist and are verified — see the
>    progress log from v5 onward, and the definition of done below, where six of seven
>    items are done. The section is kept for the record of what was true at v4; do not
>    read it as current.

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
| 9 | CLI | entry point | DONE (v5: 12 inputs, subprocess) | 1 |
| 10 | **Adaptation harness** | labeled data → head + gate → bundle | DONE (s1_adapt v3, Banking77) | 2 |
| 11 | **Escalation handler** | what `escalate` does | DONE (unit-tested; kernel leg pending) | 3 |
| 12 | **Clarify presenter** | set → question payload | DONE (unit-tested) | 3 |
| 13 | **Serving layer** | model server + prediction endpoint | DONE (v6 in-kernel, real POST) | 4 |
| 14 | **Prediction log** | inputs/outputs/version/latency | DONE (JSONL verified in-kernel) | 4 |
| 15 | **Cost accounting** | latency + throughput per decision | DONE (/stats; tokens/USD n/a local) | 4 |
| 16 | **Operational monitoring** | latency percentiles, errors | DONE (monitor.py, tested) | 5 |
| 17 | **ML monitoring** | prediction/feature distribution, per-slice | DONE (monitor.py, tested) | 5 |
| 18 | **Drift detection** | covariate/label/concept, SPC | DONE (KS + action deltas; concept needs labels) | 5 |
| 19 | **Uncertainty decomposition** | aleatoric vs epistemic | DONE (bootstrap ensemble, BALD gap) | 5 |
| 20 | **Retraining pipeline** | stateless, scheduled/triggered | PARTIAL (adapt() is the stateless train; trigger/cadence TODO) | 6 |
| 21 | **Model registry + lineage** | which data+config made this | DONE (registry.py, tested) | 6 |
| 22 | **Release strategy** | shadow → canary → rollback | DONE — shadow + rollback live-verified on v17 (n=300, 0 failures); canary leg untested | 6 |
| 23 | **Causal readout** | did the model cause the gain | TODO | 6 |
| 24 | Our own baselines | TF-IDF/LR, fine-tuned small model | TODO | 7 |
| 25 | Benchmark breadth | ~15 datasets / 7 families | TODO | 7 |
| 26 | Multimodal real-data proof | vision (+audio) | TODO | 7 |
| 27 | Multilingual slice | language-gradient risk | TODO | 7 |
| 28 | Continuous numeric output | requested-precision values | **model half DONE** — `OrdinalScorer.expected()`; the refinement loop is harness | 8 |
| 29 | llama.cpp parity + device latency | on-device runtime | TODO (you trigger) | 8 |
| 30 | Harness integration contract | where it sits in an agent loop | TODO — **harness by definition**; it binds 28/31/33 | 9 |
| 31 | Agentic memory control | System-One memory plane | OPT — **harness**; the composition half is DONE (composer) | 10 |
| 32 | README + ops runbook | how anyone uses/runs it | TODO | 11 |
| 33 | Responsible design | bias per slice, privacy, compliance | **model half DONE** (`modelcard.py`, privacy test); system card + oversight are harness | 11 |

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

### Phase 7.5 — Pre-production qualification
**Where:** Kaggle, 1 kernel. **Depends:** 7. **Blocks:** 8.

The step from "we measured a lot" to "this build may ship". A kernel verdict answers
*did the measurement land where we expected*; a qualification answers *may this build ship*,
which needs criteria fixed **before** the numbers arrive and a report able to say no.

`src/decision/qualify.py` holds the criteria as data — seven gates, three severities
(`must_pass` = a property of shipped behaviour; `must_fix` = an open capability gap that
blocks the pre-production *claim*; `report` = recorded, not gated) — plus
`src/decision/refusals.py`, the behavioural safety battery, and
`src/decision/readiness.py`, the human-interaction readiness checks.

| gate | criteria |
|---|---|
| **A. Contract & determinism** | suite green; bundle round-trips identically; **the qualification names the artifact it qualified** (bundle digest) |
| **B. Quality vs baseline** | head beats TF-IDF+LR on **both proper metrics**, every dataset; headroom recorded (ADR-0011) |
| **C. Risk control** *(the differentiator)* | coverage ≥ 1−α on both paths; **mean set size bounded**; the gate neither answers nor escalates everything |
| **D. Refusals** | no refusal case produced an answer; **the battery was not passed by refusing everything** (verbatim-option positive control) |
| **E. Slices & gaps** | multilingual holds and transfers; a non-text state carries signal; **must_fix:** two-field composition beats both singles, mnli ≥ .60 |
| **F. Operational** | promote + rollback with a non-empty shadow; **device latency (Phase 8, deferred by construction)** |
| **G. Human-interaction readiness** — *what a marginal number can hide* (`docs/research/benchmarks-and-evaluation-readiness.md`) | **G1** no populated confidence band is catastrophically under-covered (marginal coverage can hide a hard slice — MAPIE conditional-CP); **G2** deferral is well-aimed (acc on deferred ≤ asserted — the gate escalates what it would get wrong, learning-to-defer); **G3** abstention does not concentrate on a single intent *(report)*; **G4** the head is not answering surface cues (candidate-order invariant + withheld state never answered, arXiv:2609.37647) |

Two rules that make it a gate rather than a dashboard:

1. **Not measured is `deferred`, never a pass.** The same fail-open that hid a broken
   candidate behind `n = 0` in v15 must not reappear as "we did not look, so fine".
2. **A check that *raises* blocks.** A broken criterion is not absent evidence — the gate
   cannot vouch for it, so it fails closed.

**Acceptance:** `qualification.json` exists with every criterion's value, threshold, and
status; the verdict is derived from severities, not asserted; and the report names the git
commit and bundle digest it applies to.

---

### Phase 8 — On-device deployment *(you trigger this)*
**Where:** your PC, memory-gated. **Depends:** 7.

- llama.cpp parity (correct metric: **cosine agreement**, not argmax) + **real latency
  on your hardware**.
- **Continuous numeric output** (recursive range refinement) — optional.
- Privacy: inputs never leave the device.
- **Acceptance:** parity within tolerance; measured latency; memory gate passed.

### The model/harness boundary — governs components 28, 30, 31, 33

Researched rather than assumed. The field's operational definition, from a systematic
review of 896 papers and 80+ regulatory documents (arXiv:2603.10023): **"models consist
of trained parameters and architecture, while systems consist of the model plus
additional components including an interface for processing inputs and outputs."**

Every one of these four is a **harness** concern with a **thin model-side contract that
is already built**. The line is the same in each case: the model provides *calibrated
primitives over a given input*; the harness decides *what the input is* and *what to do
with the output*. That is also the paper's additive design principle — the decision layer
"can lower confidence, flag findings for review, or prune agents, but it cannot override
a deterministic validator's rejection… the worst case of System One failure is equivalent
to running without it."

| # | harness owns | model side — already built |
|---|---|---|
| 28 | the requested-precision **refinement loop** (coarse-to-fine bisection) | the ordinal distribution and `OrdinalScorer.expected()`. `Noul` is the cheap calibrated binary query that bisection needs |
| 30 | the integration contract itself — context hook, safety hook, orchestration surface | `decide(state, question) -> Prediction`, and nothing more |
| 31 | retrieval, eviction, paging, **what to include** in a state | `State` (an ordered item list) and `EmbeddingComposer` — **how** a given set is composed |
| 33 | system card, safeguards, routing, human oversight, monitoring, red-teaming the *assembled* thing | the model card and training-data disclosure; privacy (inputs never leave the device, tested) |

**Two caveats recorded now so they are not discovered later:**

1. **Continuous-output precision has a floor.** A single forward pass cannot exceed its
   declared level granularity — *"the larger the number of categories, the smaller the
   discretization error"* (Rothe et al.). Precision beyond that costs refinement queries.
   "Requested precision" is a cost decision, not a free parameter.
2. **Compliance is scope-dependent, and the model is not the regulated object.** The EU AI
   Act regulates **systems**: *"the model provider is on the hook for the model card; the
   system provider is on the hook for the system."* This model is not GPAI under the Act —
   270M frozen encoder plus a fitted head, neither "significant generality" nor the
   10^25-FLOP systemic-risk threshold. Obligations split three ways: model (ours),
   system (the harness), deployer (whoever runs it). **Get counsel before relying on any
   of this** — arXiv:2603.10023's whole premise is that the model/system boundary is
   ambiguous in current regulation, and that ambiguity *is* the compliance risk.

**What the harness inherits, as three named contracts rather than a blank page:**
compose-a-given-state, answer-calibrated-binary-queries, report-provenance (the bundle
digest). Design it against those.

---

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
| 2026-10-08 | 0 | Boundary hardened in `AGENTS.md` (R1 verbatim); local GGUF artifact removed; `llama_cpp` confirmed absent from venv. **Phase 0 done.** |
| 2026-10-08 | 1 | CLI made backend-pluggable (`--backend st\|llamacpp`, `--inputs` batch). s1_verify v5 pushed: llama.cpp leg replaced with a CLI leg (subprocess, 12 real inputs, calibrated bundle). Awaiting run. |
| 2026-10-08 | 1 | **DONE** — s1_verify v5 all green; CLI ran end-to-end in a subprocess on 12 inputs (5 answered ≥0.997, 7 escalated incl. all OOS/ambiguous). Session 5.5 min. |
| 2026-10-08 | 2 | `src/decision/adapt.py` — `adapt(texts, labels, encoder)` → head + calibrated gate + bundle + `adapt_report.json`. 6 tests (158 total green). |
| 2026-10-08 | 2 | **DONE** — s1_adapt v3 all 8 verdicts green on Banking77 (never calibrated before): head 0.995 vs zeroshot 0.975, no undercoverage, τ_in_schema fitted on 20 held-out OOS classes, reload identical 400/400, CLI leg ran. |
| 2026-10-08 | 3 | `src/decision/handlers.py` — CallableEscalation/QueuedEscalation/clarify_payload/handle dispatch; escalate falls back to a JSONL queue on System-Two failure; abstain returns a reason. 8 tests. In-kernel exercise pending next s1_verify rev. |
| 2026-10-08 | 4 | **DONE** — `serve.py` (stdlib HTTP: POST /decide, /healthz, /stats p50/p95/p99) + PredictionLogger JSONL + CLI `serve`. s1_verify v6 all green incl. in-kernel serve leg. Prediction mode recorded: online; batch via `decide --inputs`. 186 tests. |
| 2026-10-08 | 5 | **DONE (code)** — `monitor.py` (ops + ML metrics, per-slice, KS drift naming covariate/label shift, thresholded alerts) + `uncertainty.py` (bootstrap-ensemble BALD decomposition). Unit-tested; consumes serve.py JSONL — log schema aligned. |
| 2026-10-08 | 6 | **PARTIAL** — `registry.py` (versioned bundles, lineage with data_hash+config+parent, promote/rollback pointers) + `shadow_compare` (replay logged inputs through a candidate). Missing: a retrain kernel wiring adapt()→register→shadow on logged data, and the causal readout. |
| 2026-10-08 | 7 | **PARTIAL** — s1_verify v10 all eval legs green: bootstrap CIs, AURC/risk-coverage, ECE15, TF-IDF+LR baseline, similarity-band memorization probe. taskhead beats best baseline on every proper metric (acc .969 vs .906, log .118 vs .687, brier .046 vs .230, ece .006 vs .235); memorization band clean (.911 on <0.8-sim novel inputs → .993 near-duplicates). Still open: benchmark breadth (~15 datasets), multimodal proof, multilingual slice, expected-loss action rule, cost accounting, AUROC/AUPRC. |
| 2026-10-08 | — | Clean-code pass after v8→v10 push churn: eval-leg logic extracted to `src/decision/evaluate.py` (unit-tested + synthetic dry-run); cells now orchestrate. monitor dead param + label-shift naming, serve import scope, adapt explicit `hidden`/`SplitConfig`, handlers escalation `Protocol`, stub backend for weightless local CLI/serve runs. 194 tests green. |
| 2026-10-08 | 7 | **Slow-state (glial) policy A/B — measured, local, real cached data.** `src/decision/policy.py`: SlowState (astrocyte n-of-m over label-free decision statistics + drift check), Static/Glial/Recalibrate(oracle) arms, `run_stream`. On real cached CLINC150 (8 seeds, 300+300 items): under an option-set shift the glial arm cuts unsafe answers **0.304±0.019 → 0.019±0.013**, beating the fair delayed-label baseline (0.107) *and* the immediate-label oracle (0.055) without using a single label; on a healthy stream it is **identical to static** (0.150→0.120 esc, 0.071 wrong). Difficulty sweep (novelty 0.00→1.00): advantage **+0.000 → +0.302, monotone, zero phase-1 cost at every level** — the ANGN review's central regularity (Alvarez-Gonzalez et al. 2023, *Artificial glial cells in artificial neuronal networks: a systematic review*, 22 papers) reproduced on a decision layer. Zero added parameters, which is the capacity confound that review says the ANGN literature cannot rule out. |
| 2026-10-08 | — | Three defects found by *instrumenting* that A/B, not by reading it: (1) `final_thresholds` reported the base bar while the phase ran at a raised one (state deactivated on the last window) — replaced with per-phase threshold stats; (2) drift sat on the margin so the state flickered and snapped back mid-shift — replaced with a Schmitt trigger; (3) the anomaly criterion fires at the *scorer's* base rate (median healthy decision sits exactly on τ_answer), so 51% of healthy decisions looked anomalous and 32/280 healthy windows hit the constant bar — the bar is now fitted from a healthy reference stream. 223 tests green. |
| 2026-10-08 | 7 | v11 failed at cell 2: the hand-rolled bundle omitted `scripts/`, so `tests/test_evaluate.py` could not import and the contract gate aborted before any experiment. Bundling moved to `scripts/build_notebook.py` (explicit manifest, syntax check, fails if the bundle cell is missing). |
| 2026-10-08 | 7 | **v14 DONE — all 21 verdicts green, reproduces v13 exactly** (cross-task unsafe 0.143 static / 0.087 glial / 0.305 delayed-label / 0.137 oracle; breadth 9/9). The clean-architecture lock and the action-rule extraction are therefore behaviour-preserving on the real run, and the artifact matches HEAD. |
| 2026-10-08 | 7 | Cost-calibrated escalation built and measured (`src/decision/escalation.py`): answer iff P(wrong)·C_wrong < C_escalate. **Null with a diagnosis** — at matched handoff it does not beat the frozen gate (0.442 vs 0.419 at 0.40), and the diagnostic proves why: `cost-cal ≡ rank-by-margin` and `frozen ≡ rank-by-max_score`, so the difference is the *feature*, selected on the calibration target rather than the deployment one. Margin is the best label-free risk signal (held-out AUROC 0.862) vs top_prob 0.809 / max_score 0.778 / entropy 0.642 / set_size 0.500. Ranking transfers, calibration does not: at cost 10:1 the model claims P(wrong) ≤ 0.100 and realises 0.383 — the measured case for an ACI-style adaptive threshold. |
| 2026-10-08 | 7 | **v12 DONE — all 20 verdicts green, breadth on 9/9 datasets.** Head beats TF-IDF+LR on **both proper metrics in 9/9** datasets (log/brier), not just CLINC. CLINC150 reproduces v10 exactly (head acc .969, log .118, brier .046, ece .006). First measured emotion numbers: `dair-ai/emotion` 6-way .625, `tweet_eval/emotion` .798, `/sentiment` .666, `/hate` .586 — all beating TF-IDF on proper metrics, ECE .023–.166. `dbpedia_14` 14-way .970 (log .096, ece .012) is the best-calibrated set; `banking77` 77-way .930. Ordinal leg still beats zero-shot (.432 vs .318). |
| 2026-10-08 | 7 | Honest weak spot in v12: **NLI (mnli) .449** — concatenating premise+hypothesis into one string is not enough for a frozen encoder with a linear head. The two-field composer path needs its own design before NLI/verification counts as supported. |
| 2026-10-08 | 7 | **v12 pushed.** Breadth rebuilt on parquet-only ids (9 configs incl. 3 emotion, 5 tweet_eval, mnli, dbpedia) after a real-`load_dataset` local dry-run caught class-ordered caps hiding classes (banking77: 17 of 77 in the head, 62 of 77 strided) — fixed with a per-class quota plus an explicit coverage guard. |
| 2026-10-08 | 6 | **v15: operate leg live — 27/28.** Monitor on 300 logged decisions fired two real alerts (escalate_rate 0.63>0.6, top_prob KS=0.247>0.15); handlers queued 189 escalations; registry register→promote→rollback round-tripped. Only failure: `shadow n=0` — silent except in shadow_compare hid that every candidate decide raised. |
| 2026-10-08 | 6 | **v16 diagnosed, fix shipped.** Root cause was three layers: (1) the leg dropped `n_failed`/`first_error` from its report; (2) every decide raised `scores and labels must be the same length` — the bundle's head had 75 classes vs 150 options; (3) the retrain slices were halved *positionally* on a class-ordered index, partitioning by class — the breadth leg's "head slice" bug one level down. Fixes: `datasets.class_halves` (split within each class's run, singletons duplicated), `DecisionModel._align_scores` (reindex a schema-bound scorer's np.unique-ordered logits to `question.options`, fail loud on out-of-space options — without it a 150-class head would have *silently* produced wrong predictions), leg now surfaces n_failed/first_error. |
| 2026-10-08 | 6 | **v17 DONE — all 28 verdicts green, first fully-green run.** `shadow: n=300 agreement=0.370 failed=0` on live-logged traffic; registry lineage + rollback verified in-run. Definition-of-done 4 and 5 now have artifacts. Open reading: incumbent escalates 63% while the retrain answers almost all (`escalate→answer: 187`) — the gates' thresholds diverge enough that incumbent/candidate agreement is a metric to watch, not a pass/fail. |
| 2026-10-08 | 6 | **v18 DONE — all 34 verdicts green** (28 + 6 new). Cadence fired on the real drift alert (`top_prob KS=0.247 > 0.15`) and held below the sample floor; canary and causal both produced readouts. **The finding:** the retrain is far more aggressive than the incumbent — escalate rate 63.6% → 1.1% (Δ −0.623 [−0.673, −0.567], paired CI) bought with unsafe answers 0.67% → 7.67% (Δ +0.070 [+0.043, +0.100]). That is the safety/cost trade-off quantified, and it explains v17's shadow agreement of 0.370. |
| 2026-10-08 | 6 | **v18 measurement bug, mine.** The canary reported `latency_p95 x130` — a `DecisionCache` artifact, not a regression: the incumbent had served the same 300 items seconds earlier so every re-decide was a cache hit (p95 0.45 ms), while the freshly loaded candidate paid the real 58.6 ms (log pass median: 55.1 ms). The verdict (rollback) was still correct, but for one real reason (unsafe) and one fake one (latency). Fix for v19: both arms cache-free before timing. |
| 2026-10-08 | 6 | **Verified weakness: the conformal sets are vacuous on the path we ship.** Measured locally from the v18 score matrix (head path, alpha=0.10, target coverage 0.90): `t_set=1.0` (shipped, never fitted) gives `qhat=0.9998`, **mean set 26.9 of 150 labels**, singleton 12.3%, **coverage 1.0000** — over by 10 points. The cosine path is tight by comparison (0.914). Mechanism: `qhat` is the 0.90 quantile of `mass_needed`, and a very confident head (P(top)≈0.99) saturates that quantile at ~1.0, so APS must accumulate the last 0.01 of mass — which costs ~27 labels. Consequence: `k_clarify=3` almost never fires and a 27-label set is not actionable. **This is the differentiator**: stock EG2 offers no risk guarantee at all, and ours is currently trivially satisfied. **Correction to an earlier version of this entry:** it claimed sweeping `t_set` does not fix it, which was wrong — that test only tried *sharpening* (t ≤ 1), which compresses the score further. Searching *upward* does fix it: measured t=4 → 7.2 labels, t=8 → 2.4, t=32 → 1.9, and the curve plateaus from 32 through 4096 (so 32 is a plateau, not a grid boundary). Fixed in v19 by fitting the temperature; see the entry below. Still-open options if efficiency needs to go further: a regularized or margin-based conformal score (margin is the best-measured label-free signal, AUROC 0.862), and per-slice `qhat` instead of one global quantile. |
| 2026-10-08 | — | Reproducibility gap found while checking the above: `verify_scores.npz` persists only the **cosine** scores (`S @ LV.T`), not the head's logits, so the head path cannot be re-evaluated offline — contrary to the convention in `AGENTS.md`. The head had to be recovered from `bundle_h/scorer.pt` and scored against the cached embeddings. Also note `y_*` in that npz hold **raw dataset ids** (with `oos` at index 80, mid-range), not the compact 0..149 indices the head's columns use — an easy mis-read that produced a false 53% accuracy before the mapping was recovered. |
| 2026-10-08 | 6 | **v19 DONE — all 36 verdicts green, and the conformal fix is confirmed on real data.** Head-path prediction sets went **31.1 → 1.94 labels** with coverage **1.0000 → 0.9882** (target 0.90), against a local held-out prediction of 1.93/0.9867. The cosine path is untouched — 3.80 labels, coverage 0.9140 — because the fitted temperature correctly selects 1.0 for spread scores. Two new verdicts (`cosine_sets_small`, `head_sets_small`) make set size a checked outcome, so this degeneracy cannot hide behind a perfect coverage number again. Canary latency artifact fixed; head scores now persisted. |
| 2026-10-08 | 7 | **v20 DONE — 42 verdicts, 41 true. Multilingual: the pipeline holds off-English and transfer is strong.** Six non-English languages, same 60-intent taxonomy: in-language head accuracy **de .797 / es .808 / fr .819 / ru .830 / zh-CN .816 / ja .830** against English .876 — a measured **language gap of 5–8 points**, with the non-Latin scripts (zh-CN, ja) at the low end. An **English-trained head transfers at .772–.812**, within 1–3 points of in-language training: the encoder's representations are largely language-aligned. Zero-shot cosine is competitive (.777–.885), consistent with ADR-0011 — 60 intents is a shallower task than CLINC's 150. This is the first multilingual evidence in the project. |
| 2026-10-08 | 7 | **v20 multimodal: the first decision-level vision evidence, and my verdict asked the wrong question.** ScienceQA, 1500 fit / 600 test, 4-choice, vision tower loaded (peak 4.23 GiB). **image_only .4333 [.3983,.4735] vs 4-way chance .25 — the image alone carries real, significant decision information.** But `full` (image+question, mean-pooled) is **.4283**, *below* image_only, and text_only is the worst arm at .3950. So: (a) vision works at the decision level — new positive evidence; (b) mean-pooling image+text does **not** beat the best single modality, which is exactly the deficiency ADR-0003's composer exists to fix and is now empirically motivated rather than hypothetical. **My `multimodal_image_helps` verdict tested `full > text_only`, which is the wrong operationalization** — it can be false while the image carries signal. Corrected in v21 to two pre-registered claims: does the image beat chance, and does the combination beat **both** singles. |
| 2026-10-08 | 7 | v20 conformal sets on the multimodal task are degenerate (mean set 4.00 = the whole 4-option space, coverage 1.000) because the scorer is near-uniform there (log ≈ 1.35 vs uniform 1.386). The v19 `mean_set_size` diagnostic surfaces this correctly, and the `t_set` fit declined to pick a temperature — the documented fallback, since no temperature helps a score with no signal. This is the diagnostic doing its job on a task where the model has nothing to say. |
| 2026-10-08 | — | **v21 DONE — 44 verdicts, 43 true.** The one false is the *correct* answer: `multimodal_combination_beats_both_singles` fails while `multimodal_image_carries_signal` passes, which is exactly the pair the corrected operationalization produces. **`cli_adapt` ran: 1500 rows / 150 classes, headroom 0.709 (below 0.85, so a head is justified), test accuracy .904, coverage .9467, mean set 2.02 of 150, round-trip action `answer`** — definition-of-done #1 verified on Kaggle through the CLI with the real encoder, and the v19 conformal fix holding on a freshly adapted bundle. |
| 2026-10-08 | — | **Researched the model/harness boundary for components 28, 30, 31, 33 rather than assuming it.** All four are **harness** concerns with a thin model-side contract that is **already built** — which is why deferring them is architecturally correct, not just sequencing. The line is the same in each case: the model provides *calibrated primitives over a given input*; the harness decides *what the input is* and *what to do with the output*. Operational definition from arXiv:2603.10023 (896 papers, 80+ regulatory documents): models are "trained parameters and architecture", systems are "the model plus additional components including an interface for processing inputs and outputs". Model-side halves already in place: `OrdinalScorer.expected()` for 28 (the refinement loop is harness, and `Noul` is the calibrated binary query it needs), `EmbeddingComposer` for 31 (retrieval/eviction is harness), `modelcard.py` + the privacy test for 33 (system card, oversight and monitoring are harness). Two caveats recorded: continuous-output precision has a hard floor at the declared level granularity, and compliance is scope-dependent — the EU AI Act regulates *systems*, and this model is not GPAI under it. |
| 2026-10-08 | 7.5 | **v23 DONE — the first qualification ran, and it says `qualified`.** 45 verdicts, 43 true. `qualification.json`: **verdict `qualified`**, 0 must_pass failures, 17 criteria passed, **check errors none**, provenance `git_commit 56fcd30` + **`bundle_sha256 207cef05…`** — so the artifact is named, and A3 is satisfied. Open: `must_fix` E4/E5 (the composer and two-field gaps, as expected) and `deferred` F2 (device latency — Phase 8, by construction). ML Test Score coverage 14/5/1 (82.5% weighted). Refusals ran with the cost-derived boundary. |
| 2026-10-08 | 7.5 | **v23 bug, mine: the model card was never written.** `build_card` raised `AttributeError("'float' object has no attribute 'get'")` — `adapt()` writes `headroom: {"zeroshot_test_acc": x}` while the kernel's `cli_adapt` summary stores the value directly, and the card read only the first shape. The failure mode was nasty in a specific way: **the qualification still succeeded and wrote its verdict, so the run looked complete while the delivery artifact was silently absent.** Fixed with `_headroom_value`, which reads either shape; both are real. A test covers both plus the absent case. Re-run needed (v24) to actually produce the card. |
| 2026-10-09 | 7.5 | **Researched benchmark/evaluation readiness (library-first) → `docs/research/benchmarks-and-evaluation-readiness.md`.** The category's reference evaluation is in the local library (arXiv:2609.37647, 37 datasets / 346k requests, mapping onto the same Choice/Score/Noul primitives); our §4 protocol already matches most of it. The two gaps that decide human-interaction readiness — **conditional (group) coverage** and **deferral quality** — were measured by neither us nor the reference. **Wired all four checks into the qualify leg as gate G** (`src/decision/readiness.py`, pure array logic on the per-example records `policy_eval` already collected): G1 conditional coverage by confidence band (must_pass, tolerance −0.15 — a hard band legitimately dips, a collapse is the defect); G2 deferral well-aimed, deferred acc ≤ asserted acc (must_pass); G3 per-intent rejection concentration (report — legitimate difficulty isn't a defect, so it surfaces rather than gates); G4 memorization probes — candidate-order invariance on cached scores + withheld-state collapse (must_pass). 422 tests green; kernel cells all compile. Needs a v24 run to produce real readiness numbers. |
