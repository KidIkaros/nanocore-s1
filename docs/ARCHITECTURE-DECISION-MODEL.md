# NanoCore-S1 Decision Model — Architecture Map

Date: 2026-10-07. Status: **revised architecture.** This document describes the architecture as
decided, with an honest validation column. It is an inventory, not a proposal.

The revision is recorded in `adr/0007`–`adr/0012`; three earlier ADRs were amended
(`0001` partially superseded, `0003` deprecated, `0005` superseded). Measured evidence for every
claim is in `reports/runs/`.

Related: `STATE-OF-THE-PROJECT.md` (consolidated findings), `ENCODER-EFFICIENCY.md` (runtime
investigation), `RESEARCH-NOTES.md` (R1–R13), `ARCHITECTURE.md` (legacy decoder, untouched),
`ARCHITECTURE-REASSESSMENT.md` (why the decoder was set aside).

---

## 0. Execution boundary (the rule that was broken)

**Everything that imports torch, loads model weights, or executes over real data
arrays runs on Kaggle. Not here.**

This host has ~1.6 GB RAM against a 1.7 GB model and was OOM-killed during this
project by exactly this mistake. A local "dry run" of a notebook — torch import,
a 19 MB embedding array, head training — killed the working session on 2026-10-07.
It was not a judgment call; it was a violation of a stated constraint.

| Work | Where |
|---|---|
| Reading, writing, editing files | local |
| Static checks: `ast.parse`, JSON validity, glob/path assertions | local |
| `git` operations | local |
| Anything importing torch / numpy over real arrays | **Kaggle** |
| Encoder loading, embedding, head or composer training | **Kaggle** |
| Any number that appears in a results table | **Kaggle** |

A notebook must be validated by *running it on Kaggle*, not by executing it here.
Cheap fast-fail is achieved by pushing a reduced kernel, not by a local dry run.

---

## 1. What the system is for (ML suitability pass)

**Problem statement.** Given a *state* (one or more items: text, image, audio) and a
*question* over a supplied set of options, return a **typed decision**: a probability
distribution over the options, a prediction set, and an explicit action (auto / escalate /
abstain).

**Why ML and not rules.** The decision is a semantic judgment over unstructured, multimodal
input with no deterministic predicate. Rules cannot express "is this evidence sufficient to
answer this question". The learned pattern is real and the data exists.

**Business objective → ML objective.**

| Business need | ML proxy | Measured by |
|---|---|---|
| Answer cheaply and locally | frozen encoder + one fitted temperature | **89.2 ms CPU per decision** (was 416 ms) |
| Do not answer when unsure | conformal prediction sets | coverage at nominal `alpha` |
| Do not confidently err | honest probabilities | log score, Brier (gated); ECE reported |
| Improve with use | flywheel crystallization | out of scope here |

**Requirements priority** (this ordering drives every trade-off):

1. **Reliability** — a wrong confident answer is worse than an abstention, **except where
   abstention suppresses a positive** (see ADR-0009: the gate is asymmetric by domain).
2. **Latency / cost** — measured, not asserted: 89.2 ms CPU, 56.9 ms T4, 4.08× over the
   PyTorch path (ADR-0007).
3. **Maintainability** — one fitted temperature for `Choice`; the encoder is frozen.
4. **Adaptability** — flywheel, precedent, escalation to System Two.

**Tolerable failure modes.** Abstention on a decidable item, where escalation is the fallback.
**Intolerable:** a confident wrong answer on a high-cardinality option set — the documented weak
spot of the category (Laya: Banking77 0.425).

---

## 2. Component map

```
state items [str | {"image": p} | {"audio": p} | {"video": p}]
      │
      ▼  Encoder — FROZEN, served via llama.cpp GGUF Q8_0        ADR-0007
      │            mmap · SIMD · explicit threads · hard max_tokens   ADR-0012
      │            bf16 on accelerators / fp32 on CPU; fp16 REJECTED
 item embeddings  (n_items, 768)          option embeddings (k, 768), cached
      │                                          │
      │  (no composer: parked, ADR-0003)         │
      ▼  state vector s = masked mean            │
      │                                          │
      ▼  Scorer ── temperature-scaled cosine (1 parameter)        ADR-0008
      │            OrdinalScorer (CORN) for Score only            ADR-0010
      │
      ▼  ConformalGate — APS/RAPS sets, asymmetric policy         ADR-0009
      │
      ▼  Prediction{probabilities, prediction_set, action}   Choice / Score / Noul
```

| Component | File | Params | Trained? | Status |
|---|---|---|---|---|
| `LlamaCppEncoder` | `src/decision/backends.py` (Stage 1) | 271M (frozen) | no | **decided**, not yet implemented |
| `SentenceTransformerEncoder` | `src/decision/encoder.py` | 271M / 439M | no | reference backend, works today |
| `CosineScorer` | `src/decision/scoring.py` (Stage 2) | **1** | one temperature | **decided**, not yet implemented |
| `OrdinalScorer` (CORN) | `src/decision/scoring.py` (Stage 4) | ordinal head | **yes** | decided, not yet implemented |
| `ConformalGate` | `src/decision/gate.py` (Stage 3) | — | calibrated | decided, not yet implemented |
| `DecisionHead` | `src/decision/head.py` | 59,136 / ~148k | yes | **deprecated for Choice** (ADR-0008); retained as the ablation |
| `EmbeddingComposer` | `src/decision/composer.py` | 28,315,392 | never | **deprecated, parked** (ADR-0003) |
| `protocol` | `src/decision/protocol.py` | — | — | the shared evaluation instrument; 87 tests green |
| Legacy decoder | `src/model.py` | 135,266,328 | — | untouched Route-4 path |

---

## 3. Interfaces

**Current (implemented):**

```python
NanoCoreS1(encoder, composer=None, head=None, normalize_state=True)
  .state_vector(state) -> (768,)
  .decide(state, question) -> Prediction
  .decide_batch(state, questions) -> [Prediction]

Prediction(qtype, labels, probabilities, answer_confidence, entropy_confidence,
           abstention, abstention_threshold, max_score)
  .choice .noul .score
```

**Target (decided, Stages 1–3):**

```python
Encoder(Protocol):   name, dim, max_tokens
                     encode_texts(texts, prompt=None) -> (n, d)
                     encode_items(items) -> (n, d)

Scorer(Protocol):    name; scores(state_vec, option_vecs) -> (k,)
                     fit(examples) -> dict
                     ├── CosineScorer      (1 fitted temperature)
                     ├── KNNScorer         (retained; wins on classification, collapses on routing)
                     ├── InteractionScorer (wraps the retired head; ablation only)
                     └── OrdinalScorer     (CORN; Score only)

ConformalGate(alpha=0.05, method="aps", policy="escalate")
  .calibrate(scores_list, targets_list) -> dict
  .decide(scores, labels) -> GateResult(set, probabilities, action)

DecisionModel(encoder, scorer, gate=None, cache=None)
  .decide(state, question, policy=None) -> Prediction
```

**Breaking changes (accepted):** `Prediction` gains `prediction_set`, `action`, and `alpha`;
`abstention` becomes a deprecated read-only property. `State`, `Question`, `DecisionExample`,
and the `{state, questions, gold}` training schema are **unchanged**, so jev-stack interop
survives.

---

## 4. Embedding provenance, runtime, and the resolved prompt question

The vectors are EmbeddingGemma 2's output. The encoder is **modular**: 270M text/code,
+170M vision, +300M audio, 740M full — unused encoders are excluded at load time, so our text
encoder is *smaller* than Laya's 421M ModernBERT.

**Prompt pairing — resolved.** States were encoded with `Classification` and options with
`Document`, an unjustified asymmetry. Tested (kernel `s1_prompt_ablation`):

| state / option prompt | zero-shot | kNN-5 | head | head log | head Brier |
|---|---:|---:|---:|---:|---:|
| `Classification` / `Document` (incumbent) | 92.89% | 93.64% | 93.11% | 0.377 | 0.116 |
| **`SearchQuery` / `Document`** | **93.38%** | **94.58%** | **93.60%** | **0.299** | **0.103** |
| `Classification` / `Classification` | 92.05% | 93.64% | 93.11% | 0.377 | 0.116 |
| *(none)* / *(none)* | 92.89% | 94.25% | 93.51% | 0.316 | 0.104 |

The fix is the **retrieval** pairing, not symmetry — symmetric is the *worst* option. Adopt
`SearchQuery`/`Document`; earlier numbers are pessimistic by ~0.5 points.

**Also found:** the fingerprint head **ignores option text entirely** — in that mode the score
is `states @ fingerprints.T` and `option_embeddings` is never read, which is why two option
prompts produced byte-identical results.

**Runtime.** The encoder is served through llama.cpp GGUFs (ADR-0007). Half the model's 271M
parameters are a 262,144-entry vocabulary table which, under mmap, a short input never fully
reads — a structural saving a dense PyTorch forward cannot match.

---

## 5. Training regimes

| Stage | Trains | Compute | Status |
|---|---|---|---|
| Embedding precompute | nothing (frozen encoder) | Kaggle GPU | done |
| `CosineScorer` temperature | **one scalar** per question type | CPU | decided (Stage 2) |
| `ConformalGate` calibration | a quantile on held-out data | CPU | decided (Stage 3) |
| `OrdinalScorer` (CORN) | ordinal head | CPU on cached embeddings | decided (Stage 4) |
| `DecisionHead` | retired for Choice | — | **ablation only** |
| `EmbeddingComposer` | parked | — | not planned |
| Encoder | never | — | frozen by ADR-0001 |

`Choice` and `Noul` require **no training at all** beyond a scalar temperature.

---

## 6. Validation status matrix

| Capability | Implemented | Tested | Real-data measured | Verdict |
|---|---|---|---|---|
| Evaluation protocol | yes | **87 tests on Kaggle** | n/a (it is the instrument) | **validated**; regression tests for both published errors |
| EG2 text encoding | yes | via kernels | yes | **validated** (92.92% → 93.38% with correct prompts) |
| EG2 via llama.cpp Q8_0 | recipe proven | kernel | yes | **validated** — 4.08× faster, quality +0.49 |
| Order invariance | yes | yes | **yes** | **validated** — 2.38e-07 / 1.19e-07, 0 flips, 2–37 options |
| Cosine scoring | ad-hoc in kernels | — | yes | **validated** — matches the head on every proper metric |
| `DecisionHead` for Choice | yes | yes | yes | **retired** — no win on any metric; worse on routing |
| Composer | yes | contracts yes | no | **parked** (ADR-0003) |
| `Score` (ordinal) | schema only | schema only | no | **unexercised**; uses the wrong loss today |
| Conformal gate | no | no | no | decided, not built |
| Multimodality | vision smoke only | no | 100 images | **smoke only**; audio/video untested |
| Legacy decoder | yes | 52 repo tests | no | Route-4 path, untouched |

---

## 7. Measured results

**Banking77, 3,080-item test split** (wide temperature grid, ADR-0007's corrected fitter):

| Method | Params | Accuracy | Log score | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Majority | — | 1.30% | — | — | — |
| Zero-shot cosine | 0 | 92.92% | — | — | — |
| **cosine + fitted τ** | **1** | 92.92% | 0.380 | 0.117 | **0.045** |
| kNN-1 / kNN-5 + τ | 0 | 93.60% / **93.64%** | 0.570 / 0.508 | 0.126 / 0.117 | 0.057 / 0.052 |
| Fingerprint head (3 seeds) | 59,136 | 93.11% ±0.0002 | **0.377** | **0.116** | 0.046 |
| Interaction head (2 seeds) | ~148k | 92.48% ±0.0002 | 0.383 | 0.123 | 0.045 |

**BFCL dispatch routing** — 1,252 queries, 2–37 candidate tools:

| Method | Accuracy | Log score | Brier |
|---|---:|---:|---:|
| random | 32.00% | 1.202 | 0.676 |
| **zero-shot cosine** | **93.07%** | **0.176** | **0.099** |
| interaction head (3 seeds) | 90.84% | 0.253 | 0.137 |
| kNN-20 / kNN-5 | 73.07% / 71.47% | 0.630 / 0.689 | 0.343 / 0.368 |

**Latency** — same task, one variable (the runtime):

| | PyTorch fp32, CPU | llama.cpp Q8_0, CPU | T4 (PyTorch) |
|---|---:|---:|---:|
| short state | 363.5 ms | **89.2 ms** | 56.9 ms |
| long state | 6,595 ms | **1,754.8 ms** | 64.6 ms |
| throughput | 2.4 texts/s | **9.9 texts/s** | — |

Against the bars: **beats Jev's cloud p50 (236–276 ms) by ~2.9×**, loses to Laya's T4 figure
(33–40 ms). Thread scaling: 1→2 threads 1.84×, 2→4 only 1.16× more.

**Competitive position** (from the library's own benchmark paper): our 92.92% Banking77
zero-shot against Jev's **independently measured 79.7%** (CI 0.782–0.812) and Laya's 42.5%; at
50% coverage, 99.61% against Jev's 96.3% and Qwen's 94.5%. **The win belongs to the encoder,
not to this project** (ADR-0008).

---

## 8. Open questions and risks

1. **Both benchmarks are saturated** at ~93% zero-shot (ADR-0011). No architectural comparison
   is meaningful until a headroom-positive task is found — a scoped follow-up.
2. **`Score` may fail too.** It is the only remaining trained component; Stage 4's acceptance is
   explicit (CORN must beat cross-entropy on RPS, monotonicity must hold).
3. **The conformal gate assumes exchangeability.** A domain shift breaks the guarantee.
4. **Long inputs cost 19.7× short ones** (ADR-0012); the cap's accuracy cost is unmeasured.
5. **The residual latency gap is clock speed and instruction set**, not architecture — a
   2.20 GHz AVX2-only host. Untested on AVX-512 or GPU llama.cpp.
6. **Soft targets / RLCD remain untested** — the one route back to a useful `Choice` head,
   blocked on verifiable labels.
7. **Multimodality is a smoke test only.** Vision: 100 images. Audio/video: never run, despite
   being the capability neither Jev nor Laya has.
8. **The legacy decoder is a liability** — 135M parameters of unused code that still carries
   its own tests, retained deliberately as the Route-4 path.

---

## 9. Decisions

All decisions now live in `adr/`. Summary:

| ADR | Decision | Status |
|---|---|---|
| 0001 | Frozen encoder instead of from-scratch decoder | **encoder half stands**; head half superseded by 0008 |
| 0002 | Typed non-autoregressive output; never parse generated text | accepted |
| 0003 | Bidirectional composer for multi-item states | **deprecated, parked** |
| 0004 | Gate on proper scoring rules; report ECE only | accepted |
| 0005 | Abstention via a fitted max-score threshold | **superseded by 0009** |
| 0006 | Kaggle-only execution for weights and real arrays | accepted |
| 0007 | Serve the encoder through llama.cpp GGUFs | accepted |
| 0008 | Cosine scoring; retire the head for Choice | accepted |
| 0009 | Conformal abstention, asymmetric by domain | accepted |
| 0010 | Ordinal `Score` is the only trained component | accepted |
| 0011 | Benchmark admission requires a headroom check | accepted |
| 0012 | The encoder enforces a hard input-length cap | accepted |

---

## 10. Staged implementation

Specified in full in the approved plan; each stage gated on its own acceptance test.

| Stage | Change | Acceptance |
|---|---|---|
| 1 | llama.cpp encoder backend | cosine ≥0.999 vs the ST backend; ≤100 ms CPU batch-1 |
| 2 | Cosine scorer; deprecate the head | matches or beats the head on every proper metric |
| 3 | Conformal gate | nominal coverage on BFCL **and** Banking77 |
| 4 | Ordinal `Score` (CORN/CORAL) | CORN beats cross-entropy on **RPS**; monotonicity holds |
| 5 | Input-length contract | published latency/quality curve at 128/256/512 tokens |

**Rule enforced throughout:** every number in an ADR must cite a `reports/runs/` artifact, or be
re-measured or removed.
