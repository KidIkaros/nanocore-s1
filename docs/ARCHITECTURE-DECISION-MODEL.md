# NanoCore-S1 Decision Model — Architecture Map

Date: 2026-10-07. Status: **map of what exists, what is validated, and what is assumed.**
This is the document that should have existed before implementation. It is not a
plan and not a proposal — it is an inventory with an honest validation column.

Related: `ARCHITECTURE.md` (legacy decoder, untouched), `ARCHITECTURE-REASSESSMENT.md`
(why the decoder was set aside), `DECISION-MODEL.md` (build plan + first results).

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
distribution over the options, plus a confidence and an explicit abstention.

**Why ML and not rules.** The decision is a semantic judgment over unstructured,
multimodal input with no deterministic predicate. Rules cannot express "is this
evidence sufficient to answer this question". The learned pattern is real and the
data exists.

**Business objective → ML objective.**

| Business need | ML proxy | Measured by |
|---|---|---|
| Answer cheaply and locally | small frozen encoder + tiny trained head | parameters, latency (not yet measured) |
| Do not answer when unsure | calibrated P(option) + abstention gate | selective accuracy vs coverage |
| Do not confidently err | honest probabilities | log score, Brier (gated); ECE reported |
| Improve with use | flywheel crystallization | out of scope here |

**Requirements priority** (this ordering drives every trade-off):

1. **Reliability** — a wrong confident answer is worse than an abstention. Asymmetric
   loss; this is what justifies proper scoring rules and the abstention gate.
2. **Latency / cost** — the design's whole pitch is "extremely lightweight". Never yet
   measured.
3. **Maintainability** — the head is `options x dim`; the encoder is frozen.
4. **Adaptability** — flywheel, precedent, escalation to System Two.

**Tolerable failure modes.** Abstention on a decidable item (costs latency).
**Intolerable:** a confident wrong answer on a high-cardinality option set — the
documented weak spot of the category (Laya: Banking77 0.425).

---

## 2. Component map

```
state items [str | {"image": p} | {"audio": p} | {"video": p}]
      │
      ▼  StateEncoder ── google/embeddinggemma-2, FROZEN, 439M (text+vision)
      │                  bf16 on GPU / fp32 on CPU; fp16 REJECTED
      │                  MRL truncation 768→512/256/128, re-normalized
 item embeddings  (n_items, 768)          option embeddings (k, 768)
      │                                          │
      ▼  EmbeddingComposer, TRAINABLE, 28,315,392 params
      │  bidirectional attention, RoPE, QK-norm, ReLU², zero-init out-proj
      │  (composer = None ⇒ masked-mean pooling = the baseline path)
 state vector  s  (768,)  ────────────────────┐
      │                                       │
      ▼  DecisionHead, TRAINABLE              │  per-option isolated scoring
 score_j = φ(s, o_j)  → softmax(·/τ)  ◄───────┘  ⇒ order-invariant by construction
      │
      ▼  Prediction{probabilities, confidence, abstention}   Choice / Score / Noul
```

| Component | File | Params | Frozen? | Status |
|---|---|---|---|---|
| `StateEncoder` | `src/decision/encoder.py` | 439M (text+vision) | frozen | text path validated; vision smoke only |
| `EmbeddingComposer` | `src/decision/composer.py` | 28,315,392 | trainable | **never trained** |
| `DecisionHead` (fingerprint) | `src/decision/head.py` | 59,136 (`k x dim`) | trainable | measured; competitive, not superior |
| `DecisionHead` (interaction) | `src/decision/head.py` | ~147,584 (`3d x h`) | trainable | measured; competitive, not superior |
| `metrics` | `src/decision/metrics.py` | — | — | vendored from jev-stack |
| `protocol` | `src/decision/protocol.py` | — | — | **the shared evaluation protocol** — one temperature fitter, one metric block, one best-baseline verdict, headroom check, dispatch metrics |
| Legacy decoder | `src/model.py` | 135,266,328 | — | untouched Route-4 path |

---

## 3. Interfaces and data contracts

```python
State(items=[str | dict])                      # ordered list of state items
Question(qtype="choice"|"score"|"noul", options=[...], scale=None)
Prediction(probabilities, answer_confidence, entropy_confidence,
           abstention="unevaluated"|"passed"|"abstained",
           abstention_threshold, max_score)

DecisionHead(mode="fingerprint"|"interaction", dim, hidden, qtypes)
  .raw_scores(state_embedding, option_embeddings, qtype=, labels=)  # no bias
  .predict(...)  -> Prediction
  .fit(examples, ...)             # minimizes the log score (strictly proper)
  .fit_temperature(examples)      # post-hoc, leaves argmax untouched
  .fit_abstention_threshold(examples, qtype, target_precision)

NanoCoreS1(encoder=, composer=, head=)
  .state_vector(state) -> (768,)
  .decide(state, question) -> Prediction
  .decide_batch(state, questions) -> [Prediction]      # the Jev calling pattern
```

Wire/training schema `{state, questions, gold}` is shared with jev-stack, so one
harvested file feeds either head trainer. `gold` targets are **distributions**,
not labels.

---

## 4. Embedding provenance, and an unablated assumption

The cached Banking77 vectors are EmbeddingGemma 2 output. Evidence from the v13
kernel log: `parameters: 439M | dtype: torch.bfloat16 | dim: 768`, producing
`(10003, 768)`, `(3080, 768)`, `(77, 768)`. They are unit-normalized (mean norm
1.00085), so cosine similarity is a plain dot product.

**The problem, and its resolution.** States were encoded with
`prompt_name="Classification"` and options with `prompt_name="Document"` — an
unjustified asymmetry, since a dot product between two prompt-conditioned subspaces is
the scoring function of both the head and the cosine baseline.

**Tested and resolved** (kernel `s1_prompt_ablation`, first kernel to re-encode from the
encoder in-kernel):

| state / option prompt | zero-shot | kNN-5 | head | head log | head Brier |
|---|---:|---:|---:|---:|---:|
| `Classification` / `Document` (incumbent) | 92.89% | 93.64% | 93.11% | 0.377 | 0.116 |
| **`SearchQuery` / `Document`** | **93.38%** | **94.58%** | **93.60%** | **0.299** | **0.103** |
| `Classification` / `Classification` | 92.05% | 93.64% | 93.11% | 0.377 | 0.116 |
| *(none)* / *(none)* | 92.89% | 94.25% | 93.51% | 0.316 | 0.104 |

The incumbent pairing was suboptimal, and the fix is the **retrieval** pairing — not
symmetry. Symmetric `Classification`/`Classification` is the *worst* option (92.05%
zero-shot), which rules out the intuitive fix. Adopt `SearchQuery`/`Document`; earlier
numbers under the incumbent pairing are pessimistic by ~0.5 points.

**Also found: the fingerprint head ignores option text entirely.** In `fingerprint` mode
the score is `states @ fingerprints.T` and `option_embeddings` is never read — which is
why two option prompts produced byte-identical head results. ADR-0002's claim that a new
option "needs no retraining" holds **only in `interaction` mode**.

---

## 5. Training regimes

| Stage | Trains | Compute | Cost |
|---|---|---|---|
| Embedding precompute | nothing (frozen encoder) | Kaggle GPU | ~4 min for 13k texts |
| Head fit | head only, on cached vectors | Kaggle CPU | ~2 min (fingerprint), ~10 min (interaction) |
| Composer fit | composer + head, backprop into embeddings | Kaggle GPU | not yet run |
| Encoder | never | — | — |

Heads are trained on **one-hot labels** in every run so far. The soft-teacher /
RLCD objective the design cites is supported by the API but has never been used —
so current heads are linear probes, not the RLCD model.

---

## 6. Validation status matrix

| Capability | Implemented | Unit-tested | Real-data measured | Verdict |
|---|---|---|---|---|
| Evaluation protocol | yes | **80 tests on Kaggle** | n/a (it is the measuring instrument) | **validated**; two regression tests encode the bugs that produced wrong conclusions |
| EG2 text encoding | yes | no (needs weights) | yes | **validated** (92.92% zero-shot) |
| EG2 vision encoding | yes | no | 100 images | **smoke only** |
| EG2 audio / video | partial | no | no | **untested** |
| Composer composition | yes | contracts yes | no | **unvalidated** |
| Fingerprint head | yes | yes | yes | competitive, **not superior** |
| Interaction head | yes | yes | yes | competitive, **not superior** |
| Score / Noul question types | yes | schema only | no | **unexercised** |
| Order invariance | yes | synthetic yes | **yes** | **validated** (2.4e-07, 0 flips) |
| Abstention gate | yes | yes | yes | measured; best at moderate coverage, not decisive |
| End-to-end pipeline | yes | — | prompt-ablation kernel | in progress |

Reading this table: **one of three pillars is validated (EG2, text only)**; order
invariance and abstention have since been measured and are reported in §7. The typed
head is competitive but has not earned its place on accuracy. The composer is untested.
That is the honest state.

---

## 7. Measured results

**Banking77, 3,080-item test split.**

| Method | Params | Accuracy | Log score | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Majority | — | 1.30% | — | — | — |
| Zero-shot cosine | 0 | 92.92% | — | — | — |
| cosine + fitted τ | **1** | 92.92% | 0.380 | 0.117 | **0.045** |
| kNN-1 + fitted τ | 0 | 93.60% | 0.570 | 0.126 | 0.057 |
| kNN-5 + fitted τ | 0 | **93.64%** | 0.508 | 0.117 | 0.052 |
| kNN-20 + fitted τ | 0 | 93.54% | 0.480 | 0.117 | 0.048 |
| Fingerprint head (3 seeds) | 59,136 | 93.11% ±0.0002 | **0.377** | **0.116** | 0.046 |
| Interaction head (2 seeds) | ~148k | 92.48% ±0.0002 | 0.383 ±0.012 | 0.123 | 0.045 |

**Correction (2026-10-07).** The baseline rows were previously reported from a temperature
grid whose *lower bound* three of them fitted, which made cosine look catastrophically
miscalibrated (log 3.800, ECE 0.907). With a properly fitted temperature (τ = 0.0126),
cosine reaches log 0.380 / Brier 0.117 / ECE 0.045. **That earlier claim was my grid
artifact, not a property of cosine scoring.**

**Abstention and order invariance** (calibration kernel v3, complete):

| Method | Selective acc @10% cov | @50% cov | Coverage @95% precision |
|---|---:|---:|---:|
| Fingerprint head | **100.00%** | **99.61%** | 0.9659 |
| Interaction head | **100.00%** | 99.48% | 0.9510 |
| kNN-5 + τ | 99.68% | 99.16% | **0.9679** |
| cosine + τ | 99.68% | 99.03% | 0.9393 |

Order invariance: 40 states, options permuted, max |Δp| = **2.38e-07**, **0 argmax
flips**. Confirmed empirically on real embeddings.

**What the evidence supports:**

1. **Order invariance is validated.** The first design property confirmed on real data
   rather than asserted. A generative interface cannot offer this.
2. **A one-parameter cosine classifier matches the trained head.** With a properly fitted
   temperature, `softmax(cos/τ)` reaches log 0.380 / Brier 0.117 / ECE 0.045 — against the
   59,136-parameter head's 0.377 / 0.116 / 0.046. **For `Choice`, training buys essentially
   nothing.** The head's justification must come from what cosine cannot do: ordinal
   `Score`, multi-item composition, or inference cost at scale.
3. **A competent retrieval baseline is a real competitor.** kNN-5 beats the head on
   accuracy, Brier and ECE, and loses only on log score.
4. **The head is best at moderate coverage** (99.61% at 50% coverage — the highest of
   any candidate), and roughly tied with kNN-5 at 95% precision.
5. **Seed variance is negligible (±0.0002)**, so the accuracy gap is real on this
   split. Split-level variance is unmeasured.
6. **The typed-head pillar is competitive, not justified.** Its clear wins are order
   invariance and the typed interface.

**Two flaws in the analysis above, disclosed rather than buried:**

- **The temperature grid bound the baselines.** cosine, kNN-1 and kNN-5 all fitted
  τ = 0.5012, the *lower bound* of the search grid. Three methods landing on the same
  boundary means the optimum lies below it, so the baselines' log scores are not their
  best and the head's log-score win is inflated. **The log-score comparison is not
  settled** until the grid is widened and re-fitted.
- **The verdict block compared against the weakest baseline** (`cosine_tau`), which is
  why it reported a Brier win. Against the strongest baseline the head loses on Brier,
  ECE and accuracy. Verdicts must compare against the best baseline, not the most
  convenient one.

Composer: 28,315,392 params, produces a finite `(1, 768)` vector. No capability
claim is supported.

---

## 8. Open questions and risks, ranked

1. **Prompt pairing — RESOLVED.** The incumbent `Classification`/`Document` was
   suboptimal; `SearchQuery`/`Document` improves every candidate (+0.49 zero-shot,
   +0.94 kNN-5, +0.50 head, better log score and Brier). Symmetric prompts are worst.
   See §4. Adopt the retrieval pairing everywhere.
2. **kNN-5 matches or beats the head on most metrics.** The head needs a
   non-accuracy justification — order invariance (now validated), abstention at
   moderate coverage, new options without retraining, or soft targets — or it should
   not exist.
3. **The baseline temperature grid binds** (three methods fitted the lower bound), so
   the head-vs-kNN-5 log-score comparison is unsettled. Widen the grid and re-fit.
4. **The composer is unproven.** If it cannot beat mean-pooling at equal information
   on a real multi-item task, the "karpathy component" reduces to code style.
5. **Heads are linear probes, not RLCD.** The soft-target thesis is entirely untested.
6. **One dataset, one split.** No second domain, no split-level variance.
7. **No latency measurement**, despite "lightweight" being the design's pitch.
8. **Score and Noul are unexercised** — two of the three primitives have no evidence.
9. **Abstention thresholds are mode-specific** (fitted 10.6 for fingerprint vs 56.4
   for interaction on the same data), so they are not portable across head modes.
10. **End-to-end validation has only just begun** — the prompt-ablation kernel is the
    first to re-encode from the encoder rather than consume a cached artifact.

---

## 9. Decisions to record (ADR candidates)

| # | Decision | Status |
|---|---|---|
| 1 | Frozen EG2 encoder + trained typed head, instead of from-scratch decoder | accepted (evidence: reassessment) |
| 2 | Non-autoregressive typed output (Choice/Score/Noul), never parsed text | accepted |
| 3 | Bidirectional composer over EG2 item vectors for multi-item states | **proposed — unvalidated** |
| 4 | Gate on proper scoring rules (log score, Brier); report ECE only | accepted |
| 5 | Abstention via zero-bias max-score threshold at target precision | accepted, unmeasured |
| 6 | Kaggle-only execution for anything touching weights or real arrays | accepted (after a violation) |

These belong in `docs/adr/` if you want them versioned as records rather than
rows in this table.

---

## 10. Corrected validation plan

Each step is a Kaggle kernel that re-derives its own inputs — no pre-baked
artifacts standing in for the pipeline.

1. **Prompt ablation (cheap).** Re-encode Banking77 under {Classification,
   Document, none, SearchQuery} for both states and options; measure zero-shot
   cosine and head accuracy. Settles open question 1 and possibly 2.
2. **Abstention + order invariance (cheap, CPU).** Complete the run that failed:
   selective-accuracy vs coverage curves, coverage at 90%/95% precision, and an
   empirical order-invariance check — all candidates on one split.
3. **Composer ablation (GPU).** ScienceQA, multimodal Choice: state =
   `[question, context, image]`, options = answer choices. Composer vs masked-mean
   pooling at **equal information**, same head, same split. This is the decisive
   test for decision 3.
4. **Latency (CPU).** Encoder + head, text-only 270M, batch 1 — the number the
   design actually claims.
