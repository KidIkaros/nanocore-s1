# NanoCore-S1 — Architecture Re-assessment

> **As of:** 2026-10-07
> **Method:** market research (competitive landscape + technical architecture critique + build/buy route economics)
> **Question:** re-assess the NanoCore-S1 design, evaluating two possible inclusions — **Jev-style typed decision making** and **EmbeddingGemma 2** as the encoder. Laya is deliberately **not** treated as a component or a candidate; it appears only as illustrative evidence that the decision-model category ships (§5.4), plus one latency comparison (§4).
> **Supersedes the framing of:** [`ARCHITECTURE.md`](ARCHITECTURE.md) (archived 2026-10-05, "Pivot to Laya + EmbeddingGemma 2 stack")
>
> Every figure below is sourced or labelled an estimate. Figures carry an as-of date because the market moved substantially between 2026-09-15 and 2026-10-06.

---

## 1. Executive summary

**NanoCore-S1 is competent engineering aimed at the wrong interface.**

The design is a 135M-parameter decoder-only transformer that emits a typed decision by *generating* an `[ANSWER]` text block and parsing it back out. In the three weeks since the project was archived, a market has formed around exactly the job NanoCore-S1 was built for — fast, typed, calibrated decisions — and **every shipping member of it is non-autoregressive**. None of them generate the answer as text.

The independent evaluation of the category's flagship product states the failure mode plainly:

> "Using a generative/autoregressive model for such decisions is quite indirect. The decision has to be phrased as a prompt, the answer has to be parsed out of generated text, the model's probability for the answer is usually unavailable or unreliable, and every generated token is paid for."
> — Deußer, Sparrenberg & Sifa, *Evaluating and Benchmarking the System One Model Jev*, arXiv:2609.37647 (2026-09-29)

That is a description of NanoCore-S1's interface, written by authors who had never heard of it.

**Findings on the two proposed inclusions:**

| Inclusion | Verdict | Why |
|---|---|---|
| **Jev-style typed decision making** (Choice / Score / Noul, non-autoregressive, calibrated) | **Adopt** | It is the category's converged interface. It removes parsing, makes the probability the model's own output rather than a number recovered from text, and makes order invariance a structural property rather than a training hope. |
| **EmbeddingGemma 2** as encoder | **Adopt** | A frozen 270M text-only (740M full) Apache-2.0 encoder in a unified 768-d space. It removes the need to train an encoder at all, and makes multimodal states reachable without a new architecture. |

**Recommendation:** re-architect NanoCore-S1 as **frozen encoder + typed decision head**. Keep the repository's real assets — the test discipline, the quality-gate pattern, the `[STATE]`/`[CHOICE]` markup as a *wire format* — and retire the from-scratch decoder as the primary path. The decoder survives only under the narrow condition in §7.

**What would falsify this recommendation:** a target domain where nothing on the shelf has seen the data — a non-English corpus with no coverage, a proprietary symbolic format, or a hard constraint that no model with an unknown training set may touch the inputs. None of those describe "classify a support ticket".

---

## 2. What NanoCore-S1 specified

From [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`src/model.py`](../src/model.py):

| Element | Design |
|---|---|
| Family | Decoder-only, autoregressive (nanochat-derived `GPT`) |
| Depth | 12 layers, `n_embd` 768, 6 heads, head_dim 128 |
| Parameters | ~135M at the intended 32,768 vocab (§4) |
| Attention | RoPE, QK-norm, ReLU² MLP, untied embeddings, per-layer residual scalars |
| Positional / windowing | `window_pattern = "SSSL"` declared |
| Tokenizer | Custom byte-level BPE, 20 decision special tokens, target 32,768 vocab |
| Interface | Generate `[STATE] … [CHOICE] … [ANSWER] route=…, P(x)=0.92 [/ANSWER]`, then parse |
| Training | 4 stages: base (FineWeb-EDU) → mid → SFT → RLCD |
| Claimed latency | 15–35 ms CPU, 2–5 ms GPU |
| Claimed memory | ~544 MB fp16 CPU inference |
| Cost target | $0 (free GPU tiers) |

---

## 3. What was actually built and measured

This distinction matters, because the archive note says "52 tests passing, quality gate green" — which is true, and says nothing about model quality.

| Artifact | Reality | Source |
|---|---|---|
| Test suite | **52 tests** across 5 files (config 16, tokenizer 13, evaluate 10, model 7, train 6) — all passing | `tests/` |
| Code size | ~2,933 LOC across `src/`, `scripts/`, `tests/` | `wc -l` |
| Trained model | **None.** The only run is a 20-step dry run | `models/training_report.json` |
| Dry-run loss | **6.7661** on an **867**-token vocab | `models/training_report.json` |
| — what that means | `ln(867) = 6.7650`. The model is **exactly at uniform-random** | arithmetic |
| Dry-run tokens seen | 40,960 | `models/training_report.json` |
| Tokenizer | vocab **867** (target 8,192), 592 merges, **compression 1.8056×** | `models/tokenizer_meta.json` |
| Quality gate | `all_passed: true`, but the **Test Suite check is `skipped`** | `models/quality_gate_report.json` |
| Cloud entry point | `notebooks/train_colab.ipynb` is **invalid JSON** (missing comma, line 63/64) — it cannot be opened or run | verified |
| Full gate, run for real | **passes** — exit 0, 150 s, dry run 15.5 s, all six checks `True` | `reports/runs/audit/gate_b.json` |
| Gate's reported test count | **48** — the repo has **52**; `-k "not DryRun"` silently excludes all four `TestTrainingDryRun` tests | `reports/runs/audit/nanocore-audit.log` |

**The honest summary:** a well-tested training *harness* exists. No model was trained, and the one smoke test landed at chance, as a 20-step run must. There is no evidence of capability in either direction.

> **Validated on Kaggle.** Every claim in §4 that is about a *number the repository produces* was reproduced by running the repository's own code on a T4 — ten measurements, no reimplementations. See [`reports/runs/audit/`](../reports/runs/audit/) and §14.

---

## 4. Claims audit

Every quantitative claim in `ARCHITECTURE.md`, checked against a source. Status is one of **Accurate**, **Flag** (misleading or inconsistent), or **Unsupported**.

| Claim | Check | Status |
|---|---|---|
| `sequence_len 2048`, `n_head 6`, `n_embd 768`, `window_pattern "SSSL"` | Matches nanochat `GPTConfig` defaults exactly | **Accurate** |
| `vocab_size 32768` = nanochat default | True of nanochat's `GPTConfig` default. Note the *trained* nanochat uses **65,536** vocab at **4.8× compression**; NanoCore's tokenizer reached **1.8×** | **Accurate** (the gap is tokenizer quality, not the vocab number) |
| "157M params" (line 60) vs "~137M params" (line 74) | At 32,768 vocab the model is **135.3M**. 137M is close; **157M is wrong by ~14%** | **Flag** — internally inconsistent |
| Latency "15–35 ms CPU" | The design must *generate* an `[ANSWER]` block (~15–40 tokens) plus prompt prefill. For comparison: Jev is **236–276 ms p50** and Laya **32.8–39.5 ms on a T4 GPU**. A CPU autoregressive decode cannot be 15–35 ms | **Unsupported** |
| Memory "~544 MB fp16 CPU" | Weights 270.5 MB + KV cache 72.0 MiB at 2,048 tokens + activations ≈ 544 MB. Broadly consistent (doc's 64 MB KV figure is ~15% low) | **Accurate** |
| "Flash Attention 3 … with SDPA fallback" | Only `F.scaled_dot_product_attention` is used. No FA3 path exists | **Flag** |
| "Smear Gate: previous-token embedding mixing" listed under *What We Keep* | Not implemented anywhere in `src/model.py` | **Flag** |
| "Sliding Window Attention: SSSL pattern" listed under *What We Keep* | `window_pattern` is declared on the config and **never read**. Attention is plain full causal (`is_causal=True`) | **Flag** — documented efficiency feature absent |
| "Backout lambda — Remove" listed under *What We Remove* | Backout **is** implemented (`x = x - 0.2 * x_backout`, lines 277–278) | **Flag** — doc contradicts code |
| `n_kv_head 6` in the config table | `NanoCoreConfig` has **no `n_kv_head` field**; there is no GQA path | **Flag** |
| `[ANSWER] … P(security_fraud)=0.92` produced by generation, then parsed | Exactly the "indirect" path arXiv:2609.37647 names; the probability is recovered from text, not emitted by the model | **Invalidated** |
| "52 tests passing, quality gate green" | True, but the gate's Test Suite check records `skipped` | **Qualify** |
| Calibration "RLCD training needed"; option invariance "Untrained" | Correctly self-reported as weaknesses | **Accurate** |
| "No encyclopedic knowledge — facts must be in state/context" | Matches Karpathy's cognitive-core argument | **Accurate** |
| The architecture check validates the architecture | `_init_weights` zeroes `attn.c_proj` **and** `mlp.c_proj`, so `TransformerBlock(x) == x` exactly at init. The gate's "✓ Gradient flow verified" runs on an identity stack | **Flag** — validated |
| "52 tests passing" | The gate reports **48**: `pytest -k "not DryRun"` excludes all four `TestTrainingDryRun` tests, and the count comes from counting `" PASSED"` substrings | **Flag** — validated |
| The gate's evaluation step is meaningful | It reports **"ECE 0.0000"** for an untrained model and calls it "✓ All metrics valid" | **Flag** — validated |

Two further notes:
- **No test covers any of the flagged items.** `window_pattern`, `smear`, `flash`, `n_kv_head` and `backout` do not appear in `tests/`. The gaps are untested as well as undocumented.
- The archive note itself already admits this pattern. `ADR-001`'s *Corrections* section records three claims that "were wrong or unverifiable when checked against primary sources". The discipline of publishing corrections is good; the rate of them is the argument for doing this re-assessment.

---

## 5. The market, as of 2026-10-07

### 5.1 The category is real, and it formed in three weeks

| Date | Event |
|---|---|
| 2026-09-15 | TypeSafe AI releases **Jev**, its first "System One" model |
| 2026-09-22 | **2,170 public Jev projects**; **1,865 new repos** and **43,750 stars** in week one (arXiv:2609.30216) |
| 2026-09-24 | *Jev in the Wild* (arXiv:2609.30216) and the pentest/Rave paper (arXiv:2609.28940) |
| 2026-09-29 | *Evaluating and Benchmarking the System One Model Jev* (arXiv:2609.37647) |
| 2026-10-06 | Google releases **EmbeddingGemma 2** |

Karpathy's public position on it: Jev sits at "an under-invested point on the LLM Pareto curve — single-token, low-latency frontier-ish intelligence", a regime with "large revealed latent demand that was under-invested as the industry raced toward higher intelligence".

### 5.2 What the category converged on

Three primitives, and **no text generation**:

| Primitive | Question | Answer |
|---|---|---|
| `Choice` | pick one of a defined option set | the option, plus a probability per option |
| `Score` | rate on an ordered scale | probability mass across the levels |
| `Noul` | yes or no | `P(yes)` in [0, 1] |

All questions about one state are answered **in parallel in a single request**.

### 5.3 Jev — the commercial reference point

Measured, not claimed (arXiv:2609.37647, jev-1.13.0, zero-shot, 37 datasets, full eval splits):

| Metric | Value |
|---|---|
| Requests evaluated | **346,009** |
| Total cost | **US$9.15** (217.9M input tokens, ~630/request); whole study US$9.40 |
| Mean latency | **0.36 s** per request |
| Accuracy highlights | 95–99% IMDB / SST-2 / HellaSwag / ARC; **86.7% Belebele across 122 languages** |
| vs. open models | beats **Qwen3.8-27B on 27/37** datasets (no Qwen lead outside bootstrap intervals); beats **Gemma-4-E4B on 37/37** |
| Calibration | choice probabilities well calibrated and support selective prediction |
| **Order invariance** | **"Rotating the options leaves Jev's accuracy unchanged"** |
| Cost comparison | Gemma-4-E4B needed **6.4 A40 GPU-hours**; Qwen3.8-27B **17.4 A100 GPU-hours** for the same requests |

Weak spots, published alongside (the paper's own framing — a metric card with no regressions is not a card):

- Emotion **58.5%** (hashtag-derived, noisy labels); SST-5 five-way **57.9%**
- AGB-DE German legal clause analysis **F1 0.204**, AUROC 0.784
- Prompt-injection detection: **precision perfect, recall 50%** (AUROC 0.982)
- Binary probabilities rank well but are **poorly placed against a fixed 0.5 threshold**; tuning raises UNFAIR-ToS micro-F1 0.499 → 0.748
- MMLU calculation-heavy questions score *higher* than other MMLU questions (94% vs 91%), which the authors flag as possible benchmark exposure and therefore treat with caution

### 5.4 The open example

One open implementation exists and is worth naming once, as evidence the category ships rather than as a candidate: **Laya** (Convai Innovations, Apache 2.0, 421M `ModernBERT-large`) — 32.8–39.5 ms on a T4, self-hosted, fine-tunable, but **Banking77 0.425 at 77 labels** against Jev's 0.870, which is the category's known high-cardinality weakness. **OpenJev / OpenJevPro** (zhangcy122) is a design reference only — its licence is **PolyForm Noncommercial 1.0.0**, which covers internal enterprise production use.

### 5.5 EmbeddingGemma 2

Released **2026-10-06**, Apache 2.0, built on Gemma 4:

| Property | Value |
|---|---|
| Total parameters | **740M** |
| Text-only | **270M** (130M backbone + 140M embedder) |
| Vision / audio | +170M / +300M, **selectively loadable** |
| Output | **768-d native**, MRL truncation to 512 / 256 / 128 |
| Context | **8,192 tokens** (sliding window 1,024; ~5.5 min audio) |
| Languages / code | 100+ languages; ~14% better on code than its predecessor |
| Vocab | 262,144 |
| Licence | Apache 2.0 |

Hard-won constraint, already established in the sibling project: **EmbeddingGemma 2 must not run in float16** — bf16 on accelerators, fp32 on CPU. fp16 produces NaN or degraded embeddings.

---

## 6. Technical critique

### 6.1 The interface is the defect, and it is the one thing that cannot be patched

NanoCore-S1's probability is a *string that the model wrote*. `P(security_fraud)=0.92` is a token sequence; recovering `0.92` requires a parser, and the parser can fail, drift, or be fooled. The number is not a probability the model produced — it is a number the model *said*.

Jev's `Choice` probability is the softmax the model computed. There is nothing to parse and nothing to hallucinate.

This is not a stylistic difference. It determines:
- whether calibration means anything (a generated `0.92` has no calibration guarantee; a softmax does),
- whether order invariance holds (a generative model can reorder its answer; an isolated per-option scorer cannot), and
- whether the output is safe to gate on (a parse failure vs. a low-confidence signal).

### 6.2 Calibration

NanoCore-S1 correctly self-identifies this as untrained, and stages a 2-minute RLCD run as the fix. The category's answer is stronger: train against **strictly proper scoring rules** (log score, Brier, spherical; ranked probability score for ordinal `Score`), whose unique optimum is the true conditional distribution. The pentest paper's Table II rates RLHF **"Low"** security suitability — annotators prefer confident-sounding labels, which is precisely the overconfidence a decision model cannot afford. RLAIF inherits the problem; RLCD fixes it by construction.

Practical consequence already measured in the sibling project: **ECE can be gamed by sharpening**, while the log score and Brier cannot. Gate on the proper rules; report ECE.

### 6.3 Order invariance

The design says "each option is scored in isolation (no competitors in context) — guaranteeing 100% order invariance." That is the right idea and it is the category's consensus mechanism. But it is **incompatible with the rest of the design**: a model that generates one `[ANSWER]` block does not score options in isolation — it emits a single answer. The doc describes the Jev architecture in the NanoCore prompt format.

Jev measures order invariance empirically ("rotating the options leaves accuracy unchanged"); a per-option isolated head gets it at exactly 1.0 by construction.

### 6.4 Tokenizer

Measured compression is **1.8056×** against nanochat's **4.8×**. At equal corpus size, NanoCore-S1 would spend roughly **2.7× the compute per unit of text** — and its 200M-token budget is already small for its size. The 32,768 vocab number is not the problem; the merge quality from a short training run is.

### 6.5 Latency and memory

Memory is broadly honest (§4). Latency is not: a 15–35 ms CPU figure is not reachable by an autoregressive decoder that must emit a multi-token `[ANSWER]` block, and the design's own comparison table sets it beside Laya's *measured* 33 ms on a **GPU**. The realistic figure is one to two orders of magnitude higher.

### 6.6 Scale and data budget

135M parameters trained on ~200M tokens of FineWeb-EDU is far below the data budget that size wants, and — more importantly — **none of it is decision data**. The category's working models are 270M–421M encoders with decision-specific training. NanoCore-S1's stage 3 plans "10K curated decision examples"; the pentest paper's Rave design proposes **10,000+ labeled examples across 30+ CWE classes** for one decision task in one domain. The decision data, not the parameter count, is the constraint.

---

## 7. Build/buy route economics

Karpathy's *Small LLMs Setup* (§XII) gives the decision procedure:

| Route | What it is | Time | Cost | Take it when |
|---|---|---|---|---|
| 1 | Prompt an open model off the shelf | hours | $0 | the task is generic |
| 2 | Fine-tune one | days | tens of $ | you have labels |
| 3 | Distil one from a frontier teacher | 1–2 weeks | hundreds of $ | you have traffic |
| 4 | Train one from scratch | weeks | $100–$1,000 | **the domain is not English prose** |

And the anti-pattern table (§XVIII) names the failure directly: **"Training before prompting — Route 4 before route 1. Try the shelf first; most tasks stop there."**

**NanoCore-S1 took Route 4 for a job the shelf now covers.** Its stated job — "make fast, type-safe, calibrated decisions", "classify a security-fraud support ticket" — is a `Choice` over ~4 labels. That is Route 2 territory at worst.

The contrarian case is real and should be stated: Route 4 is not wrong in principle. It is correct when the shelf has not seen your data. NanoCore-S1's stated motivation was partly this — a bespoke `[STATE][CHOICE][ANSWER]` format and a $0 cost target. But the format is a serialization detail, not a domain the shelf cannot represent, and the $0 target is now met by Apache-2.0 weights rather than by training.

---

## 8. Inclusion 1 — Jev-style decision making

**What adopting it actually changes:**

| Change | Effect |
|---|---|
| Drop autoregressive generation | Latency from prefill + N decode steps → **one forward pass** |
| Drop the parser | Removes the failure mode where the answer exists but cannot be read |
| Probability becomes model output | Calibration becomes a measurable property, not a claim |
| Per-option isolated scoring | Order invariance by construction |
| Parallel questions per state | One request answers a whole decision form |
| Proper scoring rules | Training objective with the true distribution as its unique optimum |
| Typed primitives | The caller receives `{choice, P, confidence, abstained}`, not text |

**What survives from NanoCore-S1:** the `[STATE]` / `[CHOICE]` markup is worth keeping as the **wire format** for serializing `state` + `questions`. The Jev API takes exactly that pair. The markup stops being a generation target and becomes a request encoding — a demotion, not a deletion.

The tokenizer, likewise, survives only if a decoder survives (§10, Option A/C).

---

## 9. Inclusion 2 — EmbeddingGemma 2 as encoder

**What it buys:**

- **The encoder question disappears.** 270M text-only, frozen, Apache 2.0 — no encoder training, no tokenizer training, no corpus curation for the encoder.
- **A unified 768-d space.** Text, code, image, video and audio land in the same space, so the same head answers questions about a screenshot or a voicemail. That is a capability NanoCore-S1's design cannot reach without a new architecture.
- **8K context**, which partially substitutes for knowledge in the weights — the cognitive-core argument's "look it up" done cheaply.
- **MRL truncation** to 512/256/128, i.e. up to 6× cheaper vector storage for a memory plane, with re-normalization after truncation.
- **Training reduces to head-only.** Freeze the encoder, precompute embeddings, and the learning step fits on CPU: the head is `options × dim` parameters, not a transformer.

**What it costs:**

- ~0.5 GB resident for text-only; ~1.6–3 GB if the full multimodal stack is loaded. This is *more* than NanoCore-S1's claimed 544 MB, and it is a real constraint on a small host.
- **bf16/fp32 only.** fp16 NaNs. This is a deployment rule, not a preference.
- Embeddings must be precomputed and cached; option embeddings are keyed by label text, so a shared cache stays small (a measured example: 1,200 states → 49 unique option vectors).

---

## 10. Re-assessed target architecture

Three options, with the evidence for each.

### Option A — Keep the decoder, bolt on a typed head

Retain the 135M decoder; replace the generative `[ANSWER]` with a typed head over the final hidden state; use EmbeddingGemma 2 externally for retrieval.

- **For:** smallest change; preserves the tokenizer and the trained-adjacent code.
- **Against:** keeps the full training cost and the encoder-training burden, to reach a capability a frozen 270M encoder reaches for free. The decoder's own §6.6 data budget is the binding constraint. This is Route 4 with extra steps.

### Option B — Frozen EmbeddingGemma 2 encoder + typed decision head ★ recommended

Replace the decoder with a frozen encoder and a typed `Choice`/`Score`/`Noul` head.

- **For:** one forward pass; calibration by construction under proper scoring rules; order invariance at 1.0; multimodal states reachable; head trainable on CPU; encoder is Apache 2.0 and needs no training.
- **Against:** two artifacts to maintain; head-only training needs harvested labels and a held-out set before it can be judged; ~0.5 GB resident.
- **Note:** this is close to what the sibling `jev-stack` project already implements in Phase B. **This report does not claim it as a discovery.** Its contribution is the *justification and the correction record* — and the correction is that the original archive note attributed the design to "Laya + EmbeddingGemma 2", when Laya was never a component of it, and the architecture that actually works is the encoder + typed head.

### Option C — Hybrid: encoder decides, decoder drafts

Use the typed head for decisions and keep a small decoder for the "Draft" job (commit messages, summaries) that Karpathy's six-jobs list identifies as a separate small-model competence.

- **For:** honest about there being two jobs; reuses the existing tokenizer and training pipeline for the drafting half.
- **Against:** two models, two training pipelines, two failure surfaces — and the drafting job is *not* what NanoCore-S1 was built for. Only worth it if drafting is independently needed.

**Recommendation: Option B**, with Option C deferred until a drafting requirement actually exists.

---

## 11. Risks and caveats

**Contrarian evidence, stated plainly:**

- **Route 4 is sometimes correct.** If the target corpus is non-English with no coverage, a proprietary symbolic format, or subject to a hard "no unknown training set" constraint, training from scratch returns to being the right answer. None of those is "classify a support ticket", but the condition is real and should be re-checked per deployment rather than assumed away.
- **The benchmark is Jev-specific.** arXiv:2609.37647 evaluates one commercial model. Its conclusions about the category's *interface* transfer; its accuracy numbers are one vendor's.
- **Jev's own MMLU result is suspect.** The authors flag it as possible benchmark exposure and treat long-established English benchmarks with caution. Do not treat the headline accuracies as contamination-free.
- **Open weights are not free.** EmbeddingGemma 2 removes the *training* cost, not the memory, latency or engineering cost of running it.

**Caveats on this document:**

- **It is dated, and the field moves in days.** EmbeddingGemma 2 is one day old; the Jev ecosystem is three weeks old. Re-check before relying on any figure.
- **Licence traps.** OpenJev is PolyForm Noncommercial (design reference only). EmbeddingGemma 2 and Laya are Apache 2.0. LFM2.5 caps free commercial use at $10M revenue.
- **"Accurate" in §4 means "matches a source", not "the feature works".** Several rows are documented-but-unimplemented; §4 flags those.
- **No model was trained for this report.** The critique of the architecture is analytical; the critique of the *artifacts* is measured.

---

## 12. Recommendation

1. **Re-architect as Option B** — frozen EmbeddingGemma 2 + typed `Choice`/`Score`/`Noul` head, trained under proper scoring rules.
2. **Retire the from-scratch decoder as the primary path.** Keep it only under the falsification condition in §1.
3. **Keep, and demote:** the `[STATE]`/`[CHOICE]` markup becomes a request wire format; the test suite and quality-gate pattern carry over unchanged in spirit.
4. **Fix the archive record.** `ARCHITECTURE.md` attributes the design to "Laya + EmbeddingGemma 2"; Laya was never a component of the architecture that works. Correct that in place, and add the §4 claims audit so the next reader does not re-derive it.
5. **Do not gate on ECE.** Gate on log score and Brier; report ECE.

**Success condition:** a held-out decision set where the model is order-invariant at 1.0, abstains honestly on unseen options, and its stated confidence matches its accuracy — all before it is put in any path where a wrong answer costs more than a slow one.

---

## 13. Validation on Kaggle

The claims above that are about *numbers the repository produces* were not left as readings.
They were reproduced by running the repository's own code on a T4 — kernel
`mauricew/nanocore-audit` v2, source `mauricew/nanocore-s1-worktree` (the uploaded working
tree, not a clone). Ten measurements, no reimplementations, so a finding cannot be an
artifact of a reimplementation. Full record in [`reports/runs/audit/`](../reports/runs/audit/).

| Claim | Measured | Verdict |
|---|---|---|
| blocks are the identity at init | `max｜TransformerBlock(x) − x｜ = 0.0`; both `c_proj` all-zero | confirmed |
| 135.3M params, not 157M | counted `135,266,328` = by hand; line 60 off by 22M (16%) | confirmed |
| `window_pattern` is inert | identical logits for `SSSL`/`L`/`SL`, at init **and** de-zeroed | confirmed |
| decision accuracy is not a function of `(model, data)` | same model + same data → `[0.12, 0.16, 0.12, 0.10, 0.08, 0.16, 0.08, 0.16]` | confirmed |
| ECE does not respond to the model | 6 initialisations → `[0.002020, 0.002023]`; zeroed model → `0.001953` = 1/vocab | confirmed |
| "perplexity" uses unshifted targets | repo `512.30` vs unshifted `512.57` vs shifted `512.72` | confirmed (structural) |
| tokenizer is 1.7×, not 4.8× | vocab 867, 592 merges, 1.671× measured | confirmed |
| "15–35 ms on CPU" | prefill 142 ms; 24 generated tokens **3601 ms** (150 ms/token) | confirmed — off by ~100× |
| cloud entry point is invalid JSON | `JSONDecodeError: Expecting ',' delimiter: line 64 column 5` | confirmed |
| gate passes with checks skipped | exit 0, `all_passed: true`, Dry Run **and** Test Suite skipped | confirmed |

**The finding the static read missed.** `_init_weights` zeroes both sublayer output
projections, so a block computes `x + 0 + 0`. The gate's architecture check therefore runs
its forward and backward pass on an identity stack, and reports *"✓ Gradient flow verified
(loss=10.3996)"* — where `ln(32768) = 10.397`, exactly uniform-random. This also confounded
the first version of the notebook, so every affected test was re-run with `c_proj`
randomised; V6 and V1 hold in both conditions.

**What this does not settle.** The full gate genuinely passes (150 s, all six checks) — the
defect is that it cannot distinguish that from passing with checks skipped. V3 is structural:
on random data all three perplexities sit near the vocab size, so the magnitude of the error
is not demonstrated. And no model was trained, so nothing here measures capability.

---

## 14. Sources

**Papers**
- Deußer, T., Sparrenberg, L. & Sifa, R. *Evaluating and Benchmarking the System One Model Jev.* arXiv:2609.37647v1 [cs.CL], 2026-09-29.
- Ling, G., Xue, M. & Ye, Z. *Jev in the Wild: A Data-Driven Analysis of the Jev Model's Functionality, Applications and Ecosystem.* arXiv:2609.30216v1 [cs.SE], 2026-09-24.
- dos Santos, J. A. *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents.* arXiv:2609.28940v1 [cs.CR], 2026-09-24.
- Karpathy, A. *Small LLMs Setup / Small Language Model Engineering 2026* (§IV one dial; §XII four ways to get one; §XIII the 2026 shelf; §XVIII anti-patterns).

**Primary model sources**
- EmbeddingGemma 2 model card — https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
- `google/embeddinggemma-2` — https://huggingface.co/google/embeddinggemma-2
- EmbeddingGemma 2 launch — https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/ (2026-10-06)
- Laya (illustration only) — https://huggingface.co/convaiinnovations/laya
- OpenJev (licence check) — https://github.com/zhangcy122/OpenJev
- TypeSafe AI, *Introducing System One Models & Jev* — https://typesafe.ai/blog/introducing-system-one-models-and-jev
- nanochat `GPTConfig` — https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py

**Repository artifacts**
- `models/training_report.json`, `models/tokenizer_meta.json`, `models/quality_gate_report.json`
- `src/model.py`, `src/tokenizer.py`, `tests/`, `notebooks/train_colab.ipynb`
- `docs/ARCHITECTURE.md`, `docs/IMPLEMENTATION-PLAN.md`

**Validation run (§13)**
- Kernel `mauricew/nanocore-audit` — `notebooks/nanocore_audit/nanocore_audit.ipynb`
- Dataset `mauricew/nanocore-s1-worktree` — the uploaded working tree under test
- `reports/runs/audit/` — `audit_results.json`, `gate_a.json`, `gate_b.json`, `nanocore-audit.log`, `analysis.md`

**Sibling project (relationship noted, not depended on)**
- `jev-stack` — `docs/adr/0001-laya-stack.md`, `docs/adr/0002-system-one-decision-model.md`, `AGENTS.md`
