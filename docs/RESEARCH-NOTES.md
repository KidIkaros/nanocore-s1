# Research Notes — NanoCore-S1 decision model

A dated log of research that informs the architecture. Each entry states the question,
the evidence, and the implication for our plan. Companions:
`ARCHITECTURE-DECISION-MODEL.md` (what exists), `ARCHITECTURE-SHARPENING.md` (proposals
and their outcomes), `adr/` (decisions).

---

## 2026-10-07 — Four questions that decide whether the remaining pillars can work

### R1. Is Banking77 the right benchmark? **No — it is saturated.**

**Question.** Can a 92.9%-zero-shot benchmark discriminate between architectures?

**Evidence.** Our own measurements: zero-shot cosine 92.92%, linear probe 93.11%, kNN-5
93.64%. Every difference we have been chasing lives inside a **7.1% residual** — the
fraction of items the frozen encoder does not already solve.

The literature agrees on the mechanism. LP++ (CVPR 2024) reports a standard linear probe
scoring **~20% below** zero-shot in the 1-shot regime — so probing is not inherently weak;
its advantage depends on the data regime. "Why Linear Probing Works" (ECCV 2026) attributes
generalisation from few examples to the spectral structure of foundation-model features
(effective dimension), not to classifier capacity.

**Implication.** **Benchmark choice is a first-order experimental variable we have been
ignoring.** On a saturated task, a better head cannot show a better number, and a null
result is uninformative. Every architectural comparison must be run on a task with
headroom. Concretely: report zero-shot accuracy *before* choosing a benchmark, and require
meaningful headroom (say, zero-shot ≤ ~85%) for any claim that one head beats another.

### R2. Should the composer beat mean pooling? **The prior is mixed, and leans negative for classification.**

**Question.** Does attention over frozen item embeddings beat simple pooling?

**Evidence.**
- *Pooling and Attention: What are Effective Designs for LLM-based Embedding Models?*
  (arXiv 2409.02727) is a large controlled study with statistical testing: models with
  bidirectional attention **and an additional trainable pooling layer** "demonstrate
  superior performance in semantic textual similarity and information retrieval tasks but
  **underperform in clustering and classification tasks**." Our `Choice` task is
  classification-shaped.
- *Why and when should you pool?* (Findings of EMNLP 2020): pooling's advantage is largest
  in **low-resource** scenarios and when salient content sits in the middle rather than at
  the boundaries — i.e. when relevance is **uneven across items**.
- Set Transformer (Lee et al., ICML 2019) and Zaheer et al. (2017) establish that
  permutation-invariant functions are exactly `ρ(pool(φ(x)))`, so mean pooling is a special
  case — but a special case is not automatically a *worse* one, and with `φ` frozen the
  composer can only recombine, never extract.

**Implication.** The composer prediction must be stated narrowly, or the experiment will be
misread either way:

> Attention should help when state items are **redundant or act as distractors** (relevance
> is uneven), and should **not** be expected to help on uniform-relevance classification,
> where the literature reports trainable pooling *underperforming*.

ScienceQA fits the first case (question + lecture + hint + image, some irrelevant), so it is
a fair test — but a **null result is a likely outcome and must be pre-registered as
acceptable**, not spun.

### R3. How should the `Score` primitive be built? **We are currently using the wrong loss.**

**Question.** What is the right way to model an ordered rubric?

**Evidence.** CORAL (*Rank Consistent Ordinal Regression*, Raschka et al. 2020) and CORN
(2021) both exist because "many real-world prediction problems have ordinal response
variables, and **this ordering information is ignored by conventional classification losses
such as multi-category cross-entropy**." CORAL: `K-1` binary tasks sharing weights with
independent biases, giving rank-monotonicity guarantees. CORN: removes the weight-sharing
restriction via conditional training sets and the chain rule, and outperforms CORAL.

**Our current state.** `DecisionHead` scores `Score` as an ordinary softmax over ordered
options with a log-score objective — precisely the loss the ordinal literature says discards
the ordering. We already have `ranked_probability_score` in `metrics.py` and it is unused.

**Implication.** `Score` is the **clearest differentiator** in the whole design, because
temperature-scaled cosine has no notion of order at all. It also has a well-specified,
literature-backed implementation path: CORN/CORAL head + RPS evaluation + monotonicity
check. It has never been exercised in this project.

### R4. What should the abstention mechanism be? **APS/RAPS — but it is a feature, not a differentiator.**

**Question.** Which conformal method, and does it differentiate the architecture?

**Evidence.** *Classification with Valid and Adaptive Coverage* (Romano, Sesia & Candès,
NeurIPS 2020) introduces **APS**: nonconformity score = cumulative probability mass needed
to reach the true class once predictions are sorted descending, giving prediction sets with
guaranteed marginal coverage that adapt to difficulty — "easy points get tight sets,
ambiguous points get larger ones." **RAPS** (Angelopoulos et al. 2021) adds a rank-based
regulariser for smaller sets without sacrificing coverage. Empirically RAPS gives the
smallest sets at high coverage for few classes; SAPS is preferred for many classes.

Decisively for our framing, the APS paper states the method works "with any black-box
predictive model… **regardless of whether they are well-calibrated**."

**Implication.** Conformal abstention is the right engineering answer (it replaces a fitted
threshold with a guarantee, and it removes the 90%-precision saturation problem). But because
it applies to *any* scorer, it **does not differentiate** the trained head from
temperature-scaled cosine. It is a deployable feature of the interface, and it should be
reported as such — not as evidence for the model.

---

## 2026-10-07 (later) — The library contains the target domain, and it contradicts one of our design assumptions

Source: `~/Documents/Library/technical/ai-ml/Calibrated Decision Models for Autonomous
Penetration-Testing Harnesses` (Santos, 2026) — the same paper this project's typed
primitives come from, read this time for its *application*, not its primitives. Supported
by the `security/` shelf (ML for Cybersecurity, Kali Linux, AI-Powered Cybersecurity).

### R5. The target domain is LLM-driven penetration-testing agents, and the paper specifies the architecture

The paper formalizes **four decision points** where a System One model replaces a
generative LLM call, each with a concrete state, option set, and ground truth:

| | Decision | Primitive | State | Options / target |
|---|---|---|---|---|
| **DP1** | Finding adjudication | `Choice` + `Noul` | HTTP request–response pair + evidence | `{confirmed, needs-review, rejected}`; Noul: was real impact demonstrated? |
| **DP2** | Severity recalibration | `Score` | Exposed data + evidence | **four ordered levels**: none → common → … |
| **DP3** | Agent pruning | batched `Noul` | Candidate agent set (15) | `p(relevant)` per agent; drop below τp |
| **DP4** | Confirmation loop | `Choice` + `Noul` | Candidate payloads + observed filtering | next payload; did it produce the expected effect? |

It also defines a **five-level cost cascade** with abstention bands, which is the
architecture our gate is trying to be:

| Level | Mechanism | Cost | Latency | Share |
|---|---|---|---|---|
| 0 | deterministic CWE validator | $0 | µs | 40% |
| 1 | System One `Noul` (outside τlow…τhigh) | $0.00006 | 33–276 ms | 25% |
| 2 | System One `Choice` + `Score` | $0.0002 | 33–276 ms | 20% |
| 3 | LLM deliberation | $0.02–0.10 | 1.5–3.0 s | 15% |
| 4 | Human review | $15 | minutes–hours | 0% measured, 5–10% expected |

**Implication.** This is not a hypothetical target — it is a specified one, with real
states, real option sets, and real economics. It also gives **both** composition regimes a
concrete home: DP1's state is a *bag* of evidence (permutation-invariant), while DP4's state
is a *sequence* of payload attempts and filtering observations (order-bearing). That is a
domain-grounded reason for the S2 design decision to make order-sensitivity an explicit,
ablatable mode rather than baking RoPE in.

### R6. The training signal is *verifiable*, not soft — this corrects S8

Ground truth comes from **deterministic per-CWE validators** (27 in the NeuroSploit
harness) which "constitute proof of exploitability", and the paper's RLHV paradigm
(Reinforcement Learning from Human Verification) produces `(prediction, validator-verdict)`
pairs on every engagement, so "the decision model converges toward the validators' decision
boundaries."

**Implication.** S8 assumed we needed *soft teacher distributions* — the expensive,
hard-to-obtain part of the RLCD story. In this domain we do not: we need **verifiable
labels**, and the harness already emits them. Proper scoring rules still apply (Brier and
log score are proper on hard labels too), so ADR-0004 survives — but the framing "we need a
teacher to distil" was wrong. The bottleneck is verifiable labels, and the domain supplies
them.

### R7. **Abstention is not safe in this domain** — and our design assumes it is

The `designing-ml-systems` pass recorded that "the tolerable failure mode is an abstention,
not a confident wrong answer." The paper contradicts that for this domain:

> "A miscalibrated System One model cannot cause false positives: fabricated findings cannot
> pass the deterministic validators that gate the pipeline. However, it can cause **false
> negatives**… The worst case of System One failure is therefore **not symmetric**: false
> positives are blocked, but false negatives are possible and **may be worse than running
> without the layer**, because the LLM-only path would have reported those findings."

**Implication — this is a design defect, not a nuance.** In this domain, abstaining on a
"is this real?" judgment is not a deferral: it **suppresses a finding**, which is an active
negative decision. A max-score threshold that silently drops low-confidence positives is
exactly the failure mode the paper warns about. Our gate must therefore be **asymmetric**:
low confidence on a positive claim escalates, and it must never silently drop. That is a
stronger requirement than the conformal coverage guarantee alone (R4) — conformal
prediction-set abstention is closer to the right shape than thresholded argmax, because a
set can widen instead of collapsing to "no".

The paper's own mitigations are worth adopting: preprocess the state to strip obvious
injection patterns, monitor for anomalous probability distributions ("all Nouls below 0.3
for a surface with known vulnerabilities"), and flag LLM/System One disagreements.

### R8. Latency targets exist; we have never measured ours

Jev: **236–276 ms** p50. Laya: **33–40 ms**. The cascade's economics depend on this — DP3
pruning of 15 agents is 22–45 s at LLM latency versus 276 ms (Jev) or 72 ms (Laya) batched,
"the difference between agent-pruning being a bottleneck and being invisible."

**Implication.** "Extremely lightweight" now has a number to hit: to sit at Level 1–2 the
model needs to be tens of milliseconds, not hundreds. We have never timed our encoder + head
at batch 1, so the design's central claim is still unquantified (S7, still untested).

### Also worth recording: the security argument for our non-autoregressive choice

> "The non-autoregressive architecture eliminates the text-generation attack surface: there
> is no autoregressive chain to redirect and no system prompt to override."

That is an independent, adversarial-robustness argument for ADR-0002 — but the paper also
notes the **state input is attacker-influenced**, so adversarial perturbation can push the
probability distribution toward suppression. A threat model for the encoder and composer,
not just the output layer.

---

## 2026-10-07 (latest) — What should a general-purpose assistant's model decide?

**The question.** If the goal is a SOTA personal assistant — "jack of all trades, ready for
anything" — what should the model decide?

**The answer, from the framework this project is named after.** Karpathy's *Small LLMs
Setup* (in the library) lists six jobs a ~500M model is actually good at, and states the
constraint directly:

> "Six jobs where small, fast and predictable beats large, slow and general.
> **None of them is a general assistant, and that is the point.**"
>
> "Classification is the clearest case. A bounded label set needs no world knowledge, and a
> small model that answers in milliseconds for a fraction of a cent beats a frontier call
> that does the same job slower, more expensively, and with more variance. **Routing is the
> same shape: deciding which model should handle a request is a smaller problem than
> handling it.**"

The six jobs: **Classify** (typed decisions over a fixed label set, no world knowledge),
**Route** (pick which big model, tool or queue a request belongs to), **Draft** (first pass
on a bounded template), **Run on device**, **Absorb a distillation**, **Teach**.

### R9. A general assistant's model should decide **dispatch**, not content

"Jack of all trades" is a property of the **stack**, not of any single model. The small
model is the switchboard that makes a heterogeneous set of tools, models and skills behave
like one assistant. Its decisions are meta-decisions:

| Decision | Primitive | Notes |
|---|---|---|
| **Route** — which model/tool/skill handles this? | `Choice` over a supplied tool set | order-invariant by construction; new tools need no retraining in `interaction` mode |
| **Escalate** — is the cheap path sufficient? | `Noul` / confidence gate | the designed fallback is the strong model |
| **Classify** — intent, domain, policy class | `Choice` over a fixed label set | no world knowledge needed |
| **Verify** — is this output good enough to return? | `Noul` | post-generation quality estimation |
| **Authorize** — is this action permitted? | `Noul` | policy gate |
| **Clarify** — ask, or act? | `Noul` | is the request underspecified? |

**The routing literature says the quality estimator is the bottleneck.** *A Unified Approach
to Routing and Cascading for LLMs* (arXiv 2410.10347) proves optimality for routing and
cascading and concludes that "**good quality estimators [are] the critical factor** for the
success of model selection paradigms." *RouteLLM* reports **85% cost reduction at 95% of
GPT-4 quality** on MT-Bench. *Cluster, Route, Escalate* (arXiv 2606.27457) retains **97–99%
of the strongest model's accuracy** with a quality-estimation cascade, and needs only
**task-correctness labels**. A semantic router for reasoning requirements gains **+10.2
accuracy points while cutting latency 47% and tokens 48.5%**.

**Why this finally makes the architecture coherent.** Every design choice that looked
unjustified on Banking77 has a purpose here:

| Property | Why it matters for dispatch |
|---|---|
| Calibrated probabilities | the quality estimator *is* the product — the literature's stated bottleneck |
| Abstention | safe here: the fallback is escalation to the strong model, so abstention is the design, not a failure |
| Order invariance | tool sets arrive in arbitrary order; permuting them must not change the routing |
| Multi-item state | conversation + tool inventory + retrieved candidates + prior outputs — genuinely multi-item, in **both** regimes (a bag of tools, a sequence of turns) |
| Multimodality | requests arrive as text, screenshots, audio and files — one shared space, no fusion training |
| Low latency | 33–40 ms to decide versus ~2 s to generate *is* the entire value proposition |
| Headroom | routing and quality estimation are nowhere near saturated — unlike Banking77's 92.9% zero-shot |

**This reverses R7 for this domain.** In the pentest harness, abstention suppressed findings
and was dangerous. For assistant dispatch, abstention *is* escalation — the safe path — so
the original `designing-ml-systems` assumption holds here. Abstention safety is a property
of the domain, not of the model.

**Why the design "hasn't come to fruition."** The common failure is trying to make a small
model *answer*, which puts it in direct competition with frontier models it cannot beat. The
viable design makes it *dispatch*, which is a strictly smaller problem (Karpathy: "deciding
which model should handle a request is a smaller problem than handling it") — and the
"general assistant" behaviour emerges from the stack the dispatcher controls.

---

## 2026-10-07 (dispatch run) — R10: saturation is general, and the headroom is in the *quality estimator*

First run of the dispatch benchmark (`notebooks/s1_dispatch`), measured entirely with
`src/decision/protocol.py`. 1,252 BFCL routing queries (2–37 candidate tools) and 1,118
irrelevance queries.

### The headroom problem is not a Banking77 quirk

| benchmark | zero-shot accuracy | headroom | usable |
|---|---:|---:|---|
| Banking77 intent classification | 92.92% | 6.9% | no |
| **BFCL tool routing** | **93.07%** | **6.9%** | **no** |

`headroom_check` flagged both as unusable. **EmbeddingGemma 2 zero-shot reaches ~93% on
realistic tool routing**, so a frozen encoder plus a temperature-scaled cosine is already a
complete router. This is the most consequential result so far: it is not that our head is
bad — it is that on these tasks there is nothing left for a head to win.

### The trained head loses on every metric, and retrieval collapses

| method | accuracy | log score | Brier |
|---|---:|---:|---:|
| random | 32.00% | 1.202 | 0.676 |
| **zero-shot cosine** | **93.07%** | **0.176** | **0.099** |
| interaction head (3 seeds) | 90.84% | 0.253 | 0.137 |
| kNN-20 votes | 73.07% | 0.630 | 0.343 |
| kNN-5 votes | 71.47% | 0.689 | 0.368 |

`compare_to_best` reported **no winners on any metric**. Two distinct causes, both pointing
the same way: the task is saturated, *and* 877 training examples is the low-data regime
where a probe is documented to underperform zero-shot.

**kNN collapses on dispatch (71–73%) after winning on classification (93.6%).** The
query→tool mapping is not locally smooth in embedding space — similar queries need
*different* tools. So "retrieval is the strong baseline" is task-dependent, which corrects
the earlier framing: retrieval wins on classification and loses badly on routing.

### Order invariance holds on real variable option sets

80 states, **2–37 options varying per request**, max |Δp| = **1.19e-07**, **0 flips**.
Previously validated only on a fixed 77-label set. This is the real test, and it passes.

### The headroom is in relevance/quality estimation, not routing

Abstention signal from the zero-shot raw score:

| | value |
|---|---:|
| mean score, relevant | 0.7401 |
| mean score, irrelevant | 0.6428 |
| **separation** | **0.0973** |
| auto-handled at 90% precision | **3.46%** |
| cost saved vs always-strong | **3.4%** |

To hold 90% precision the gate must escalate 96.5% of traffic — that is not a product.
But unlike routing this task is **genuinely unsaturated**, and BFCL's irrelevance set is
*constructed* to be hard (tempting-but-wrong tools). The routing literature names the
quality estimator as the critical factor for exactly this reason.

**Consequence.** Close the routing-accuracy track — it is saturated and cannot discriminate.
Open the **quality-estimator track**: re-frame the head as a *relevance* judge (is this tool
relevant to this request?) rather than a router, which is a different, unsaturated task with
real labels and a measurable product outcome (cost saved at fixed precision).

### Protocol defect found by the run, and fixed

Several fitted temperatures landed on **0.000333**, the lower edge of the *refinement*
window — the same grid-edge bug as the original, surviving one level down. Both stages of
`fit_temperature` are now wide enough (coarse 10^-6 … 10^2, refinement ±1 decade) that an
edge hit would require an optimum below 10^-7, with a regression test asserting a
temperature below 0.01 is recovered. 87 tests pass on Kaggle.

---

## 2026-10-07 (market check) — R11: how Qwen and MiniCPM actually handle these problems

Asked to check how the competition handles the issues we hit. Two reference classes, and
both are informative because they answer our questions differently than we assumed.

### Qwen3 — the market's answer to "should we think harder?" is a **knob, not a router**

- Dense sizes **0.6B, 1.7B, 4B**, 8B, 14B, 32B, plus MoE 30B-A3B and 235B-A22B.
- **Hybrid thinking mode**: one checkpoint switches between chain-of-thought "thinking" and
  efficient "non-thinking" via `enable_thinking`, *without changing models*.
- Headline capabilities are **agentic**: precise external tool integration and Model Context
  Protocol support, in both modes. 119 languages.

**The escalation decision is exposed to the caller as a flag.** Nobody trains a dispatcher
to decide it. That is the single most useful fact in this section: it means a *learned*
escalation predictor is either genuinely novel or genuinely unnecessary — and that the
market's current answer is that a caller who knows their workload can set the knob.

### MiniCPM — competes on **efficiency at fixed capability**, and measures tokens and latency

- **MiniCPM-V 4.6**: **1.3B total** (SigLIP2-400M + a Qwen3.5-0.8B LLM). Beats
  Gemma4-E2B-it; ~**1.5× token throughput** vs Qwen3.5-0.8B. Intra-ViT early compression
  cuts visual encoding cost **>50%**; mixed **4×/16× visual token compression**. Runs on
  iOS, Android and HarmonyOS with edge adaptation code open-sourced.
- **MiniCPM-V 4.0**: 4.1B, OpenCompass **69.0** — above GPT-4.1-mini — at **<2 s
  first-token latency and >17 tok/s on an iPhone 16 Pro Max**.
- **MiniCPM4**: 0.5B and 8B; InfLLM v2 trainable sparse attention; **CPM.cu** inference
  system combining sparse attention, quantization and speculative sampling; trained on 8.8T
  tokens; MiniCPM4.1 adds a hybrid reasoning toggle.

**Two lessons.** First, their competitive axis is *tokens, latency and memory at fixed
capability* — first-token latency and throughput are first-class reported metrics, not
afterthoughts. Second, **the model sizes for usable multimodal capability are falling fast**:
the MiniCPM-V paper's thesis is that GPT-4V-level performance on-device is arriving, and
they are engineering for it (sparse attention, quantization, speculative decoding, visual
token compression).

### What this means for us

| Our assumption | What the market does |
|---|---|
| "A small model should dispatch, not answer" | Qwen ships agentic/tool-use capability *in* the small model; MiniCPM ships small models that **answer** (OCR, multi-image, video) |
| "Escalation needs a learned predictor" | Qwen exposes `enable_thinking` as a **caller-set flag** |
| "Competitive axis is accuracy" | MiniCPM's axis is **token throughput, TTFT, visual token count, quantization** |
| "Frozen encoder + tiny head is the lightweight path" | The market trains **small generative multimodal models** end-to-end (1.3B–8B) and wins on efficiency engineering |
| Our latency claim (33–40 ms, never measured) | MiniCPM-V 4.0: **<2 s TTFT, >17 tok/s on a phone** — a different regime, and a published one |

**The uncomfortable conclusion.** Our benchmarks were saturated *because* a frozen embedder
solves classification and routing. The tasks the market actually competes on — OCR,
document and video understanding, visual QA — are **not** saturated by embeddings, because
they require generation and grounding. A frozen encoder plus a linear head cannot enter
that class of task at all.

**But there is a real gap the market leaves open.** None of these models ships a **typed
decision interface** — `Choice`/`Score`/`Noul`, probabilities as first-class outputs,
calibrated abstention. Qwen gives a thinking flag; MiniCPM gives generation. Jev sells typed
decisions, commercially, in the cloud. So the defensible niche is narrow and specific:
**on-device typed decisions with calibrated abstention**, not content, and not a router.

### Consequences for the four options

- **A (ship the interface, not a model)** — strengthened. The interface is the part the
  market does not have, and it is the only part with no unsupported claims.
- **B (ordinal `Score`)** — strengthened. Still categorically different from anything the
  market exposes.
- **C (find harder tasks)** — reframed. "Harder" should mean *generative/grounded* tasks
  (OCR, documents, video), which is where the market's benchmarks live and where embeddings
  are not already at ceiling. But that is a different, much larger build.
- **D (composer)** — unchanged, and still only measurable on a task with headroom.
- **New, and cheap: measure latency.** We claim "extremely lightweight" and have never
  produced a number. MiniCPM's published figures are the bar. This is the cheapest way to
  make or break the design's central claim.

---

## 2026-10-07 (competitive check) — R12: we beat the typed-decision competitors, and the encoder is why

Triggered by the latency result (a larger competitor being faster). Checked the library's own
benchmark paper, which evaluates **Jev and open models on the same tasks we measured**.

### The category benchmark, with sources

| system | Banking77 accuracy | source |
|---|---:|---|
| **EG2 zero-shot + fitted temperature (ours)** | **92.92%** | our measurement, 3,080-example test split |
| Jev — **independently measured** | **79.7%** (CI 0.782–0.812), ECE 0.087 | *Evaluating and Benchmarking the System One Model* |
| Jev — **self-reported** | 87.0% | quoted in the pentest paper |
| Laya | **42.5%** | pentest paper, comparison table |

At **50% coverage** — the selective-prediction metric the benchmark paper itself reports:

| system | accuracy at 50% coverage |
|---|---:|
| **ours** (fingerprint head, calibration v3) | **99.61%** |
| Jev | 96.3% |
| Qwen (open model) | 94.5% |

**Two things follow.** First, Jev's independently measured number (79.7%) is **7.3 points below
its self-reported 87.0%** — worth remembering when vendor figures are quoted. Second, our
frozen-encoder approach beats Jev's measured accuracy by **13.2 points**, Laya's by **50
points**, and Qwen's selective accuracy by 5 points at the same coverage.

**The uncomfortable corollary:** the win comes from **EmbeddingGemma 2 zero-shot**, not from
anything this project built. The encoder is the asset; the head has never beaten it.

### Why the encoder is not the whole story: size is modular

EG2 is a **modular** model — 270M text/code, 440M text+vision, 570M text+audio, **740M full
multimodal** — and the unused encoders are excluded at load time via `config_kwargs`. We load
text-only (measured 271,002,624 params) or text+vision (439M). So the 740M headline figure is
the all-modalities configuration, and our text encoder is in fact **smaller than Laya's 421M**.

### Why Laya is faster, and whether it matters

**Correction (2026-10-07, see `docs/ENCODER-EFFICIENCY.md`).** This was first written as "our
EG2 descends from a decoder stack with full attention at every layer". **That was wrong.**
EG2's own `config.json` shows alternating local/global attention at **5:1**, a **1024-token**
sliding window, **GQA/MQA** (4 query heads, 2 local / 1 global KV), and a **512** hidden size —
the same efficiency playbook as ModernBERT. The claim is retracted.

The real explanation has two parts:

1. **EG2's sliding window is 1024 tokens and our states are far below it.** Within one window
   every token still attends to all preceding tokens up to the bound, so attention is
   *effectively quadratic* for short and medium inputs — exactly matching the measured 18×
   penalty for a ~500-token state versus a 20-token one. The window only pays off past 1024
   tokens, which our inputs never reach.
2. **We run the unoptimized reference path** — no quantization, no export, no thread tuning, no
   sequence cap. We measure **7.7× above the memory-bandwidth floor on CPU and 32.6× on GPU**,
   and Google publishes the same model at **~191 MB quantized on a Pixel 11 Pro**.

**Verdict:** worth attention, not alarm — and the gap is deployment engineering, not the
encoder. Laya remains text-only, so it cannot be multimodal at all.

### What this means for the `Score` track

The benchmark paper reports Jev on SST-5: **57.9% five-way accuracy** but **ρ = 0.851**
probability-weighted ordinal correlation, noting that "treated as an ordinal scale, SST-5 is
thus solved considerably better than its 57.9% five-way accuracy suggests."

That is a published target for exactly the primitive we were about to build, and it is the
cleanest statement of the thesis: **for ordinal decisions, the distribution carries more than
the argmax.** Our `Score` experiment should be measured against both numbers.

---

## 2026-10-07 (encoder efficiency) — R13: llama.cpp closes 4.08×, and two claims retracted

Asked to research encoder efficiency deeply after the latency result showed a *larger* competitor
faster on the same hardware. Full write-up in `docs/ENCODER-EFFICIENCY.md`; conclusions are now
ADR-0007 and ADR-0012.

### The measurement

On the same 4-core Xeon (2.20 GHz, **AVX2 only**), same task, llama.cpp Q8_0 vs our PyTorch fp32:

| | PyTorch fp32 | llama.cpp Q8_0 | speedup |
|---|---:|---:|---:|
| short state, batch 1 | 363.5 ms | **89.2 ms** (p95 97.4) | **4.08×** |
| long state, batch 1 | 6,595 ms | **1,754.8 ms** | **3.76×** |
| throughput | 2.4 texts/s | **9.9 texts/s** | **4.12×** |

**Quality is preserved**: Banking77 zero-shot **93.41%** under Q8_0 against 92.92% for PyTorch
fp32 (+0.49, within noise). Thread scaling saturates: 1→2 threads 1.84×, 2→4 only 1.16× more.

Against the bars: **beats Jev's cloud p50 (236–276 ms) by ~2.9×**, still loses to Laya's T4 figure
(33–40 ms) and to our own PyTorch-on-T4 (56.9 ms). The residual **3.9×** is explicable by clock
speed and instruction set — a modern AVX-512 CPU at 4+ GHz would plausibly reach ~30–45 ms.

### Retraction 1: EG2 does *not* lack modern efficiency features

I claimed EG2 was "a decoder stack with full attention at every layer, quadratic". Its own
`config.json` shows **alternating local/global attention at 5:1**, a **1024-token** sliding
window, **GQA/MQA**, and a **512** hidden size — the ModernBERT playbook. Retracted.

The real reason the window does not help us: **our states are far below 1024 tokens**, and within
a single window every token still attends to all preceding tokens up to the bound, so attention
is *effectively quadratic* for our inputs. That is exactly the measured 18–19.7× penalty for a
~500-token state versus a ~20-token one.

### Retraction 2: weight traffic is not the binding constraint

I predicted Q8_0's 1.8× smaller weights (0.31 GB vs 0.56 GB) would give a proportional speedup.
Measured: **89.2 ms vs 92.3 ms — 3% apart.** With mmap the 262,144-entry embedding table (134M of
the model's 271M parameters) is never fully read, and the runtime is **compute-bound** on an
AVX2-only 2.20 GHz part. The bandwidth-floor analysis is retracted as the primary explanation.

### A deployment trap worth recording

`llama-cpp-python` — PyPI 0.3.36 *and* its git master, which still reports 0.3.36 — **cannot load
the EmbeddingGemma 2 GGUFs** (`Failed to load model from file`). It vendors a llama.cpp revision
that predates the architecture. `ggml-org/llama.cpp` master (verified at `bd4eeaa`) loads them
fine, and the GGUF's own `convert.log` confirms standard conversion via
`convert_hf_to_gguf.py`. **Use `llama-server` from source, not the pip bindings.**

### What this changes

The design's central claim — "extremely lightweight" — was an adjective for the whole project and
is now a measurement with a condition: **true with an accelerator or a modern CPU, false on a
2.20 GHz AVX2 core.** Runtime became a first-class design axis, and it was worth 4× — more than
any head or composer decision in this project's history.

---

## Synthesis: what the research says about the three pillars

| Pillar | Verdict from research |
|---|---|
| **EG2 encoder** | Validated; saturates Banking77 (R1), which is why it looked like the head added nothing |
| **Jev typed head** | Not justified for `Choice` *classification* (R3); but **dispatch** is a different problem — routing, escalation and verification are exactly typed decisions over supplied options, and the literature names the quality estimator as the bottleneck (R9) |
| **Karpathy composer** | Prior mixed-to-negative for uniform classification (R2); **dispatch states are multi-item in both regimes** — a bag of tools, a sequence of turns (R9) |
| **Abstention** | APS/RAPS (R4); **safe in dispatch** because the fallback is escalation (R9), unsafe where it suppresses findings (R7) |
| **Benchmark** | **Both** Banking77 and BFCL routing are saturated at ~93% zero-shot (R1, R10). The unsaturated task is **relevance / quality estimation** (R10) |

**The strategic consequence.** The model is not a general assistant and should not try to be
— it is the **dispatcher** whose decisions make a heterogeneous stack behave like one. But
the dispatch *routing* decision is already solved by a frozen encoder, so the contribution
cannot be routing accuracy. It has to be the **quality estimator** — "will the cheap path
suffice?" — which is hard, unsaturated, and named by the routing literature as the critical
factor. That is also where the design's distinctive machinery (calibrated probabilities,
abstention, proper scoring rules) actually applies.

## Consequences for the plan

1. **Close the routing-accuracy track.** Both candidate benchmarks are saturated (R10); no
   head can show a difference, so further work there produces null results by construction.
2. **Open the quality-estimator track.** Re-frame the head as a *relevance* judge — is this
   tool relevant to this request? — with BFCL irrelevance as the negative class. Current
   signal: separation 0.097, 3.5% auto-handled at 90% precision. That is a real, unsaturated
   target with a product outcome (cost saved at fixed precision).
3. **Add a task-selection criterion to the protocol** (already implemented as
   `headroom_check`): measure zero-shot first and refuse to draw conclusions without
   headroom.
4. **Redesign the gate for its domain** (R7, R9): asymmetric where abstention suppresses a
   positive; plain escalation where abstention *is* the fallback.
5. **Composer experiment** (R2, R9): pre-register the prediction, accept a null, and test
   both regimes — but only on a task with headroom.
6. **`Score` track** (R3): CORN/CORAL + RPS + monotonicity check. Ordinal is the one thing
   temperature-scaled cosine cannot do at all, so it is the last standing differentiator for
   a trained component.
7. **Measure latency** (R8): 33–40 ms (Laya) to 236–276 ms (Jev) are the targets.

## Sources

- Lee et al., *Set Transformer*, ICML 2019; Zaheer et al. (2017) — permutation invariance.
- *Pooling and Attention: What are Effective Designs for LLM-based Embedding Models?*,
  arXiv 2409.02727.
- *Why and when should you pool? Analyzing Pooling in Recurrent Architectures*, Findings of
  EMNLP 2020.
- Huang et al., *LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP*, CVPR 2024.
- *Why Linear Probing Works: Non-Vacuous Generalization Bounds via Effective Dimension*, 2026.
- Cao, Mirjalili & Raschka, *Rank Consistent Ordinal Regression* (CORAL), Pattern Recognition
  Letters 2020; Shi, Cao & Raschka, *CORN*, arXiv 2111.08851.
- Romano, Sesia & Candès, *Classification with Valid and Adaptive Coverage* (APS), NeurIPS
  2020; Angelopoulos et al., *RAPS*, 2021.
- SCoRE, *Conformal Selective Prediction with General Risk Control*, arXiv 2603.24704.
