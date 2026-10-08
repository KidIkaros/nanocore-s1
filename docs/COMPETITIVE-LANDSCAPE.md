# Competitive Landscape — NanoCore-S1 Revised Design

*Generated: 2026-10 | Sources: 14 external + 4 internal artifacts | Confidence: High on claims marked verified, Medium where flagged*

Methodology note: this report follows the deep-research workflow — sub-questions, multi-source
search, deep-read of primary sources. Internal numbers are verified against `reports/runs/`
artifacts and the jev-stack repository; external claims are cited inline.

---

## Executive summary

The revised NanoCore-S1 design — frozen EmbeddingGemma 2 via llama.cpp GGUF, temperature-scaled
cosine, typed Choice/Score/Noul, conformal APS/RAPS abstention — occupies a real but narrow
niche: **typed decisions with calibrated probabilities and coverage-guaranteed abstention,
runnable on-device**. Verification of our prior research mostly holds: the Jev benchmark paper
(arXiv:2609.37647) confirms every number we cited, EG2's architecture matches the official model
card, and Qwen3's `enable_thinking` is indeed a caller-set flag. One prior claim needs correction
(MiniCPM-V 4.6's throughput comparison target). The most important new findings are competitive
weaknesses we had not surfaced: **Laya ships a broken temperature (0.1005, clamped with a warning)
and a max-probability drift of 0.85 under option permutation** — two places where our design is
demonstrably better. The largest remaining gaps: conformal abstention and ordinal Score are
unimplemented while Jev already ships equivalents; multimodal is claimed but never measured; and
our benchmarks are saturated, hiding whether any learned component can win.

## 1. Prior research — verification results

| Claim | Status | Detail |
|---|---|---|
| EG2 arch: 24 layers, hidden 512, FFN 2048, sliding window 1024, 5:1 local:global, GQA/MQA | **Verified** | Official model card matches exactly; GGUF metadata confirms `sliding_window 1024` and per-layer pattern ([model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), [HF](https://huggingface.co/google/embeddinggemma-2)) |
| EG2 modular params: 270M text / 440M +vision / 570M +audio / 740M full | **Verified** | Card decomposes as backbone 130M + embedder 140M + vision 170M + audio 300M ([developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)) |
| Qwen3 thinking = caller-selected flag | **Verified** | `enable_thinking` is a per-request hard switch; `/no_think` `/think` are soft switches ([QwenCloud docs](https://docs.qwencloud.com/developer-guides/text-generation/thinking)) |
| MiniCPM-V 4.6 ≈1.3B (SigLIP2-400M + Qwen3.5-0.8B), edge-first | **Verified, one correction** | ~1.5× throughput claim is **vs its own Qwen3.5-0.8B backbone**, not Gemma — earlier note misattributed it. Also 4.6 ships **two checkpoints** (Instruct + Thinking), replacing 4.5's single hybrid flag ([HF card](https://huggingface.co/openbmb/MiniCPM-V-4.6), [cookbook](https://opensqz.github.io/MiniCPM-V-CookBook/site/en/v4.6/overview.html)) |
| Jev Banking77: 79.7% → 96.3% @50% coverage; Qwen 94.5% @50% | **Verified** | [arXiv:2609.37647](https://arxiv.org/abs/2609.37647) (Deußer, Sparrenberg & Sifa) — confirmed verbatim in paper text |
| Jev SST-5: ρ = 0.851 | **Verified** | Paper §4.4: "ρ=0.890 for STS-B, 0.851 for SST-5, 0.841 for […]" |
| Laya Banking77: 0.425; Jev self-reported 0.870 | **Verified** | `jev-stack/docs/adr/0001-laya-stack.md`; `reports/runs/benchmark/analysis.md` |
| Conformal APS/RAPS is standard machinery | **Verified** | Reference impl ([aangelopoulos/conformal_classification](https://github.com/aangelopoulos/conformal_classification)), [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) `RAPSConformityScore`, TorchUncertainty `ConformalClsRAPS` — our planned score formula matches the RAPS reference |
| RAPS/APS rarely seen in production | **Supported** | "Applying conformal prediction in real-world industrial use cases is rare" — Husqvarna/MAPIE study ([PMLR v204](https://proceedings.mlr.press/v204/uddin23a.html)); LLM conformal abstention is active research (CAP, CIC, ExAUL) but not productized |

## 2. The competitive map

The "typed decision" space now has a clearer shape: two incumbent typed-decision systems
(Jev, Laya), a cluster of **generative-model routers** (RouteLLM, NotDiamond, Martian,
OpenRouter), and capability-model competitors (Qwen3, MiniCPM-V) that solve different problems.

| | NanoCore-S1 (revised) | Jev (TypeSafe AI) | Laya | RouteLLM | NotDiamond | Qwen3 hybrid | MiniCPM-V 4.6 |
|---|---|---|---|---|---|---|---|
| **What it is** | Typed decision layer | Commercial typed decisions | Open typed decisions | LLM router | Hosted routing layer | Mode flag on an LLM | Small MLLM |
| **Output** | Choice/Score/Noul + prediction set | Same triple | Same triple | Model choice | Model recommendation | Text | Text |
| **Probabilities** | Temp-scaled cosine + conformal | Calibrated (indep.-verified) | **Shipped temp 0.1005 invalid** | Routing probs | Cost-quality tradeoff | None surfaced | None |
| **Abstention** | Conformal APS/RAPS (planned) | Selective prediction (proven: 79.7→96.3%) | Threshold | None | None | Caller flag | None |
| **Multimodal** | text/image/audio (unmeasured) | text | text only | text | text | text | vision+video+audio |
| **On-device** | yes, GGUF ~310MB | no — cloud, 0.36s mean | yes | self-host | no (+100–150ms hop) | self-host | yes (phones) |
| **Banking77** | 93.4% (Q8_0) | 79.7% | 42.5% | n/a | n/a | ~? (94.5% @50% cov) | n/a |

### New competitor findings

**RouteLLM** (LMSYS, Apache-2.0) — the closest *academic* precedent for learned routing:
trained routers between strong/weak LLMs, claiming ~85% cost reduction at 95% GPT-4 quality
([paper](https://arxiv.org/abs/2406.18665), [repo](https://github.com/lm-sys/RouteLLM)).
**But a reproduction found the authors' BERT routers overfit the majority class — macro F1
0.23–0.35** ([Liqs-v2 reproduction](https://github.com/Liqs-v2/RouteLLM)). Learned routers are
fragile — this validates retiring our trained Choice head rather than mourning it.

**NotDiamond / Martian** — hosted routing-intelligence layers. NotDiamond trains a custom
router (random forest on embeddings) on your eval data, adds **100–150ms latency**, charges
$0.05/M tokens routed, claims 20–40% savings — vendor-reported, unverified
([docs](https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post),
[comparison](https://www.orcarouter.ai/blog/not-diamond-vs-martian)). A neutral benchmark
(RouterArena) **ranks a leading commercial router 12th** — vendor routing numbers are inflated
([dreaming.press analysis](https://dreaming.press/posts/2026-06-21-routellm-vs-notdiamond-vs-martian.html)).

**xLAM / ToolACE** — Salesforce's action models top BFCL (xLAM-1b: 78.94%) but that metric is
**call generation accuracy (AST match), not tool selection** — our BFCL number measures
choosing the right tool from a candidate set. Different task; do not compare headline numbers
([xLAM paper](https://arxiv.org/html/2409.03215v1), [ToolACE](https://arxiv.org/html/2409.00920)).

## 3. Design smells — us vs. them

### Smells we have that competitors don't (our gaps)

1. **Multimodal is claimed, never measured.** EG2's vision (+170M) and audio (+300M) encoders
   load, but no non-text decision has ever been benchmarked. It's our best differentiator —
   neither Jev nor Laya can do it — and it's currently a promise, not a result.
2. **Score and the conformal gate are unimplemented.** Jev already ships both equivalents:
   ordinal rubric scoring (ρ=0.851 SST-5) and selective prediction (79.7→96.3% @50% coverage).
   Our versions are designs, not artifacts.
3. **No moat around the asset.** The 93% accuracy is EmbeddingGemma 2's, not ours. Our value-add
   is the typed interface + calibration + conformal guarantee — which is real but thin until
   the gate exists.
4. **Saturated benchmarks.** 92–93% ceilings on Banking77/BFCL leave ~7% headroom — too tight
   to demonstrate any learned component's value (ADR-0011 encodes this as a gate).
5. **On-device is asserted, not shown.** Measured only on a Kaggle Xeon; no mobile-class ARM
   or consumer-hardware number exists. MiniCPM publishes phone benchmarks; we publish a server CPU.

### Smells they have that we don't (our verified advantages)

1. **Laya's calibration is broken at ship time.** Temperature `choice:11+ = 0.1005` is invalid
   and clamps to 0.5 — *"a 0.24 top probability is published as 0.99"* (jev-stack AGENTS.md;
   the checkpoint emits a RuntimeWarning on load). Our temperature-fit + conformal path is
   honest by construction.
2. **Laya's probabilities move 0.85 under option permutation.** Max-prob drift 0.8517 with the
   argmax stable only ~98.6% of the time (benchmark analysis). Since any gating stack reads
   probabilities, not argmax, this is unsafe to gate on. Our order invariance is at the
   *distribution* level: 2.38e-07, zero flips, over 2–37 options.
3. **Jev's numbers are inflated ~7 points by the vendor.** Self-reported 87.0% Banking77 vs
   independently measured 79.7%. Our numbers are our own Kaggle artifacts.
4. **Jev is cloud-only: 0.36s mean latency, $ per request.** Our 89ms on-device has no network
   hop and no marginal cost.
5. **Learned routers don't reliably reproduce.** RouteLLM's own routers hit macro F1 0.23–0.35
   under reproduction. Our scorer is 1 temperature parameter — nothing to overfit.
6. **Nobody ships conformal-guaranteed prediction sets.** Jev has selective prediction; routers
   have heuristics; conformal abstention for LLMs is still a 2025–26 *research* topic (CAP, CIC,
   ExAUL — [arxiv 2607.04430](https://arxiv.org/html/2607.04430v1), [2506.14067](https://arxiv.org/html/2506.14067),
   [2606.29054](https://arxiv.org/html/2606.29054v1)). Heuristic abstention "misses user-specified
   risk targets by 7.5–12.5%" — the exact gap conformal closes.

### A fair admission

Order invariance is **not** a differentiator vs Jev — the paper confirms rotating options leaves
Jev's accuracy unchanged (per-option scoring, like ours). It's a differentiator vs Laya only
(91.7% sequential layout). Claim it narrowly.

### Convergent evidence — the "glial" lineage

A separate research tradition independently lands on our architecture: **freeze a substrate,
modulate at the edge**. CAM-Brain (de Garis, ATR 1993–2001) gave glial cells the modulation/
maintenance role over an evolved CA neural substrate. The modern neuron–glia literature makes
the same split measurable: ANAN (NeurIPS 2024 workshop) matches CNN fine-tuning by optimizing
**4 astrocyte-modulation parameters on a frozen network**; Kozachkov et al. (PNAS 2023) show
neuron–astrocyte networks implement the Transformer block's core computation; GliaNet (CVPR
2025) uses glia-driven structure learning for SoTA accuracy with fewer parameters; cortical
cell-type motifs improved ViT OOD generalization ~20% as a fixed module (PNAS 2025).

Our equivalent, arrived at by measurement rather than bio-inspiration: frozen EG2 (substrate)
+ a thin adaptive layer — fitted temperature, conformal quantile, meta-routing, drift monitor
(the glial/homeostatic functions). Two consequences:

- **Cross-validation**: the architecture thesis has independent support from a field that
  doesn't know we exist.
- **What we deliberately don't adopt**: glial units *inside* the network target training
  dynamics and structure learning — problems a frozen encoder doesn't have. The one idea
  worth borrowing is already planned: glial modulation integrates slow-timescale activity
  history — the CAP per-instance-α upgrade path is exactly that, driven by cache statistics
  rather than the current input alone.

## 4. What needs the most work — ranked

| # | Gap | Why first | Evidence |
|---|---|---|---|
| 1 | **Conformal gate implementation** | The claimed differentiator doesn't exist yet. Jev's selective prediction already proves the value (79.7→96.3%); conformal's *guarantee* is what nobody ships | ADR-0009; paper §selective |
| 2 | **Multimodal verification** | Our only capability Jev/Laya categorically lack — never exercised once. One image- or audio-decision benchmark settles whether the claim is real | EG2 card; `reports/runs/` (none exist) |
| 3 | **Ordinal Score (CORN/RPS)** | The only planned trained component; Jev sets the bar at ρ=0.851 SST-5 and explicitly calls SST-5 "solved considerably better" ordinally — our exact thesis | ADR-0010; paper §4.4 |
| 4 | ~~A headroom-positive benchmark~~ | **Done** — GoEmotions ran (headroom 0.713); the fitted head won by +0.168 macro-F1 at the fine-tuned-BERT frontier, and won at every data scale down to n=100 | `s1_goemotions`, `s1_goemotions_head`, `s1_goemotions_scaling` |
| 5 | **Backend productization** | The llama.cpp recipe is proven in a kernel but not in `src/`; needs pinned commit, health probe, contract tests | Plan Stage 1 |
| 6 | **Sequence-length cap benchmark** | Cheap; closes the 18× long-state penalty | ADR-0012 |

## Key takeaways

- **Our competitive position is better than we reported** — Laya's shipped calibration and
  probability drift are verifiable weaknesses, and our order-invariance and honest calibration
  are advantages worth stating plainly.
- **Our differentiation is narrower than claimed** — abstention isn't novel (Jev's selective
  prediction works); the *conformal guarantee* and *on-device multimodality* are the parts
  nobody else has. Both are currently unimplemented or unmeasured.
- **The "just use the encoder" risk is real** — our accuracy is EG2's. Until the conformal
  gate and Score land, our defensible surface is thin.
- **The next benchmark should be GoEmotions-class** (fine-grained, ~28 labels, everyone
  degrades) — that's where a trained component can actually demonstrate value, and where
  Jev's macro-F1 of 0.243 leaves real headroom.

## Sources

1. [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647) — Deußer, Sparrenberg & Sifa; the independent benchmark (37 datasets, 346k requests)
2. [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2) — official architecture
3. [EmbeddingGemma 2 developer guide](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/) — modular params, MRL
4. [QwenCloud thinking docs](https://docs.qwencloud.com/developer-guides/text-generation/thinking) — `enable_thinking` semantics
5. [MiniCPM-V 4.6 HF card](https://huggingface.co/openbmb/MiniCPM-V-4.6) + [cookbook](https://opensqz.github.io/MiniCPM-V-CookBook/site/en/v4.6/overview.html)
6. [RouteLLM](https://arxiv.org/abs/2406.18665) + [repo](https://github.com/lm-sys/RouteLLM) + [reproduction](https://github.com/Liqs-v2/RouteLLM)
7. [NotDiamond docs](https://docs.notdiamond.ai/reference/token_model_select_v2_modelrouter_modelselect_post), [OrcaRouter comparison](https://www.orcarouter.ai/blog/not-diamond-vs-martian), [dreaming.press analysis](https://dreaming.press/posts/2026-06-21-routellm-vs-notdiamond-vs-martian.html)
8. [OpenRouter Model Router Benchmarks](https://openrouter.ai/blog/announcements/model-router-benchmarks/)
9. [xLAM](https://arxiv.org/html/2409.03215v1), [ToolACE](https://arxiv.org/html/2409.00920), [xLAM-1b-fc-r](https://huggingface.co/Salesforce/xLAM-1b-fc-r)
10. [aangelopoulos/conformal_classification](https://github.com/aangelopoulos/conformal_classification), [MAPIE RAPS](https://github.com/scikit-learn-contrib/MAPIE), [Husqvarna PMLR v204](https://proceedings.mlr.press/v204/uddin23a.html)
11. Conformal-for-LLM research: [CIC](https://arxiv.org/html/2607.04430v1), [ExAUL](https://arxiv.org/html/2506.14067), [CAP](https://raw.githubusercontent.com/mlresearch/v304/main/assets/tayebati26a/tayebati26a.pdf), [CRC bounds](https://arxiv.org/html/2606.29054v1)
12. Internal: `jev-stack/docs/adr/0001-laya-stack.md`, `jev-stack/reports/runs/benchmark/analysis.md`, `jev-stack/reports/runs/phase_b_interaction/analysis.md`, `jev-stack/AGENTS.md`
13. Glial/neuromodulatory lineage: [CAM-Brain](https://www.jstage.jst.go.jp/article/sicejl1962/33/2/33_2_128/_article/-char/en) + [CBM](https://dl.acm.org/doi/10.1023/A:1011286308522); [ANAN](https://research.latinxinai.org/papers/neurips/2024/pdf/Ana_Ribas-Rodriguez.pdf); [neuron–astrocyte Transformers](https://www.pnas.org/doi/10.1073/pnas.2219150120); [GliaNet](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_GliaNet_Adaptive_Neural_Network_Structure_Learning_with_Glia-Driven_CVPR_2025_paper.pdf); [cortical motifs + neuromorphic](https://www.pnas.org/doi/abs/10.1073/pnas.2504164122)

## Methodology

10 search queries across web sources; deep-read of the Jev benchmark paper (arXiv:2609.37647)
text, the EG2 model card and GGUF metadata, and jev-stack's internal reports. Sub-questions:
prior-claim verification, typed-decision competitors, LLM-routing competitors, conformal in
production, encoder-efficiency recheck. Gaps acknowledged: Martian's internals are closed;
RouterArena's full leaderboard wasn't read directly; MiniCPM-V 4.0's iPhone TTFT claim from the
earlier session was not re-verified in this pass.
