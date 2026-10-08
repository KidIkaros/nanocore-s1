# State of the Project — what the experiments established

Date: 2026-10-07. **Status: consolidation.** Six Kaggle runs, four on the decision model
itself. This document states what is established, what was withdrawn, what has never been
exercised, and what the evidence implies. It is written so that the next decision can be
made from a clean position.

Companions: `ARCHITECTURE-DECISION-MODEL.md` (components and validation matrix),
`RESEARCH-NOTES.md` (R1–R10), `ARCHITECTURE-SHARPENING.md` (proposals and outcomes),
`adr/` (decisions), `reports/runs/` (raw artifacts for every run below).

---

## 0. The one-paragraph answer

EmbeddingGemma 2 zero-shot reaches **~93%** on both benchmarks we tried — Banking77 intent
classification (92.92%) and BFCL tool routing (93.07%). A frozen encoder plus a
temperature-scaled cosine is therefore a **complete solution** for both tasks, and no trained
head can show an improvement because there is nothing left to win. The one design property
that is solidly validated is **order invariance**. The abstention gate is implemented but its
signal is weak (separation 0.097, 3.5% auto-handled at 90% precision). The composer and the
ordinal `Score` primitive have **never been exercised**. The binding constraint on every
architectural claim was **benchmark saturation**, not model quality — until the GoEmotions
run found a task with real headroom (macro-F1 0.287, AUROC 0.824) and exercised `Noul` for
the first time, tying Jev's tuned micro-F1 (0.386 vs 0.387).

---

## 1. Claim ledger — established, with evidence

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 1 | EG2 zero-shot is ~93% on Banking77 and on BFCL tool routing | 92.92% / 93.07%; both flagged `usable: false` by `headroom_check` | **high** |
| 2 | Order invariance holds on real embeddings, fixed and variable option sets | Banking77: 2.38e-07, 0 flips / BFCL: 1.19e-07, 0 flips over 2–37 options | **high** |
| 3 | A temperature-scaled cosine matches the trained head on Banking77 | cosine+τ 0.380 / 0.117 / 0.045 vs head 0.377 / 0.116 / 0.046 (1 param vs 59,136) | **high** |
| 4 | The trained head is *worse* than zero-shot on dispatch routing | 90.84% vs 93.07%; no winners on any metric from `compare_to_best` | **high** |
| 5 | kNN wins on classification and collapses on routing | 93.6% on Banking77 vs 71.5–73.1% on BFCL | **high** |
| 6 | The prompt pairing materially affects results, and the incumbent was wrong | `SearchQuery`/`Document` gains +0.94 kNN, +0.50 head, +0.49 zero-shot; symmetric is *worst* | **high** |
| 7 | Head initialization from option embeddings has no effect | 93.12% vs 93.11% random, log 0.377 both | **high** |
| 8 | Normalizing fingerprints + training is harmful | accuracy 93.11% → **79.66%** | **high** |
| 9 | The abstention gate's relevance signal is weak | separation 0.097; 3.46% auto-handled at 90% precision | **medium** (single split) |
| 10 | The legacy decoder path is not viable | 20 steps at loss 6.7661 vs `ln(867)=6.7650`; inert `window_pattern`; decorative `head_dim`; synthetic decision accuracy; ~150 ms/token vs a claimed 15–35 ms | **high** |
| 11 | The fingerprint head ignores option text entirely | two option prompts produced byte-identical head results; `_scores_batch` never reads `option_embeddings` | **high** |
| 12 | **"Extremely lightweight" holds on a T4 (56.9 ms/decision, inside Jev's 236–276 ms cloud p50) and fails on CPU (420 ms)** | same harness, one variable: the device. Encoder is 86.5% (CPU) / 97% (T4) of the decision | **high** |
| 13 | Batching gives 137× on GPU and nothing on CPU; long states cost 102× more on CPU | 53.6→2.95 ms/text vs a flat ~400 ms; 6,595 ms vs 64.6 ms for a ~500-token state | **high** |
| 14 | Matryoshka buys nothing for decision latency | scoring 77 options costs 0.94 ms against a 420 ms decision — 0.2% | **high** |
| 15 | **llama.cpp is 4.08× faster than PyTorch on CPU, with quality preserved** | 89.2 ms vs 363.5 ms short; 1,754.8 vs 6,595 ms long; 9.9 vs 2.4 texts/s; Banking77 zero-shot **93.41%** under Q8_0 vs 92.92% | **high** |
| 16 | The Python bindings cannot load these GGUFs | `llama-cpp-python` 0.3.36 (PyPI *and* git master) fails; `ggml-org/llama.cpp` master (`bd4eeaa`) loads them | **high** |
| 17 | Weight traffic is **not** the binding constraint at batch 1 | Q8_0 (0.31 GB) 89.2 ms vs BF16 (0.56 GB) 92.3 ms — 1.8× less traffic, 3% faster | **high** |
| 18 | **A headroom-positive benchmark exists: GoEmotions** | zero-shot Noul battery macro-F1 0.287 (headroom 0.713, gate `usable`); macro-AUROC 0.824 — ordering strong, placement weak | **high** |
| 19 | Tuned micro-F1 on GoEmotions **ties Jev** | 0.386 vs Jev's 0.387; above Qwen 0.317 and Gemma 0.268 (Jev paper Table 4 protocol) | **high** |
| 20 | `Noul` works as a scored primitive | 28 binary questions/comment on 5,427 test items — natural multilabel formulation | **high** |
| 21 | Bare emotion names beat descriptive label templates | AUROC 0.824 vs 0.789 — extra words dilute the signal | **medium** (single task) |
| 22 | **A task-fitted head wins decisively on a headroom-positive task** | five-arm ladder on GoEmotions: Platt 0.287 → `mlp_emb` 0.455 macro-F1 (+0.168, ±0.003 across seeds); every learned arm beats the oracle threshold bound (0.297) | **high** |
| 23 | The dominant head win is label co-occurrence | +0.111 of +0.168 comes from joint linear fitting on 28-dim scores alone; raw state adds +0.036 | **high** |
| 24 | The learned gain is real ranking, not thresholds | macro-AUROC climbs 0.824 → 0.901 → 0.908 → 0.914 → 0.922 with each arm | **high** |
| 25 | Task-fitted `mlp_emb` reaches the fine-tuned-BERT frontier | 0.455 macro / 0.525 micro vs ~0.46 for trained BERT-era systems; above Jev tuned 0.353 and Qwen tuned 0.323 (different regime — task-fitted vs generative) | **high** |
| 26 | **The head beats Platt at every data scale measured** | scaling curve n=100…43k on cached real embeddings: mlp_emb wins at n=100 (+0.08); Platt saturates at ~0.31 by 16k while heads still climb; `linear_emb` is the sweet spot below ~16k | **high** |
| 27 | **OOS inputs are separable by score magnitude** | CLINC150: `max_sim` AUROC 0.934 — the in-schema check meta-routing needs (ADR-0013) | **high** |
| 28 | **Conformal coverage holds at the unsharpened temperature** | CLINC150: 0.913 @ α=0.10, T=1.0 | **high** |
| 29 | **`clarify` resolves with measured precision** | P(true intent ∈ set \| \|set\|≤3) = 0.975, on 27.9% of in-scope inputs | **high** |
| 30 | **Log-loss temperature fitting can destroy conformal sets** | kernel fit T=0.02 → q̂=0.9975, mean set 56.7, trivial 0.997 coverage; scorer and gate need separate temperature treatment | **high** — the sharpening trap in a new guise |
| 31 | **Set-size action triggers fail on flat softmaxes** | `s1_policy`: `answer`-on-singleton unreachable at T=1.0 (top prob ~0.008 < q̂=0.0235); ambiguity policy lost −0.26 to threshold gating; triggers must key on calibrated confidence | **high** |
| 32 | **A linear TaskHead nearly solves 150-way CLINC150** | 0.970 in-scope, OOS AUROC 0.968 via head max_prob; but degenerate conformal sets (q̂≈1.0, mean set 29.7) — overconfident heads need their own set calibration | **high** |
| 33 | **The ambiguity policy is scorer-dependent** | `s1_policy_v2` (confidence triggers, dual temps, 3-way splits): on flat cosine it still loses 0.482 vs 0.719 — a 90%-precision τ admits only ~33% of a flat 150-way softmax and `clarify` fired on 1% of items; honest uncertainty on wide schemas *is* escalation | **high** |
| 34 | **The identical policy on a task head resolves 0.918** | +0.20 over threshold gating, +0.30 over always-answer; head also improves the in-schema check (86.9% OOS at 6.4% FPR vs cosine 63.0% at 3.9%). The full 3-action policy needs a confidence-meaningful scorer — head is prerequisite on wide schemas | **high** |
| 35 | First non-degenerate conformal coverage on a policy run | 0.914 @ α=0.10 with dual temperatures (T_prob floored 0.25, T_set=1.0); head sets still degenerate (q̂≈1.0, coverage 0.9996) — 4th sharpening-pathology sighting, per-scorer set calibration is required | **high** |

---

## 2. Claims made and withdrawn — the corrections

Recorded because the pattern matters more than the individual errors.

| Claim we published | What it actually was | Corrected to |
|---|---|---|
| "Naive retrieval calibration fails badly: cosine log 3.800, ECE 0.907" | **my temperature grid's lower bound** — three baselines fitted it | cosine+τ reaches log **0.380**, ECE 0.045 |
| "The head beats the baseline on Brier" | **verdict compared against the weakest baseline** of four | against the best baseline it loses on Brier, ECE and accuracy |
| "Headroom is a Banking77 problem" | assumed task-specific | BFCL routing is **also** saturated at 93.07% |
| "Initialize the head from option embeddings (highest expected value)" | never tested when proposed | tested: **no effect** |
| "Normalize fingerprints per the Zero-Bias paper" | paper's remedy presumes a norm-controlled parameterization | applied as a forward-pass transform: **harmful** |
| "The 90%-precision gate" | saturates when base accuracy exceeds 90% | gate must be set above the base error rate |

**Three of the six came from the measuring instrument, not the model.** That is why the
protocol now exists as tested code with regression tests rather than as repeated care.

---

## 3. Never exercised

| Capability | State |
|---|---|
| `EmbeddingComposer` | implemented (28,315,392 params), contracts tested, **never trained** |
| `Score` (ordinal) | schema tested, **never run on real data**; currently uses cross-entropy, the loss the ordinal literature says discards ordering |
| `Noul` | **exercised** — GoEmotions battery + trained-head ladder (§1 rows 18–25) |
| Multimodality | vision: 100-image smoke test. audio/video: **untested** |
| End-to-end deployment | never attempted; every number comes from an offline split |

---

## 4. The binding constraint: saturation

This is the strategic finding, and it is not about our code.

If a frozen 270M encoder already answers ~93% of a task, then:
- a trained head has **7% of headroom** to work with;
- differences of 0.2–0.5 points are the whole available signal;
- a null result is **uninformative** — it may mean "no head can help here", not "our head is bad".

Both benchmarks we selected hit this wall. The practical consequence for any future
experiment: **measure zero-shot first, and refuse to draw architectural conclusions without
headroom.** `protocol.headroom_check` now enforces that mechanically.

**The wall has now been breached once — and the fight is settled.** GoEmotions
(`s1_goemotions`) — a 28-label Noul battery — returns zero-shot macro-F1 **0.287** with
macro-AUROC **0.824**: strong ordering, weak placement. The follow-up ladder
(`s1_goemotions_head`) shows the gap is *not* a calibration problem — Platt already sits
at the score-thresholding ceiling (oracle bound 0.297) — and a task-fitted head wins
decisively: `mlp_emb` reaches **0.455 macro / 0.525 micro / 0.922 AUROC** (+0.168 over
Platt, ~the fine-tuned-BERT frontier). The dominant signal is label co-occurrence
(+0.111 from a linear map on the 28 scores alone); the raw state adds +0.036. The head is
label-schema-bound and data-hungry (43k examples) — a **per-deployment** component, not a
universal one — but it is the first trained component with a measured win.

---

## 5. What the evidence implies for the architecture

**Supported by evidence:**

```
frozen EmbeddingGemma 2  →  typed interface (Choice / Score / Noul)
                         →  temperature-scaled cosine scoring
                         →  abstention gate
```

No trained head. This is not a defeat: it is a 270M-parameter, training-free dispatcher that
routes at 93% with calibrated probabilities, and the whole cost argument (33–40 ms to decide
versus ~2 s to generate) survives intact.

**Not supported:** the trained head for `Choice` and for routing *on saturated tasks*.
Three experiments, no win over the best baseline on any proper metric — at 93% ceilings
there is nothing to learn. **Supported, newly:** a task-fitted head for `Noul`-family
tasks where labeled data and headroom coexist (§4): +0.168 macro-F1 on GoEmotions, at
parity with fine-tuned-BERT-era results. The two findings are consistent — the head
loses where the task is saturated and wins where it is not.

**Newly qualified by measurement:** the latency claim holds **only with an accelerator**
(56.9 ms on a T4 versus 420 ms on CPU). The design is accelerator-requiring, which weakens
"on-device" unless the device has an NPU. And a *larger* competitor is faster on the same
hardware — Laya's 421M ModernBERT does typed decisions in 33–40 ms on a T4 against our
270M's 56.9 ms — which makes **encoder efficiency a design axis we have never examined**.

**Still open, and the only places a trained component could earn its place:**

1. **Ordinal `Score`** — temperature-scaled cosine has no notion of order at all. This is the
   strongest remaining candidate and the only capability that is categorically different.
2. ~~Multilabel `Noul` heads~~ — **resolved**: the fitted head won, at every data scale
   down to n=100 (§1 rows 22–26). Open follow-up is head→conformal composition only.
3. **Composition** of multi-item states — unproven, prior leans negative for
   classification-shaped tasks, positive when relevance is uneven.

---

## 6. Decisions taken

The four options below were the open question when this document was first written. **They have
since been decided and recorded** in `adr/`:

| Option | Outcome | ADR |
|---|---|---|
| **A. Ship the interface, not a model** | **Taken** — cosine scoring replaces the head for `Choice`; the encoder is served through llama.cpp | [0007](../docs/adr/0007-llamacpp-runtime.md), [0008](../docs/adr/0008-cosine-scoring.md) |
| **B. Build the ordinal `Score` track** | **Taken** — `Score` is the only trained component, with CORN/CORAL + RPS | [0010](../docs/adr/0010-ordinal-score.md) |
| **C. Find harder tasks** | **Scoped as a follow-up**, and made mandatory by the headroom rule | [0011](../docs/adr/0011-headroom-check.md) |
| **D. Test the composer** | **Deferred** — parked with an explicit revisit condition | [0003](../docs/adr/0003-bidirectional-composer.md) (deprecated) |

Abstention moved from a fitted threshold to conformal prediction sets
([0009](../docs/adr/0009-conformal-abstention.md)), superseding ADR-0005, and the encoder now
enforces a hard input-length cap ([0012](../docs/adr/0012-input-length-cap.md)).

The revised architecture is described in `ARCHITECTURE-DECISION-MODEL.md` §2.

---

## 7. Instrument status

The measuring apparatus is now in better shape than the model.

| Artifact | State |
|---|---|
| `src/decision/protocol.py` | one temperature fitter (two-stage, edge-safe), one metric block, ragged support for variable option sets, headroom check, dispatch metrics, best-baseline verdicts |
| `tests/test_decision_protocol.py` | 26 tests, including regression tests for both published errors and the residual grid-edge defect |
| **149 tests** | pass locally; suite re-run inside `s1_verify` on the Kaggle image |
| `notebooks/s1_tests` | reusable CPU fast-fail harness — no GPU, no internet, no quota |
| Execution boundary | ADR-0006; local is reading/writing/static checks only. Violated once, structurally fixed |

Every number in this document comes from a Kaggle run with artifacts in `reports/runs/`.

---

## 8. Quality control, benchmarking, and evaluation

Added after the fact — the experiment log had evidence but no standing harness.
This section defines what "checked" means going forward.

### The QC harness

`notebooks/s1_verify/` is the single verification kernel — one GPU session that runs,
in order: the full test suite on the Kaggle image → real encoder load → CLINC150
end-to-end through the shipped classes → policy evaluation in both scorer modes →
deployable-bundle round-trip → the live typed interface. It asserts against the
measured bands below; a regression in shipped code fails the run, not a discussion.

**One kernel per concern, iterated by version** (AGENTS.md) — `s1_verify` is the
recurring QC run; experiment families keep their own notebooks; no per-revision
kernels.

### The standing benchmark suite

| Dataset | What it establishes | Status |
|---|---|---|
| Banking77, BFCL | saturated zero-shot routing (~0.93) — the "no head needed" regime | measured |
| GoEmotions | headroom-positive multilabel; head ladder + data-scaling curve | measured |
| CLINC150 | OOS detection + ambiguity policy + the degenerate-set trap | measured |
| SST-5 | ordered `score` — CORN vs zero-shot level names | wired into s1_verify |
| Audio/vision | the multimodal differentiator | **gap — smoke test only** |
| Device latency | the on-device claim (llama.cpp, real hardware) | **gap — Kaggle-measured only** |

### Regression bands (fail the verify run if missed)

- resolved fraction: cosine leg 0.48±0.05, head leg 0.92±0.06 (CLINC150)
- in-scope set coverage: 0.85–0.95 at α=0.10 — **a band, not a floor**
  (≥0.99 means degenerate sets, the fourth sharpening sighting)
- bundle round-trip: 200/200 identical decisions post save→load
- calibration gates on proper scoring rules (log/Brier); ECE is report-only

### Coverage gaps (ranked)

1. **Multimodal eval on real data** — the capability no competitor has, still
   unproven on a scored task.
2. **True-device latency** — llama.cpp numbers are from a Kaggle Xeon, not a
   consumer device.
3. **A deployment soak** — every number is an offline split; nothing has run
   under real call patterns.
4. **Ambiguous-phrasing eval** — `clarify` fired on 1% of CLINC items; it needs
   a dataset with genuine in-schema ambiguity to be judged.
