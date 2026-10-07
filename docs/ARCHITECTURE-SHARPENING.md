# Architecture Sharpening — research-backed proposals

Date: 2026-10-07. Status: **proposals, not decisions.** Each item states the change, the
evidence, the expected effect, the cost, and how to falsify it.

Companion to `ARCHITECTURE-DECISION-MODEL.md` (what exists and what is validated) and
`adr/` (decisions taken). Sources: the project library at
`~/Documents/Library/technical/ai-ml/` plus targeted online research.

---

## Outcomes (kernel `s1_sharpening`, 2026-10-07)

| Proposal | Outcome |
|---|---|
| **S3** head init from option embeddings | **REJECTED** — identical to random init to three decimals (93.12% vs 93.11%, log 0.377 both). The optimizer converges to the same solution from either start, so the initialization washes out. |
| **S5** normalize fingerprints | **REJECTED as implemented** — training the normalized head dropped accuracy from 93.11% to **79.66%**. |
| **S7** Matryoshka 256-d | untested (kernel died before that cell) |
| **S9** protocol fixes | **ADOPTED, and it overturned a conclusion.** Widening the temperature grid took cosine+τ from log 3.800 to **0.380**. |

**The consequence is bigger than any single proposal:** with a properly fitted temperature,
a **one-parameter** cosine classifier reaches log 0.380 / Brier 0.117 / ECE 0.045, against
the 59,136-parameter trained head's 0.377 / 0.116 / 0.046. **For `Choice`, the trained head
buys essentially nothing.** The architecture's justification must therefore come from what
cosine scoring cannot do — ordinal `Score`, multi-item composition, or inference cost at
scale — not from the head.

---

## 0. What the measurements say is weak

| Weakness | Evidence |
|---|---|
| The head does not beat kNN-5 on accuracy, Brier or ECE | calibration v3; kNN-5 93.64% vs fingerprint 93.11% |
| Head initialization is random, throwing away a 92.92% zero-shot prior | implementation |
| Abstention is a fitted heuristic with no guarantee, and saturates at 90% precision | calibration v3, disclosed flaw 3 |
| The composer is order-**sensitive** (RoPE) for set-like states | `composer.py` |
| A separability diagnostic was built and never used on real data | `fingerprint_distance_matrix`, unrun |
| "Extremely lightweight" has never been measured | no latency or dimension study |
| Heads train on one-hot labels, not the RLCD objective the design cites | implementation |

---

## S1. Abstention → conformal selective prediction

**Change.** Replace the fitted max-score threshold with split-conformal selective
prediction: calibrate a quantile on held-out scores so the **error rate among accepted
decisions** is controlled at a chosen level α, with a finite-sample, distribution-free
guarantee.

**Evidence.** SCoRE, *Conformal Selective Prediction with General Risk Control*
(arXiv 2603.24704): derives trust/abstain decisions with "finite-sample error control"
among trusted cases, "ensured by data exchangeability without requiring any modeling
assumptions." Selective conformal inference with false-coverage-rate control
(arXiv 2301.00584) is the sibling formulation. Angelopoulos & Bates give the review.

**Why it matters here specifically.** A grep of the reference paper this project is
built on — *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses*
(JEV/Laya) — returns **zero occurrences of "abstain" or "abstention."** That paper
handles calibration thoroughly (temperature scaling, per-question-type temperatures,
ECE and Brier reporting) but provides no principled abstention mechanism. Our
abstention gate is therefore a genuine differentiator, and conformal selective
prediction would put it on a theoretical footing the category's own literature lacks.

**Expected effect.** Turns the gate from "we fitted a threshold once" into "the
accepted decisions carry a stated error bound." Also dissolves the 90%-precision
saturation problem: α is chosen relative to the base error rate, so it cannot be
trivially satisfied.

**Cost.** Low. Needs the calibration split we already hold and a quantile computation.
CPU-only over cached embeddings.

**How to falsify.** Report the *realized* error rate among accepted decisions against
nominal α across several α values. If realized ≫ nominal, exchangeability is violated
and the guarantee does not hold.

**Risk.** Conformal validity assumes exchangeability between calibration and test. Our
split is iid, but a deployed domain shift breaks the assumption — the same caveat the
JEV paper raises for per-domain temperature scaling. Mitigation: state the assumption,
monitor shift, recalibrate per domain.

---

## S2. Composer → permutation-invariant set encoder; make order-sensitivity explicit

**Change.** Three parts:

1. **Drop RoPE for set-like states.** A state is a *set* of evidence items; positional
   encoding makes `[a, b]` differ from `[b, a]` for no semantic reason. Use Set
   Attention Blocks instead.
2. **Replace masked-mean pooling with Pooling by Multihead Attention (PMA)** — a
   learned seed vector attends over the items, so the aggregator is learned rather than
   fixed.
3. **Keep role/order encoding as an explicit, ablatable option** for states where order
   is real (conversation turns), rather than baking it in.

**Evidence.** Lee et al., *Set Transformer* (ICML 2019): SAB/ISAB are
permutation-equivariant and PMA pools permutation-invariantly; ISAB reduces
self-attention from quadratic to linear in set size. Zaheer et al. (2017) proved all
permutation-invariant functions can be written as `ρ(pool(φ(x)))` with sum pooling — so
**our masked mean is a special case of the Set Transformer family**, and attention's
job is to learn a better `ρ∘pool`.

**A consequence worth recording.** Because `φ` — the encoder — is **frozen**, the
composer can only *recombine* item features. It cannot extract new per-item
information. This bounds what attention can buy: the composer's value is **selection
and weighting of evidence**, not feature learning. That sharpens the D3 hypothesis
precisely:

> The composer should help when items are **redundant or vary in relevance** (some
> items are distractors), and should **not** be expected to help when the task requires
> composing *new* features, which frozen per-item vectors cannot provide.

That is a falsifiable prediction, and it says the right benchmark is one with
distractor items — which ScienceQA has (a question, a lecture, a hint, an image).

**Cost.** Moderate: rewrite the composer, retrain on GPU.

**How to falsify.** ScienceQA multimodal Choice, three arms at equal information and
the same head: mean-pool (baseline) / SAB+PMA permutation-invariant / RoPE+mean
(current). If neither attention arm beats mean pooling, ADR-0003 is rejected and the
composer is deleted.

---

## S3. Initialize the head from the option embeddings (highest expected value)

**Change.** Initialize `fingerprint_j ← o_j` — the option's own encoder embedding —
instead of random noise. Optionally add a constraint penalising deviation from those
prototypes.

**Evidence.** CLAP / **ZS-LP** (*A Closer Look at the Few-Shot Adaptation of Large
Vision-Language Models*): zero-shot prototypes "provide a strong baseline that should
not be discarded during few-shot learning"; the CLAP constraint keeps the adapted probe
from deviating excessively from them, preserving the prior. **LP++** (*A Surprisingly
Strong Linear Probe for Few-Shot CLIP*, CVPR 2024) reaches highly competitive accuracy
by making the classifier weights learnable *functions of the text embedding* rather
than free parameters.

**Outcome: rejected.** Measured identical to random initialization — 93.12% vs 93.11%
accuracy, log 0.377 for both. With 9,003 examples and 120 epochs the optimizer reaches
the same solution from either start, so the initialization does not survive training.
The hypothesis that random init "throws away the prior" was wrong for this task.

**Original rationale (kept for the record).** Zero-shot cosine already scores **92.92%**; a
randomly initialized head must rediscover that solution from scratch inside 120 epochs.

**Expected effect.** Faster convergence, a better optimum, and — critically — it makes
the head a *learned correction to the zero-shot similarity* rather than a from-scratch
classifier. It also directly attacks the measured failure.

**Cost.** Trivial: a few lines of initialization. No new data, no new compute.

**How to falsify.** Same protocol as calibration v3 (same split, same seeds, same
baselines) with the grid flaw fixed. If the initialized head does not beat both the
random-init head and kNN-5 on the proper scores, this proposal is dead.

---

## S4. Parameterize the head as a function of the option embedding

**Change.** Instead of one free vector per option, let the option's scoring vector be
`W·o_j` — a learned projection of its embedding, with a class-wise multiplier blending
option-text and state knowledge.

**Evidence.** LP++ (CVPR 2024), as above: the "surprisingly strong" probe is one whose
weights are functions of the text embedding.

**Expected effect.** Shares statistical strength across options; keeps the
"new option needs no retraining" property (ADR-0002) while reducing parameters.

**Cost.** Low–moderate. **How to falsify.** Same protocol as S3.

---

## S5. Fingerprint normalization, and use the diagnostic we already built

**Change.** (a) Unit-normalize fingerprints so `score_j` is an exact cosine; (b) run
`fingerprint_distance_matrix` on the real 77 intents and publish the most confusable
pairs **before** measuring accuracy.

**Correction (2026-10-07).** This section originally said "run the separability
diagnostic we built and never used." That is true of **jev-stack**, not of this
repository: when the head was ported into `nanocore-s1`, the diagnostics methods
(`fingerprint_distance_matrix`, `fingerprint_norms`, `separability_report`) were not
carried across. The kernel died on `AttributeError` proving it. The claim was wrong.

**Outcome: rejected as implemented.** Training with forward-pass normalization dropped
accuracy from 93.11% to **79.66%**, while the untrained normalized head reproduced the
zero-shot cosine baseline (92.86%, log 3.800). Hypothesis, not established: normalizing
inside the forward pass while optimizing the unnormalized parameter introduces a
`1/|v|` gradient factor, and Adam at lr = 0.05 rotates the fingerprints away from the
prior into a badly conditioned solution. The paper's remedy presumes a parameterization
that keeps the norm controlled; we applied it as a forward-pass transform instead. A
retry would project the parameters onto the sphere after each step rather than
normalizing in the forward path.

**Evidence.** Liu et al., *Zero-Bias Deep Learning* (2021), **Corollary 1**: if
fingerprint magnitudes vary, the zero-bias layer is biased toward specific classes. The
paper gives **two** remedies — regularize magnitude variance (we have `magnitude_reg`),
**or replace the layer equation with a normalized form (we do not)**. **Corollary 2**:
mutual fingerprint distances track separability, "fingerprints are distantly separated
with higher accuracy."

**Expected effect.** Removes a known bias channel in confidence, and converts a
built-but-never-run diagnostic into a pre-deployment check: if two intents have nearly
parallel fingerprints, that is visible before it becomes wrong answers.

**Cost.** Trivial. **How to falsify.** Report fingerprint norms and the top confusable
pairs; check whether the confusable pairs predict the observed errors.

---

## S6. Option text enrichment

**Change.** Replace humanized intent names (`card arrival`) with richer option
descriptions.

**Evidence.** RaLP (arXiv 2212.10391): label prompts are encoded and matched to the
input; "potentially poorly descriptive labels in their original format" are a known
limiter, and retrieval-augmented label prompts beat much larger baselines.

**Why it matters.** Option text is the only thing the zero-shot baseline sees, and it
becomes the head's initialization under S3. Improving it lifts both at once.

**Cost.** Low–moderate (authoring or retrieving descriptions per option).
**How to falsify.** Zero-shot cosine accuracy is the immediate, training-free signal.

---

## S7. Matryoshka dimension — measure the "lightweight" claim

**Change.** Evaluate at 256-d (truncate + re-normalize) instead of 768-d; report
accuracy alongside storage and latency.

**Evidence.** The EmbeddingGemma 2 developer guide: at 256 dimensions "most of the full
quality of the original embedding on text and code is retained," at roughly a third of
the storage; 128-d retains ~90% on text but degrades multimodal retrieval to ~75%.

**Expected effect.** The design's central pitch is "extremely lightweight," and it has
never been measured. This is the cheapest way to make it a fact rather than an
adjective.

**Cost.** Trivial (truncate cached vectors, re-normalize).
**How to falsify.** Accuracy at 256-d vs 768-d on the same split; if it collapses, 768
stays.

---

## S8. Soft targets — the largest remaining gap

**Change.** Train on soft teacher distributions instead of one-hot labels.

**Evidence.** The JEV/Laya paper: "RLCD (Reinforcement Learning for Calibrated
Decisions) trains against strictly proper scoring rules … the Brier score's unique
minimum is the true conditional distribution," and it documents RLCD with the Brier
score as a core capability.

**Current state.** Our heads train on one-hot labels, which reduces the log-score
objective to ordinary cross-entropy on a linear probe. **This is the biggest gap between
the design as documented and as implemented** — ADR-0004's whole argument for proper
scoring rules is not yet exercised.

**Cost.** Higher: requires harvesting teacher distributions.
**How to falsify.** Compare head calibration (log score, Brier) trained on one-hot vs
soft targets, same split.

---

## S9. Protocol fixes (from the flaws disclosed in calibration v3)

| Fix | Reason |
|---|---|
| Widen the temperature grid to 10^-3 … 10^2 | three baselines fitted the lower bound, so their log scores were not their best |
| Verdicts compare against the **best** baseline | our verdict compared against the weakest (`cosine_tau`) and printed a false win |
| Choose abstention thresholds **above** the base error rate | at 93% base accuracy a 90%-precision criterion is satisfied by answering everything |
| Report split-level variance, not just seed variance | ±0.0002 seed spread says nothing about a different sample |
| Measure latency | the "lightweight" claim is otherwise unquantified |

---

## What the evidence says NOT to do

- **Do not enlarge the head.** A linear probe is already competitive; the literature
  says *initialization and constraint* are what matter, not capacity.
- **Do not add positional encoding to set-like states.** It breaks permutation
  invariance for no semantic benefit.
- **Do not fine-tune the encoder yet.** ADR-0001 alternative 3 stays deferred until the
  frozen path is exhausted — and S3/S4 are untried.
- **Do not build a synthetic multi-item task** that flatters attention (ADR-0003 risk).

---

## Prioritized roadmap

| # | Proposal | Cost | Expected value |
|---|---|---|---|
| 1 | **S3** head init from option embeddings | trivial | high — attacks the main measured weakness |
| 2 | **S9** protocol fixes | trivial | high — restores trust in every comparison |
| 3 | **S5** fingerprint normalization + FD matrix | trivial | medium — removes a known bias, uses a built diagnostic |
| 4 | **S1** conformal abstention | low | high — heuristic → guarantee; fills a gap the reference paper leaves |
| 5 | **S2** Set Transformer composer | moderate | decisive for ADR-0003 |
| 6 | **S7** Matryoshka 256-d | trivial | medium — makes the pitch measurable |
| 7 | **S6** option text enrichment | low–moderate | medium — lifts baseline and initialization together |
| 8 | **S8** soft targets | higher | high, but blocked on label harvesting |

Items 1–4 are cheap enough to combine into a single CPU kernel, since they all run on
cached embeddings.

---

## Sources

**Project library** (`~/Documents/Library/technical/ai-ml/`)
- Liu, Wang, Li et al., *Zero-Bias Deep Learning for Accurate Identification of IoT
  Devices*, IEEE IoT-J (2021) — Corollaries 1 and 2, threshold selection.
- Santos, *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses:
  JEV and Laya as System One Decision Layers* (2026) — RLCD, proper scoring rules,
  temperature scaling. **No abstention mechanism.**
- *Multimodal Deep Learning* (2023) — contrastive/shared-space alignment (CLIP, ALIGN,
  Florence); the basis for characterising this design as pre-aligned late fusion.
- Karpathy, *Small LLMs Setup* (2026) — scale discipline and the Route-4 argument.
- Hardt & Recht, *Patterns, Predictions, and Actions* — distribution shift, calibration.

**Online**
- Lee et al., *Set Transformer: A Framework for Attention-based Permutation-Invariant
  Neural Networks*, ICML 2019 — SAB/ISAB/PMA.
- Zaheer et al. (2017) — permutation-invariant function representation.
- *Conformal Selective Prediction with General Risk Control* (SCoRE), arXiv 2603.24704.
- *Selective Conformal Inference with False Coverage-statement Rate Control*,
  arXiv 2301.00584.
- *LP++: A Surprisingly Strong Linear Probe for Few-Shot CLIP*, CVPR 2024.
- CLAP / ZS-LP, *A Closer Look at the Few-Shot Adaptation of Large Vision-Language
  Models*.
- *Empowering Sentence Encoders with Prompting and Label Retrieval for Zero-shot Text
  Classification* (RaLP), arXiv 2212.10391.
- EmbeddingGemma 2 developer guide — Matryoshka dimension guidance.
