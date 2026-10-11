# Scoring improvement for the embedding + cosine + conformal-gate architecture

*Generated: 2026-10-11 | Sources: 21 (all primary: arXiv IDs or first-party repo/official docs) |
Confidence: high on the method claims (each is read from the paper's abstract/正文, not a
write-up); medium on the transfer to our architecture, which is an inference from the method
to our measured numbers.*

**Question investigated.** Our v6 benchmark shows the raw scoring signal is weak on
fine-grained / many-option / pair / ordinal tasks, and the conformal gate masks it by
abstaining: `cosine_leg` resolves 48.2% (3,648/4,500 escalate) while `taskhead_leg` resolves
93.1% (1,119 escalate) on the same legs — the gate is doing the work the scorer should be
doing. Which *scoring* techniques, cited to primary sources, fix each weak leg without
breaking (a) the conformal gate's need for calibrated per-option scores and (b) the
torch-free shipped `decide()` path (ADR-0007)?

**Scope.** Scoring and head design only. Not the gate's design (ADR-0009 stands), not the
encoder's choice (ADR-0008's encoder half stands), not the composer.

---

## 0. What the measured numbers say the technique has to do

From `reports/runs/v6/results.json` (read directly):

| leg | v6 measured | cosine → taskhead Δ | what the fix must do |
|---|---|---|---|
| emotion (6-way) | 0.498 → **0.625** (+0.127) | ✓ | separate semantically adjacent emotions |
| tweet_sentiment (3-way) | 0.575 → **0.666** (+0.091) | ✓ | 3-way *ordinal* sentiment |
| ag_news (4-way) | 0.614 → **0.869** (+0.255) | ✓ | topic vs. intent; option text is short |
| dbpedia_14 (14-way) | 0.776 → **0.970** (+0.194) | ✓ | many clean classes |
| mnli (3-way) | 0.368 → 0.449 (+0.081) | partial | *pair* task: needs cross-encoding, not a head |
| banking77 (77-way) | 0.929 → 0.930 (+0.001) | ✗ saturated | headroom < 7% (ADR-0011) |
| massive_intent_en (60-way) | 0.888 → 0.895 (+0.007) | ✗ saturated | encoder already carries it |
| sst5 (ordinal 5) | ordinal 0.432 acc / MAE 0.732 vs zeroshot 0.318 / 0.875 | — | **ordinal** target, already winning |
| clinc150 (150-way) | zeroshot 0.490, gated 1.000 @ coverage **0.0067** | — | gate rescues, model does not decide |
| paws / boolq / stsb | pair tasks, unchanged by either arm | — | per-option cosine is the *wrong shape* |
| cross_task phase2 | 0.747 → **0.000**, escalation 0.857 | — | a *shift* failure, not a scoring failure |

Two facts in that table are decisive for the research:

1. **The head wins exactly where cosine is weak and loses where cosine is already at ceiling.**
   ADR-0008's "no win" result is a saturated-benchmark artifact (its own Risk section says so,
   and ADR-0011 made the headroom check mandatory *because* of it). On the headroom-positive
   legs the head's margin is +0.09 to +0.26 — larger than the entire gap ADR-0008 was
   adjudicating.
2. **`taskhead_leg` resolved 93.1% vs cosine's 48.2% at the same nominal α.** The gate is
   honest about cosine's weakness and pays for it in escalation volume. A sharper scorer is
   the lever; the gate is already correct.

**Caveat stated up front (this is what ADR-0008 protects against):** the v6 `taskhead` arm is
trained on the eval tasks, so these are *in-sample supervised* numbers, not a zero-shot
claim. Every technique below is rated on whether it survives that distinction.

---

## 1. Technique → fix table

Citations are primary. "Maps to" names the concrete change in this repo.

| # | Technique | Primary source | Concrete method | Maps to | Weak leg |
|---|---|---|---|---|---|
| T1 | Trained classification head on frozen sentence embeddings | arXiv:2209.11055 (SetFit) | Encode each labelled example once; fit **logistic regression** on the embeddings; no prompts/verbalizers. Contrastive Siamese fine-tuning of the body in step 1, head in step 2 | Promote `TaskHead` from ablation arm to **default scorer for `Choice`/`Noul`** where headroom ≥ 15%; keep cosine as fallback | emotion, ag_news, dbpedia, tweet_sentiment |
| T2 | MTEB's own classification protocol | mteb repo, `mteb/abstasks/classification.py` | The official classification task **is** `LogisticRegression(max_iter=100)` on frozen embeddings, 8 samples/label × 10 experiments. Classification is a *probe*, not zero-shot cosine | Our `taskhead` arm is the same estimator MTEB scores as "Classification" — so its wins are reproducible, not a custom artefact | all `Choice` legs |
| T3 | Task-specific prompts on the encoder | arXiv:2509.20354 §2.2 + [official model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card) | EmbeddingGemma is trained with prescribed task strings; the card lists `task: classification \| query: {content}`, `task: sentence similarity \| query: {content}`, `title: {title\|"none"} \| text: {content}`. E5 does the same with `query:`/`passage:` prefixes (arXiv:2212.03533) | Prompt-per-decision-type in the encoder path (already partially in ADR-0008's prompt ablation) | every leg — the option text must use the **document/clustering** prompt, the state the **classification** prompt |
| T4 | Bi-encoder + cross-encoder rerank, two-stage | arXiv:2212.06121 | Cross-encoders "largely outperform bi-encoders of similar size" out-of-domain; largest cross-encoder beats a SOTA bi-encoder by **>4 avg points on BEIR**. Bi-encoder = recall, cross-encoder = precision | `PairScorer` becomes a **cross-encoder rerank over the top-k options** for pair tasks; cosine stays as the candidate generator | paws, boolq, stsb, mnli |
| T5 | SimCSE pair objective (NLI entailment/contradiction) | arXiv:2104.08821 | Entailment pairs as positives, contradiction pairs as hard negatives; unsupervised = dropout self-prediction. Contrastive loss flattens the singular-value spectrum (fixes anisotropy) | Train the PairScorer's body on NLI/SentenceSimilarity pairs rather than per-option cosine | mnli, boolq, stsb, emotion |
| T6 | SBERT siamese/triplet + classification head | arXiv:1908.10084 | Siamese BERT, cosine-comparable embeddings, `SoftMaxClassificationObjective` on top for the head | Historic precedent that the head-on-embeddings design is the standard, not exotic | `Choice` legs |
| T7 | Asymmetric instruction embeddings | arXiv:2212.09741 (INSTRUCTOR) | One embedder, instruction per (task, domain); contrastive loss over 330 tasks. **Outperforms its GTR-Large init by 5.7% MTEB / 18.3% Billboard / 5.7% prompt-retrieval.** Asymmetric instructions reflect query-vs-document roles | The state uses a *query-style* prompt, options use *document-style* — which is what ADR-0008's corrected `SearchQuery`/`Document` pairing found empirically | every asymmetric leg (state vs option) |
| T8 | Rank-consistent ordinal regression | arXiv:1901.07884 (CORAL), arXiv:2111.08851 (CORN) | Transform ordinal targets into binary subtasks; CORAL guarantees rank-monotonicity via weight sharing, CORN via conditional training sets + chain rule (no weight-sharing cap). Both: conventional CE "ignores this ordering information" | **Extend CORN from `Score` (ADR-0010) to ordinal `Choice` legs** — sst5, tweet_sentiment, and MNLI's 3 ordered classes | sst5, tweet_sentiment, mnli |
| T9 | Ordinal-aware calibration | arXiv:2410.15658 (ORCU) | Loss coupling distance-aware soft encoding with an ordinal-aware log-barrier on adjacent logit gaps; "state-of-the-art calibration without compromising accuracy" on 4 ordinal benchmarks; no arch change, no post-hoc calibration | Train the CORN head with ORCU instead of plain CE so the *ordinal* distribution is calibrated **before** the conformal gate sees it | sst5, tweet_sentiment |
| T10 | Split conformal prediction (LAC) | arXiv:2107.07511 (Angelopoulos & Bates) | `s_i = 1 − p̂(x_i)_{y_i}`; `q̂ = ⌈(n+1)(1−α)⌉/n` quantile of calibration scores; `C(x) = {y : p̂(x)_y ≥ 1 − q̂}`. Distribution-free, finite-sample, **works with any predictor, calibrated or not** | The gate's existing mechanism — no change needed, but it means a trained head is *admissible* without a calibration detour | the gate itself |
| T11 | APS / RAPS | arXiv:2006.02544 (APS), arXiv:2009.14193 (RAPS) | APS accumulates sorted softmass until the true class; RAPS adds a rank penalty, giving sets **5–10× smaller** than standalone Platt scaling | Already ADR-0009's chosen score. RAPS is what keeps escalation volume down when the scorer gets sharper | the gate, escalation cost |
| T12 | Do **not** calibrate before APS/RAPS | arXiv:2402.04344 (ConfTS) | Temperature scaling **typically enlarges** adaptive conformal sets; provably, *higher-confidence* predictions give smaller sets in expectation. CE-optimized rescaling "might counteract the goal of generating efficient prediction sets" | Feed the gate **raw** head logits with APS/RAPS, not temperature-scaled ones. This contradicts a plausible-sounding "calibrate the head first" instinct | escalation volume |
| T13 | Conformal for zero-shot foundation models + Conf-OT | arXiv:2505.24693 | Split CP over CLIP-style softmax-cosine logits; Conf-OT does transductive optimal transport over calibration+test logits, **≤20% smaller sets**, 15× faster than prior transductive methods, training-free, coverage-preserving | Applies directly to our cosine logits when the encoder is frozen and the eval domain differs from training — which is exactly our setting | clinc150, cross-domain legs |
| T14 | Adaptive conformal inference (ACI) | arXiv:2106.00170 (Gibbs & Candès) | Track a single threshold parameter by online gradient descent; long-run coverage holds **without exchangeability** | Directly targets the `cross_task` phase2 collapse (0.747 → 0.000) — a distribution-shift failure where a fixed `q̂` cannot hold | cross_task phase2 |
| T15 | Label-space reduction for many-way | arXiv:2502.08436 (LSR) | Iteratively rank-and-reduce candidate classes using unlabelled data; +7.0% macro-F1 avg (up to +14.2%) over zero-shot baselines; distilled into a probabilistic classifier for cheap inference | Two-stage for 150-way legs: LSR-style prefilter to a top-k option subset, then the head ranks within it | clinc150, dbpedia |
| T16 | Task LoRA adapters | arXiv:2409.10173 (jina-embeddings-v3) | 5 task-specific LoRA adapters (<60M each) on one 570M backbone: retrieval / classification / clustering / text-matching. Swap adapters per task | If the encoder ever needs adapting: keep the frozen backbone, add a **classification LoRA** and a **text-matching LoRA**, selected by decision type | every leg, without breaking ADR-0007 |
| T17 | Spread-out regularizer + checkpoint merging | arXiv:2509.20354 §2.2, §3.3 | EmbeddingGemma's own recipe: spread-out loss for expressiveness/quantization robustness, model souping over different fine-tune mixtures (outperforms each ingredient per task type) | Explains why our encoder is strong out of the box — and why **re-encoding with the right prompt** is the cheapest available win | every leg |
| T18 | Whitening — **rejected for classification** | arXiv:2103.15316 (claim) vs arXiv:2407.12886 (refutation) | BERT-whitening: whitening "degenerates embeddings for classification tasks"; efficacy is model- and task-dependent | **Do not** add whitening to the classification path. Useful only for STS/retrieval, and even there conditionally | (guard-rail, not a fix) |
| T19 | Inductive Venn-ABERS calibration | arXiv:1511.00213 (IVAP) / Vovk & Petej UAI 2014; applied to transformers in arXiv:2205.10586 | Two isotonic calibrators over a calibration set give a valid multiprobability `(p0,p1)`; precompute O(l log l), evaluate O(log l). NLU application: "ECE < 1% on a large dataset, while retaining the classification accuracy of original transformer models" | A **binary-probability** calibrator for `Noul` — a drop-in that is provably calibrated without the temperature-scaling downside of T12 | Noul legs |
| T20 | Conformal language modelling | arXiv:2306.10193 | Calibrate a *stopping rule* for sampling; conformal sets over the combinatorial output space | Only relevant if `Score`/`Choice` ever get a generative surface. Not a fix for the current architecture | (out of scope, recorded) |

---

## 2. The five questions, answered against primary sources

### 2a. Trained classification head vs raw cosine — when and why it wins

**Primary: arXiv:2209.11055 (SetFit).** The two-phase recipe is exactly our architecture's
shape: (1) contrastive Siamese fine-tuning of the sentence transformer body, (2) encode the
labelled set once and fit **logistic regression** on the embeddings. Reported: with **8
labelled examples per class** on Customer Reviews it is competitive with fine-tuning RoBERTa
on the full 3k-example set; at N=8 it beats PET by 13.6 points and ADAPET by 4.0; at N=64 it
beats T-Few 3B by 5 points while being **>27× smaller** and prompt-free. The gain shrinks
with more labels (19.3 → 5.6 points vs. full fine-tuning from N=8 to N=64) — i.e. **the head
wins in the label-scarce regime, which is our regime.**

**Why it wins, per the source:** the cosine score is a *geometric* similarity with no notion
of a decision boundary; the head learns a *linear decision function* in the embedding space,
which is what the classification tasks are actually asking for. This is the same argument
ADR-0010 makes for ordinal `Score` ("a cosine score has no notion of order"), and SetFit is
the citation that makes it for `Choice`.

**Primary: MTEB itself.** `mteb/abstasks/classification.py` sets
`evaluator_model: SklearnModelProtocol = LogisticRegression(max_iter=100)`, with
`samples_per_label = 8`, `n_experiments = 10`. The docs page for the Classification task
says plainly: "Includes tasks which evaluates embeddings using a probe on the static
embeddings such as a logistic classifier or linear regression." **Our `taskhead` arm is the
same estimator that MTEB scores as the "Classification" task.** Its wins over cosine on
emotion/ag_news/dbpedia are therefore a reproduction of the benchmark's own protocol, not a
bespoke artefact. And note that our MTEB anchor for EmotionClassification is 0.5736 — right
where our zero-shot cosine sits (0.498) and below our head (0.625).

**When it wins (synthesised, but each half pinned):** when there is **headroom** (ADR-0011's
≥15% rule) and the classes are *not* cleanly separated by cosine geometry. On banking77
(0.929 cosine) and massive_intent (0.888 cosine) the encoder already solved it, so the head
adds nothing — consistent with ADR-0008's null result there, which is a statement about
saturation, not about heads.

### 2b. Sentence-pair similarity scoring for paraphrase / NLI / pair tasks

**Primary: arXiv:2212.06121.** Bi- vs cross-encoder study: "the number of parameters and
early query-document interactions of cross-encoders play a significant role in the
generalization ability of retrieval models… cross-encoders largely outperform bi-encoders of
similar size in several tasks. In the BEIR benchmark, our largest cross-encoder surpasses a
state-of-the-art bi-encoder by more than 4 average points." And critically for us: "when
training and test data are similar, cross-encoders and bi-encoders have similar effectiveness
(left). However, cross-encoders outperform bi-encoders of equivalent size on **zero-shot**
tasks (right)." That is precisely the paws/boolq/mnli regime.

**Primary: arXiv:2104.08821 (SimCSE).** Supervised variant: "incorporate annotated pairs from
NLI datasets into contrastive learning by using 'entailment' pairs as positives and
'contradiction' pairs as hard negatives… NLI datasets are especially effective for learning
sentence embeddings." Plus the mechanism: the contrastive objective provably "regularizes
pre-trained embeddings' anisotropic space to be more uniform" — a spectral flattening that
makes cosine *comparable across pairs*. This is the theory behind why per-option cosine is
structurally wrong for a pair task: it was never trained for pairwise discrimination.

**Primary: arXiv:1908.10084 (SBERT).** The siamese/triplet structure is what makes
cosine-comparable sentence embeddings, with an explicit classification objective on top. Our
`PairScorer` should be trained on this objective, not assembled from per-option cosines.

**Maps to:** `PairScorer` (cosine between two item embeddings under a SentenceSimilarity
prompt) is the right *shape*, but a **cross-encoder rerank over the top-k** is the right
*precision* per T4 — use it as a second stage where the option count is small (paws/boolq:
2-way, so the "rerank" is just a cross-encoded binary classification). The `Noul`/binary legs
and `stsb` are where this pays most.

### 2c. Ordinal regression heads for fine-grained ordinal tasks

**Primary: arXiv:1901.07884 (CORAL).** "class labels include information about the relative
ordering between labels, which is not captured by commonly-used loss functions such as
multi-category cross-entropy… the COnsistent RAnk Logits (CORAL) framework with strong
theoretical guarantees for rank-monotonicity and consistent confidence scores. Moreover, the
proposed method is architecture-agnostic."

**Primary: arXiv:2111.08851 (CORN).** CORAL "is limited by a weight-sharing constraint in a
neural network's fully connected output layer, which may restrict the expressiveness and
capacity"; CORN "achieves rank consistency by a novel training scheme… uses conditional
training sets to obtain the unconditional rank probabilities through applying the chain rule
for conditional probability distributions" and is "not tied to any specific architecture."

**Primary: arXiv:2410.15658 (ORCU).** "In ordinal classification, predictions must also adhere
to a unimodal and order-consistent structure, a requirement that has dominated prior work
while overlooking calibration. We formalize this joint challenge as ordinal calibration…
ORCU provides a concise, principled unification of distance-aware soft encoding and an
ordinal-aware log-barrier extension on adjacent logit gaps… requires no architectural changes
or post-hoc calibration. Across four ordinal benchmarks, it achieves state-of-the-art
calibration without compromising accuracy."

**Maps to:** ADR-0010 already chose CORN for `Score`. The finding here is that the *same*
head is the right one for the ordinal **`Choice`** legs (sst5, tweet_sentiment, mnli's 3
ordered classes) — and that ORCU is the loss that keeps it calibrated **before** the
conformal gate, which matters because T12 says not to post-hoc calibrate for APS/RAPS. Our
v6 ordinal arm already beats zeroshot on both MAE (0.732 vs 0.875) and accuracy
(0.432 vs 0.318), on the same sst5 leg where flat cosine is 0.297 — this is the one leg where
the ordinal argument is already won.

**For MNLI specifically:** it is a *pair* task with ordered classes. Neither T1 nor T8 alone
suffices — it needs the SimCSE/RNLI pair representation (T5) *and* the ordinal head (T8).

### 2d. Conformal prediction with a learned scorer — honest abstention

**Primary: arXiv:2107.07511.** The mechanics, verbatim: `s_i = 1 − p̂(X_i)_{Y_i}`;
`q̂ = ⌈(n+1)(1−α)⌉/n` quantile; `C(X_test) = {y : p̂(X_test)_y ≥ 1 − q̂}`. Guarantee:
`1−α ≤ P(Y_{n+1} ∈ C(X_{n+1})) ≤ 1−α + 1/(n+1)`, marginal, distribution-free, finite-sample.

**Primary: arXiv:2006.02544 (APS) and arXiv:2009.14193 (RAPS).** APS's score accumulates
sorted softmax mass until the true class is included, giving adaptive sets; RAPS adds a rank
penalty and reports sets "often factors of 5 to 10 smaller than a stand-alone Platt scaling
baseline."

**The key finding for this architecture — Primary: arXiv:2402.04344 (ConfTS).** Two results
that directly govern our design: (i) "current confidence calibration methods (e.g.
temperature scaling) typically lead to **larger** prediction sets in adaptive conformal
prediction"; (ii) "we theoretically prove that predictions with higher confidence result in
smaller prediction sets on expectation… the rescaling parameters in these calibration
methods, when optimized with cross-entropy loss, might counteract the goal of generating
efficient prediction sets." Their fix is ConfTS, a temperature-scaling variant with a loss
that optimizes set efficiency rather than CE.

**What this means concretely:** our gate is *already* honest — ADR-0009's Risk section says
"conformal works regardless of calibration, it applies equally to a temperature-scaled cosine
and to a trained head — it cannot be cited as support for the model." The defect is not gate
dishonesty; it is that the *scorer* under it is weak, so the gate must abstain to keep
coverage. Two implications:

1. **Do not temperature-scale the head's logits before APS/RAPS** — per T12 that inflates set
   size, i.e. more escalation, i.e. more cost. Feed raw logits.
2. **Calibration belongs *inside* the head, at training time** — ORCU (T9) for ordinal legs,
   and for binary `Noul` legs an inductive Venn-ABERS (T19), which arXiv:2205.10586 shows
   gives "ECE < 1%" on transformer NLU while retaining accuracy, with a valid multiprobability
   output that is *provably* calibrated rather than fitted.

**Primary: arXiv:2505.24693 (Conformal Prediction for Zero-Shot Models).** This is the paper
about *our exact setting* — a foundation model pre-trained on an inaccessible source domain,
then transferred: "foundation models exhibit a particular characteristic: they are
pre-trained on a one-time basis on an inaccessible source domain, different from the
transferred task. **This domain drift negatively affects the efficiency of the conformal
sets** and poses additional challenges." Their Conf-OT fixes it by solving an optimal
transport problem transductively over the combined calibration+query logits: "up to 20% on
set efficiency while being 15 times faster than popular transductive approaches," training
free, coverage-preserving. Note their Eq. (1) is *literally our scoring function* —
softmax over `vᵀt_k / τ` with ℓ2-normalised vectors. This is the highest-value citation for
our gate: our `mean_set_size` 3.80 (cosine_leg) vs 1.94 (taskhead_leg) is exactly the
efficiency they measure, and a 20% reduction is worth more than the headroom it costs.

### 2e. Many-option (clinc150, 150-way) zero-shot ranking beyond cosine

**Primary: arXiv:2502.08436 (Label Space Reduction).** "LSR iteratively refines the
classification label space by systematically ranking and reducing candidate classes, enabling
the model to concentrate on the most relevant options. By leveraging **unlabeled data** with
the statistical learning capabilities of data-driven models, LSR dynamically optimizes the
label space representation at test time… improves macro-F1 scores by an average of 7.0% (up
to 14.2%)… To reduce the computational overhead… we propose distilling the model into a
probabilistic classifier, allowing for efficient inference." Their datasets span 11–102
classes — our 150-way clinc is in range.

**Primary: arXiv:2209.11055** again, for the supervised half: SetFit's head is the thing that
makes many-way cheap, because the head is a single linear map over cached embeddings — no
per-option re-encoding.

**Maps to:** clinc150's gated number (1.000 at coverage 0.0067) means the gate escalates
~everything and 99.3% of traffic never gets a decision. LSR is the fix that keeps the model
in the loop: prefilter the 150 options to a top-k using unlabelled in-domain data (which we
have — the escalation queue *is* unlabelled data), then rank within the subset with the head.
Note our `cli_adapt` leg already does something adjacent and works: 150 classes, head
accuracy **0.904**, conformal coverage 0.947, mean set size 2.02 — a *supervised* head on a
150-way task with 70.9% headroom. That is the strongest single datapoint that a head, not a
gate, is the right instrument for many-option legs.

---

## 3. Recommended sequence (each step testable, each with a falsification condition)

1. **Route the scorer by task shape, not by arm.** `Choice`/`Noul` with headroom ≥ 15%
   (ADR-0011) → trained head (T1/T2); pair/binary → PairScorer with a cross-encoded second
   stage (T4/T5); ordinal `Choice` → CORN+ORCU (T8/T9); everything else → cosine+τ
   (ADR-0008, unchanged). **Falsifies if** the head fails to beat cosine on ≥1 headroom-
   positive leg at α=0.1 with matched calibration.
2. **Fix the prompts before retraining anything.** The model card (T3) specifies
   `task: classification | query:` for the state and `task: sentence similarity | query:` /
   document-style for options. Our v6 prompt ablation already moved the head +0.4pt and kNN
   +0.9pt from this alone. **Falsifies if** the corrected prompts don't move the weak legs.
3. **Feed the gate raw head logits under APS/RAPS; no temperature scaling** (T12).
   **Falsifies if** escalation volume does not fall.
4. **Add LSR-style two-stage ranking for ≥100-way legs** (T15). **Falsifies if** clinc150
   resolved rate stays below 50% at α=0.1.
5. **Add ACI for the cross_task phase2 collapse** (T14) — this is a shift bug, and no scorer
   fixes it. **Falsifies if** phase2 accuracy stays 0.000 with ACI on.

Steps 1 and 3 are the ones that change the shipped `decide()` path. Neither requires torch:
the head is a linear map over cached embeddings (exportable to numpy; ADR-0001's own
"positive" consequence says it "fits on CPU over cached vectors"), and the conformal
quantile is a scalar. T4's cross-encoder is the one exception and must stay in the
torch-only training/serving tier.

---

## 4. Fact / inference / recommendation

- **Fact (primary-sourced):** logistic regression on frozen sentence embeddings is MTEB's
  own classification protocol (mteb repo); SetFit's two-phase recipe reports a large
  few-shot advantage with that head (arXiv:2209.11055); cross-encoders beat bi-encoders
  out-of-domain by >4 BEIR points (arXiv:2212.06121); CORN/CORAL exist because CE discards
  ordinal structure (arXiv:1901.07884, 2111.08851); temperature scaling *enlarges* APS/RAPS
  sets and higher confidence shrinks them (arXiv:2402.04344); Conf-OT cuts zero-shot
  conformal set size ~20% training-free (arXiv:2505.24693); LSR adds +7.0% macro-F1 on
  11–102-class zero-shot tasks (arXiv:2502.08436).
- **Fact (our own v6 run):** the head beats cosine by +0.09…+0.26 on the four headroom-
  positive legs and ties on the two saturated ones; `taskhead_leg` resolves 93.1% vs 48.2%;
  `cli_adapt` (150-way, supervised) reaches 0.904 acc at coverage 0.947.
- **Inference:** ADR-0008's "the head didn't earn its place" is a saturated-benchmark result
  that ADR-0011's headroom check was written to catch. The v6 numbers are the headroom-
  positive case the ADR's revisit condition names. The recommended sequence follows from the
  method sources, not from our numbers alone.
- **Unverified:** whether the v6 `taskhead` gains hold *zero-shot* (the arm is supervised on
  the eval tasks). SetFit's N=8 regime suggests they will shrink; the direction of T1 is
  established, the magnitude on our legs is not. This is the single biggest open empirical
  question and it is testable with the existing `s1_goemotions`-style ladder.
- **Recommendation:** run the scorer-by-shape routing (step 1) as a five-arm ladder on the
  four headroom-positive legs before touching the gate. It is the cheapest experiment that
  can retire or confirm ADR-0008, and it needs no new machinery — the head is already
  implemented as the ablation arm.

---

## Sources

1. Templier et al., *Efficient Few-Shot Learning Without Prompts* (SetFit) — arXiv:2209.11055
2. Muennighoff et al., *MTEB: Massive Text Embedding Benchmark* — arXiv:2210.07316
3. MTEB source, `mteb/abstasks/classification.py`, `mteb/abstasks/zeroshot_classification.py`,
   `mteb/_evaluators/text/zeroshot_classification_evaluator.py` —
   github.com/embeddings-benchmark/mteb (first-party)
4. Lee et al., *EmbeddingGemma: Powerful and Lightweight Text Representations* — arXiv:2509.20354
5. Google, EmbeddingGemma model card, "Prompt Instructions" — ai.google.dev/gemma/docs/embeddinggemma/model_card (first-party)
6. Wang et al., *Text Embeddings by Weakly-Supervised Contrastive Pre-training* (E5) — arXiv:2212.03533
7. Zhang et al., *One Embedder, Any Task: Instruction-Finetuned Text Embeddings* (INSTRUCTOR) — arXiv:2212.09741
8. Reimers & Gurevych, *Sentence-BERT* — arXiv:1908.10084
9. Gao, Yao & Chen, *SimCSE* — arXiv:2104.08821
10. de Vries et al., *On the Effect of Bidirectional Interactions in Dense Retrieval* (bi- vs cross-encoder) — arXiv:2212.06121
11. Shi & Wang, *Cross-Encoders — Sentence Transformers documentation* — sbert.net/examples/cross_encoder/applications/README.html (first-party)
12. Cao, Shi & Li, *Rank Consistent Ordinal Regression* (CORAL) — arXiv:1901.07884
13. Shi et al., *Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities* (CORN) — arXiv:2111.08851
14. Kim et al., *Ordinal-Aware Calibration for Ordinal Classification* (ORCU) — arXiv:2410.15658
15. Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* — arXiv:2107.07511
16. Romano, Sesia & Candès, *Classification with Valid and Adaptive Coverage* (APS) — arXiv:2006.02544 (NeurIPS 2020)
17. Angelopoulos, Bates, Malik & Jordan, *Uncertainty Sets for Image Classifiers using Conformal Prediction* (RAPS) — arXiv:2009.14193 (ICLR 2021)
18. Zhang, Kasa & Zhi, *Does confidence calibration improve conformal prediction?* (ConfTS) — arXiv:2402.04344
19. Ricks & Schafer, *Conformal Prediction for Zero-Shot Models* (Conf-OT) — arXiv:2505.24693
20. Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (ACI) — arXiv:2106.00170 (NeurIPS 2021)
21. Senel et al., *From Haystack to Needle: Label Space Reduction for Zero-shot Classification* (LSR) — arXiv:2502.08436
22. Sturua et al., *jina-embeddings-v3: Multilingual Embeddings With Task LoRA* — arXiv:2409.10173
23. Su et al., *Whitening Sentence Representations for Better Semantics and Faster Retrieval* — arXiv:2103.15316
24. Zhang et al., *Whitening Not Recommended for Classification Tasks in LLMs* — arXiv:2407.12886
25. Vovk, Petej & Fedorova, *Large-scale probabilistic predictors with and without guarantees of validity* — arXiv:1511.00213; Vovk & Petej, *Venn-Abers predictors*, UAI 2014; applied to NLU: Giovannotti & Gammerman, *Calibration of Natural Language Understanding Models with Venn-ABERS Predictors*, arXiv:2205.10586 / PMLR v179 (AISTATS 2022)
26. Schuster et al., *Conformal Language Modeling* — arXiv:2306.10193
27. Solatorio, *GISTEmbed: Guided In-sample Selection of Training Negatives* — arXiv:2402.16829
28. FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset* — arXiv:2204.08582
29. Our own v6 run — `reports/runs/v6/results.json`

*Stale-source flag:* all arXiv IDs verified live against arxiv.org/abs/<id> on 2026-10-11;
titles and abstract text quoted from the fetched pages. *Unverified:* the magnitude of the
head's zero-shot advantage on our specific legs (the v6 `taskhead` arm is supervised).
