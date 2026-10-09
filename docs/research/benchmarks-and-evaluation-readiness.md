# Benchmarks and evaluation readiness: research report

*Generated: 2026-10-09 | Sources: 12 | Confidence: High on the reference protocol (the
category's own published evaluation, read directly) and on the conformal/deferral methods
(published and coded). Medium on what "ready" means operationally — that is partly the
deployment's call, not the literature's.*

**Question investigated.** What benchmarks and evaluation protocols determine whether a
*decision model* — frozen encoder + fitted head + conformal gate, emitting `Choice` /
`Score` / `Noul` and the actions `answer`/`clarify`/`escalate`/`abstain` — is ready for
human interaction? And what specifically is missing from this project's evidence?

Scope is deliberately narrow: what is *buildable and measurable* for `nanocore-s1` — a
typed-decision layer, not a generative model. Reader / dashboard / case-study delivery,
and any question of whether a number is publishable, are out of scope.

---

## Executive summary

**There is a reference evaluation for exactly this model shape, and it is in the local
library.** `EVALUATING AND BENCHMARKING THE SYSTEM ONE MODEL` (arXiv:2609.37647, Deußer,
Sparrenberg & Sifa, University of Bonn / Lamarr / Fraunhofer IAIS) is a 37-dataset,
346,009-request evaluation of Jev that **maps its tasks onto the same three primitives we
use** — `Choice` for classification and multi-choice, `Score` for ordinal, `Noul` for
binary detection and multi-label. Its metrics, template discipline, and memorization
probes are what our §4 evaluation standard was derived from.

**Our evidence already covers most of that protocol.** What it does *not* cover is the
two things that decide whether the model is safe to put in front of a person:

1. **Conditional (group-conditional) coverage.** Our conformal results are all *marginal*
   — correct on average across the test set. MAPIE's conditional-conformal work shows a
   single global cutoff can hide **local undercoverage in a hard group** — the very thing
   that matters when the group is, e.g., a low-resource language or a rare intent. We have
   never measured per-slice coverage.
2. **Deferral / handoff quality.** The learning-to-defer literature (Geifman & El-Yaniv
   2017; Madras, Pitassi & Zemel 2018; Mozannar et al. 2023) exists to evaluate *whether
   deferring to a human actually helps* — and it names two failure modes we have not
   measured: **over-reliance** (humans accept a confident-but-wrong assertion) and
   **minority over-rejection** (selective classifiers reject proportionally more from hard
   groups).

**The honest finding:** no published standard defines a numeric threshold for "ready for
human interaction" for a classifier-shaped model. What exists is a *protocol* — a set of
measurements — and a set of named failure modes to exclude. "Ready" reduces to: the
protocol run, the failure modes measured and bounded, and the deployment's own acceptance
criteria stated. The thresholds are ours to set, which is a finding, not a gap in the
research.

---

## 1. The reference evaluation — what it does (arXiv:2609.37647)

This is the category's own benchmark. Its structure is the answer to "what does a proper
evaluation look like for this model".

**Request construction — maps onto our schema primitive-for-primitive:**

| task | primitive |
|---|---|
| classification | `Choice` over the dataset's labels (with one-sentence descriptions where a label name alone is ambiguous) |
| multiple-choice | `Choice` keyed A, B, C… — *not* numeric keys (the key "36" once coincided with the answer) |
| binary detection (spam, toxicity, prompt injection, grounding, void clauses) | **`Noul`** |
| multi-label | **one `Noul` per label in a single request** — 28 for GoEmotions, 8 each for OpenAI moderation and UNFAIR-ToS |
| ordinal | **`Score`** with described levels (SemEval STS guidelines; G-Eval/SummEval/HelpSteer2 definitions) |

CLINC150 specifically gets an explicit out-of-scope option. ToxiGen is posed as *both* a
Noul and a five-level Score in the same request.

**Template discipline.** One template per dataset, tested on 20 training/validation
examples, **frozen before the evaluation split runs** — no prompt tuning on eval data, no
in-context examples.

**Metrics (Appendix A).** Per dataset: accuracy, **ECE over 15 equal-width bins** of the
top-option probability (P(yes) for Nouls), **Brier**, and selective prediction — **rank by
reported confidence, then accuracy at 80% and 50% coverage and AURC**. Customary metrics
elsewhere: positive-class F1 for imbalanced binary, macro-F1 (GoEmotions), micro-F1
(UNFAIR-ToS, per LexGLUE), mean AUPRC (OpenAI moderation), per-source balanced accuracy
(LLM-AggreFact), **Spearman for `Score`** tasks.

**Memorization probes.** Rotating the option order leaves accuracy unchanged; withholding
the question drops it to near chance. Together they rule out *shallow* position/length
memorization — but not memorized question–answer pairs.

**Calibration finding that transfers to us.** *"Binary probabilities rank well but are
poorly placed relative to a fixed 0.5 threshold"* — AUROC/AUPRC are good while a fixed
cutoff is not. Thresholds tuned on training data raised UNFAIR-ToS micro-F1 from 0.50 to
0.75. **This is the same thing we found the hard way:** the ranking signal is good and the
*absolute* threshold is what needs fitting.

**Cost.** 346,009 requests for under US$10 — the thing that makes a decision-model
evaluation economical is that each request is a single forward pass, not a generation.

**They release the code, harness and all raw responses** — the harness is a primary source.

## 2. What our evidence covers, and what it does not

| protocol element | the reference eval | ours | status |
|---|---|---|---|
| per-dataset customary metric | acc / F1 / AUPRC / Spearman | acc, log, Brier, ECE15 | **covered** |
| calibration summary | ECE15 + Brier | ECE15 + Brier | **covered** |
| selective prediction | rank by confidence; Acc@80/50, AURC | identical (AURC, acc@50/80) | **covered** |
| frozen template, no eval tuning | yes | yes | **covered** |
| memorization probes | rotate options; withhold question | half the verbs are in-scope class | **partial** (weaker) |
| bootstrap CIs | yes | yes (95% percentile, 500 resamples) | **covered** |
| headroom/saturation | n/a — they use real SOTA | ours: no benchmark claim with <15% headroom | **ours is stricter** |
| **conditional / group coverage** | **not reported** | **not reported** | **neither — the gap** |
| **deferral/handoff quality** | not evaluated | not evaluated | **neither — the gap** |
| multimodal | not covered | image vs chance measured | **ours exceeds** |
| multilingual slice | 122 languages (Belebele) | 7 languages | **partial** |

Two things to flag honestly. First, the reference eval's datasets and ours overlap
(CLINC150, Banking77, Emotion, MMLU, AG News…) but **the splits and templates differ, so
the numbers are not comparable** — that is the project's own stop-doing rule. Second, our
memorization probe (in-scope class membership) is weaker than theirs (option rotation +
question withholding); theirs is a sharper control and worth adopting.

## 3. The two things that decide "ready for human interaction"

### 3a. Conditional coverage — measured nowhere, our marginal numbers can hide it

MAPIE's conditional-conformal work exists because *"the single global conformal cutoff is
still not sufficient for the hardest group"* — a standard split-conformal set achieves its
coverage on average while a hard group sits under-covered. The fix is group-conditional
coverage, `P{Y ∈ C(X) | G = g} ≥ 1 − α` (Mondrian / `ConditionalSplitConformalClassifier`).

This is directly load-bearing for us: our headline claim is calibrated abstention, and a
group-conditional failure would be invisible in the `.914` / `.9882` marginal numbers we
report. **Concretely: coverage per intent class and per language.** One caveat the method
itself documents — a group's conformalization set needs ~200+ samples to be stable, so
small slices should be reported with their size, not just their coverage.

### 3b. Deferral quality — a whole literature exists, with two named failure modes

There is a real body of work on *whether deferring to a human helps*, and it is not
interchangeable with "accuracy at coverage":

- **Geifman & El-Yaniv, NeurIPS 2017** — selective classification with a guaranteed-risk
  "dial"; the foundational frame.
- **Madras, Pitassi & Zemel, NeurIPS 2018** — *learning to defer*: the model can say PASS
  and the metric is **system error** (the Human-AI team), not model accuracy.
- **Mozannar et al. 2023** — the system-error goal is genuinely hard: *"humans over-rely on
  the AI when it is incorrect"*, and naive approaches *"rarely achieve performance higher
  than either the human or AI alone."*
- **Pugnana & Ruggieri** — a documented *bias* in abstention: selective classifiers reject
  proportionally more from the minority/hard class.

**Implication for us.** "Ready for human interaction" is not a number — it is that these
two failure modes are *measured and bounded*. We currently measure that the model
*escalates* (the action), not whether escalation makes the *team* more accurate, and not
whether escalation concentrates on a group.

### 3c. Refusals — two-sided, already aligned

XSTest (Röttger et al., NAACL 2024) defines the two-sided design: safe prompts that must
**not** be refused alongside unsafe ones that must. Our `refusals` battery implements
this — the positive control is exactly the anti-over-refusal control. **Covered.**

## 4. What cannot be established offline

Three things no benchmark can supply, which should be recorded as unclaimable rather than
implied:

- **offline↔online correlation** — no deployment exists, so we cannot claim that our
  offline metrics predict anything in production
- **real handoff quality** — the Human-AI team error requires humans; offline we can only
  *stage* it (simulate the deferral decision and measure the resulting system error, if
  the labels allow)
- **the deployment's own acceptance thresholds** — "ready" is partly their call, not ours
  or the literature's

## 5. The plan

| what to run | what it establishes | source | acceptance criterion | cost |
|---|---|---|---|---|
| **conditional coverage per slice** | marginal coverage isn't hiding a hard group | MAPIE conditional-CP / Mondrian | coverage in [1−α, 1−α+0.05] per group with n≥200 | Kaggle (recompute on cached scores) |
| **deferral simulation** | does escalating actually help? | Mozannar et al. 2023 | system error < model-only error | local (pure logic on logs) |
| **minority over-rejection** | abstention isn't concentrated on a slice | Pugnana & Ruggieri | rejection rate ratio within a band | Kaggle (on cached scores) |
| **sharpened memorization probes** | the model isn't answering the surface | arXiv:2609.37647 | option-rotation accuracy flat; question-withheld ≈ chance | Kaggle |
| **Score tasks on real ordinal data** | `Score` carries graded signal | SemEval STS / G-Eval | Spearman ρ > 0 | Kaggle |
| **the released harness as a check** | our numbers vs the reference's | arXiv:2609.37647 harness | directional agreement | Kaggle |
| **Noul-vs-Choice on the same task** | the binary gate works as a Noul | arXiv:2609.37647 | threshold-free AUROC high; fixed 0.5 tuned on train | Kaggle |

## 6. Fact, inference, recommendation

**Fact.** A published reference evaluation for this model shape exists and is releasable
(arXiv:2609.37647). Our §4 protocol matches most of it. Two measured properties are absent
from both: conditional coverage and deferral quality. The deferral literature names two
failure modes we have not measured (over-reliance, minority over-rejection).

**Inference.** "Ready for human interaction" is a *protocol* plus *named failure modes
excluded*, not a numeric threshold. Our two open `must_fix` items (mnli .449, multimodal
composition) are capability gaps; the genuinely blocking gap for human interaction is
*conditional coverage* — because it can hide inside a good-looking marginal number, which
is the class of bug this project keeps catching.

**Recommendation.** Before Phase 8, run the four cheap ones: conditional coverage per
slice, deferral simulation, minority over-rejection, and the sharpened memorization
probes. All four compute on cached scores or logs — no re-encoding. They are the
difference between "the model is accurate" and "the model is safe to defer to a person",
which is the actual question.

## Sources

1. Deußer, Sparrenberg & Sifa, *Evaluating and Benchmarking the System One Model Jev*
   (arXiv:2609.37647, Univ. Bonn / Lamarr / Fraunhofer IAIS) — the reference evaluation.
2. Ye, Liu & Jiang, *NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision
   Trees* (arXiv:2609.28587) — training-free continuous readout over the primitives.
3. Deng, Fan, Zhang & Xie, *Jev for Scientific Decisions* (arXiv:2609.24965) — the
   model's evaluation in a downstream pipeline.
4. Ling, Xue & Ye, *Jev in the Wild* (arXiv:2609.30216) — ecosystem / latency / cost.
5. Barbosa, *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses*
   (arXiv:2609.28940) — the model/harness boundary and the cost-derived decision rule.
6. MAPIE — conditional / Mondrian conformal (`ConditionalSplitConformalClassifier`),
   group-conditional coverage guarantees; the ~200-samples-per-group caveat.
7. Geifman & El-Yaniv, *Selective Classification for Deep Neural Networks* (NeurIPS 2017)
   — the guaranteed-risk dial.
8. Madras, Pitassi & Zemel, *Predict Responsibly* (NeurIPS 2018) — learning to defer.
9. Mozannar et al., *Who Should Predict?* (ICML 2023) — system error, human over-reliance.
10. Pugnana & Ruggieri (in the MLR selective-classification benchmark) — minority
    over-rejection.
11. Röttger et al., *XSTest* (NAACL 2024) — the two-sided refusal protocol.
12. Hardt & Recht, *Patterns, Predictions, and Actions* — decision theory; Chip Huyen,
    *Designing Machine Learning Systems* — production readiness (library).

*Stale-source flag:* none — all are 2024–2026. *Unverified:* the released harness's exact
URL and the per-dataset numbers beyond what is quoted; the conditional-CP group-size
threshold (~200) is documented but its rigidity is an inference.
