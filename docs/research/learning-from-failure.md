# Learning from bad answers without punishing them: research report

*Generated: 2026-10-08 | Sources: 24 | Confidence: High on the mechanism and failure
modes (multi-source), Medium on individual production numbers (single-source each)*

**Question investigated.** Can a model learn from its bad answers — wrong, overconfident,
or unanswerable — *without* a punitive objective, and self-regulate? Which observable
signal carries "badness" when no label is available? What stops such a loop from
entrenching its own errors?

Scope is deliberately narrow: what is buildable in `nanocore-s1`'s decision layer, where
the encoder and head stay frozen and the only moving part is policy.

---

## Executive summary

The idea is correct, has a name, and has been deployed — but every version that works
carries an **external anchor**, and the versions that omit one are measurably harmful.

1. **"Don't punish, learn from the error" is active inference.** Replacing external reward
   with intrinsic free-energy minimisation is argued explicitly as the fix for reward
   engineering's scaling bottleneck ([Missing Reward, 2025](https://arxiv.org/html/2508.05619)).
2. **The failure signal must be externally grounded, not introspective.** Intrinsic
   self-correction *degrades* performance without external feedback — earlier positive
   results used oracle labels ([Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)).
   The bottleneck is **finding** the error, not fixing it; a small out-of-domain classifier
   finds mistakes better than prompting a large model
   ([ACL Findings 2024](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.826.pdf)).
3. **The label-free carrier exists and is standard.** Softmax response is among the best
   certainty proxies for a trained network, alongside functional margin and distance to
   nearest neighbours ([SelectiveNet, ICML 2019](http://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf);
   [Art of Abstention, ACL 2021](https://aclanthology.org/2021.acl-long.84.pdf)).
4. **The anchor has a proof.** Adaptive conformal inference achieves target coverage under
   *arbitrary* distribution shift by tracking a single parameter
   ([Gibbs & Candès, NeurIPS 2021](https://arxiv.org/html/2106.00170v3)); the 2024 JMLR
   extension tunes the step size online, so it adapts to both the size and type of shift
   with provably small regret ([JMLR 2024](https://www.jmlr.org/papers/v25/22-1218.html)).
5. **The failure mode has a name.** Confirmation bias: "those confident but wrong
   predictions would be used to guide subsequent training, leading to a loop of
   self-reinforcing errors" ([Arazo et al. 2019](https://ar5iv.labs.arxiv.org/html/1908.02983)).
6. **Learning from failure without punishment is already deployed** in production routers:
   cluster escalation failures → targeted distillation → retrain, in a closed loop
   ([RouteNLP, 2026](https://arxiv.org/html/2604.23577)).

**Bottom line for us:** the idea is buildable, and our measured result already points the
right way — but the mechanism we built (a hand-tuned Schmitt trigger) is the *heuristic*
version of a thing that has a provable form (ACI/DtACI), and the durable loop should be a
**mistake notebook with a keep/revert gate**, not a weight update.

---

## 1. The idea has a name, and the framing is right

Active inference replaces external reward with intrinsic free-energy minimisation. The
2025 position paper argues the "Era of Experience" still depends on hand-engineered reward
functions — shifting the bottleneck "from data curation to reward curation" — and that
free-energy minimisation removes that dependency, with LLMs as generative world models
inside the AIF loop ([Missing Reward, 2025](https://arxiv.org/html/2508.05619)).

The precursor result is explicit that **no reward notion is needed**: "we do not need to
invoke the notion of reward, value or utility" — agents minimise free energy and reproduce
RL-optimal policies on the mountain-car benchmark
([Friston et al., RL or Active Inference?](https://www.fil.ion.ucl.ac.uk/~karl/Reinforcement%20Learning%20or%20Active%20Inference.pdf)).
Free energy also has the right mathematical character for a regulator: it is a Lyapunov
function for neuronal dynamics ([Process Theory](https://activeinference.github.io/papers/process_theory.pdf)).

Our own prior work sits here too: kindred implemented predictive coding
(`free_energy.rs`) with glial drives modulating *how much* to learn from surprise, and
never measured it.

## 2. Four patterns that learn from failure without punishing — all with a gate

| pattern | how it uses failure | the gate that makes it safe |
|---|---|---|
| **Agent-R** (2025) — explicitly "unlike traditional methods that reward or penalize actions solely based on correctness", it uses MCTS to **recover correct trajectories from erroneous ones** | converts a failure into a recoverable trajectory rather than a penalty | tree search finds the first error step — an external search, not introspection ([2501.11425](https://arxiv.org/html/2501.11425)) |
| **Mistake Notebook Learning** (2025) — self-curate generalisable guidance from **batch-clustered failures**, distilled into "mistake notes" in external memory | failure becomes durable guidance, training-free | "updating an external memory **only when batch performance improves** to ensure stability" ([2512.11485](https://arxiv.org/pdf/2512.11485)) |
| **Failed-action-aware objective** (ACL 2025) — assign **zero return** at failure steps instead of a penalty, then perturb unsuccessful trajectories into successful ones | suppresses the negative impact of failure while still experiencing it | returns inferred from **textual feedback** ([ACL 2025](https://aclanthology.org/2025.acl-long.1526.pdf)) |
| **Early Experience** (2025) — interaction data where future states serve as supervision **without reward signals** | the environment's own next state is the teacher | "in environments with **verifiable rewards**", early experience is the bridge to RL ([2510.08558](https://arxiv.org/pdf/2510.08558v2.pdf)) |

Also: failure-based reward densification, where a discriminator measures dissimilarity
from failures to generate dense reward without expert demonstrations
([AAMAS 2025](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2792.pdf)).

The pattern across all four: **no penalty, but always a gate** — an external search, a
batch-improvement criterion, textual feedback, or verifiable reward. "Don't punish" is
viable; "don't verify" is not.

## 3. The label-free carrier: what signal predicts an error

The selective-prediction literature is the canonical answer, and it is unglamorous:
**softmax response** is among the best certainty proxies for an already-trained network,
alongside functional margin and nearest-neighbour distance
([SelectiveNet, ICML 2019](http://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)).
For NLP specifically, softmax response and MC-dropout are the standard estimators, and
error-regularisation during training improves confidence estimation without much extra
compute ([Art of Abstention, ACL 2021](https://aclanthology.org/2021.acl-long.84.pdf)).
A rigorous survey across six GLUE tasks found the field emphasises **ranking** predictions
over absolute confidence ([ACL 2023](https://aclanthology.org/2023.acl-long.437.pdf)).

**This validates our proxies and bounds their ambition.** Margin, set size, and entropy
are the standard family; we should expect them to rank well, not to be perfectly
calibrated — which is exactly why the escalation threshold needs calibration against
outcomes rather than being read off the score.

The open-set variant matters for us: **UniEnt** minimises entropy on pseudo-in-distribution
samples and **maximises** it on pseudo-OOD samples
([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Unified_Entropy_Optimization_for_Open-Set_Test-Time_Adaptation_CVPR_2024_paper.html)).
Our cross-task stream is exactly an open-set situation, and the asymmetry (minimise on
in-schema, maximise on novel) is the shape a correct signal should take.

## 4. The anchor: adaptive conformal inference

This is the most important find, because it is the rigorous version of what we built.

ACI forms prediction sets online when the data-generating distribution varies arbitrarily,
by **modelling the shift as a single parameter that is continuously re-estimated**, and
provably achieves the target coverage frequency over long intervals *irrespective of the
true data-generating process* ([Gibbs & Candès, NeurIPS 2021](https://arxiv.org/html/2106.00170v3)).
The 2024 JMLR extension removes ACI's dependence on knowing the rate of change by tuning
the step size online, with provably small regret over local intervals
([JMLR 2024](https://www.jmlr.org/papers/v25/22-1218.html)).

Map onto our mechanism:

| ours | ACI / DtACI |
|---|---|
| Schmitt trigger, hand-tuned `activate_at` / `release_at` / `drift_margin` | single tracked parameter, **coverage** is the objective |
| no guarantee; we measured a flicker bug when drift sat on the margin | provable coverage under arbitrary shift |
| step size fixed by `tau_step` | step size **tuned online** (DtACI) |

We independently discovered the *shape* (a slow single-parameter update on observable
statistics, acting on a threshold, not weights). ACI/DtACI supplies the guarantee and
removes the three constants we found fragile.

Complementary deployed evidence that this framing is the right one:

- **UCCI** maps token-margin uncertainty to a per-query **error probability** by isotonic
  regression, then picks the escalation threshold by **constrained cost minimisation**;
  proves threshold policies on the calibrated score are cost-optimal under three stated
  assumptions; 31% cost cut at micro-F1 0.91 with ECE 0.12 → 0.03 on 75k production
  queries — and **beats entropy thresholding, split-conformal routing, and a
  FrugalGPT-style learned threshold** ([UCCI, 2026](https://arxiv.org/html/2605.18796)).
- **RACER** frames routing as minimising expected set size subject to a risk bound, with
  distribution-free risk control and abstention
  ([RACER, 2026](https://arxiv.org/html/2603.06616v1)).
- **HCMA** calibrates abstention with ~50–100 labelled examples, cutting ECE 50% versus
  Platt scaling; and notes **chain-of-thought is ineffectual for selective prediction**
  while zero-shot prompting drives error to 0% on TruthfulQA at high abstention
  ([2410.02173](https://arxiv.org/pdf/2410.02173)).
- Operational framing: set escalation and abstention thresholds from **held-out evidence
  and explicit human-review capacity, not self-reported confidence**
  ([vdf.ai](https://vdf.ai/blog/local-model-router-calibration-escalation-abstention/)).

## 5. The failure mode: intrinsic self-correction degrades

The counter-evidence is strong and specific:

- **"LLMs cannot self-correct reasoning yet"** — without external feedback, performance
  often *degrades* after self-correction; improvements in earlier work came from using
  oracle labels to guide it ([Huang et al., ICLR 2024](https://arxiv.org/abs/2310.01798)).
- **The bottleneck is error location, not correction** — given the error location,
  correction is robust; and a small out-of-domain classifier outperforms prompting a large
  model at finding mistakes
  ([ACL Findings 2024](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.826.pdf)).
- **The dark side** — intrinsic self-correction causes answers to waver, introduces prompt
  bias and human-like cognitive bias, and can overturn *correct* answers
  ([2412.14959](https://arxiv.org/html/2412.14959)).
- **Confirmation bias** — the formal name for the entrenchment risk: confident-but-wrong
  predictions guide subsequent training, and the errors accumulate
  ([1908.02983](https://ar5iv.labs.arxiv.org/html/1908.02983); mitigations in
  [Robust LR](https://arxiv.org/pdf/2112.02960), [TaMatch](https://ar5iv.labs.arxiv.org/html/2409.18316),
  [TrustMatch](https://openaccess.thecvf.com/content/ICCV2025W/BISCUIT/papers/He_TrustMatch_Mitigating_Pseudo-Label_Bias_in_Semi-Supervised_Learning_with_Trust-Aware_Refinement_ICCVW_2025_paper.pdf)).
- **Uncontrolled entropy minimisation makes models overconfident** and can produce
  degenerate solutions in online test-time adaptation; noisy samples destabilise the
  update ([Protected TTA, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9b35a0a20d617dc68ae98a7a57df2f51-Paper-Conference.pdf);
  [REALM, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Seto_REALM_Robust_Entropy_Adaptive_Loss_Minimization_for_Improved_Single-Sample_Test-Time_WACV_2024_paper.pdf)).

Notably, the NeurIPS 2024 TTA paper claims precisely our requirement — it "improves
test-time accuracy under distribution shifts **while maintaining accuracy and calibration
in their absence**" — and achieves it by *matching* the test entropy distribution to the
source's, rather than by minimising entropy. That is a direct alternative to our
"become conservative under drift" rule: hold the distribution invariant instead of moving
the threshold.

## 6. The glial side: what is published, and how it lines up

The glia-inspired computing literature has moved fast and, importantly, has converged on
**few parameters, no retraining, and structure over weights**.

| work | mechanism | result |
|---|---|---|
| [GliaNet, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_GliaNet_Adaptive_Neural_Network_Structure_Learning_with_Glia-Driven_CVPR_2025_paper.pdf) | Glia unit (oligodendrocytes select neurons, astrocytes modulate) adaptively optimises **network structure during training** | advances SOTA while significantly reducing parameters |
| [ANAN, NeurIPS 2024 workshop](https://research.latinxinai.org/papers/neurips/2024/pdf/Ana_Ribas-Rodriguez.pdf) | artificial astrocytes modulate a **pre-trained** CNN's synaptic weights, **only four parameters optimised, no retraining** | statistically significant gains on four datasets; "resource-saving alternative to fine-tuning" |
| [Gong et al., PLOS Comp Biol 2024](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012186) | nested neuron–astrocyte feedback loops over **separated time-scales**; astrocytic modulation as **meta-plasticity** | networks with astrocytes learn **far more reliably across multiple fluctuating contexts** in bandit RL |
| [Neuron–astrocyte associative memory, PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2417788122) | astrocytes as Dense Associative Memory units; memories stored in **astrocytic process networks, not just synapses** | **superior memory scaling law**; framework spans Dense Associative Memory and Transformer as limiting cases |
| [Astrocyte SNN, 2025](https://arxiv.org/pdf/2503.06798) | astrocyte-like units in a liquid state machine | neurons+astrocytes together are critical; **highest learning rate at ~2:1 astrocyte:neuron** |
| [Volume transmission, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/da4dffbe0ea5181d0ad8c59e1641cecd-Paper-Conference.pdf) | neuromodulation implements **context factorization**; multiplicative gating | **targeted online credit assignment**, localising updates to relevant contexts; emergent RPE encoding |
| [What can astrocytes compute?](https://www.biorxiv.org/content/10.1101/2021.10.20.465192v2) | calcium waves as a function approximator | universal, at **higher computational complexity** |

Three things line up with what we measured:

1. **The benefit is conditional on fluctuating context.** The PLOS paper's claim — astrocytic
   modulation enables learning "where fluctuations in task parameters may occur much more
   slowly than within-task requirements" — is exactly our result: **zero advantage at
   novelty 0.00, +0.302 at 1.00**, and SSM's D11 null on static data. The literature does
   not claim a win on stationary input either.
2. **Few parameters, no retraining.** ANAN's four-parameter modulation of a frozen model is
   structurally our slow state. This is the published, defensible framing for what we built.
3. **Memory scaling is where glia are claimed to add most.** PNAS 2025 claims a superior
   memory scaling law with memories held in astrocytic process networks rather than
   synapses. That is the strongest published support for your memory intuition — and it is
   a *capacity* claim, testable, not a metaphor.

## 7. What this means for nanocore

Concrete, ordered by (value ÷ effort):

1. **Replace the hand-tuned anomaly criterion with a calibrated error-probability model.**
   Isotonic regression from observable signals (margin, set size, entropy, raw max score)
   → P(wrong), as UCCI does. Then set the escalation threshold by constrained cost
   minimisation — which is our roadmap's "expected-loss action rule", and UCCI proves
   cost-optimality under three stated assumptions. This also fixes the thing we measured:
   our label-driven arm optimised `tau_answer` while the binding constraint was
   `tau_in_schema`.
2. **Adopt ACI/DtACI for threshold adaptation** instead of the Schmitt trigger. Same shape,
   but with a coverage guarantee and a self-tuning step size — removing the three
   constants (`activate_at`, `release_at`, `drift_margin`) we found fragile.
3. **Build the durable loop as a mistake notebook, not a weight update.**
   Cluster failures in batches → distill guidance → update memory **only when batch
   performance improves** (MNL). Training-free, which is the only option compatible with a
   frozen encoder, and it has the keep/revert gate built in.
4. **Keep the failure signal externally grounded.** Our in-schema boundary plus coverage is
   the external anchor; confirmation bias is the documented consequence of dropping it.
5. **Do not build intrinsic self-correction.** The evidence says it degrades, and it is the
   exact thing that looks like your idea but is not — "learn from bad answers" works when
   badness is measured externally, and fails when the model is asked to judge its own
   answers without an anchor.

**One caution on the emotion/glial analogy.** The 2024–25 glial-ML papers are about
structure, meta-plasticity, and memory capacity. None of them validate affect as a signal;
the emotion-glial link remains kindred's unmeasured extension, not a supported design.

---

## Key takeaways

1. **Your instinct is right and has a literature** — active inference explicitly replaces
   punitive reward with free-energy minimisation, and failure-driven learning without
   penalty is deployed in production routers.
2. **"Badness" must be observable, or it is not a signal.** Error location, not correction,
   is the bottleneck — and a small out-of-domain classifier beats a large model at it.
3. **The anchor is the whole game.** Every working pattern has one: a batch-improvement
   gate, an external search, textual feedback, verifiable reward, or conformal coverage.
   Without one, confirmation bias accumulates confident errors.
4. **Our mechanism is the heuristic version of a provable one.** ACI/DtACI gives the
   coverage guarantee and removes the fragile constants we had to hand-tune.
5. **The escalation threshold is a cost decision, not an accuracy decision** — calibrated
   against held-out evidence and review capacity, never self-reported confidence.
6. **Glia's strongest published claim is memory capacity** (PNAS 2025, superior scaling
   law), which makes the memory direction the best-supported of your glial ideas.

## Sources

1. [The Missing Reward: Active Inference in the Era of Experience](https://arxiv.org/html/2508.05619) — AIF replaces reward engineering; grounded-agency gap (2025)
2. [Reinforcement Learning or Active Inference?](https://www.fil.ion.ucl.ac.uk/~karl/Reinforcement%20Learning%20or%20Active%20Inference.pdf) — no reward/value/utility needed
3. [Active Inference: A Process Theory](https://activeinference.github.io/papers/process_theory.pdf) — free energy as a Lyapunov function
4. [Free Energy Projective Simulation](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0331047) — internal rewards only, interpretable
5. [Gradient-Free De Novo Learning](https://pmc.ncbi.nlm.nih.gov/articles/PMC12468873/) — active inference for structure learning
6. [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) — intrinsic self-correction degrades
7. [LLMs cannot find reasoning errors, but can correct them given the error location](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.826.pdf)
8. [Understanding the Dark Side of LLMs' Intrinsic Self-Correction](https://arxiv.org/html/2412.14959)
9. [Pseudo-Labeling and Confirmation Bias in Deep SSL](https://ar5iv.labs.arxiv.org/html/1908.02983)
10. [Robust Label Refurbishment](https://arxiv.org/pdf/2112.02960) — errors accumulate under confirmation bias
11. [TaMatch: debiased SSL](https://ar5iv.labs.arxiv.org/html/2409.18316)
12. [TrustMatch (ICCV 2025 workshop)](https://openaccess.thecvf.com/content/ICCV2025W/BISCUIT/papers/He_TrustMatch_Mitigating_Pseudo-Label_Bias_in_Semi-Supervised_Learning_with_Trust-Aware_Refinement_ICCVW_2025_paper.pdf)
13. [The Art of Abstention (ACL 2021)](https://aclanthology.org/2021.acl-long.84.pdf) — selective prediction for NLP
14. [SelectiveNet (ICML 2019)](http://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf) — softmax response as a certainty proxy
15. [On the Evaluation of Neural Selective Prediction for NLP (ACL 2023)](https://aclanthology.org/2023.acl-long.437.pdf)
16. [Selective prediction-set models with coverage guarantees](https://ar5iv.labs.arxiv.org/html/1906.05473)
17. [Adaptive Conformal Inference Under Distribution Shift (NeurIPS 2021)](https://arxiv.org/html/2106.00170v3)
18. [Conformal Inference for Online Prediction with Arbitrary Distribution Shifts (JMLR 2024)](https://www.jmlr.org/papers/v25/22-1218.html)
19. [UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing](https://arxiv.org/html/2605.18796)
20. [RouteNLP: Closed-Loop LLM Routing](https://arxiv.org/html/2604.23577) — clusters escalation failures, retrains the router
21. [RACER: Risk-Aware Calibrated Efficient Routing](https://arxiv.org/html/2603.06616v1)
22. [Efficiently Deploying LLMs with Controlled Risk (HCMA)](https://arxiv.org/pdf/2410.02173)
23. [Protected Test-Time Adaptation via Online Entropy Matching (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9b35a0a20d617dc68ae98a7a57df2f51-Paper-Conference.pdf)
24. [REALM (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/papers/Seto_REALM_Robust_Entropy_Adaptive_Loss_Minimization_for_Improved_Single-Sample_Test-Time_WACV_2024_paper.pdf)
25. [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://ar5iv.labs.arxiv.org/html/2006.10726)
26. [UniEnt: Open-Set Test-Time Adaptation (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Unified_Entropy_Optimization_for_Open-Set_Test-Time_Adaptation_CVPR_2024_paper.html)
27. [GliaNet (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Han_GliaNet_Adaptive_Neural_Network_Structure_Learning_with_Glia-Driven_CVPR_2025_paper.pdf)
28. [ANAN: astrocyte-inspired CNNs without training (NeurIPS 2024 workshop)](https://research.latinxinai.org/papers/neurips/2024/pdf/Ana_Ribas-Rodriguez.pdf)
29. [Astrocytes as a mechanism for contextually-guided network dynamics (PLOS Comp Biol 2024)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012186)
30. [Neuron–astrocyte associative memory (PNAS 2025)](https://www.pnas.org/doi/10.1073/pnas.2417788122)
31. [Characterizing Learning in Spiking Neural Networks with astrocytes](https://arxiv.org/pdf/2503.06798)
32. [Volume Transmission Implements Context Factorization (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/da4dffbe0ea5181d0ad8c59e1641cecd-Paper-Conference.pdf)
33. [What can astrocytes compute?](https://www.biorxiv.org/content/10.1101/2021.10.20.465192v2)
34. [Modeling neuron-astrocyte interactions (PLOS Comp Biol)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013503)
35. [When Should a Local Model Router Escalate or Abstain?](https://vdf.ai/blog/local-model-router-calibration-escalation-abstention/)
36. [Agent-R: reflect via iterative self-training](https://arxiv.org/html/2501.11425)
37. [Mistake Notebook Learning](https://arxiv.org/pdf/2512.11485)
38. [Teaching Text Agents to Learn from Failure (ACL 2025)](https://aclanthology.org/2025.acl-long.1526.pdf)
39. [Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558v2.pdf)
40. [On-Policy RL From Failure via Sparse Reward Densification (AAMAS 2025)](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2792.pdf)

## Methodology

Eight search queries across the mechanism (active inference, failure-driven learning),
the label-free carrier (selective prediction, abstention, uncertainty calibration), the
anchor (adaptive conformal inference), the failure mode (self-correction limits,
confirmation bias), the glial literature (2024–25 astrocyte/glia-inspired computing), and
deployment practice (LLM cascade routing and abstention thresholds). Four sources read in
full: the 2025 active-inference position paper, the PLOS 2024 astrocyte paper, the
selective-prediction/ACI cluster, and the deployment-router cluster. 40 sources listed.

**Limitations.** `firecrawl` and `exa` MCP servers are not configured in this environment,
so searches used the built-in web search and fetch rather than the skill's preferred tools;
coverage is therefore narrower than a full firecrawl crawl would give. Production numbers
(UCCI 31%, RouteNLP 58%, HCMA 30%) are single-source and vendor- or author-reported.
Two claims are flagged as unverified single-source: the PNAS "superior memory scaling law"
and the ~2:1 astrocyte:neuron optimum — both are consistent with the wider literature but
rest on one paper each.
