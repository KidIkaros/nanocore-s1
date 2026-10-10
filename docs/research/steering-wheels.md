# What steers a model — the output-selection mechanism per family

*Date: 2026-10 | Status: architectural survey | Question: across model families, what is
the "steering wheel" — the mechanism that selects WHAT the output is, as opposed to the
engine (how internal state evolves) and the brake (whether/how much to trust it)?*

Working decomposition, used throughout:

- **Engine** — how the internal state evolves (attention, recurrence, diffusion steps).
- **Steering wheel** — the mechanism that maps state → a selection from an *answer
  space*. The wheel is always the same shape: **a distribution over a discrete or
  continuous answer space, plus a selection rule**. Families differ in *what the answer
  space is* and *who controls it*.
- **Gauges** — entropy, temperature, confidence: readouts/warps on the wheel, never the
  wheel itself. (In NanoCore, `obs.entropy` is a gauge we record and don't yet read.)
- **Brake** — whether to act on the selection (gates, thresholds, abstention).

---

## Per family

### Autoregressive transformers (GPT / Claude / Gemma / Qwen)
**Wheel: the LM head** — hidden state → logits over the full vocabulary (~100k+), softmax
→ sample or argmax, re-steered *every token*. The answer space is open and enormous; RLHF/
instruct tuning reshapes the landscape the wheel rolls over. Content selection and content
*production* are fused in one matrix — the wheel both steers and speaks.
**Engine:** attention over the KV cache.
**Brake:** none built in — RLHF refusals are the closest thing, and they're a learned
behavior, not a mechanism.

### SSMs / recurrent (Mamba-2, Zamba2, RWKV, GRU/LSTM)
**Wheel: identical** — an LM head over the vocabulary. Mamba-Jev verified this is the
load-bearing fact: swapping the engine (recurrent state for attention) leaves the
steering untouched. The SSM/attention distinction lives entirely in the engine — state
dynamics, memory shape, compute profile — never in the wheel.
**Implication we already rely on:** "which trunk" questions are engine questions; they
do not change what the model can express at the output.

### Encoders (BERT, EmbeddingGemma 2, sentence-transformers)
**Wheel: none.** The model emits coordinates in an embedding space, not an answer.
Steering is whatever consumes the vector — cosine similarity, a trained head, an index.
This is why EG2 + scorer is the honest description of NanoCore's fast path: the wheel is
*ours*, and it's a comparison in geometry, not a projection over vocabulary.

### Classifier heads (any encoder + head)
**Wheel: logits over a caller-fixed label set → argmax.** A bounded wheel — the answer
space is declared, so the model cannot steer outside it. This is the property TypeSafe
markets as "can't hallucinate," and it's structural, not trained.

### Decision models (Jev, Laya, Clef, NanoCore-S1)
**Wheel: scores over a caller-declared, per-request option set → argmax.** The answer
space isn't even fixed per model — it arrives with the request (`questions[].criteria`).
The wheel is bounded AND supplied: the caller draws the road, the model picks the lane.
**Brake (differentiator):** Jev = calibrated-confidence heuristic; NanoCore = conformal
set + typed actions — coverage with a guarantee, not a vibe.

### Diffusion / score-based generative models (image, discrete diffusion LMs)
**Wheel: the score function ∇log p(x)** — literally a gradient field; each denoising step
steers the sample toward the data manifold. Steering is *iterative correction*, not a
single softmax — the wheel is turned hundreds of times per output, in a continuous answer
space. Terminological delight worth noting: diffusion's steering wheel is *called* the
score; our scorer is the same idea discretized.

### Masked LMs (BERT-class fill-in)
**Wheel: per-position logits over vocabulary**, conditioned bidirectionally. A bounded-
scope wheel: steers only masked positions, per position.

### RL / world-model agents (MuZero, Dreamer, LeCun's JEPA line)
**Wheel: the policy head** — state → distribution over *actions*. The answer space is the
action space. Two extra instruments surround it: a **value head** (a gauge — how good is
the trajectory) and **planning** (MCTS / latent rollout — steering by *looking ahead*:
the wheel gets turned on imagined futures before committing). JEPA adds energy
minimization in latent space — steering = descent on a compatibility landscape rather
than a softmax.
**Lesson:** this family proves a wheel can be *learned over an internal action space* —
the direct precedent for a wheel over abilities.

### Mixture-of-Experts (Mixtral, Switch, modern MoE LLMs)
**Wheel: two of them.** The token wheel (LM head, as above) plus the **router** — a
learned per-token gating network that scores which experts fire. The router is a steering
wheel *inside* the model, over internal pathways, learned end-to-end.
**This is the closest shipped analogue to the glial regulator** — a context-conditioned,
learned dispatch between abilities — except it routes per token over FFN experts, where
ADR-0014 routes per decision over abilities. Nobody ships the system-level version.

### Energy-based models (EBMs)
**Wheel: ∇E(x)** — no softmax at all; inference is descent on a learned energy surface.
Selection = optimization. The wheel and the landscape are the same object.

*Postscript (2026-10-10)*: the decision model **is** a tractable EBM — `softmax(scores/t_prob)`
over declared options is a Boltzmann distribution with energy = −score; `t_prob` is the
temperature parameter and `fit_temperature` is energy-scale learning. The EBM catch
(intractable partition function Z) evaporates because the support is enumerable by
construction — the conformal gate brakes *on* a Boltzmann distribution the bounded
option space made cheap. The EBM family didn't get rejected; it got domesticated.

### Neuro-symbolic / program synthesis
**Wheel: search + constraints** — enumeration or SAT/constraint pruning over a formal
space. The wheel is a solver; legibility is total, coverage is brittleness.

### RAG / retrieval stacks
**Wheel: the retriever's similarity ranking** (top-k over an index) — selects *what
context enters*; the generator's token wheel then steers within it. Two wheels in series:
what to read, then what to write.

---

## The generalization

Every family's wheel = **(answer space) × (scoring function) × (selection rule)**:

| Family | Answer space | Scoring | Selection |
|---|---|---|---|
| AR LLM | full vocabulary | LM-head logits | sample/argmax, per token |
| SSM LLM | full vocabulary | LM-head logits | identical — engine differs only |
| Encoder | *none — downstream* | (none) | (none) |
| Classifier | fixed labels | head logits | argmax |
| Decision model | per-request options | scorer over options | argmax + gate |
| Diffusion | continuous sample space | ∇log p(x) | iterative descent |
| RL agent | action space | policy logits | argmax/sample + planning |
| MoE router | experts | gating scores | top-k routing |
| EBM | configuration space | learned energy | gradient descent |
| Symbolic | formal space | constraints | search |

Freedom lives in the *answer space* column: open vocab (LLM), declared options
(decision model), action space (RL), experts (MoE), manifold (diffusion/EBM).
Cost, latency, calibratability, and hallucination-risk all track how open that space is.

## What it means for NanoCore-S1

1. **The fast path's wheel is already the bounded kind** — per-request options, argmax,
   conformal brake. That part is settled architecture.
2. **Entropy/temperature are gauges, not wheels** — the thing the user called entropy is
   a dashboard instrument; the wheel is the score→argmax over the declared options.
3. **ADR-0014's regulator is "a wheel over wheels"** — the MoE-router pattern lifted one
   level: a learned, context-conditioned scorer over the ability space (decide /
   converse / escalate / abstain / glial-modulate). The RL family proves wheels over
   internal action spaces are learnable; MoE proves learned dispatch composes cleanly at
   scale. The glial contribution from `glial-regulator.md` supplies *when* it runs
   (slow timescale, consolidated) and *what* it conditions on (regional competence).
4. **The articulated/conversational channel is another wheel position** — "which way to
   express this decision" is a bounded-option question of the same shape; articulation
   rides the same mechanism, not a new one.
5. **The open question the survey surfaces:** every wheel above is *trained on its own
   task signal*. The wheel-over-abilities' training signal is outcome-joined logs —
   component 42 remains the prerequisite that turns this survey into a build order.

## Canonical references

Vaswani et al. 2017 (attention); Gu & Dao 2023 / Dao & Gu 2024 (Mamba); Song et al. 2021
(score-based diffusion); Schrittwieser et al. 2020 (MuZero); Hafner et al. (Dreamer);
LeCun 2022 (JEPA/energy); Shazeer et al. 2017 (MoE); Lewis et al. 2020 (RAG);
arXiv:2609.37647 + TypeSafe jev-1.13.0 docs (System One decision models);
Alvarez-Gonzalez et al. 2023 (artificial glia); de Garis 1996 (CAM-Brain);
Lee et al. 2026 (LLM sleep, arXiv:2605.26099).
