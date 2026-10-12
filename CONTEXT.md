# NanoCore — Domain Glossary

*Canonical vocabulary. Every term here has one meaning. Terms are resolved as the
grill-with-docs session settles them; implementation details live in code and ADRs,
not here. Derived from the owner's source library (41 books) — see
`docs/research/book-synthesis.md` for the sourced mechanisms behind each term.*

## The organism (what NanoCore is)

- **Cognitive core** — the small *trained* model at the center. Karpathy lineage:
  sacrifices encyclopedic knowledge for capability, looks facts up externally, is
  always-on and tool-using. This is the spine; everything else is an organ on it.
  NOT a frozen encoder (that was the shortcut that drifted into the foundation).

- **Organ** — a subsystem of the one model, not a separate project. The core plus
  its organs are one trained/learned whole. "Multiple systems used together, not
  separate."

- **Hybrid** — the core's organs working as one model: a trained decision core, a
  learned regulator, a composer, and memory. NOT "decision model + a frontier LLM
  bolted on," NOT "two abilities sequenced." One model, internal structure.

## The decision layer (System One)

- **System One** — the fast, typed, calibrated decision layer. Returns typed
  verdicts (Choice/Noul/Score), never generated text. Jev lineage.

- **Choice** — a probability distribution over a caller-supplied option set, plus
  confidence `c = (K·p_max − 1)/(K−1)`.

- **Noul** — `P(yes)` for a binary judgment. The cheapest gate.

- **Score** — a probability-weighted position on an ordered rubric of described
  levels; returns the full level distribution.

- **Judge–executor separation** — the decision core judges; the trained model (or
  an LLM) executes. They share no parameters, no training data, no inductive bias.
  This is what breaks self-evaluation bias. The executor never overrides the judge;
  the layer is strictly additive (can downgrade, never resurrect).

- **Calibration** — reported probabilities match observed frequencies
  (`P[Y=1|R=r]=r`). Measured by ECE and Brier. A decision layer is only as good as
  its calibration; accuracy without calibration is not a usable signal.

- **Abstention options** — explicit typed outputs `not_stated` / `conflicting` /
  `other` with precedence `contradiction > outside > missing`. Uncertainty is a
  typed output, not a guess.

## The regulator (the brake / steering)

- **Glial regulator** — the learned layer that modulates which ability engages and
  how the core behaves, per region of latent state. Biological glial-cell lineage:
  support cells that tune neurons without being neurons. Today's hand-written
  `SlowState` is the embryonic version; the target is *learned*, regional.

  **Sourced mechanism (Artificial Glial Cells review, 22 papers):** a *second,
  slower-dynamics feedback layer*, not signal-path neurons. Per hidden unit a
  scalar state `a_i` on a slow clock; scale its outgoing weight by `(1 + k·a_i)`.
  Triggered by *sustained thresholded firing*, not spikes. Pulse propagation
  across neighbors (PGC calcium-wave) with per-unit refractory periods escapes
  local minima better than noise. **Key negative result: gains come only when the
  astrocyte influence is LEARNED (per-problem, per-neuron), never a fixed global
  gain.** So the regulator's gains are trained, not hand-tuned.

- **Brake** — the part that controls *whether* an answer leaves, never *what*. The
  conformal gate + threshold shifts. Distinct from the wheel.

- **Steering wheel** — the part that controls *what* — which pathway/ability
  engages per context. The wheel-over-abilities is the learned-regulator target.

- **Engine** — the trained core that produces the decision/content.

- **Residual learning** (PINN lineage) — put the structure you already know (a
  rule, invariant, operator) in the loss; let the trained core absorb only the
  correction. The scaling bargain a tiny model needs: learn the residual, not the
  whole function.

## Memory & composition

- **Composer** — turns a multi-item / multi-turn state into one latent vector `z`.
  The substrate the conversational ability and trajectory memory read.

- **Trajectory** — `(state, typed action, outcome, next state)`; the corpus that
  trains the regulator and the transition head. Emitted by every decision.

- **Consolidation** — the "sleep" pass that folds trajectories into durable state
  (cluster → summarize → forget). Biological sleep lineage.

  **Sourced mechanism (Do LLMs Need Sleep):** a fast-weight state per memory
  layer updated by a *gated outer-product rule* `S ← α·S + β·v kᵀ` with both
  gates data-dependent. When the working buffer fills, run **N ≥ 2 offline
  recurrent passes** over the buffer that refine only `S`, then clear the buffer —
  `S` is the only thing that survives. Wake-time prediction stays a single pass.
  Gradient flows through the *refined fast weights*. Measured: gains concentrate
  on the deepest-reasoning instances (cellular automaton t=32: ~10% → >30% with
  3-4 loops). Loop only the *middle* layers to keep cost near-linear in N.

- **Shared latent state `z`** — the one space all organs read and write. EG2 space
  today; a learned subspace is the target.

## The frontier (post-v1 — charted, not built)

Resolved by the frontier grill from the owner's library (world-model / memory
books). **Convergent finding across independent clusters: the literature does not
support a *generative* organ. It supports a *predictive* organ and a *typed
memory* organ — both cleanly separable from generation.** Decide-on-embeddings is
not the ceiling, but the ceiling is crossed by depth over typed history, not by
decoding.

- **World model** — in this literature means **predicting a decision-relevant
  scalar readout**, not reconstructing or generating. Two sources:
  - *Consequence prediction* (Language Models as AI Research World Models):
    `M(state, action, history) → median measured gain`. No rollout, no tokens —
    one scalar by design. **Key ablation: knowledge beats capacity beats effort**
    — the weakest backbone *with* typed records beat the strongest *without*
    (0.638 vs 0.621); adding knowledge moved Spearman 0.609→0.892 vs effort's
    +0.04. Payoff is in **record volume, not head capacity**. This is a frozen
    model + typed history + a predictive readout — structurally v1's trajectory
    log already.
  - *Offline recurrent consolidation / "sleep"* (Do LLMs Need Sleep): fast-weight
    state `S ← α·S + β·v kᵀ` (data-dependent gates), N offline recurrent passes
    refine `S` before buffer eviction, **single-pass prediction preserved at
    decision time** — the cleanest proof that recurrence ≠ generation. **Caveat:
    gains concentrate on the *deepest* instances and saturate on easy ones** — run
    the diagnostic (hold info fixed, vary decision depth) before building.
  - *MuZero statement* (cited): a recurrent model that predicts rewards, values,
    policies **without reconstructing observations**. The value readout *is* the
    world model. This is the non-generative branch; latent rollouts / imagination
    (Dreamer) is the generative branch — **not** needed unless multi-step planning
    is required.

- **Memory-type routing** (MemoType): a single shared-encoder similarity metric
  has a **provable ceiling** `1 − (C−1)δ/2C` on query–doc alignment that tightens
  as the type space grows (Theorem 3.3) — exactly the geometry of a typed decision
  head. Fix: **partition the memory by the type the head already emits, and score
  each partition with a strategy matched to it.** The router needs no generation
  (BERT/RoBERTa within ~1–3 pts of a 1.7B LLM router); Recall@1 +6–9. Per-type
  scoring: event-like → decompose-then-align; preference/state → reverse-HyDE.

- **Typed multi-relational memory plane** (Jev-Mem): one node set, four typed edge
  views (semantic / temporal / causal / entity), deterministic insertion, edge
  weight threshold. Multi-hop 0.528→0.623, adversarial 0.742→0.962. Fully
  separable from generation (typed edges + weights, no tokens).

- **Adaptive stopping** (Jev-Mem): terminate when evidence is sufficient *or*
  expected utility of more search < threshold — the accuracy lever that also
  guards against distractor memories degrading the decision. Two scalar
  thresholds.

- **Valence channel** (Do LLMs Feel?): emotion *readout* is a linear probe on the
  frozen encoder's pooled vector (99.65%-controllable, ~2–4 MLP neurons/layer) —
  a legitimate **decision-gating feature** (bias escalate/abort/confirm). The
  *intervention* half is out of scope (needs a decoder path the frozen encoder
  lacks). Readout only, not control.

- **Consolidation is typed, not sleep-weight** (for a frozen encoder): Jev-Mem's
  consolidation is a *controller decision* (merge / promote / keep-separate over a
  bounded candidate set), not a weight update. Fast weights would mean retraining
  the head — a much bigger step than the framing suggests.

## The training discipline

- **Depth dial** — one integer sets width, LR, token budget, and cost. The
  experimentation unit is minutes, not days (Karpathy).

- **Staged training** — pretrain (representation) → midtrain (format/interface) →
  SFT (answer quality) → RL (only where a verifier exists). Each stage is a cheap
  checkpoint.

- **Distillation** — a teacher (EG2, a frontier model) trains the student; the
  student ships behind a gate that escalates uncertain cases to the teacher, and
  those escalations become the next training set. EG2's permanent role: *teacher
  and cold-start scaffold*, never the inference-time foundation.

- **CORE-style eval** — averaged across many datasets, not validation loss.
  Publish the regressions, not just the wins.
