# Recursive structure — PSI & TRM mapped onto NanoCore

**Date**: 2026-10-10 · **Status**: research note, feeds component 47 + ADR-0015 design study
**Sources**: PSI — arXiv:2509.09737 (Stanford NeuroAI, Yamins lab);
TRM — arXiv:2510.04871 (Samsung SAIL Montréal); PTRM/GRAM variants noted via
secondary sources only — treat as directional, not primary evidence.

## What each paper contributes

**PSI (Probabilistic Structure Integration)** — a virtuous cycle, not a model:

1. *Predict*: train Ψ as a **random-access conditional predictor** — the
   complete set of conditionals p(any variable | any subset of others), not a
   single next-token head.
2. *Extract*: pull intermediate structures out of Ψ zero-shot via **causal
   prompting** — counterfactual/hypothetical interventions (what would X be if
   Y were forced to v).
3. *Integrate*: turn extracted structures into **new token types** and mix them
   back as conditioning *and* prediction targets. Repeat — each cycle enlarges
   control surfaces with no architecture growth.

**TRM (Tiny Recursive Model)** — a shape, not a scale:

- One 2-layer, 7M-param network carrying `(x, y, z)`: embedded input, candidate
  answer, **opaque latent workspace**.
- Inner loop refines `z` n times given `(x,y,z)`; outer loop refines `y` given
  `(y,z)`; up to ~16 supervision steps.
- 45% ARC-AGI-1 / 8% ARC-AGI-2 — beats LLMs at <0.01% of the params on ~1k
  examples. Finding: recursion is where capability lives on small data; the
  latent needs no biological justification, it is iterative error-correction.
- PTRM/GRAM variants add stochastic sampling / multi-hypothesis trajectories —
  i.e., *sampling the refinement path*, which is where probabilistic dispatch
  would live.

## The mapping — where each touches NanoCore

| Paper element | NanoCore analogue | Status |
|---|---|---|
| Ψ, random-access predictor | transition head `p(z'|z,a)` (component 43) | planned **single-conditional** — one conditional too narrow |
| Structure extraction | Observation stats; regional competence field (probe: errors cluster spatially, kNN AUC 0.978) | exists |
| Integration (structures → conditioning) | glial modulation of thresholds | hand-coded modulation only; never a conditioning input |
| `(x,y,z)` recursive loop | SlowState: window → anomaly count → streak → τ move | the *shape* exists, written as fixed arithmetic — no learned `z` |
| Deep supervision per step | delayed outcome joins (`/outcome`, component 42) | infrastructure exists; supervision semantics open |
| Multi-hypothesis refinement (PTRM/GRAM) | — | absent |

## The gap, named

**Two gaps, one root.** NanoCore has the *extraction* half of the PSI cycle
(label-free statistics, a spatially-real competence field) but:

1. **The predictor is planned single-conditional.** `p(z'|z,a)` answers one
   query. PSI's lesson: random-access conditioning is what makes extraction,
   integration, and *counterfactual dispatch* (`p(z'|z, do(a=clarify))` — a
   causal prompt, not a reweighting hack) all fall out of one object. Build `f`
   as spec'd and we rebuild it for dispatch later.

2. **The recursive loop is hand arithmetic, not a learned workspace.**
   `SlowState` is literally TRM's `(x,y,z)` with `z` replaced by a streak
   counter and the update rule hand-derived. TRM's result says the learned
   version wins at our data scale. The "internal monologue" is the right frame:
   `y` inspectable at each step, `z` opaque-and-carried — and our `z` is
   currently a hand-rolled integer.

## What to steal vs. what doesn't transfer

- **Steal**: the random-access formulation (maskable variable set over
  trajectory fields — ~10 vars, thousands of rows, a masked head is nearly
  free); the cycle discipline (extract → integrate → enlarge); the (x,y,z)
  shape for a learned regulator.
- **Doesn't transfer**: TRM's dense per-step supervision (puzzles have
  verifiable answers; decisions have sparse delayed outcomes). Architecture
  transfers, supervision semantics are an open design question — candidate
  signals: self-consistency across recursion steps, conformal-correctness
  targets, delayed outcome labels via the join infra (42).
- **Doesn't transfer**: PSI's scale (1.4T video tokens). Steal the cycle, not
  the size.
- **Validates**: frozen EG2 + semantic-latent transitions — neither paper
  requires touching the encoder.

## Where it lands

- **Component 47**: random-access semantic transition predictor — maskable
  trajectory-variable head, not single-conditional `f`.
- **ADR-0015 criterion 1** gains the formulation question and criterion 4 gains
  the "counterfactuals as causal prompts" mechanism.
- The TRM learned-workspace question (a learned `z` replacing hand streak
  arithmetic) is recorded as a design-study question, not a build ticket — its
  supervision signal is genuinely open.
