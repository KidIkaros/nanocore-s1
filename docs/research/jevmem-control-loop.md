# How others built it — Jev-Mem's System-One control loop, as a blueprint

**Date**: 2026-10-10 · **Status**: research note, feeds component 38 (composer factory)
+ the retrieval proposer (T1) + the loop's stopping rule (commit `05e31c2`)
**Source**: Jev-Mem — arXiv:2609.23986 (Jiang, Li & Li, UT Dallas, 2026-09-21);
LoCoMo eval, code public. Companion to `proposer-and-loop.md`.

## Why this paper, specifically

Jev-Mem is the closest *built* system to what NanoCore is becoming: a
System-One typed-decision layer acting as a **control plane** around a
generative reasoner, with the loop's stopping decided by the same typed
decisions that do the work. It is not a proposal — it ran, and the numbers
are the argument: LoCoMo LLM-as-judge 0.777 (+11% over the strongest
baseline), memory construction 6.6× faster, query latency −36.7%.

## The one idea to steal: memory control is a first-class control loop

The paper's thesis sentence: *"many memory-control operations are semantic
but not generative… they produce bounded outputs such as labels,
probabilities, or scores rather than free-form text."* So it moves every
high-frequency decision — typing, routing, budget, traversal, scoring,
**stopping** — out of the generative path and into a typed System-One
controller `J(S, Q)` that returns probabilities over a *known, small output
space*. System Two (the generator) is invoked only for synthesis.

That is exactly the `decide`-serves-the-loop shape we just built. The
inspiration is in **how they close the loop**, which is the piece our
`run_loop` is missing.

## The retrieval loop, concretely (their §3.3)

Retrieval is a **closed-loop control process, not a single top-k search**:

1. **Route** — predict per-view relevance `p_g(q)` over relation types
   {semantic, temporal, causal, entity}; activate a view when `p_g ≥ θ_act`.
   Multiple views can fire at once (independent probabilities, not one
   discrete intent).
2. **Budget** — distribute a total expansion budget B across active views
   proportionally to `p_g(q)^γ`. Don't spend equal effort everywhere.
3. **Anchor** — fuse vector + lexical rankings (reciprocal-rank fusion,
   κ=60) to seed the frontier cheaply before any expensive graph step.
4. **Expand + score** — each candidate scored on five dimensions:
   embedding similarity `z_v`, query relevance `a_v`, relation usefulness
   `ℓ_v`, novelty `n_v`, evidence support `c_v` — combined with learned
   weights λ into one transition score.
5. **Stop** — the load-bearing part. After each round the controller
   estimates four scalars: evidence **sufficiency** `s_d`, expected
   **utility of more** `u_d`, **missing** evidence `m_d`, unresolved
   **contradiction** `c_d`. Terminate when
   `s_d ≥ θ_suff ∧ m_d < θ_cont ∧ c_d < θ_cont`, **or** when
   `u_d < θ_cont` — "more search won't help even if evidence is
   incomplete."

## The mapping onto NanoCore — line by line

| Jev-Mem | NanoCore | Status |
|---|---|---|
| System-One controller `J(S,Q)` | `decide()` + gate | BUILT |
| Typed, bounded output space | Choice/Noul/Score qtypes | BUILT |
| Multi-view routing `p_g(q)` | glial regulator: which ability per region of z | DECIDED (ADR-0014) |
| Budget allocation `b_g` | retrieval-budget / candidate-count cap | absent |
| Anchor = RRF(vector, lexical) | PairScorer cosine + option scoring | BUILT (pair) |
| 5-dim candidate score | Observation stats + regional field | partial |
| **Stop rule** (sufficiency ∧ ¬missing ∧ ¬contradiction ∨ ¬utility) | gate action `answer`/`abstain` | **the gap to close** |
| Memory plane (canonical nodes + typed edges) | trajectory log `(z_t, action, z_{t+1})` | PARTIAL (component 42) |
| System Two synthesis | `converse`/articulate | MISSING (Tier 3) |

## The three concrete inspirations

**1. The stop rule is four typed estimates, not one threshold.** Our
`run_loop` stops on a single gate action. Jev-Mem stops on a *conjunction*:
enough evidence AND nothing missing AND no contradiction — OR the expected
value of continuing is below a floor. That second clause (`u_d < θ_cont`,
"stop even if incomplete, because more won't help") is the one we lack, and
it is exactly the glial regulator's job: a learned estimate of "is another
iteration worth it in this region of z." Our gate already produces the raw
material (top_prob, set_size, margin, entropy); the stop rule composes them
the way Jev-Mem composes its four scalars.

**2. Routing is per-view probability, not one intent.** Jev-Mem activates
several relation views at once because the probabilities are independent.
Our glial regulator is specified the same way (ADR-0014: a regional
competence field that *conditions* the fast path, not a dispatcher that
picks one ability). Jev-Mem is the proof that this shape works in
production — it validates the ADR-0014 direction against a shipped system.

**3. The memory plane is canonical nodes + typed edges — which is our
trajectory log.** Jev-Mem keeps the original observation, never
store-or-discard at ingestion; selectivity comes later, at construction and
retrieval. Our component-42 trajectory log already does this: `raw_scores`
is the canonical node, `state_hash` the fingerprint, `prev_id`/`seq` the
edges. We are closer to Jev-Mem's memory plane than it appears — the edges
just aren't *typed* yet (semantic/temporal/causal vs our single "prev").

## What does NOT transfer

- **The multi-relational graph.** Four typed edge sets is right for a memory
  system querying a long history; for a single-device decision core the
  trajectory is short and mostly temporal. Steal the *typed-edge idea* (tag
  each `prev_id` with what kind of continuity it is), not the full graph.
- **RRF anchor fusion.** Assumes a lexical index we don't run. Our anchor is
  the composed state; the cosine PairScorer already covers the vector half.
- **The λ-weighted 5-dim score.** Five learned weights over candidate
  features is the glial field's eventual job, not a hand-rolled formula —
  build it as the learned regulator (component 41), don't hard-code λ.

## The recommendation, sharpened

The retrieval proposer (T1 from `proposer-and-loop.md`) should be built as a
**Jev-Mem-style closed loop**: propose a candidate → `decide` → score the
candidate on the gate's own statistics → **stop when the composed stop rule
fires**, not on a single action. The stop rule is the concrete thing to
steal: sufficiency ∧ ¬missing ∧ ¬contradiction ∨ ¬utility, estimated from
the Observation row we already log. That turns `run_loop` from
"iterate until one action says stop" into "iterate until the loop knows
another round won't help" — which is the difference between a loop and a
*controlled* loop, and it is the piece that makes the glial regulator
load-bearing rather than decorative.
