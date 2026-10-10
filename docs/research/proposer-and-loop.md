# The proposer — what fills the `converse` slot, and why it is not a generator

**Date**: 2026-10-10 · **Status**: research note, feeds component 38 (composer factory)
+ the `converse` ability (Q6) + the loop's propose side (commit `05e31c2`)
**Sources**: Gulli, *Agentic Design Patterns* (Prompt Chaining, Routing, Reflection,
Planning, Tool Use, Memory, Learning); Roitman, *Hitchhiker's Guide to Agentic AI*
§18.5 (ReAct, Plan-and-Execute), §12.3 (Reflexion, off-policy trajectory filtering);
dos Santos, *Calibrated Decision Models for Autonomous Pentest Harnesses*
arXiv:2609.28940 (DP4 confirmation loop); Jev-Mem arXiv:2609.23986 (System-One
control plane); recursive-structure.md (PSI/TRM, in-repo).

## The question

The decision loop (commit `05e31c2`) has a spine — propose → decide → feed the
verdict back — but the **proposer is a stub**. Without it the loop confirms on
iteration 1: there is nothing to iterate. The missing ability is the `converse` /
articulate slot in ARCHITECTURE-SYSTEM.md, and the instinct is "add a generator."
This note argues that is the wrong first move, and names what to build instead.

## The finding: the proposer is a *retrieval + composition* problem, not a generation problem

Every loop pattern in the library separates **what proposes** from **what decides**,
and in every case the proposer is cheap and structured, not a generative LLM:

| Pattern | Proposer | Decider | Source |
|---|---|---|---|
| DP4 confirmation loop | candidate set + filtering behavior (code) | System One Choice/Noul | pentest paper |
| Jev-Mem control plane | query routing, graph traversal (structured) | System One scoring + adaptive stopping | Jev-Mem |
| ReAct | Thought = scratchpad chain-of-thought | Act = tool call, Observe = env | Roitman §18.5.1 |
| Plan-and-Execute | Planner emits a subtask list | Executor runs each (cheaper model ok) | Roitman §18.5.2 |
| Reflexion | verbal self-critique → episodic buffer | next attempt injects the lesson | Roitman §12.3 |

The proposer's job is to **narrow the candidate space using the prior verdict** —
DP4's "informed by the observed filtering behavior," Reflexion's "lessons
injected into the prompt." That is retrieval and composition over what the loop
has already seen, not free generation. Our frozen-encoder + composer architecture
is *already shaped* for this: the composer (component 38) turns the loop's history
`[turn1, verdict1, turn2, verdict2]` into one state vector, and the next
proposal is a retrieval/composition over that composed state.

## Why a generator is the wrong first move

1. **It breaks the frozen-encoder invariant.** ADR-0001 freezes EG2 because
   plastic perception voids every calibrated artifact downstream. A generative
   head over the latent is a *second* learned model with its own calibration
   story — a much bigger commitment than the decision head, which is a linear
   readout.
2. **The loop is correct without it.** commit `05e31c2` runs end-to-end with a
   stub proposer; the gate's action drives iteration. The proposer improves
   *what* gets proposed, not *whether* the loop works.
3. **The library's own efficiency result.** DP4: "each iteration costs one System
   One call (33–40 ms), the loop can evaluate 10–30 candidates in the time a
   single LLM call would take." The proposer is meant to be *cheaper* than the
   decider. A generator inverts that.

## What to build instead — three tiers, in order

**Tier 1 — Retrieval proposer (build now, local, no weights).** The proposer
selects the next candidate from a *candidate set* using the prior verdicts.
Concretely: given the loop's history and a pool of candidates (options, prior
states, retrieved neighbors), score each by composed-state similarity and offer
the best not-yet-tried one. This is `PairScorer` + the composer doing proposal.
It is the DP4 "candidate set + filtering behavior" with our existing pieces.
*Unblocks*: the loop actually iterating on real tasks; Q2's multi-turn legs.

**Tier 2 — Composer factory (component 38).** The substrate Tier 1 reads. Without
it the loop's history is mean-pooled (ordering and who-said-what destroyed — the
same bug as the sentence pair). The composer composes `[cand, verdict, cand,
verdict]` into one vector that preserves structure. This is the load-bearing
piece and it unblocks E4 *and* conversation *and* the loop's memory.

**Tier 3 — Articulate/render (the genuine generator).** Only after 1 and 2 exist
and are measured does a generative head earn consideration — and even then, scoped
to *rendering* a composed state into text, not to proposing from scratch. PSI's
lesson (recursive-structure.md): capability comes from the predict→extract→
integrate cycle, not from a bigger generator. TRM's: recursion over a learned
workspace beats scale on small data.

## The mapping onto what exists

| Loop role | NanoCore piece | Status |
|---|---|---|
| Propose (narrow on verdict) | retrieval over composed state (Tier 1) | **build now** |
| Carry history with structure | composer (component 38, Tier 2) | TODO — E4 blocker too |
| Decide (typed, calibrated) | `decide` + gate | BUILT |
| Loop control | gate action (answer/clarify/escalate/abstain) | BUILT (`05e31c2`) |
| Learn the loop | glial regulator over trajectory (z_t, action, z_{t+1}) | DECIDED (ADR-0014) |
| Articulate/render | `converse` | MISSING — Tier 3, last |

## What to steal vs. what doesn't transfer

- **Steal**: DP4's candidate-set proposer (code-driven, cheap, verdict-informed);
  Reflexion's episodic lesson buffer (our trajectory log *is* this buffer);
  Plan-and-Execute's "re-plan from current state" (= re-propose with the verdict).
- **Doesn't transfer (yet)**: ReAct's generative Thought scratchpad — that is
  Tier 3, and PSI/TRM say recursion over a learned workspace gets there cheaper
  than a generator. Multi-agent orchestration — out of scope for a single-device
  decision core.

## The recommendation

**Build Tier 1 (retrieval proposer) + Tier 2 (composer factory) together** — they
are the same substrate. The retrieval proposer is ~local, stub-testable, and makes
the loop real. The composer is the piece that has blocked E4, conversation, and now
the loop's memory; building it once pays three times. Defer the generator (Tier 3)
until the loop runs on real tasks and we can measure what articulation would add.

This keeps the architecture's identity: a frozen encoder, typed decisions, a
learned slow regulator, and a loop whose proposer is retrieval-and-composition —
not a from-scratch generator bolted on.
