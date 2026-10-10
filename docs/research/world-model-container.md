# NanoCore as world-model-container — survey of the field + what it changes

*Date: 2026-10 | Status: synthesis | Source: JuanRafap's HF "world-models" collection
(14 papers) + canonical WM literature | Question from the owner: what if NanoCore's
abilities lived in a state/latent space — a world model as a container the system calls
when it needs to?*

---

## The finding that legitimizes the framing

**The field's own emerging definition is already the container view.** OpenWorldLib
(Zeng et al., Peking/Kuaishou/Tsinghua/NUS, Apr 2026, arXiv:2604.04707) proposes the
field's unified definition:

> *"a world model is a model or framework **centered on perception, equipped with
> interaction and long-term memory capabilities**, for understanding and predicting
> the complex world."*

and implements it as **modules**: Operator, Representation, Synthesis, Reasoning,
Memory, Pipeline. A world model is no longer claimed to be a monolithic net — it's a
perception-centered framework with an internal model of state. NanoCore-as-container
is not a crazy stretch; it's the vocabulary the field converged on.

## What a world model actually is, mechanically

Stripped to the load-bearing piece (Ha & Schmidhuber 2018 → PlaNet/Dreamer → RSSM):
a world model is a **learned transition function in latent space**:

```
z_{t+1} = f(z_t, a_t)      — given current latent state and an action, predict the next
```

plus decoders for whatever must be read out (reward, observation, answer). The magic
is *imagination*: roll the transition function forward without touching the world —
evaluate candidate actions by their predicted consequences before committing.

## The three collection papers that matter most for NanoCore

### 1. MobileWorldBench / MobileWorld (UCLA/Panasonic/Salesforce, arXiv:2512.14014)
**Semantic world modeling** — state transitions described in *language/structured
semantics* instead of pixels. For GUI agents, predicting "what changes" semantically
beat pixel-space prediction and lifted AndroidWorld success +7.4%.
**Why it matters to us:** the world model doesn't have to render the world — it has to
predict the *decision-relevant* state change. NanoCore's world is a decision domain,
not a screen. A WM over EG2 *embedding-space transitions* is a legitimate member of
the semantic-WM class — and it inherits the on-device relevance: this paper's setting
is literally mobile agents.

### 2. Looped World Models (FaceMind, arXiv:2606.18208)
Parameter-shared transformer block iteratively refines latent environment state —
**adaptive compute depth** scaling to the difficulty of each step, ~100× parameter
efficiency. **Why it matters:** iterative latent refinement is the same axis as the
sleep paper's consolidation and our "do real work between calls" constraint — a tiny
model can buy capability with *loop iterations*, not parameters. Fits the on-device
budget story.

### 3. WorldMM (KAIST, arXiv:2512.02425)
Three complementary memories (episodic multi-timescale, semantic KG, visual) plus an
**adaptive retrieval agent that iteratively selects which memory to query** — a learned
dispatcher over internal resources, exactly the dispatch-over-abilities shape. Their
finding: adaptive selection of *which memory* beats fixed retrieval — evidence that
learned dispatch over internal components pays.

### Honorable mentions from the collection
WMPO (policy optimization *inside* the world model — policy and WM co-trained);
In-Context World Modeling (ICL as the dynamics model — no learned transition at all);
EchoWM (omnimodal enterable WMs); Robot Learning from a Physical WM (WM as training
ground for a separate policy).

## What the container framing buys NanoCore — and what it costs

### The compelling version of the idea

```
                 ┌──────────── NANO CORE ───────────────────────────┐
                 │   shared latent state  z  (EG2 space, or a        │
                 │   learned subspace of it)                         │
                 │                                                   │
                 │   abilities = heads/modules over z:               │
                 │     decide     — scorer over caller options       │
                 │     converse   — articulation channel             │
                 │     predict    — transition head: z' = f(z, a)    │
                 │     evaluate   — value/competence heads           │
                 │                                                   │
                 │   glial regulator — modulates which heads fire    │
                 │   and how much each is trusted, per region of z   │
                 └───────────────────────────────────────────────────┘
```

The transition head is the piece that turns the regulator from reactive to
*deliberative*: instead of scoring "should I clarify" from features, the system can
simulate — roll `z' = f(z, clarify)` forward, measure whether the imagined next
state resolves the ambiguity, and pick the ability by predicted consequence.
That's model-based dispatch — MuZero one level up. "Come to its own conclusions"
has a mechanism: imagination over the shared latent.

### What stays identical — the honest part

- **EG2 stays frozen perception**; the latent `z` is its output (or a projection of
  it). The container doesn't replace the encoder — it gives the encoder's outputs a
  place to live and evolve.
- **Scorer, gate, conformal, calibration — untouched.** They're the `decide` ability
  reading `z`. The bundle, qualification, rollback — all survive.
- **The decision model isn't demoted; it's promoted** — from "the model" to "the
  best-developed ability inside the model," which is the user's actual thesis.

### The costs — honest

- **The transition model needs sequential (state, action, outcome) traces.** Deepens
  component 42: logging must record trajectories, not just rows — `z_t`, the action
  taken, and the resulting `z_{t+1}`. Until that exists, the WM is an empty frame.
- **Latent dynamics quality is measurable and must earn its slot**: does predicted
  `z'` carry decision-useful signal (e.g., does `f(z, clarify)` separate resolved
  from unresolved continuations)? An E-leg, not an assumption.
- **Scope risk is real.** "World model" invites pixel-prediction ambition; NanoCore's
  is the semantic/latent class — state transitions over embeddings, nothing more.
  The ADR should bound it explicitly or it drifts toward video generation nonsense.
- **Nobody in the decision-model category ships this** — same differentiation argument
  as the glial regulator, one level deeper.

## The taxonomy this clarifies

NanoCore-S1 under this framing = **a semantic world model over its own interaction
domain**, where:

- **perception** = frozen EG2 (multimodal → `z`)
- **transition** = learned `z' = f(z, action)` over abilities' effects (NEW — the
  piece worth researching/building)
- **abilities** = heads over `z` (decide/converse/evaluate)
- **regulation** = glial layer modulating ability usage per region of `z`
  (ADR-0014 — which now reads as *"the WM's action-selection substrate"*)
- **memory** = decision log + consolidation (component 42 → trajectories)

The three missing organs, in dependency order: **trajectory logging → transition
head → deliberative dispatch** (the glial regulator's imagination mode).

## Recommendation

1. Record the framing as a candidate architectural evolution — do NOT rewrite
   ARCHITECTURE.md around it yet; it inherits the same earn-the-slot rule as every
   component.
2. Extend component 42's spec: log trajectories (`z_t, action, z_{t+1}`), not just
   decisions — it feeds both the glial regulator and any transition head.
3. First cheap evidence for the WM leg: on cached score matrices + action streams,
   test whether `(z_t, action) → z_{t+1}` is learnable at all — a probe, not a model.
4. If the framing survives contact, write ADR-0015 scoping "world model" to *semantic
   latent-state transitions only* — bounded so it can't drift into generative scope.

## Sources

- OpenWorldLib (definition + modular framework) — arXiv:2604.04707
- MobileWorldBench/MobileWorld (semantic WM beats pixel WM; mobile/on-device) — arXiv:2512.14014
- LoopWM (iterative latent depth, adaptive compute, 100× param efficiency) — arXiv:2606.18208
- WorldMM (learned dispatch over internal memories) — arXiv:2512.02425
- Collection: huggingface.co/collections/JuanRafap/world-models (14 papers)
- Canonical: Ha & Schmidhuber 2018; Hafner et al. PlaNet/DreamerV3; Schrittwieser
  et al. MuZero; internal: docs/research/{glial-regulator,steering-wheels}.md
