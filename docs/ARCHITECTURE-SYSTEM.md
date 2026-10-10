# NanoCore-S1 — System Architecture

*Date: 2026-10-10. Status: **the system map.** This document describes NanoCore as a
*system that makes efficient decisions* — not as a decision model. The decision model
is the most mature *ability inside* the system. For its internals see
`ARCHITECTURE-DECISION-MODEL.md`; this document is the layer above it.*

> **Honesty note.** Two drawings live below: **what exists** (a pipeline, measured)
> and **the committed direction** (an organism, mostly unbuilt). Read the gap between
> them as the real status. Anything in the second drawing is a claim about intent,
> not a claim about code.

The anatomy vocabulary is from `research/steering-wheels.md`: every model family is
**engine + steering wheel + gauges + brake**. This doc applies that anatomy to
NanoCore as-built, and marks what is built vs. decided vs. proposed.

Honest status legend: **[BUILT]** shipped + qualified · **[DECIDED]** ADR accepted,
not yet built · **[PROPOSED]** research direction, gated · **[MISSING]** identified
gap, no plan yet.

---

## The anatomy answer first — what is our steering wheel?

| Role | In NanoCore | Status |
|---|---|---|
| **Engine** | EmbeddingGemma 2 — frozen multimodal encoder; states become points in a shared 768-d space | BUILT |
| **Steering wheel (content)** | scorer → `argmax softmax(scores / t_prob)` over **caller-declared options** — bounded wheel, can't hallucinate | BUILT |
| **Gauges** | top_prob, set_size, entropy, margin, max_score (the `Observation` row) | BUILT (entropy plumbed but unread) |
| **Brake** | conformal gate + `SlowState` threshold shifts — controls *whether* an answer leaves, never *what* | BUILT |
| **Steering wheel (abilities)** | which pathway engages per context — **does not exist yet**; today a fixed rule fires the only ability there is | DECIDED (ADR-0014, glial regulator) |
| **Long-term state** | `SlowState` 20-observation window — a learned-latent workspace is the TRM-shaped upgrade; trajectory log emits real rows since v27 | BUILT (embryonic) / PARTIAL (corpus) |
| **Container/distribution** | bundle dir + manifest today; OLC6 sealed-slot vault prototyped | BUILT (dir) / PROTOTYPED (OLC6) |

The literal trace, no metaphor:

```
state → EG2 → scorer → scores → P = softmax(s/t_prob), set = APS(s/t_set,q̂)
      → obs = (top_prob, set_size, entropy, pred, max_score, margin)
      → slow.observe(obs) shifts τ_answer / k_clarify / τ_in_schema
      → action = answer | clarify | escalate | abstain   ← the brake's output
```

The glial number is **a bias on the action boundary, nothing more** — it never
touches the score row, the embedding, or the chosen label. That is a brake, not a
wheel. The wheel-over-abilities is the gap ADR-0014 fills.

---

## What actually exists (measured, as-built)

```
state ──► EG2 ──► scorer ──► gate ──► Prediction {answer|clarify|escalate|abstain}
              frozen    cosine/     conformal
              encoder   head        + SlowState threshold nudges (a brake)

ops shell around it: bundle / serve / registry / canary / monitor / qualify
```

One encoder, one scorer, one gate, one ability. The slow layer observes six
scalars and moves three thresholds. That is the entire system today — linear,
stateless per call, single-ability. Everything below this line is direction, not
fact.

## The committed direction (ADR-0014) + proposed extension (ADR-0015)

```
┌─────────────────────────── NANOCORE-S1 ───────────────────────────────────┐
│                                                                          │
│  PERCEPTION                                              [BUILT, frozen] │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │ EmbeddingGemma 2 — text/code/image/audio → 768-d unified space   │     │
│  │ bf16 on accelerator / fp32 on CPU; frozen = calibration holds    │     │
│  └──────────────────────────────┬─────────────────────────────────┘     │
│                                 │ item vectors                           │
│                         ┌───────▼────────┐                               │
│                         │   COMPOSER     │  multi-item state → one vec    │
│                         │                │  [SLOT BUILT, factory 38 TODO] │
│                         └───────┬────────┘                               │
│                                 │ z = state_vec                          │
│  ┌──────────────────────────────┼───────────────────────────────────┐    │
│  │              SHARED LATENT STATE  z  (EG2 space)                │    │
│  │        — the "world" the abilities all read —                   │    │
│  │                                                               │    │
│  │   ABILITIES = heads over z                                    │    │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │    │
│  │   │   decide    │  │  converse   │  │   predict           │    │    │
│  │   │ scorer→gate │  │ articulate/ │  │  p(vars|subset)     │    │    │
│  │   │ →Prediction │  │ render      │  │  random-access head │    │    │
│  │   │  [BUILT]    │  │  [MISSING]  │  │  [PROPOSED ADR-0015]│    │    │
│  │   └─────────────┘  └─────────────┘  └─────────────────────┘    │    │
│  │                          ▲                                    │    │
│  │   engages / modulates per region of z                         │    │
│  │   ┌───────────────────────┴───────────────────────────────┐   │    │
│  │   │  GLIAL REGULATOR — competence field over z            │   │    │
│  │   │  today: SlowState (brake-bias)           [BUILT]      │   │    │
│  │   │  target: learned, regional, consolidates in "sleep"   │   │    │
│  │   │  [DECIDED ADR-0014]  substrate: SSM/GRU/field — E-leg │   │    │
│  │   │  + candidate: learned (x,y,z) recursive workspace     │   │    │
│  │   │  (TRM-shaped; SlowState is the hand-written version)  │   │    │
│  │   │  premise measured: errors cluster spatially           │   │    │
│  │   │  (kNN AUC 0.978 vs per-class 0.940 — research §6)     │   │    │
│  │   └───────────────────────────────────────────────────────┘   │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  MEMORY  [partial]   SlowState live window [BUILT] · trajectory log      │
│                      (z_t, action, z_{t+1}) + outcomes [PARTIAL, c.42]   │
│                                                                          │
│  SYSTEM TWO (outside)   escalation target — frontier LLM / human / app   │
│                      [interface BUILT: escalate action + cost gate]      │
│                                                                          │
│  CONTAINER  bundle dir [BUILT] → OLC6 sealed vault [PROTOTYPED, c.44]:   │
│             abilities as slots, generations as seals, delta updates,     │
│             capability projections, sync-to-device                       │
│                                                                          │
│  OPS SHELL  save/load · serve · registry · canary · monitor · qualify ·  │
│             adapt · modelcard                               [BUILT]      │
└──────────────────────────────────────────────────────────────────────────┘
```

## Why "efficient decisions" is a *system* property

The economy of deciding — decide cheap when confident, clarify when ambiguous,
escalate only what merits the cost, abstain when nothing fits — is produced by
the *whole loop*, not the scorer. The scorer alone can't spend its economy; it
needs the gate to know when cheap wins, the regulator to learn where it's
competent, memory to accumulate evidence, and escalation to cash out what the
fast path can't afford. Every box above exists to serve that economy.

## The three gaps, in dependency order

1. **Memory/corpus** (component 42): trajectory logging `(z_t, action, z_{t+1})`
   + outcome joins. Feeds everything learned — regulator and transition head
   alike. First real rows emitted in v27.
2. **Glial regulator** (component 41, ADR-0014): the wheel-over-abilities —
   regional competence field, consolidates in "sleep". Premise measured (spatial
   error structure exists); substrate is an E-leg ablation. PSI/TRM sharpened
   the spec (`research/recursive-structure.md`): the field is the *extraction*
   half of a predict→extract→integrate cycle — its output must eventually feed
   back as *conditioning inputs*, not only threshold modulation; and a learned
   `(x,y,z)` workspace (TRM) is now a named substrate candidate alongside
   SSM/GRU/field — `SlowState` is that loop written as hand arithmetic.
3. **Transition head** (components 43→47, ADR-0015 *proposed*): not
   single-conditional `p(z'|z,a)` but **random-access** — a maskable predictor
   over trajectory variables (PSI lesson: counterfactual dispatch
   `p(z'|z,do(a))`, missing-stat imputation, and structure extraction fall out
   of one object; ~10 vars, a masked head, not a model-scale problem). Gated on
   the design study + feasibility probe; semantic latent transitions only,
   scope-bound.

## What stays fixed (the invariants that prevent drift)

- **The schema is the contract** — `{state, questions}` → typed answers; every
  ability conforms to it, so abilities compose without bespoke wiring.
- **EG2 stays frozen** — plastic perception invalidates every calibrated
  artifact downstream (ADR-0001). The container consumes the encoder; nothing
  redefines it.
- **Earn-the-slot adoption** — every learned component ships only if it beats
  the frozen baseline at matched coverage. Rollback by construction.
- **The fast path stays fast** — `decide()` remains valid with regulator,
  transition head, and container all absent. They are additions, never
  requirements.

## Non-functional requirements & the ops budget

*(System-design Quick Diagnostic: 5/10 → this section completes it. Deployment
posture is single-process, on-device — the web-scale rows are reframed for
that, not skipped.)*

**Budget** (order-of-magnitude; corpus terms from `TRAJECTORY-LOGGING.md` §capacity):

| Quantity | Budget | Note |
|---|---|---|
| decide() latency | encode-dominated; measured per-run in `preds.jsonl`; cache hit ~µs | no hard SLA yet — Phase 8 sets the device number |
| Corpus growth | ~3.5 KB/row → ~35 MB/day @10k/day → ~13 GB/yr | rotation required |
| Memory | EG2 ~1.1 GB resident (Q8_0) + scorer/gate ~MB-scale | Phase 8 measures |
| Availability | process-local: decide or escalate — no HA tier by design; escalation to System Two *is* the availability story for hard inputs | |

**The diagnostic, on-device reframing:**

1. **Requirements** — this block + the schema contract. Written.
2. **Capacity** — corpus estimate above; decision-rate = per-decision latency,
   measured per run, budget set at Phase 8.
3. **Redundancy** — artifact-level, not infra-level: frozen base thresholds
   (rollback), fast-path-valid-without-regulator, bundle restore, `cache=None`
   fallback, escalate→System Two as the offload tier. Written here so it's the
   strategy, not a coincidence.
4. **Data scaling** — the corpus is the datastore: rotation by size or age on
   the JSONL pair; `decision_id` is globally unique so late outcomes join
   across rotated files — the rotation contract must preserve joinability, not
   just bound size. (Spec written here; the mechanism is unbuilt — small.)
5. **Cache** — `DecisionCache`. Built.
6. **Async** — named boundary, not a queue: consolidation/"sleep" is an
   *offline batch* over the corpus file — reads `preds`+`outcomes`, writes a
   candidate artifact that must qualify. At this scale the file itself is the
   queue. (Boundary named here; the job is components 41/46 territory.)
7. **Monitoring/alerting** — `monitor.py` emits drift, unsafe-rate, and
   coverage-by-action signals; alert *delivery* is the host app's concern by
   design — the monitor surface is the contract.
8. **Deployment** — registry + shadow/canary/rollback, live-verified.

## Related docs

- `ARCHITECTURE-COMPONENTS.md` — **the numbered component map** (Mermaid): every
  roadmap component 1–47 on the architecture, status-colored — the tweakable
  twin of this document
- `ARCHITECTURE-DECISION-MODEL.md` — the decide ability's internals (as-built)
- `research/steering-wheels.md` — the engine/wheel/gauge/brake taxonomy across
  model families
- `research/glial-regulator.md` — the cohesion mechanism + the measured premise
- `research/world-model-container.md` — the container framing + field survey
- `research/recursive-structure.md` — PSI/TRM: random-access prediction +
  learned recursive workspace, mapped onto components 43/47 and the glial spec
- `adr/0014`, `adr/0015` (proposed) — the two commitments this map records
- `prototypes/olc-container/` — the sealed-container distribution spike
- `ROADMAP.md` — components 41–44 and the progress log
