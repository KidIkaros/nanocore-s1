# The glial regulator — synthesis of three papers toward the cohesion layer

*Date: 2026-10 | Status: synthesis — mechanism proposals are hypotheses until measured |
Feeds: ADR-0014, roadmap components 41–42*

The question this note answers: NanoCore's abilities (decision heads, conformal gate,
multimodal encoding, slow state) each work; **the coupling between them is hardcoded
threshold rules**. What does the literature say about learned coupling, and what shape
should it take?

---

## 1. CAM-Brain (de Garis, ATR, 1996) — the existence proof of the failure mode

`CAM-Brain: ATR's Billion Neuron Artificial Brain Project — A Three Year Progress Report`
(scanned PDF, pages read as images; 5 pp).

- Predecessor result (LIZZY): evolved single neural modules each controlled a behavior
  (walk, turn, peck, mate). **Inter-module dispatch was hand-wired** — "switching between
  behaviors involved taking the outputs from one neural net module and feeding them into
  the inputs of the next module," via "production rule" modules. It worked until it didn't:
  *"every time the author added a neural net module to the Lizzy simulation, its speed on
  the screen was slowed."*
- The project's own conclusion: *"The complexity of CAM-Brain will make it largely
  undesignable, so a (directed) evolutionary approach called 'evolutionary engineering' is
  being used."* Even the *interconnections* between modules were to be evolved — grown via
  CA growth-signal chromosomes — not specified.
- **Contribution to our question:** the modules aren't the bottleneck; the wiring is.
  Hand-chaining abilities (our gate's fixed feature→action rules) is Lizzy's topology, and
  it is demonstrated — not argued — to stop scaling as the ability space grows.

## 2. Artificial glial cells review (Alvarez-Gonzalez, Porto-Pazos, Cedron, Pazos — *AI
Review* 2023, systematic review of 22 papers)

- **Timescale separation is the mechanism.** ANGN models: astrocytes activate *after*
  intense neuronal transmission and regulate synaptic plasticity on a slower timescale
  (seconds vs ms). Glia do not carry payload signals; they **condition the channels** —
  potentiation/depression beyond neurotransmitter release.
- **Glia form a network, not a per-neuron accessory** — "astrocytes form a network in
  themselves, not distributed independently." A connected substrate with its own dynamics.
- **SONG-NET: calcium waves select preferential circuits** — glial state determines which
  neural circuit is active. In the biology, **regulation IS routing**: dispatch among
  pathways emerges from the regulatory layer's spatial state.
- **Benefit scales with complexity** — all 22 reviewed works: glial modulation helps more
  on harder problems; mirrors the astrocyte:neuron ratio rising with brain complexity
  (Herculano-Houzel). The regulator matters *more* as the ability space grows.
- **Glial parameters are learned jointly** — ANGN fits astrocyte behavior by cooperative
  coevolution with the network, not hand-tuning.
- **Honest caveat:** the evidence base is Iris/Two-Spirals/UCI-class MLPs. Direction, not
  proof at our scale. The transfer earns its keep only by measurement.

## 3. "Do Language Models Need Sleep?" (Lee, McLeish, Goldstein, Fanti — CMU/UMD 2026,
arXiv:2605.26099)

- Sleep-like consolidation: when the context window fills, the model takes N offline
  recurrent passes over accumulated context, updating **persistent fast weights** in SSM
  blocks via a learned local rule; wake-time inference stays single-pass. More sleep =
  better post-sleep reasoning, largest gains on deeper-reasoning examples.
- **Contribution:** the *when*. Regulation/consolidation is heavyweight computation that
  runs **between** fast-path calls — affordable precisely because it never blocks
  `decide()`. Also the substrate hint: SSM-style persistent state is a *learned* slow
  state — the same role our hand-rolled `SlowState` window approximates today.

## 4. The synthesis — what the regulator is

```
        ┌─────────────────────── fast path (ms) ───────────────────────┐
state → EG2 (frozen perception) → scorer/head → conformal gate → Prediction
                                    ↑   ↑   ▲
                                    │   │   │  potentiates / depresses
        ┌───────────────────────────┴───┴───┘  (thresholds, pathway gains)
        │   glial layer (slow, networked, learned):
        │   regional competence field over the EG2 embedding space
        └──── consolidates on logged decisions+outcomes ("sleep"), never blocks decide()
```

- **Competence is a field over embedding space, not a global statistic.** The glial layer
  tracks "the decision head errs *in this region of state space*" — locally potentiating
  or depressing pathways — rather than nudging one global threshold. This mechanizes
  "understanding of context builds competency": competence is represented as a property
  of *where the input lives*, which generalizes to contexts never enumerated in rules.
- **Dispatch emerges from regulation.** Which ability engages (decide / converse /
  abstain / escalate / composer arm) is determined by which pathways the glial field has
  potentiated — calcium-wave routing, not a policy lookup over canned actions.
- **The regulator trains in "sleep."** Consolidation passes over the decision log between
  inference — no latency cost, no hardcoded rules.
- **Tripartite topology preserved:** observe the fast path's transmissions, condition its
  channels, never carry the payload. `SlowState` is the embryonic form — a scalar window
  that observes and nudges; the regulator grows it into a regional, learned network.

## 5. What this does NOT imply

- **The transformer encoder stays.** CAM-Brain's lesson is about interconnection, not
  component architecture. EG2 is perception; freezing it is what makes calibration work
  (ADR-0001). An SSM/GRU/recurrent substrate is a candidate for the *regulator* — where
  persistent state is native — not for the encoder.
- **No new component by analogy alone.** The substrate (SSM vs GRU vs learned
  field-memory over embedding regions) is an ablation question for an E-leg, not a
  foregone conclusion. Roadmap rule: earn the slot by artifact, or don't ship.
- **Same prerequisite as every learned layer:** outcome-joined decision logs
  (component 42) — the regulator has nothing to learn from until they exist.

## 6. First falsification probe (cheap, next run)

The regional-competence premise is testable on artifacts we already save: per-example
score matrices (.npz) plus wrong/right labels. **Do errors cluster in score-space
regions?** If yes → a regional regulator has structure to learn. If errors are spatially
uniform → the field hypothesis is weak and the regulator degenerates to global
modulation. One clustering leg on cached data, no new architecture required.
