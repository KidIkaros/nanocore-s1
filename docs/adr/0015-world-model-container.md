# ADR-0015: NanoCore framed as a semantic world-model container — abilities as heads over a shared latent state

**Date**: 2026-10-10
**Status**: proposed — **research only; do not accept without the design study below**
**Deciders**: owner direction; synthesis in `docs/research/world-model-container.md`
**Scope bound**: this ADR claims the *semantic/latent* world-model class only —
state transitions over EG2 embeddings. It claims nothing about pixel/generative
world simulation, and any drift toward that scope is a defect, not ambition.

## Context

The owner's framing: NanoCore was never meant to *be* a decision model — it was meant
to *make efficient decisions*, and the decision model is one ability. ADR-0014
committed the glial regulator as the cohesion layer, but left open a deeper question:
**what is the substrate the abilities and the regulator operate over?**

The world-model literature answers this with a specific mechanism rather than a
metaphor: a learned transition function in latent space, `z_{t+1} = f(z_t, a_t)`,
which lets a system *simulate* candidate actions before committing — deliberative
dispatch, not reactive scoring. OpenWorldLib's unified definition (perception +
interaction + long-term memory as a *modular framework*) is the container the owner
described; MobileWorldBench proves the semantic (non-pixel) class is viable and
on-device-relevant; WorldMM shows learned dispatch over internal resources beats
fixed routing.

Current state: NanoCore has perception (EG2), one mature ability (decide), gauges
(conformal sets, entropy), and a brake (gate + slow thresholds). It lacks: a shared
evolving state, a transition model, a second ability, and any mechanism that could
*imagine* rather than react.

## Decision (proposed — pending design research)

**NanoCore's unifying frame is a semantic world-model container**: a shared latent
state `z` (EG2 embedding space, or a learned subspace), over which abilities live as
heads — `decide`, `converse`, `evaluate`, and a `predict` transition head
`p(z' | z, a)` (a *distribution* over next states — see criterion 3 below). The
glial regulator (ADR-0014) becomes the layer that selects and
modulates which ability engages per context region — and, once the transition head
exists, can *simulate* candidate actions' consequences in latent space before
committing.

This ADR is **proposed, not accepted**. Acceptance requires the design study and
evidence below; the roadmap (component 43) records it as a research direction whose
prerequisite is component 42's trajectory logging.

### Required before acceptance (the design study)

1. **Architecture design pass** — what is `z` concretely (EG2 vec vs learned
   projection vs recurrent state)? Which "actions" populate `a` (the gate's four?
   ability-level? both)? How does the transition head train — next-embedding
   prediction on logged trajectories, contrastive, masked? Written design doc, not
   a sketch. **PSI finding (arXiv:2509.09737) changes the default**: the predictor
   should be *random-access* over trajectory variables (maskable inputs; any
   variable predictable from any subset), not a single-conditional `p(z'|z,a)` —
   random access is what makes structure extraction, integration-as-conditioning,
   and counterfactual dispatch (`p(z'|z, do(a))` as a causal prompt) fall out of
   one object. At our scale (~10 variables, thousands of rows) this is a masked
   head, not a large model. The design study must either adopt random-access or
   record why single-conditional suffices. **TRM finding (arXiv:2510.04871)
   opens a second question**: `SlowState` is already TRM's `(x,y,z)` recursion
   written as hand arithmetic (window→streak→τ); whether a learned latent
   workspace should replace it is an open question — the supervision signal
   (dense per-step targets vs our sparse delayed outcomes) is unresolved and
   must be answered in the study, not assumed.
2. **Feasibility probe** — on trajectory logs (component 42), is
   `(z_t, a_t) → z_{t+1}` learnable at all? Reported as held-out prediction error
   on next-state embeddings, vs. a "predict `z_t` unchanged" baseline. If the
   transition isn't learnable, the WM leg has no foundation and this ADR dies here.
3. **`f` outputs a distribution, not a point** — deliberation without uncertainty
   can't be braked; a point prediction gives the glial layer nothing to weigh.
   Minimum contract: `p(z' | z, a)` as a heteroscedastic Gaussian (predicted
   mean + variance per dimension). Model-family selection (genAI foundations):
   Gaussian regression head is the named baseline — the degenerate VAE whose
   latent is already given; upgrade to VAE/mixture-density only if residuals are
   multi-modal; flows are a candidate for the *glial* anomaly job (exact density
   → off-manifold trajectories), not `f`'s; diffusion is rejected (iterative
   sampling cost is the antithesis of efficient decisions); GAN/EBM is rejected
   (no likelihood → cannot power the brake). The probe must report calibrated
   predictive uncertainty (e.g., held-out NLL or coverage of the predicted
   ellipsoids), not only MSE.
4. **Off-policy correction named** — trajectory logs give `p(z'|z,a)` *under the
   logging policy*: the gate's own confidence selects which actions appear, so
   naive fitting learns the gate's habits, not the world's dynamics. The design
   study must name the counterfactual estimator (inverse-propensity or doubly-
   robust over the logged propensities component 42 records) — how action values
   for *unlogged* actions get estimated. Performativity acknowledged: actions
   shift the input distribution, so the corpus is generated by the policy being
   studied — one more reason propensities are load-bearing, not metadata.
   **PSI offers a complementary mechanism**: a random-access predictor answers
   `p(z'|z, do(a))` as a causal prompt — intervention by masking-and-fixing the
   action variable — which covers unlogged actions as *model queries*. The two
   aren't exclusive: estimator corrects logged-data bias, random-access queries
   the learned dynamics; the study should say which is primary.
5. **Boundary spec** — the fast path (`decide()`) must remain valid with the
   transition head absent. The WM is an *addition* to the system contract; the
   system's shipped baseline must never require it. Verified by a bundle that loads
   and decides with no transition artifact present.
6. **Drift guard** — written confirmation that "world model" here means semantic
   latent transitions only; any component that starts predicting *inputs* (text,
   pixels, audio) rather than *states* has left scope.

### What stays identical under this framing

- EG2 stays frozen perception; `z` is its output space.
- Scorer, gate, conformal, calibration, bundle, qualification — untouched; they
  become the `decide` ability reading `z`.
- The decision model isn't demoted; it's the most mature ability in the container.
- ADR-0014's regulator is *extended*, not replaced: it gains a deliberation channel
  (imagination via `f`) alongside its modulation channel.

## Alternatives Considered

### Alternative 1: Keep the decision model as the whole architecture
- **Pros**: nothing new to build; the current system is qualified.
- **Cons**: cohesion remains a brake inside one ability — context never enters the
  action rule, abilities can't be added without hand-wiring; the architecture is
  "one organ + gauges," which the owner explicitly rejected.
- **Why not**: the container frame is what "more than a decision model" means
  mechanically — it names the substrate abilities share.

### Alternative 2: Full generative world model (pixel/frame prediction)
- **Pros**: the flagship interpretation of "world model" (Genie, Sora, Dreamer).
- **Cons**: wrong scale (video diffusion), wrong domain (we model decision states,
  not environments), wrong budget (violates the on-device constraint by orders of
  magnitude).
- **Why not**: MobileWorldBench measured semantic state transitions beating
  pixel-level prediction at the task that matters to us; scope is explicitly bound
  against this in the ADR header.

### Alternative 3: World model as the encoder (replace EG2)
- **Pros**: one unified learned representation.
- **Cons**: a plastic perception layer invalidates every calibrated artifact
  downstream — the frozen-encoder argument (ADR-0001) holds identically here.
- **Why not**: CAM-Brain's lesson was wiring, not components. The WM consumes the
  encoder; it doesn't replace it.

## Consequences

### Positive
- The container frame names where abilities *live*: heads over `z`, so adding a
  converse/evaluate ability is adding a head, not rewiring the system.
- Deliberative dispatch is a real mechanism — simulating `f(z, clarify)` answers
  "should I clarify?" with a predicted consequence rather than a score heuristic.
- The research doc's taxonomy (`perception / transition / abilities / regulation /
  memory`) matches the field's own modularity — we're using established vocabulary
  for a designed architecture, not inventing a metaphor.
- On-device fit: LoopWM's iterative-latent-depth result says capability can be
  bought with loops (time) instead of parameters (memory) — the budget thesis.

### Negative
- The transition head has no corpus until component 42 logs trajectories — deeper
  dependency than ADR-0014's (which needs rows; this needs *sequences*).
- A second learned artifact *plus* a stateful component to qualify, version, and
  fingerprint — two artifacts on the earn-the-slot treadmill.
- "World model" is a scope magnet. Without the header's scope bound, this ADR
  could be read as license for generative ambition it does not grant.

### Risks
- **Transition unlearnable**: logged decision streams may not carry enough
  next-state signal to train `f`. Mitigation: the feasibility probe is cheap and
  runs on component 42's first corpus — it kills the ADR early if the premise
  fails.
- **Scope creep toward generation**: the name invites it. Mitigation: the scope
  bound is in the ADR header, and the design-study gate requires the boundary
  spec before acceptance.
- **Architectural overreach**: three organs (regulator, transition, converse) at
  once is more than the corpus supports. Mitigation: the ordering is hard —
  trajectories → feasibility probe → transition head → deliberation — and each
  step is gated on the prior earning its slot.

## Related

- `docs/research/world-model-container.md` — the survey and the honest bounds.
- ADR-0014 — the glial regulator; this ADR extends its role from modulation to
  (candidate) deliberation.
- Component 42 — trajectory logging, the prerequisite corpus.
- `docs/research/glial-regulator.md` — the three-paper synthesis (CAM-Brain /
  glial review / sleep) that motivates the slow-learned substrate.
