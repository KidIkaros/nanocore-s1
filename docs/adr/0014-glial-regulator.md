# ADR-0014: The glial layer is the cohesion mechanism — learned, slow, regional

**Date**: 2026-10-09
**Status**: accepted
**Deciders**: owner direction; synthesis in `docs/research/glial-regulator.md`

## Context

NanoCore's abilities each work — frozen EG2 encoder, typed decision heads, conformal
gate, slow state — but the coupling between them is fixed threshold rules: a mapping from
frozen features (margin, set size, entropy) to actions, fitted once on calibration data.
`escalation.py`'s own docstring admits the static limit: the feature that wins on the
calibration target is not the one that wins on the shifted stream. CAM-Brain's Lizzy is
the demonstrated precedent for this failure mode — hand-chained modules stop scaling as
the ability space grows ("largely undesignable" — de Garis 1996). The owner directive:
hardcoding this level of nuance will not suffice; the glial network is the regulator.

## Decision

**The cohesion layer is a learned glial regulator — not a dispatch policy.** It maintains
a regional competence field over the EG2 embedding space and conditions the fast path by
potentiation/depression (modulating gate thresholds and pathway usage per context
region), consolidating from logged decisions and outcomes between inference calls. The
fast path (`EG2 → scorer → gate → Prediction`) stays stateless and single-pass; the
regulator never carries the decision payload. `SlowState` is its embryonic form.

Deliberately open: the substrate — SSM hidden state, GRU, or learned field-memory over
embedding regions — is an ablation question decided by an E-leg, not by analogy.

Adoption rule (unchanged discipline): the regulator earns its slot only if it beats the
frozen gate at matched coverage on `sel_acc`/risk-coverage; the static gate remains the
shipped default and the rollback.

## Alternatives Considered

### Alternative 1: Keep threshold rules, tune them better
- **Pros**: zero new machinery; interpretable; already qualified.
- **Cons**: fixed functions of fixed features cannot represent "the same uncertainty means
  clarify here and abstain there" — context never enters the rule; proven to lose under
  shift (escalation.py null result).
- **Why not**: this is the exact failure mode; tuning polishes the lookup table.

### Alternative 2: Contextual bandit dispatch policy
- **Pros**: learnable, principled, cost-parameterized action selection.
- **Cons**: still a bolted-on action lookup — picks among canned actions rather than
  conditioning the system's abilities; needs counterfactual/off-policy correction on
  biased logs before its gradients mean anything.
- **Why not**: dispatches *around* the components instead of regulating *through* them;
  the bandit selects actions, the regulator builds the competence the actions draw on.

### Alternative 3: Evolve a fixed interconnection (CAM-Brain's own answer)
- **Pros**: historically validated direction for undesignable wiring.
- **Cons**: produces a frozen coupling per deployment — we ship one artifact, not a
  population; no runtime adaptation to drift.
- **Why not**: regulation keeps the wiring plastic at runtime; evolution compiles it once.

## Consequences

### Positive
- The cohesion question has a home: a layer whose job is competence-in-context, generalizing
  to situations no rule enumerated.
- Tripartite topology keeps the fast path untouched — rollback = disconnect modulation.
- The slow layer can afford real computation ("sleep" consolidation) without touching
  decide() latency — the on-device budget is unaffected on the hot path.

### Negative
- The regulator cannot learn until outcome-joined logs exist (component 42) — the
  architecture is committed before the corpus is.
- A second learned artifact to qualify, version, and fingerprint (composer.pt precedent —
  regulator state must join bundle identity when it ships).

### Risks
- **Analogy risk**: the glial evidence base is toy-scale MLPs; sleep-SSM is in-forward-pass
  consolidation, not cross-call regulation. Mitigation: the regional-error clustering probe
  (research §6) tests the premise on cached artifacts before any regulator is built.
- **Scope creep**: "the regulator" must not absorb the fast path's jobs. If it ever carries
  the payload, the tripartite boundary has failed — treated as a design defect, not a feature.
