# ADR-0013: Ambiguity-aware decision policy — `clarify` action and meta-routed heads

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

The motivating observation: the dominant quality gap in decision systems is not raw accuracy
but **intent resolution** — things lost in translation, or failure to understand a user's
intent from an underspecified input. Users with poorly-formed requests should not be
required to phrase them well; the system should tolerate bad input by detecting ambiguity
and resolving it cheaply, rather than committing confidently to a misread intent.

This is directly supported by the project's measured evidence:

- **Ambiguity is measurable.** When the state embedding cannot separate candidate
  interpretations, option scores go flat — which is exactly the signature conformal
  prediction sets were designed to expose (ADR-0009). A `prediction_set` of size > 1 is
  the machine-readable form of "it's one of these; the context doesn't say which."
- **Clarification is affordable.** A local noul question costs one encoding (~57 ms on a
  T4, ~89 ms llama.cpp CPU — `reports/runs/s1_llamacpp_server`). A model fast enough to
  ask a follow-up can tolerate sloppy phrasing: ambiguity gets a clarifying question,
  not a wrong answer.
- **Head selection is a decision too.** Task-fitted heads are label-schema-bound
  (ADR-0011; +0.168 macro-F1 on GoEmotions) and wrong outside their schema. Requiring
  the caller to know which head applies is the same usability failure as requiring a
  well-formed prompt — the system should decide.
- **The failure mode to avoid is demonstrated by a competitor.** Laya's shipped
  temperature is invalid (0.1006, clamped) and its probabilities drift 0.85 under option
  permutation — it commits confidently to whichever intent it lands on. Speed at
  resolving context without honest uncertainty is the dangerous version of this idea.
- **The "modulation layer" pattern has independent support.** The neuron–glia literature
  (ANAN, NeurIPS 2024; Kozachkov et al., PNAS 2023; GliaNet, CVPR 2025) repeatedly finds
  that a thin adaptive layer modulating a *frozen* substrate matches much larger learned
  changes — the same split this design reached by measurement. The clarify/meta-routing
  layer is our glial equivalent: it modulates how decisions are computed and which
  components fire, without ever being the decision itself.

## Decision

**1. The gate action enum gains a fourth value: `clarify`.**

`action ∈ {answer, clarify, escalate, abstain}`, ordered by cost:

| action | meaning | trigger |
|---|---|---|
| `answer` | commit to argmax | prediction_set size 1, calibrated confidence high |
| `clarify` | return the ambiguity set as candidate interpretations | prediction_set size 2..k_clarify — resolvable by one cheap question |
| `escalate` | hand to a bigger model / human | set too large to phrase as a question, or OOS signal |
| `abstain` | no answer possible | empty set / policy forbids escalation |

`clarify` and `escalate` are distinct because they have different costs: a follow-up
question is ~one encode; escalation is a bigger model call. The caller's policy controls
whether `clarify` is offered (a headless router wants `escalate` directly).

**2. Scorer selection is a meta-decision, not a caller obligation.**

Before any fitted scorer runs, an in-schema check decides the path:

```
in-schema for a fitted TaskHead?   → TaskHead
in-scope but no fitted head?       → zero-shot CosineScorer
out-of-scope for the deployment?   → gate (clarify / escalate / abstain)
```

The in-schema test is itself a noul-style decision (score against domain anchors +
conformal set), reusing the same machinery — no new component type is introduced.

**3. "Knowing when" is a benchmark property, measured before it is claimed.**

The acceptance evidence is out-of-scope detection: does the ambiguity signal
(set size, calibrated max-prob) separate out-of-schema inputs from in-scope ones? This
is measured on CLINC150 (150 intents + explicit OOS class — the benchmark built for
exactly this) before the `clarify` action is claimed in documentation.

## Alternatives Considered

### Alternative 1: Leave escalation as the only fallback
- **Pros**: simpler; fewer states.
- **Cons**: collapses the cheapest resolution path. A two-option ambiguity is a
  60 ms follow-up question; treating it like a model escalation is a 30× cost
  error and the difference between a system that tolerates bad input and one that
  routes around it.

### Alternative 2: Make head selection the caller's problem (config flag)
- **Pros**: no meta-decision machinery.
- **Cons**: pushes the schema-matching burden onto the user — the exact "must write a
  good prompt" failure this ADR removes. Also silently unsafe: a fitted head applied
  out-of-schema produces confident garbage with no signal back.

### Alternative 3: Generative clarification (ask the LLM to rephrase)
- **Pros**: handles ambiguity this interface can't type.
- **Cons**: requires a generator in the hot path, breaking the non-autoregressive
  contract (ADR-0002). Clarification by *candidate set* (this ADR) covers the case the
  scores can see; the rest is `escalate` by definition.

## Consequences

- `Prediction` schema gains `action ∈ {answer, clarify, escalate, abstain}` and the
  ambiguity set doubles as the clarifying options — the output is self-describing.
- Stage 3 (conformal gate) becomes the load-bearing stage: `clarify` exists only if set
  sizes are meaningful, which requires the gate.
- The headroom gate (ADR-0011) gains a sibling at runtime: the in-schema noul check,
  same machinery, different operand.
- Honesty boundary preserved: the system may *report* ambiguity; it must never be
  silently confident on an input the scores can't separate — which is precisely the
  competitor failure this design is aimed at.
