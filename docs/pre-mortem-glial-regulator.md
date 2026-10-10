# Pre-Mortem: NanoCore-S1 Glial Regulator + Escalation Handler

**Date:** 2026-10-10  ·  **Failure horizon:** 14 days after the glial regime (ADR-0014) ships
**Product:** NanoCore-S1 decision layer with the slow-state regulator as the cohesion mechanism
**Context:** ADR-0014 (glial regulator) and ADR-0015 (world-model container transition) are about to be committed. The mechanism audit (docs/mechanism-audit.md) identified 9 real gaps. This pre-mortem imagines the shipped glial regime has failed and works backward.

---

## Pre-Mortem Framing

**It is 14 days after the glial regulator shipped, and the system is failing in production. Users report that the system escalates everything and no one is reviewing the escalations. Why?**

---

## Risk Registry

| # | Risk | Category | Urgency | Evidence | Mitigation | Owner | Decision Date |
|---|------|----------|---------|----------|-----------|-------|--------------|
| 1 | Slow-state ratchet escalates forever under sustained input shift, with no re-baseline | 🐯 Tiger (Launch-Blocking) | Launch-Blocking | systems-evaluation.md §1.3: `_anomalous` is permanently pinned to `self._base`; SlowState can only tighten, never re-baseline (Shifting-the-Burden trap); component 46 (recalibrate rung) NOT BUILT | Do not ship ADR-0014 without the recalibrate rung; ship a circuit breaker that forces re-baselining or falls back to static after N consecutive escalates | owner | before ADR-0015 acceptance gate |
| 2 | `escalate` action emits with no mandatory System Two handler — escalations vanish into "nobody" | 🐯 Tiger (Launch-Blocking) | Launch-Blocking | handlers.py:141-144 — `escalate_to is None` returns `resolved_by="nobody"` without raising; serve.py has no enforcement that a handler is configured | Make `escalate_to` non-None at gate-construction; raise at startup if the gate can emit `escalate` without a handler | deployer | before Phase 8 on-device |
| 3 | `abstain` is declared in the action contract but never produced by `action_for` | 🐯 Tiger (Launch-Blocking) | Launch-Blocking | gate.py ACTIONs = ("answer","clarify","escalate","abstain"); schema.py `abstention` defaults to "unevaluated" and is deprecated read-only; action_for has zero paths to "abstain" | Either implement the abstain path OR remove "abstain" from the declared contract — a dead state is a lying contract | owner | before ADR-0014 ships |
| 4 | Cache returns stale `action` + `thresholds` on hit, bypassing the live slow state | 🐯 Tiger (Launch-Blocking) | Launch-Blocking | model.py:171-174 — cache hit returns `Prediction(**hit)` verbatim; gate.decide (which calls slow.observe) is never reached on hit; cache key binds policy+alpha+composer but NOT the slow state's dynamic thresholds | Cache hit must re-run `gate.evaluate` with the current thresholds (raw_scores allow this); or invalidate cache entries whose logged thresholds ≠ active thresholds | owner | v28 |
| 5 | `tau_in_schema = None` silently disables out-of-schema detection on bundles adapted without OOS examples | 🐯 Tiger (Fast-Follow) | Launch-Blocking | slow.py:283-285 — `_anomalous` guards `if ... is not None`; adapt.py:172-175 — `tau_in` stays None if no oos_texts; the gate then never escalates on novel input | Make `tau_in_schema` default to a conservative constant if no calibration occurs, or require OOS examples in the adaptation contract | owner | Phase 7.5 |
| 6 | `t_set` is required by `observe_row` but not in `SlowStateConfig` — the public `attach_slow_state` API can silently miscompute anomalies | 🐯 Tiger (Fast-Follow) | Fast-Follow | slow.py SlowStateConfig has no `t_set` field; gate.py `attach_slow_state` takes only `SlowStateConfig`; callers must pass t_set separately to `observe_row` — the v26/v27 regression class | Add `t_set` to `SlowStateConfig` as a required field; remove the separate argument from `observe_row` | owner | before v28 |
| 7 | Propensities are not logged in the trajectory corpus — ADR-0015's off-policy counterfactual estimation is impossible | 🐯 Tiger (Fast-Follow) | Fast-Follow | serve.py writes `raw_scores` but no propensity field; TRAJECTORY-LOGGING.md:58 calls raw_scores "the counterfactual substrate" — true for threshold-replay but not for policy-replay; ADR-0015 §4 demands IPS/DR estimation | Add propensity logging to the corpus spec; either instrument action sampling or record the gate's counterfactual propensities explicitly | owner | before transition head design study |
| 8 | V27 units bug class re-emerges: `tau_in_schema` clamped to probability ceiling in a new modulation path | 🐙 Paper Tiger | Track | v27 postmortem: TAU_CEILING=0.999 applied to raw-score `tau_in_schema` (~9.8); fixed in thresholds() only; the guard is not in the type, calibration, or serialization paths | Introduce typed scalars (Probability vs RawScore) or add a runtime assertion that `tau_in_schema > 1.0` when `tau_in_schema is not None` | owner | before v28 |
| 9 | The glial field (ADR-0014) is a learned regional competence field, but the substrate (SSM/GRU/field-memory) is an E-leg ablation with no concrete probe scheduled | 🐘 Elephant | Track | ADR-0014: "substrate is an ablation question decided by an E-leg" — but no E-leg is scheduled in the roadmap; SlowState is the hand-written version with no path to the learned field | Explicit E-leg gate in ROADMAP.md before ADR-0014 ships — the learned field must beat the frozen gate at matched coverage, or it doesn't ship | owner | before v28 design study |

---

## Risk Classification Summary

### 🐯 Tigers (5 Launch-Blocking, 2 Fast-Follow)

**Launch-Blocking:**
1. Slow-state ratchet escalates forever under sustained shift (no re-baseline)
2. `escalate` has no mandatory handler → escalations vanish into "nobody"
3. `abstain` is declared but unreachable → dead contract state
4. Cache bypass of live slow-state thresholds → stale actions on repeated inputs

**Fast-Follow:**
5. `tau_in_schema = None` silently disables OOS detection on adaptation without OOS examples
6. `t_set` not in `SlowStateConfig` → silent anomaly miscomputation risk (v26/v27 recurrence)

### 🐙 Paper Tigers (1)

7. V27 units bug reappears in a new modulation path — already fixed once, localized fix; low probability of recurrence but high cost if it does (the slow state would silently disappear)

### 🐘 Elephants (1)

8. The learned glial field has no concrete E-leg scheduled — we're shipping a hand-written ratchet as "the cohesion layer" with a learned field as the "embryonic form." If the learned field never ships, the architecture ships without its central claimed mechanism.

---

## Elephants Requiring Escalation

**E8: We are about to commit ADR-0014 ("the glial layer is the cohesion mechanism") while shipping SlowState — a hand-arithmetic streak counter — as its concrete instantiation.** The ADR says the regulator "consolidates from logged decisions and outcomes between inference calls" and is "a regional competence field over EG2 embedding space." The implemented `SlowState` is a windowed anomaly counter over global threshold crossings. These are not the same thing. The regional-error probe (research/glial-regulator.md §6) supports the *premise* (errors cluster spatially, kNN AUC 0.978), but the *implementation* is a scalar ratchet. If we ship the ratchet and call it "the glial regulator," we have committed to an architecture whose central component exists only as a research plan — and the roadmap has no E-leg scheduled to close it.

**E8 decision needed:** Either (A) ADR-0014 is amended to say "the cohesion layer is a slow-state ratchet, and the learned field is a Phase 9 extension that may earn the slot," or (B) an E-leg for the learned field is added to the Phase 8 deliverables with acceptance criteria (must beat the frozen gate at matched coverage on sel_acc/risk-coverage, with a concrete substrate choice: SSM vs GRU vs field-memory).

---

## Mitigations

### Launch-Blocking (Act Before Ship)

1. **Ratchet re-baseline circuit breaker.** The current `SlowState._anomalous` is pinned to `_base`. Under sustained drift, the streak grows without bound, `max_shift` caps `tau_answer` at +0.30, and `k_clarify` bottoms at 1 — everything escalates forever. Before shipping ADR-0014, add a circuit breaker: if the streak exceeds `window * 3` (sustained), trigger either a re-baseline candidate (component 46) or a fallback-to-static mode with a logged alert. The system must not escalate forever.

2. **Mandatory escalation handler.** `handlers.py:handle` must raise — not return `"nobody"` — when `pred.action == "escalate"` and `escalate_to is None`. The gate emits `"escalate"` as a guarantee to the caller that a stronger model or human will review; shipping without enforcement means the guarantee is void.

3. **Kill the `abstain` lie.** Either implement the abstain path in `action_for` (the schema's asymmetric brake, ADR-0009) or remove `"abstain"` from `ACTIONS` and the `Prediction.abstention` field. A dead action in the contract is not defensive design — it is a trap for every caller who writes an `abstain` handler that never fires.

### Fast-Follow (v28)

4. **Cache must re-evaluate with live thresholds.** A cache hit currently returns the stored `action` verbatim. The slow state's entire purpose is context-sensitive threshold modulation — bypassed entirely for repeated inputs. Fix: on cache hit, replay `raw_scores` through `gate.evaluate` with the *current* `active_thresholds()`, not the cached ones. The raw scores already support this (model.py:196); the cache just doesn't use them.

5. **Bake `t_set` into SlowStateConfig.** The v26/v27 bug class was the anomaly rule reading a different conformal set than the action was chosen on, because `t_set` was passed separately. Make `t_set` a required field of `SlowStateConfig` and have `observe_row` read it from the config, eliminating the parameter-passing surface.

### Track

6. **Log propensities or scope the counterfactual claim.** TRAJECTORY-LOGGING.md:58 claims `raw_scores` is "the counterfactual substrate." For threshold-replay (component 46), this is true. For policy-replay (component 43/47, the transition head), it requires propensities — the probability that action `a` would have been chosen under a different policy. Since the gate is deterministic, the propensity of the logged action is 1.0 and all others are 0.0. Either instrument propensity logging in the corpus, or explicitly scope the counterfactual claim to threshold-replay only.

7. **Schedule the E-leg for the learned field.** ADR-0014 commits to "a learned glial regulator" as the cohesion layer but ships `SlowState` (hand arithmetic) as the impl. The regional-error probe supports the premise but doesn't validate the field. Schedule an explicit E-leg: "the learned field must beat the frozen gate at matched coverage on sel_acc/risk-coverage, with SSM/GRU/field-memory substrate selected." Do not ship the learned-field claim without it.

8. **Typed scalars for threshold units.** The v27 bug (probability ceiling on raw-score boundary) happened because `tau_in_schema: float` unifies logits (~9.8) with probabilities (~0.6). Introduce a `RawScore` newtype or a runtime assertion that `tau_in_schema > 1.0` when set, so the unit confusion cannot recur in any new modulation path.

---

## Elephants Requiring Escalation

**E8: We are shipping SlowState — a hand-written streak counter — as "the glial regulator" (ADR-0014) while the learned regional competence field exists only as a research plan with no E-leg scheduled.** The ADR says the regulator "consolidates from logged decisions and outcomes between inference calls" and is "a regional competence field over EG2 embedding space." The implemented `SlowState` is a windowed anomaly counter over global threshold crossings. If we ship the ratchet and call it "the glial regulator," we have committed to an architecture whose central component exists only as a research plan — and the roadmap has no E-leg scheduled to close it.

**E8 resolution options:**
- **(A)** Amend ADR-0014: "the cohesion layer is a slow-state ratchet; the learned regional field is a Phase 9 extension that earns its slot." Honest about what ships.
- **(B)** Add an E-leg to Phase 8 deliverables: "the learned field must beat the frozen gate at matched coverage, with a concrete substrate choice (SSM/GRU/field-memory)." Block ADR-0014 acceptance on the E-leg's result.
