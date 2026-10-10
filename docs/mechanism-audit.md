# Static Mechanism Audit — NanoCore-S1 Governance

**Scope:** The decision-gating and adaptation mechanisms in NanoCore-S1, audited
as a protocol-governance system. This is a *static* audit (no simulation) — it reads
the mechanism graph, walks mitigations against a hostile actor, and verifies
enforcement claims against code. Findings are split into **documented-accepted-costs**
(things the ADRs already record as intentional trade-offs) and **real gaps** (things
documented as guarantees that the code does not actually enforce, or mitigations whose
preconditions the system does not provide).

**Governance surface under audit.** The "mechanism" here is how the system decides
*whether to answer, clarify, escalate, or abstain*, and how it adapts its thresholds
over time. The actors are: the caller (who declares the option set and policy), the
gate (the frozen policy + the slow-state brake), the slow state (the only stateful
component), and a *possible* hostile input stream that shifts distribution or crafts
inputs to exploit threshold logic.

---

## Findings Table

### Documented-accepted-costs (no action needed; recorded by design)

| # | Claim | Where | Status |
|---|-------|-------|--------|
| D1 | Fast path is stateless per call; the slow state never blocks `decide()` | ADR-0014, ARCHITECTURE-SYSTEM.md | accepted |
| D2 | `EG2` stays frozen; a plastic encoder invalidates all calibrated artifacts | ADR-0001 | accepted |
| D3 | The conformal gate assumes exchangeability; domain shift breaks the coverage guarantee | ARCHITECTURE-DECISION-MODEL §9 | accepted |
| D4 | `clarify` and `escalate` need harness configuration to reach a real destination | handlers.py | accepted |
| D5 | The slow-state ratchet only tightens — it escalates more under sustained drift, never re-baselines | slow.py `_anomalous`, systems-evaluation.md §1.3 | accepted (component 46 is the documented fix) |
| D6 | Multimodal is a smoke test only (100 images); audio/video untested | ARCHITECTURE-DECISION-MODEL §9 | accepted |
| D7 | `SlowState` moves three thresholds — `tau_answer`, `k_clarify`, `tau_in_schema` — using hand-arithmetic, not a learned field | ADR-0014 §5 ("SlowState is its embryonic form") | accepted (the learned field is ADR-0014's *intent*, gated on component 42) |
| D8 | `bundle_digest` is content-addressed but does not bind the encoder's runtime behavior (the encoder is identified, not embedded) | model.py:41 | accepted — the encoder is frozen and pinned by commit; this is an ops concern, not a governance one |

### Real gaps (HIGH / MEDIUM / LOW)

| # | Severity | Mechanism | Enforcement claim vs. code | Rational-actor argument |
|---|----------|-----------|---------------------------|------------------------|
| G1 | **HIGH** | `escalate` has no mandatory handler | `handlers.py:handle` — if `escalate_to is None`, returns `resolved_by="nobody"` with a reason payload. The gate says "escalate" but nothing in the mechanism *forces* a System Two target to exist. | A caller that omits the handler gets a clean failure (no crash), so they have an incentive to omit it — the system appears to work but escalations sit in a JSONL queue or vanish. The mitigation (handler) depends on the deployer's goodwill, not on the protocol. |
| G2 | **HIGH** | `abstain` is never the gate's natural output | `gate.py:action_for` — `abstain` appears in `ACTIONS` and the schema but the action ordering (`action_for`) has **no path** that returns `"abstain"`. The gate returns `answer`, `clarify`, `escalate` only. `Predict.abstention` defaults to `"unevaluated"` and is a deprecated read-only property (schema.py). | A caller reading the `Prediction` schema can believe the gate will abstain under a documented condition. It never does. The "abstain" branch in the action contract is dead code — an unreachable mitigation. |
| G3 | **HIGH** | Slow state ratchet is permanently pinned to the frozen base | `slow.py:_anomalous` — compares to `self._base` thresholds; `SlowState.thresholds()` computes `schema_bar = self._base.tau_in_schema + shift * reference`, but the `_anomalous` predicate itself never moves. | Under sustained distribution shift, the system escalates forever (Shifting-the-Burden trap, systems-evaluation.md §1.3). The slow state can only tighten the *output* bars; it cannot redefine what counts as "normal" for its own anomaly detector. A hostile input stream that sustains a shift indefinitely forces unbounded escalation. Documented as component 46 (deferred); no runtime guard prevents the ratchet from running to exhaustion. |
| G4 | **MEDIUM** | Cache returns stored results with no freshness or drift check | `cache.py` and `model.py:decide` — on a cache hit, `Prediction(**hit)` is returned verbatim, including the stored `action` and `probabilities`. No timestamp, no drift re-evaluation, no validity window. | An input that was borderline yesterday, cached as "clarify", will return "clarify" forever even if the slow state has since tightened thresholds. The cache is keyed on state-content + knobs (correct), but a cached action can be stale relative to the live slow-state bars. A caller crafting inputs near the decision boundary gets deterministic, stale, conservative responses — the slow state's movement is bypassed for any repeated input. |
| G5 | **MEDIUM** | `tau_in_schema` clamping bug — v27 recurrence risk | `slow.py:_v27_bug` — `TAU_CEILING = 0.999` was applied to `tau_in_schema` (a raw-score boundary at ~9.8 for head logits), collapsing the out-of-schema check to "always pass." Fixed in v27, but the guard is *only in `thresholds()`*, not in `calibrate()` or `fit_in_schema_threshold`. | The clamping is now conditionally skipped for `tau_in_schema` (slow.py:315-317 comment: "No TAU_CEILING here"). But a future code path that re-clamps the boundary — e.g., serialization/deserialization, or a new modulation path — reintroduces the bug silently. The fix is localized, not structurally prevented. |
| G6 | **MEDIUM** | `PolicyThresholds.tau_in_schema = None` disables OOS detection entirely | `gate.py:evaluate` and `slow.py:_anomalous` — both guard `if ... is not None`. If `fit_in_schema` is never called (no OOS examples at adaptation time, see adapt.py:172), `tau_in_schema` stays `None` and the gate has no out-of-schema boundary. | A caller that provides no OOS examples during adaptation gets a gate that never escalates on novel input — all novel items are treated as in-schema and answered if confidence is high. The in-schema check is opt-in, not default-on with a conservative fallback. |
| G7 | **LOW** | `log_scores=True` is the default, bloating the prediction log | `serve.py:PredictionLogger.__init__` — `log_scores=True` writes raw score rows (~3 KB/row at k=150). The spec (TRAJECTORY-LOGGING.md:81) documents `log_scores=False` for constrained deploys, but the default is the verbose one with no guard on log volume. | A deployer that forgets to set `log_scores=False` silently fills disk at ~35 MB/day (spec §capacity). There is no size-based auto-rotation or a warning threshold in the logger itself. |
| G8 | **LOW** | `raw_scores` in the prediction log is the full score vector, not propensities | `serve.py:PredictionLogger.record` writes `pred.raw_scores` (the full `(k,)` score row). `monitor.join_outcomes()` reads them but no code path computes or stores *propensities* (the probability that action `a` was chosen under an alternative policy). | ADR-0015 criterion 4 names counterfactual estimation (IPS/doubly-robust) as the off-policy correction needed for the transition head. But the log schema writes `raw_scores` and calls it sufficient — there is no propensity field. The trajectory log claims to be "the counterfactual substrate" (TRAJECTORY-LOGGING.md:58) but it records actions, not the propensities that make off-policy evaluation valid. |
| G9 | **LOW** | `set_size` is computed from `t_set` but `StreamConfig` does not default it | `policy.py` imports `observe_row` from `slow.py`, and `observe_row` requires `t_set`. The `StreamConfig` dataclass (policy.py:254) has `t_set: float` as a required field — callers must supply it. `StreamConfig` in `slow.py` does *not* exist; it lives only in `policy.py`. | The observation statistics the slow state consumes depend on `t_set`, and `t_set` is fitted during calibration. But the `SlowStateConfig` in `slow.py` (which the shipped gate uses) has **no `t_set` field** — it is passed to `observe_row` at the call site. If a caller constructs `SlowState` directly (the public API in `gate.py:attach_slow_state`), they must remember to pass the right `t_set` separately, or the anomaly rule reads a different set than the action was chosen on. This is the exact class of bug that produced the v26/v27 threshold-collapse regressions. |

---

## Mechanism-by-mechanism analysis

### 1. The brake: `ConformalGate.action_for` (gate.py:37)

The gate's action ordering is the system's core governance rule — it decides what happens to every input:

```
1. if max_score < tau_in_schema → escalate   (out-of-schema)
2. if top_prob >= tau_answer → answer          (confident enough)
3. if set_size <= k_clarify → clarify          (narrow enough to ask)
4. else → escalate
```

**Verified against code.** `action_for` is the single source of truth — both
`ConformalGate._gate_action` and `policy.py`'s stream runner delegate to it
(gate.py:261, policy.py:317). No drift. ✓

**Gap: `abstain` is unreachable.** `ACTIONS = ("answer", "clarify", "escalate", "abstain")`
and `Prediction.abstention` exists, but the action ordering has exactly zero paths to
`"abstain"`. It is declared as a possible output but the mechanism never produces it.
This is a naming-promise gap: the contract says "four actions" and the docs describe
the asymmetric brake (ADR-0009), but the code only ever produces three. A caller
implementing a handler for "what to do when the model abstains" writes dead code.

### 2. The slow-state ratchet: `SlowState._anomalous` (slow.py:283)

The ratchet's logic:

```python
def _anomalous(self, obs):
    below_schema = (self._base.tau_in_schema is not None
                    and obs.max_score < self._base.tau_in_schema)
    return (obs.top_prob < self._base.tau_answer
            or obs.set_size > self._base.k_clarify
            or below_schema)
```

**Verified against code.** The anomaly predicate reads `self._base` — the frozen,
fitted thresholds — exclusively. `_next_streak()` increments based on this predicate.
`thresholds()` then moves the *output* bars. The predicate itself never adapts. ✓

**Gap: permanent Shifting-the-Burden.** Under sustained drift, `_anomalous` keeps
firing at the same rate (since the base bars don't move), the streak grows without
bound, `max_shift` caps `tau_answer` at 0.30 above base, and `k_clarify` bottoms at 1.
The system escalates everything indefinitely. This is the trap Meadows & Huyen
both name. Component 46 (recalibrate rung) is the documented fix, but it is
not built, and there is **no circuit breaker** — no maximum stale-escalation window,
no fallback to re-baselining. The mechanism's only response to sustained failure is
"escalate more," which is also the input distribution growing hostile.

### 3. The cache: `DecisionCache` (cache.py) + `model.py:decide` (model.py:171)

**Verified against code.** On a cache hit, `Prediction(**hit)` is returned. The hit
includes `action`, `probabilities`, `prediction_set`, and the `thresholds` *at the time
of caching* (but not the *current* active thresholds). The slow state is **not
consulted** on a cache hit — `decide()` returns before reaching
`self.gate.decide()` (which calls `self.slow.observe(obs)`). ✓

**Gap: stale thresholds on cache hit.** A repeated input near the decision boundary
gets its *original* action, ignoring the slow state's current bars. This is
documented as "cache hit returns the stored result verbatim" (cache.py:7), but it
means the slow state's adaptation has no effect on cached items — a form of
**action atrophy** where the system becomes less responsive to drift for repeated
inputs. The cache key binds policy + alpha + composer (correct), but not the
slow state's dynamic threshold values (which aren't known at key-construction time).

### 4. Escalation: `handlers.py:handle` (handlers.py:120)

**Verified against code.** `escalate_to` is optional, defaulting to `None`. When `None`,
the function returns `resolved_by="nobody"` with a reason — it does not raise. ✓

**Gap: escalation is opt-in, not enforced.** The gate emits `"escalate"` as an action,
but the system has no mechanism to require a handler exists. A deployer can serve
the gate, get escalations, and never wire System Two — the escalations silently
accumulate in the "nobody" resolution. Contrast with `clarify`: `resolved_by="caller"`
is the honest admission that the caller must handle it. Escalate has no equally
honest default — "nobody" is presented as a valid resolution state.

### 5. Bundle identity: `bundle_digest` (model.py:41)

**Verified against code.** SHA-256 over sorted file paths + file contents. Changes to
the scorer or gate calibration change the digest; touching only mtimes does not. ✓

**No gap on content-addressing itself.** The digest correctly binds the artifact.
However, it does not bind the *encoder runtime* (the encoder is identified by name,
not embedded). This is accepted — the encoder is frozen and pinned by commit
(ADR-0001). No governance concern at the mechanism level.

### 6. Outcome logging: `PredictionLogger` + `OutcomeLogger` (serve.py:38)

**Verified against code.** `decision_id` is a UUID returned in the `/decide` response.
`/outcome` joins by it. `join_outcomes` reports `coverage_by_action` — the imbalance
is visible. ✓

**Gap: propensities are not recorded.** The spec calls the corpus "the counterfactual
substrate" because `raw_scores` + `thresholds` allow replaying any policy. But off-policy
evaluation (the prerequisite for learning the transition head, ADR-0015 §4) requires
**propensities** — the probability that action `a` was chosen under an alternative policy,
which the gate does not compute or log. The gate is deterministic given thresholds
(no exploration noise, no stochastic action selection), so the propensity of the
*logged* action is 1.0 and the propensity of any *counterfactual* action is 0.0.
This makes IPS/Doubly-Robust estimation impossible from the log as-is. The "counterfactual
substrate" claim is true for threshold-replay (component 46) but **not** for policy-replay
(components 43/47). This is a documented-acceptance for threshold modulation but
an **undocumented gap** for the transition head's off-policy training.

---

## Summary

**Real gaps: 9** (2 HIGH, 3 MEDIUM, 3 LOW, 1 LOW-but-structurally-risky)

The system's strongest property is the tripartite topology (engine/wheel/gauge/brake)
— the fast path is pure, stateless, and well-tested. Its weakest property is that
**three of the four actions it declares it can take are either unreachable (abstain)
or incompletely enforced (escalate needs a handler, clarify needs a caller, abstain
needs nothing because it never fires).**

The most critical finding is **G1 + G2**: the gate emits `"escalate"` as if it
guarantees a human/stronger-model review, and declares `"abstain"` as a valid
action, but neither is backed by enforcement code. A malicious or negligent deployer
can serve the model, get escalations, and never wire the handler — the system
produces an action it cannot fulfill, with no mechanism to demand the dependency.

The second most critical finding is **G3**: the slow-state ratchet cannot re-baseline,
which is a **live operational risk** under sustained distribution shift — not a future
possibility but the documented, measured behavior (systems-evaluation.md §1.3:
"_anomalous is permanently pinned to self._base → escalate forever").
