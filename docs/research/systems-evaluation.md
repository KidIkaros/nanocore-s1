# Systems evaluation — the as-built NanoCore through three frameworks

*Date: 2026-10-10 | Status: diagnosis of the **shipped** system only — no
aspirational components. Frameworks: Huyen's Designing ML Systems passes,
Meadows' Thinking in Systems, and the system-design Quick Diagnostic.*

The exercise: evaluate what exists, not the roadmap. Every claim below was checked
against code or measured artifacts, not inferred from intent.

---

## 1. Meadows — structure diagnosis (stocks, flows, loops, delays, traps)

### Stocks (what accumulates)

| Stock | State |
|---|---|
| `SlowState._window` (20 obs deque) | the only live stock |
| `SlowState._streak` | anomaly counter |
| Calibrated artifacts (q̂, t_prob, t_set, thresholds) | frozen stocks — correct |
| Decision log (serve/monitor writes) | **dead stock — read by nothing** |
| Trajectory corpus (component 42) | **missing entirely** |

### Loops

- **Balancing loop (shipped)**: anomalies ≥ fitted bar → streak++ →
  τ_answer ↑ / k_clarify ↓ / τ_in_schema ↑ → more escalate/abstain → wrong
  answers ↓. Bounded (`max_shift`), released on sustained calm, Schmitt-triggered
  against flicker. Verified in `slow.py`: `_anomalous` compares against
  `self._base` (frozen) thresholds — the ratchet does **not** self-reinforce.
  Textbook hysteresis control; genuinely well-built.
- **The open loop (the gap)**: outcome → adaptation does not exist. The system can
  detect "the distribution shifted" but can never register "I was right/wrong."
  Meadows calls this a missing **information flow** — the cheapest high-leverage
  intervention on the ladder. Component 42 is exactly this loop, unclosed.

### Trap identified: Shifting the Burden

`_anomalous` is permanently pinned to `self._base`. Under sustained shift the
system escalates *forever* — it can tighten but can never re-baseline or grow.
The intervention (escalation to System Two) substitutes for the system's own
capacity, indefinitely. As-built, the trap is structural, not hypothetical.

### Leverage ranking of current interventions

| Proposed change | Rung | Verdict |
|---|---|---|
| Tune `tau_step`/`window`/`drift_margin` | parameters | weakest — polish, not structure |
| SlowState itself | balancing loop | shipped, mid-leverage |
| Component 42 (trajectory + outcome logging) | **information flows** | **highest buildable leverage — and cheapest** |
| Glial regulator (ADR-0014) | rules of the system | deep, but blocked on 42's stock |
| "Efficient decisions" reframe | goals/paradigm | highest possible — already made by the owner |

## 2. System-design Quick Diagnostic (adapted: embedded component + serve shell)

| Row | Status |
|---|---|
| Requirements listed | Partial — roadmap items exist; explicit QPS/latency/availability SLAs never written |
| QPS/storage estimate | **FAIL** — never estimated: serve throughput, bundle growth, log volume, on-device footprint |
| Redundancy | Pass (adapted) — rollback/canary verified (comp 22); bundle versions are the artifact-level equivalent |
| State scaling | Partial — `_Stats.latencies` bounded; trajectory-log growth unestimated |
| Caching | Pass — encoder/embedding cache exists (`.nanocore-cache`) |
| Async/decoupling | Partial — consolidation is conceptually async; `observe()` runs inline in `decide()` (cheap but synchronous) |
| Monitoring/alerting | Partial — `monitor` records; nothing alerts |
| Deployment strategy | Pass — shadow/canary/rollback live-verified |

**Score ≈ 5/10.** Works, but skips estimation and observability — the framework's
standard verdict for systems designed by feel rather than by budget.

## 3. Huyen's four passes (condensed)

| Pass | Verdict |
|---|---|
| ML suitability | **Pass** — learnable pattern measured (.92 Banking77); adaptability correctly identified as dominant requirement; failure cost priced (REFUSAL_LOSS) |
| Training data | **Fail** — the learned components' corpus doesn't exist; dispatch label semantics unwritten; imbalanced outcomes by construction |
| Deployment | **Pass** — online fast path + offline consolidation split; release strategy verified on hardware |
| Drift response | **Half** — detection exists (SlowState IS the prescribed SPC detector); response is single-mode: can only tighten, never re-baseline |

## The convergence — three lenses, two gaps

**Gap 1 — the open loop.** Meadows' missing information flow = Huyen's incomplete
Training Data pass = component 42. Highest-leverage buildable move by every lens;
no new model required — record what the system already computes plus what happened.

**Gap 2 — single-mode response.** Meadows' Shifting-the-Burden = Huyen's missing
response ladder = the code fact that `_anomalous` is pinned to `self._base`
forever. The response space needs a "recalibrate" rung between "brake" and
"retrain" — the simplest articulation yet of why thresholds alone can't be the
cohesion layer.

**Meta-finding:** the system's real weakness isn't inside the pipeline — the design
has never faced an ops budget (QPS, log volume, on-device memory). Every
learned-component parameter (window size, field granularity, trajectory schema)
depends on numbers that have never been estimated.

## What component 42 must now specify (the frameworks' concrete additions)

1. **Dispatch label semantics** — "should have clarified" needs a ground-truth
   definition *before* the log schema: hand-labeled, or natural labels
   (clarification-resolved, escalation-confirmed, user-corrected). The label
   choice shapes what must be logged.
2. **Trajectory shape** — `(z_t, action_t, z_{t+1})` with propensities + outcome
   joins, not decision rows alone (feeds 41 and 43 alike).
3. **Capacity estimate** — rows/day × bytes/row × retention, written down. The
   first ops budget in the project; required before any storage choice.
4. **Imbalance expectation** — outcomes will be dominated by "answered correctly";
   the informative minority (justified abstentions, missed clarifications) must be
   preserved deliberately, not averaged away.

---

## Meadows — completed pass (2026-10-10, full playbooks)

The earlier section ran an abbreviated diagnosis; this completes trap
identification (all 8 archetypes) + resilience (the playbook never opened).

**Trap inventory.** Two traps were found and fixed before we had their names:
*tragedy of the commons* (Kaggle session slots — retry-push deadlock; AGENTS.md
rule 1 is "regulate the commons") and *rule beating* (benchmarks green /
behavior untested — the stimulus surface is the documented way out). **The live
trap is success to the successful**: `decide` is qualified, so it attracts all
eval investment, so it becomes more qualified — converse/glial/transition starve.
Q2's diversification is the way out. *Shifting the burden* remains armed (frozen
`_anomalous` base → escalate forever; the recalibrate rung is unbuilt). *Drift
to low performance* is guarded (must_fix items must not become furniture; E4's
absolute 5-pt margin is the "hold goals absolute" move). *Seeking the wrong
goal* was fixed early (proper scoring rules, ADR-0004; Laya's 0.1006 temperature
is the cautionary artifact).

**Leverage placement.** The roadmap is climbing Meadows' ladder in the correct
direction — top rungs settled first: paradigms/goals (the "system of abilities"
reframe) → information flows (42, in flight — the cheapest high-leverage point)
→ rules (45) → self-organization (41/43, which IS rung-4 leverage: the system
changing its own structure). Parameter tuning (tau_step, window) sits at the
weakest rung — correct that it got the least attention.

**Resilience pass.** The hidden stock is real and named: fast path valid
without the regulator, frozen `_base` thresholds, bundle roundtrip, snapshots.
Two warnings worth binding: (a) *suboptimization* — the decide ability
dominating eval investment is a subsystem's goal dominating the whole, same
object as the success-to-the-successful trap; (b) *optimizing away resilience* —
the efficiency goal could strip the redundant capacity: clarify/abstain/escalate
ARE the resilience stock, not overhead. Any future "tighten for latency" move
that erodes them is this pitfall firing.

---

## Wilson (ML Engineering in Action) — four-pass check (2026-10-10)

- **Plan & Scope: strong.** Kaggle runs ARE the demo cadence (each ends in a
  decision, not status); ADR-0015's design-study gate is Wilson's "experimental
  scoping" verbatim; the fallback plan is architecture-native (earn-the-slot,
  static gate default). Gap: no time-boxing on research phases.
- **Experimentation: strong.** Baseline discipline is the religion — tfidf_lr in
  every breadth leg, "predict `z_t` unchanged" required in the transition probe,
  cosine scorer as baseline head, ADR-0011 headroom check pre-fit.
- **Modularization: good.** Smell audit RAN: zero bare excepts, no mutable
  module state, broad excepts all degrade to explicit fallbacks (HF→None,
  psutil→/proc/meminfo→None, fallback handlers). The monolith item IS component
  39 — the flat 28-module package, still TODO.
- **Attribution & Drift: measurement strong, response single-mode.** `causal.py`
  paired readouts are A/B attribution machinery; MODEL_CARD is automated honest
  reporting; SlowState detects drift but the *response* is one-rung.
  **Third framework to name the same gap** — Huyen's response ladder, Meadows'
  Shifting-the-Burden, Wilson's response plan: the recalibrate rung between
  "brake harder" and "retrain" is now triple-witnessed and unbuilt.

---

## Hardt & Recht (Patterns, Predictions, Actions) — four-pass check (2026-10-10)

- **Frame: exemplary.** Loss-before-model at both levels (proper scoring for the
  predictor, REFUSAL_LOSS for the action layer); headroom check = distance to the
  achievable bound; risk-coverage eval not accuracy. Caveat: all accuracy claims
  are benchmark-population claims until s1_stimulus measures the real one.
- **Build: strong.** Frozen-EG2 = the representation-first extreme the book
  prescribes; baseline-first everywhere.
- **Audit: mechanics good, dataset critique pending** — CLINC is clean intent
  data; humans aren't. The stimulus corpus is the corrective.
- **Act: the gap this lens uniquely exposed.** Dispatch is already MDP-shaped
  (trajectory schema = dynamics, transition head = model-based policy eval,
  correctly avoiding model-free RL). But trajectory logs give `p(z'|z,a)` under
  the *logging policy* — the gate's confidence selects which actions appear, so
  naive fitting learns the gate's habits. ADR-0015 criterion added: the design
  study must name the counterfactual estimator (IPS/doubly-robust over logged
  propensities — which is why 42 records them). Performativity named: actions
  shift the input distribution; the corpus is generated by the policy being
  studied.

## Pass 5 — Seven Languages (Tate): the internal monologue, paradigm-fit

Applied the paradigm-fit method to the system's internals as languages the
components speak to each other — the "internal monologue" frame.

**The monologue inventory**: encoder→state vector (immutable), scorer→score row
(immutable), gate→Observation (fact), slow state→PolicyThresholds (mutable,
read only by the gate), gate→GateResult, logger→trajectory row (event-sourced).
The skeleton is correct: pure functional perception, immutable facts, an
ordered-rules engine (action_for), event-sourced memory, one locked mutable
island (SlowState).

**Findings**:

1. **An unobserved channel.** The v27-E6 bug: the slow state's threshold
   utterances had no listener — the schema bar collapsed to 0.999 and nothing
   noticed. A monologue nobody reads is muttering. Fixed: the leg logs
   `active_thresholds()` per phase.
2. **Untyped units.** `tau_in_schema: float` unified a 9.8-logit boundary with
   a 0.999 probability ceiling — a category error the type system can't see.
   Haskell lesson: units belong in types (Probability vs RawScore newtypes).
   The monologue must be able to distinguish what its own numbers mean.
3. **Written but not read; dispatch hardcoded one level up.** The trajectory
   corpus is narrative memory with no reader yet — 42→43 is literally "the
   model reads its own monologue." Ability dispatch (decide/converse/clarify)
   is qtype-keyed — the same facts-and-rules shape as action_for is needed at
   the ability-selection level, as Observation-style context features feeding
   ordered rules, not an enum switch.

**Verdict**: the grammar is sound; the system has a voice and a diary. It does
not yet have a self-model that reads the diary — that is precisely the 42/43
rung, confirming the roadmap's ordering from the language side.

## Pass 6 — System Design Quick Diagnostic (Xu): completed, not just scored

Re-scored the 8-row diagnostic after component 42's capacity estimate landed.
Result: 4/8 satisfied as-written → the three failures (redundancy, datastore
scaling, async) shared one property — **the right answer existed in the system
but was never written as the strategy**. For an on-device single-process
component, the web-scale answers (multi-AZ, Kafka, read replicas) are wrong
answers; the correct move is reframing, not importing machinery.

Resolution (written into ARCHITECTURE-SYSTEM.md as "Non-functional
requirements & the ops budget"):

- **Redundancy → artifact-level**: frozen base (rollback), fast-path-valid-
  without-regulator, bundle restore, `cache=None` fallback, escalate→System
  Two as the offload tier. Strategy now stated, not coincidental.
- **Datastore → the corpus itself**: rotation spec written — joinability
  across rotated files is the load-bearing property (`decision_id` is global),
  not just size bounding. Mechanism unbuilt (small).
- **Async → named boundary, not a queue**: consolidation/"sleep" is an
  offline batch over the corpus file — the file *is* the queue at this scale.
  The job itself is 41/46 territory.
- **NFRs**: latency = encode-dominated, budget deferred to Phase 8 (the
  honest place to set a device number); corpus ~13 GB/yr → rotation required;
  availability = process-local by design, escalation is the availability
  story for hard inputs.

Post-completion: all 8 rows carry a stated strategy; 2 carry honest
"spec'd-not-built" tags (rotation mechanism, consolidation job). Meta-finding
from the earlier pass resolved: the design now has an ops budget.

## Pass 7 — Functional Design (Martin): the hidden-write defect, fixed

**OO-Smell Pass**: clean. Dataclasses where data suffices (Observation,
PolicyThresholds, GateResult, Prediction); Protocols not hierarchies
(Policy, EscalationHandler); one shared `action_for` — no GoF ceremony;
no bare mutation (all writes wrapped + lock-guarded); no recursion/lazy-seq
issues.

**Data-Flow Design**: the system already *is* a pipeline —
`encode → scores → {P, set, obs} → action → Prediction` — pure stages,
effects at edges (model load, file append, serve). One defect found:

**`gate.decide()` hid a write in a read signature** — `slow.observe(obs)`
inside an evaluation call. Latent bug, not hypothetical: component 46's
earn-the-slot mechanism replays logged trajectories; replay through
`decide()` would feed phantom observations into the live regulator —
streak inflates, thresholds move mid-replay, and the candidate measures a
moving target while corrupting the state it was supposed to evaluate.

**Fix shipped**: `evaluate(scores, labels, thresholds=None)` — the pure
stage (explicit thresholds = off-policy replay) — and `decide()` = evaluate
+ wrapped observe, the write now marked in the API itself. Two regression
tests: evaluate-leaves-state-untouched, and replayed-row-under-logged-
thresholds reproduces the logged action exactly (the component-42 corpus
contract, now a tested API instead of a convention). 78 targeted + 254
dry-run green.

This closes the loop with the seven-languages finding: the monologue now
has a pure read path — the system can *re-read* its own diary without
writing in it.
