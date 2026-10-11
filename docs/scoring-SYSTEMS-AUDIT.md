# Scoring Subsystem — Systems-Thinking Audit (Meadows Leverage Lens)

**Scope**: the scoring path of the hybrid decision system — scorer (cosine / TaskHead /
OrdinalScorer / PairScorer), the decision gate (conformal), the glial regulator
(label-free slow state), and the monitor. Question: *is the scoring solid in this hybrid
system, considering the decision head and the glial system?*

**Method**: Meadows leverage-point ladder. Read the actual code, ran behavioral probes
(operational proof, not text). Stocks/flows/loops mapped, gaps ranked by leverage.

---

## System map (not a file list)

**Stocks (what accumulates):**
- `SlowState._window` (deque, maxlen=window) — the glial memory; *is* the time constant
- `SlowState._streak` — consecutive activated windows (Schmitt trigger state)
- `SlowState._activate_at/_release_at` — fitted calibration bars (stock, set once)
- gate fitted params: `t_prob, t_set, qhat, tau_answer, tau_in_schema, k_clarify` (stock, frozen post-calibration)
- `RecalibratePolicy._labeled` — labeled observation buffer (stock, **only in experiments**)
- prediction log (jsonl) — the externalized decision record (stock, append-only)

**Flows (rates):**
- `decide()` → `evaluate()` (pure) → `slow.observe(obs)` (the one write) — decisions/sec
- `slow.observe` → `_next_streak` → `thresholds()` — threshold drift rate
- labeled outcomes → `RecalibratePolicy.observe_labeled` → refit (labeled balancing loop, **not wired to serve**)
- monitor `check()` reads the log → drift report → alerts (read-only, periodic)

**Feedback loops:**
- **B1 (balancing, glial, label-free)**: anomaly count in window → streak → thresholds move → fewer anomalies. Restoring. *Verified: returns to base.*
- **B2 (balancing, labeled)**: `RecalibratePolicy` refits thresholds from gold labels. Restoring. **Exists but is experiment-only — not in the served path.**
- **B3 (balancing, monitor)**: drift KS + escalate_rate → alerts → human acts. Restoring, but human-in-loop and periodic (delay).
- **No reinforcing loop** in scoring (correct — none wanted).

**Information flows (who learns what):**
- decision → log (thresholds recorded per-decision) ✓
- log → monitor → alerts ✓
- glial activation → **nobody** (silent — see F1)
- gold labels → **only RecalibratePolicy experiments**, not the served gate (see F2)

---

## Findings (severity-ranked, leverage-cited)

| # | Leverage point | Gap | Severity | Effort |
|---|---|---|---|---|
| F1 | **#5 Information flows** | Glial activation is silent: `SlowState.observe()` returns `None`, fires no event/callback/alert when the regulator activates and moves thresholds. An operator watching the served model sees decisions change behavior with no signal that the *regulator* did it. The per-decision log records `active_thresholds()`, so the move is recoverable post-hoc, but nothing surfaces "the glial state just activated." | **HIGH** | S |
| F2 | **#5 Information flows** | The labeled balancing loop (B2, `RecalibratePolicy`) exists and is measured, but is wired **only into the kernel experiment**, never into `serve`/`adapt`. The served gate runs the label-free glial loop (B1) only. Gold labels arrive at the logger (`gold=`) and are **never consumed** to correct the scorer/gate in production. The one loop that can see *concept* drift (label-dependent failure) is disconnected from the served path. | **HIGH** | M |
| F3 | **#4 Rules** | `auto` scorer selection picks raw `argmax` of cal accuracy with **no non-inferiority margin**. A TaskHead that beats cosine on cal by 0.001 ships, discarding the zero-shot path. ADR-0008's own lesson (cosine ties the head on banking77) argues for a margin: prefer the simpler scorer unless the head wins by a real gap. | **MED** | S |
| F4 | **#8 Buffers** | No NaN/inf guard in the shipped path. A degenerate encoder (NaN logits) flows through cosine → gate → **returns `action: unevaluated` with `nan` probabilities to the client**. Not a crash (fail-soft), but `nan` is logged and returned as a "decision"; the argmax of a NaN vector silently picks index 0. | **MED** | S |
| F5 | **#7 Balancing loops** | Label-free glial loop (B1) is **structurally blind to concept drift** — verified: 30 confident-but-wrong decisions leave the regulator calm (streak 0, base thresholds). This is *by design* (label-free), and documented in `slow.py`/`monitor.py` as "concept drift needs labels." It is not a bug, but it is the **boundary of B1's coverage**, and F2 (the labeled loop that covers it) is disconnected. F5 is the reason F2 matters. | **MED** (design boundary, becomes HIGH via F2) | — |
| F6 | **#10 Stock-and-flow structure** | No parallel/twin scorer path — `legacy.NanoCoreS1` and `DecisionModel` **share** `composer.compose_items` (verified: both import it). The E6 twin-vs-shipped drift class is closed here. **Discharged — not a gap.** | — | — |

### Discharged after data-flow verification (per audit discipline)
- **F6 (twin scorer)**: FALSE POSITIVE on first read — `legacy.py` looks like a parallel path, but both it and `model.py` call the same `compose_items`. No drift risk.
- **slow_state persistence**: the fitted bar rides in `gate.json` via `fitted_config()` and `gate.load` re-attaches it. Verified behaviorally: a glial bundle reloads with `gate.slow` set and feeds the window on `decide()`. Not a gap.

---

## What IS solid (verified, not assumed)

- **B1 restoring branch is real**: drove drift → tau_answer climbed 0.5→0.8, k_clarify 3→2, tau_in 0→3.0; returned to healthy → **all three returned exactly to base** ( Schmitt trigger releases, `thresholds()` recomputes from `_base`). No one-way ratchet.
- **Bounded movement**: `shift = min(max_shift, streak*tau_step)` — the regulator cannot run away. `k_clarify` floors at 1.
- **E6 units fix holds**: `tau_in_schema` moves in raw-score units (`+ shift*reference`), no `TAU_CEILING` clamp on the raw boundary (the v27 collapse bug stays fixed).
- **Calibration floor**: `min_n` guard refuses an underfed gate (fail-closed on too-little calibration data).
- **t_set inheritance**: the glial state inherits the gate's fitted `t_set` at attach (G9/pre-mortem #6 closed) — set_size is measured at the same temperature the action uses.
- **Off-policy replay**: `evaluate(thresholds=...)` is pure; replaying logged scores under logged thresholds reproduces the action without feeding the live regulator. The trajectory-logging invariant holds.
- **Scorer promotion is torch-free** (ADR-0007): TaskHead/OrdinalScorer export numpy weights, bundle ships `scorer.npz`, load + infer run without torch.

---

## The one-paragraph answer

**The scoring is structurally sound but information-poor.** The balancing loops work and
restore correctly (verified behaviorally), the units bugs are fixed, there is no twin-path
drift, and the scorer promotion is torch-free and gated. The weakness is not in the loops —
it is in **what the loops can see and who they tell**. The glial regulator does its job
(label-free, bounded, restoring) but activates *silently* (F1), and the one loop that could
see concept drift — the labeled recalibration — is measured but disconnected from the served
path (F2). Gold labels arrive at the logger and die there. That is the highest-leverage gap:
a small information-flow change (surface activation; consume gold labels in serve) closes
the loop the system was designed around but never wired end-to-end.

---

## Phased plan (gated: alpha-blocking = F1, F2)

**Phase 0 — surface the glial activation (F1, S).** `SlowState.observe` returns the
post-observe streak (or an `activated: bool`); `serve` logs an `activation` event and the
monitor alerts on "regulator active." Highest leverage, smallest change: an information
flow, not a structural rewrite. Verify behaviorally: drive drift through a served model,
confirm an activation event reaches the log/monitor.

**Phase 1 — wire the labeled loop into serve (F2, M).** When `gold` arrives at the
logger, feed it to a `RecalibratePolicy` attached to the served gate (delay-bounded refit).
This connects B2 to the live path so concept drift is correctable, not just detectable.
Promote-and-retire: keep B1 as the fast label-free loop, add B2 as the slow labeled
corrector. Verify: inject wrong-but-confident decisions + gold labels, confirm thresholds
recalibrate.

**Phase 2 — non-inferiority margin on auto-scorer (F3, S).** `auto` ships the head only if
it beats cosine by a margin (e.g. 1pt); else cosine. Honors ADR-0008's "don't assume the
head wins."

**Phase 3 — NaN/inf guard (F4, S).** `decide()` rejects non-finite scores with a clear
error (fail-closed) rather than returning `nan` probabilities as `unevaluated`.

F5 needs no code — it is the documented boundary of the label-free design, and Phase 1 is
its remedy. F6 is discharged.
