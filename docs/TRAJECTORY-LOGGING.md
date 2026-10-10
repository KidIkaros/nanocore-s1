# Trajectory logging — component 42 spec

*Date: 2026-10-10. Status: **spec + first implementation.** Required by the
systems evaluation (`research/systems-evaluation.md`): label semantics written
before schema, capacity estimated before storage choice, imbalance designed for
deliberately.*

This closes the **open loop** the three frameworks independently found: the
system observes distributions but never registers "I was right." Every served
decision is now a trajectory unit; outcomes join back by `decision_id`.

## Label semantics — what "should have X'd" means

The dispatch-label problem (the eval's hardest requirement) resolves through
**natural labels wherever possible, derived counterfactual labels where not**:

| outcome type | who writes it | what it validates |
|---|---|---|
| `gold` | benchmark harness / eval stream | the answer was right/wrong |
| `clarification_resolved` | whoever serves the clarifying answer — the *next* decision on the same `stream_id` | the clarify action produced a state the fast path could then answer — **the "should have clarified" natural label, free of hand annotation** |
| `escalation_confirmed` / `escalation_overruled` | the System Two consumer | escalation was warranted / wasted |
| `user_corrected` | human review | explicit relabel — carries corrected gold as payload |

**Derived counterfactual labels** (computed from joined rows, not logged):
- *should have clarified* = `action=answer` ∧ outcome shows wrong ∧ gold ∈
  `prediction_set` — the correct answer was in the set the model could have
  offered
- *should have answered* = `action∈{clarify,escalate,abstain}` ∧ argmax was
  correct — overcaution measured, not just underconfidence
- *regulator regret* = rows where the logged action under logged thresholds
  differs from the action under an alternate policy — replayable offline
  because `raw_scores` + `thresholds` are both in the row

## Row schema (what each decision now writes)

```json
{
  "ts": 1760..., "model_id": "...", "decision_id": "a1b2...",
  "stream_id": "conv-7", "seq": 3, "prev_id": "9f8e...",
  "input": "...", "qtype": "choice", "labels": [...],
  "policy": "full", "action": "clarify",
  "top_prob": 0.41, "prediction_set": [...], "alpha": 0.1,
  "latency_ms": 88.2, "state_hash": "c3d4...",
  "obs": {"top_prob": 0.41, "set_size": 2, "entropy_norm": 0.62,
          "max_score": 0.83, "margin": 0.11},
  "thresholds": {"tau_answer": 0.62, "k_clarify": 3, "tau_in_schema": 0.70},
  "raw_scores": [...],
  "gold": "..."            // when caller knows it (eval streams)
}
```

- `decision_id` — outcome join key, returned in the `/decide` response
- `stream_id`/`seq`/`prev_id` — explicit trajectory edges (caller supplies
  `stream_id`; the logger chains `prev_id`)
- `obs` — the observation row the slow state saw (margin derived from probs)
- `thresholds` — thresholds **in force** at decision time (slow-state-adjusted,
  via `gate.active_thresholds()`) — replay must reproduce the logged action
- `raw_scores` — the counterfactual substrate: replay any policy offline
  without re-encoding
- `state_hash` — sha256[:16] of the state vector; the vector itself is never
  stored because the frozen encoder makes it reproducible (~3KB/row saved)

Outcomes live beside the log at `<log>.outcomes.jsonl`
(`POST /outcome {decision_id, outcome, source, payload}`):
```json
{"ts": ..., "decision_id": "a1b2...", "outcome": "escalation_overruled",
 "source": "sys2-review", "payload": {"correct_option": "..."}}
```

## Capacity estimate (the project's first ops budget)

| term | size | note |
|---|---|---|
| base row | ~0.6 KB | input + ids + obs + thresholds |
| `raw_scores` (k=150) | ~3 KB | dominates the row |
| **row total** | **~3.5 KB** | k=10 schemas → ~1 KB |
| 10k decisions/day | ~35 MB/day | ≈ 13 GB/yr → **rotation required** |
| state vector | 0 (hash only) | frozen encoder ⇒ reproducible; storing it would cost +3 KB/row |
| outcomes side file | ~0.2 KB/outcome | sparse by design |

`PredictionLogger(log_scores=False)` drops `raw_scores` for constrained
deployments (~5× smaller rows); corpus-building deployments keep it on.
Rotation/retention remains the deployer's concern (existing docstring contract).

## Imbalance design

Outcomes are structurally imbalanced — `answer` rows get outcomes cheaply
(benchmarks know gold); deferred rows mostly don't, because "what would have
happened" has no observation. `monitor.join_outcomes()` reports
`coverage_by_action` so a corpus can't silently average the informative
minority (justified abstentions, missed clarifications) into the easy majority.
For regulator training, the corpus spec is: stratified export by action ×
outcome-type, never pooled accuracy.

## API additions

- `POST /decide` accepts optional `stream_id`, `gold`; response echoes `decision_id`
- `POST /outcome {decision_id, outcome, source?, payload?}` → `{ok, recorded}`
- `monitor.join_outcomes(log_path)` → joined rows + coverage-by-action
- `gate.active_thresholds()` — public accessor (was `_active_thresholds`)
- `Prediction.state_hash`, `Prediction.raw_scores` — additive optional fields
