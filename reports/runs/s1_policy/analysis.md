# Ambiguity-aware policy, end-to-end — analysis

**Run**: `mauricew/nanocore-s1-policy` v1 — Kaggle T4, completed clean.
**Artifacts**: `results.json`, `REPORT.md`, `policy_scores.npz`, kernel log.
**Post-hoc**: diagnosis recomputed locally on cached scores.

## Result — the policy as parameterized *loses*

| policy | resolved | encodes/item | actions |
|---|---:|---:|---|
| always_answer | 0.619 | 1.00 | answer: 5500 |
| threshold (τ on max_sim) | **0.712** | 1.00 | abstain 1507, answer 3993 |
| ambiguity_aware | 0.452 | 1.28 | **answer 0**, clarify 1535, escalate 3965 |
| ambiguity_aware + TaskHead | 0.393 | 1.14 | answer 401, clarify 764, escalate 4335 |

`clarify_pays: false`, `gate_pays: true`. The ambiguity-aware design as specified in
ADR-0013's first draft — `answer` only when the APS set is a singleton — is **worse than
the plain threshold gate by −0.26**.

## Root cause: the `answer` trigger was unreachable

At T=1.0 over 150 intents the softmax is flat: the test top-prob distribution sits at
~0.008, while q̂=0.0235. An APS set can only be a singleton if the top option's mass
exceeds q̂ — never true. Every input fell to `clarify` (≤3) or `escalate` (>3); the
resolved-fraction math then punts 72% of traffic into costly escalation.

The mechanism beneath is still sound — clarify's assignments resolve at ~97%, matching
`s1_clinc150_oos` (97.5%). What failed is the **trigger geometry**: absolute set size is
the wrong key for `answer` on a wide flat distribution. The conformal guarantee is about
coverage, not action triage; the action boundaries need confidence-space thresholds
(`P(top-1)` calibrated, or top1/top-k mass ratio), with the *set* serving as the clarify
payload.

## Second finding: the TaskHead leg

| | cosine scorer | linear TaskHead (15k train) |
|---|---:|---:|
| in-scope accuracy | 0.756 | **0.970** |
| OOS signal AUROC | 0.934 (max_sim) | **0.968** (max_prob) |
| APS mean set (in-scope) | 3.7 | 29.7 |
| coverage @α=0.10 | 0.913 | 0.9996 (q̂=0.9999 — degenerate) |

A linear head nearly solves 150-way CLINC150 and detects OOS *better* than the zero-shot
score — a third headroom-positive win for task-fitted heads. But its sharpened output
distribution collapses the conformal machinery: q̂ ≈ 1.0, mean set 29.7, trivial
coverage. **Third sighting of the temperature pathology** (Laya's shipped 0.1006, our
own T=0.02 fit, now a cross-entropy head): any score map trained toward confidence needs
its own set-construction calibration. This is now a hard requirement for Stage 3, not a
note.

## Consequences for ADR-0013

- `clarify` as a *mechanism*: validated again — ambiguity sets contain the answer ~97%
  of the time.
- `clarify` as a *trigger*: **rejected in current form.** `answer` must be keyed on
  calibrated confidence (or relative set mass), not `|set|==1`. The policy mapping needs
  a redesign pass before implementation — candidate fix: `answer` when calibrated
  top-prob ≥ τ_answer; `clarify` when below it but the set is small; `escalate` on OOS
  or genuinely diffuse sets.
- In-schema check: holds (OOS caught 85.3% at Youden-τ; head max_prob lifts it to 0.968
  AUROC).
- A policy test at this level was the right experiment — the failure is in the action
  table, not the measurements, and it was caught *before* implementation. That is the
  process working.
