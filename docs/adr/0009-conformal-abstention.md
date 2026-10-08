# ADR-0009: Use conformal prediction sets (APS/RAPS) for abstention, asymmetric by domain

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent
**Supersedes**: [ADR-0005](0005-abstention-zero-bias-threshold.md)

## Context

ADR-0005 chose a zero-bias max-score threshold fitted to a target precision. Measurement
showed the mechanism does not work as a product.

**On BFCL irrelevance** — 1,252 relevant (query, tool) pairs against 2,300 irrelevant — the raw
cosine score separates the two classes by only **0.097** (0.7401 vs 0.6428). To hold 90%
precision the gate must escalate **96.5%** of traffic, saving 3.4% of cost. At 95% precision it
auto-handles 2.36%.

**On Banking77** the operating point saturates: every candidate reported
`coverage_at_90pct_precision = 1.0` because base accuracy (93%) already exceeds 90%, so a
verdict was briefly drawn from an artifact. `protocol.coverage_at_precision` now refuses to
report a saturated number instead.

Two further facts shaped the decision. Conformal classification (**APS**, Romano, Sesia &
Candès, NeurIPS 2020) provides prediction sets with guaranteed marginal coverage that adapt to
difficulty — "easy points get tight sets, ambiguous points get larger ones" — and works "with
any black-box predictive model… **regardless of whether they are well-calibrated**." **RAPS**
(Angelopoulos et al., 2021) adds a rank-based regulariser for smaller sets.

And the domain determines whether abstention is *safe*: in a dispatch setting the fallback is
escalation to a stronger model, so abstention is the designed outcome; where a low-confidence
verdict **suppresses a positive** (a finding, an alarm, a candidate), abstaining is an active
negative decision and can be worse than not running the model at all.

## Decision

**Replace the fitted max-score threshold with conformal prediction sets**, using APS as the
default nonconformity score and RAPS where set size matters, calibrated on a held-out split.
The gate is **asymmetric by domain policy**:

| policy | behaviour on low confidence |
|---|---|
| `escalate` | widen the prediction set and hand off; the fallback is a stronger model |
| `drop-safe` | **never** silently discard; require an explicit rejection decision with its own error bound |

Calibrate on the same split used for temperature fitting, and report the **empirical coverage**
achieved alongside the nominal `alpha`.

## Alternatives Considered

### Alternative 1: Keep the threshold and improve the relevance signal
- **Pros**: no new machinery; the weakness may be in the features rather than the mechanism.
- **Cons**: the separation is 0.097 — a threshold cannot manufacture signal that is not there,
  and better option text only helps the cosine and interaction paths, not the fingerprint head
  (which ignores option text entirely, measured).
- **Why not**: it optimizes the wrong quantity. The problem is the *shape* of the decision, not
  the cut point.

### Alternative 2: A learned "none of the above" option
- **Pros**: the model decides abstention itself; one interface.
- **Cons**: requires abstention examples in training data; the reject class competes with real
  options in the softmax and absorbs probability mass from them.
- **Why not**: it changes the meaning of the option distribution. May be revisited once real
  escalation data exists.

### Alternative 3: Drop abstention; return a distribution and let the caller decide
- **Pros**: simplest; no calibration step; no guarantee to maintain.
- **Cons**: removes the one behaviour that makes a cheap model safe to put in a production path;
  the routing literature names the quality estimator as the critical factor precisely because
  the caller cannot judge it from a bare distribution.
- **Why not**: it deletes the feature that justifies the architecture.

## Consequences

### Positive
- Coverage becomes a **stated, testable property** with a distribution-free finite-sample
  guarantee, rather than a threshold fitted once and hoped for.
- The saturated-operating-point problem disappears: `alpha` is chosen relative to the base error
  rate, so it cannot be trivially satisfied.
- Prediction-set **size** is a useful signal in itself — an ambiguous request yields a larger
  set, which is exactly what a router wants to know.

### Negative
- Adds a calibration stage and a held-out split to maintain.
- The output shape changes: a `Prediction` now carries a `prediction_set` and an `action`
  alongside its probabilities, which is a breaking interface change (accepted; see the plan).
- Set size is not free — a larger set means more escalation, i.e. more cost.

### Risks
- **Conformal validity assumes exchangeability between calibration and test.** A deployed domain
  shift breaks it. Mitigation: state the assumption, monitor shift, and recalibrate per domain.
- **This is a feature, not architecture evidence.** Because conformal works regardless of
  calibration, it applies equally to a temperature-scaled cosine and to a trained head — it
  cannot be cited as support for the model. Recorded here so it is not misused later.
- **Abstention safety is a property of the domain, not the model.** The asymmetric policy exists
  because the same mechanism is safe in one setting and harmful in another.

## Sources

- `reports/runs/s1_dispatch/` — relevance separation 0.097, escalation at 90%/95% precision
- `reports/runs/s1_calibration/` — selective curves, and the saturated 90% operating point
- Romano, Sesia & Candès, *Classification with Valid and Adaptive Coverage* (APS), NeurIPS 2020
- Angelopoulos et al., *RAPS*, 2021
- `docs/RESEARCH-NOTES.md` R4, R7 — conformal mechanics and the domain-asymmetry finding
