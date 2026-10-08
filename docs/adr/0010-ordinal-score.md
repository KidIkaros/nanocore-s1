# ADR-0010: Ordinal `Score` is the only trained component; CORN/CORAL with RPS

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

`Choice` is not differentiable from a temperature-scaled cosine (ADR-0008), and the composer has
never been trained (ADR-0003, now deprecated). That leaves one primitive where a trained
component could be categorically better rather than marginally so: **ordinal `Score`**.

The reason is structural. A cosine score has **no notion of order** — it measures similarity
between two vectors and nothing else. Asking it to respect that "critical" is further from "low"
than "medium" is requires machinery it does not have.

Our current implementation has the opposite problem: `DecisionHead` scores `Score` as an
ordinary softmax over ordered levels trained with cross-entropy. The ordinal-regression
literature exists precisely because that is wrong — CORAL (*Rank Consistent Ordinal
Regression*, Raschke et al., 2020) and CORN (2021) both begin from the observation that
conventional losses "ignore this ordering information."

`Score` has **never been exercised on real data** in this project. `ranked_probability_score`
sits unused in `metrics.py`.

There is also a published target to beat. The independent benchmark of the category's flagship
product reports, on SST-5, **57.9% five-way accuracy but ρ = 0.851** probability-weighted
ordinal correlation, noting that "treated as an ordinal scale, SST-5 is thus solved
considerably better than its 57.9% five-way accuracy suggests." That is the thesis in one line:
**for ordinal decisions the distribution carries more than the argmax.**

## Decision

**Build `Score` as an ordinal head using CORN**, with CORAL as the ablation, trained and
evaluated with the ranked probability score. Specifically:

- **CORN** as the default: rank-consistent via conditional training sets and the chain rule,
  which removes CORAL's weight-sharing restriction.
- **RPS** as the primary metric (a proper scoring rule for ordinal targets), reported alongside
  mean absolute error of the expected scale value and within-one-level accuracy.
- **A monotonicity check** as a correctness property: cumulative probabilities must be
  non-increasing across ordered levels.
- **`Score` is the only trainable component** in the revised architecture.

## Alternatives Considered

### Alternative 1: Keep softmax over ordered levels with cross-entropy (today's behaviour)
- **Pros**: already implemented; one code path for all three primitives; no new machinery.
- **Cons**: discards the ordering — the entire point of an ordinal rubric. Two rubrics with the
  same level count but different semantic spacing are indistinguishable to the loss.
- **Why not**: it is the specific defect the ordinal literature was written to fix, and it makes
  the `Score` primitive indistinguishable from `Choice` with ordered labels.

### Alternative 2: Regression on the ordinal position, then map to a distribution
- **Pros**: simple; directly models the scale; well understood.
- **Cons**: produces a point estimate, not a distribution, so it cannot be scored by a proper
  scoring rule or feed a conformal gate; and it assumes the levels are equally spaced.
- **Why not**: the architecture needs a *distribution* over levels — that is what `Score`
  returns and what the gate consumes.

### Alternative 3: CORAL instead of CORN
- **Pros**: rank-monotonicity and consistent confidence scores with theoretical guarantees;
  architecture-agnostic; lower training complexity through parameter sharing.
- **Cons**: the weight-sharing constraint in the output layer "may restrict the expressiveness
  and capacity" of the network, and CORN was introduced specifically to remove it, with
  substantial measured improvement over CORAL.
- **Why not**: kept as the ablation. If CORAL matches CORN here, the simpler model wins on
  maintenance grounds.

### Alternative 4: Leave `Score` unimplemented until a real rubric task appears
- **Pros**: avoids building against a synthetic target.
- **Cons**: abandons the only remaining differentiator, and the primitive is already in the
  published interface.
- **Why not**: the benchmark target is published and the dataset is ungated, so the experiment
  is cheap and decisive.

## Consequences

### Positive
- Targets the one capability a cosine **categorically cannot** provide, so success is meaningful
  rather than marginal.
- A proper ordinal metric (RPS) already exists in `metrics.py` and becomes load-bearing.
- Gives the project a defensible answer to "what is the trained component for?" — the question
  that ADR-0008 leaves open.

### Negative
- Adds the first genuinely new training code since the head: CORN's conditional training sets
  and the chain rule, plus monotonicity enforcement.
- Requires an ordinal dataset. `SetFit/sst5` is verified ungated (5 ordered levels, 8.5k train /
  2.2k test); it is a proxy for a real rubric, not the real thing.

### Risks
- **It may also fail to beat its baseline.** The acceptance criterion is explicit: CORN must beat
  cross-entropy on **RPS**, and monotonicity must hold. If it does not, `Score` joins the head
  and the composer as a recorded negative result rather than being quietly retained.
- **SST-5 is sentiment, not a decision rubric.** A good result there does not transfer to
  severity or priority grading without a domain dataset. The claim will be scoped accordingly.
- **The published 57.9% / ρ=0.851 pair comes from a different system and harness**, so it is a
  reference point, not a like-for-like comparison.

## Sources

- Cao, Mirjalili & Raschka, *Rank Consistent Ordinal Regression* (CORAL), Pattern Recognition
  Letters 2020
- Shi, Cao & Raschka, *Deep Neural Networks for Rank-Consistent Ordinal Regression Based On
  Conditional Probabilities* (CORN), arXiv 2111.08851
- *Evaluating and Benchmarking the System One Model* — SST-5 at 57.9% accuracy, ρ = 0.851
- `src/decision/metrics.py` — `ranked_probability_score`, `expected_value`, `within_one_level`
- `docs/RESEARCH-NOTES.md` R3 — the ordinal-loss finding
