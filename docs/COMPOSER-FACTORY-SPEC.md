# Component 38 — composer factory: train a composer, ship it through adapt()

**Date**: 2026-10-10 · **Status**: design + build spec · feeds the loop's history,
conversation (Q6), and E4's composition arm
**Depends on**: `src/decision/composer.py` (BUILT, untrained), `adapt()` (BUILT,
no composer path), `DecisionModel.save/load` (BUILT, `composer.pt` round-trips)
**Blocking**: E4 (composition indistinguishable), Q6 (converse), the loop's
structured history (`05e31c2`), the retrieval proposer (T1)

## The gap, precisely

`DecisionModel` has a `composer` slot. `save()` persists it to `composer.pt`;
`load()` rebuilds it. `compose_items()` is the single composition path, shared by
`DecisionModel._state_vec` and the legacy model. **Everything except the thing
that trains a composer exists.** `adapt()` takes no `composer` argument and
produces no composer bundle — so no composer ever ships, which is why:

- E4 measured `full` (mean-pooled) *below* `image_only` and gated on
  non-inferiority — there is no trained composer to test.
- The loop's history is mean-pooled (ordering/who-said-what destroyed — the
  sentence-pair bug one level up).
- Conversation degrades to history-as-text.

## The build — one factory function, one config flag

`adapt_composer(item_embs, targets, encoder, cfg) -> (composer, report)`:

1. **Input**: per-item embeddings `(n_items, dim)` for multi-item training states
   + the downstream target (the head's labels, or a held-out composition metric).
2. **Train** the `EmbeddingComposer` so its composed state vector reproduces what
   mean-pooling would have produced *plus* the structure mean-pooling loses.
   The objective is the load-bearing design choice — see below.
3. **Return** the composer + a report (params, init-vs-final composition drift,
   the non-inferiority margin vs mean-pool).

`AdaptConfig` gains `composer: Optional[ComposerAdaptConfig] = None`. When set,
`adapt()` builds the composer, composes each training state through it, fits the
head on *composed* vectors, and saves `composer.pt` into the bundle. When `None`,
behavior is byte-identical to today (mean-pool) — the earn-the-slot default.

## The design decision that needs the owner: the training objective

Three options, and this is the one real choice:

**(a) Reconstruction** — composer(compose(items)) ≈ the pooled vector, plus a
regularizer that preserves per-item retrievability. Cheap, self-supervised (no
labels), but it only learns to *not destroy* the pooled signal — it does not
learn composition that helps the decision.

**(b) End-to-end through the head** — compose → head → loss, backprop into both.
The composer learns exactly the composition that improves the decision. This is
the "joint fusion" the Multimodal DL taxonomy names, and it is what E4 needs to
beat. Cost: the composer and head train together, so the headroom check and the
head fit both move onto composed vectors.

**(c) Contrastive** — composed states of *related* items close, *unrelated* far.
Learns structure without labels, but the notion of "related" is task-defined and
we have no such signal for arbitrary states.

**Recommendation: (b), with (a) as the init.** The composer already starts as the
identity (zero-init output projections), so early epochs *are* mean-pool — (a) is
the starting point, not a separate objective. (b) is the only option that makes
E4's composition claim measurable: it produces a composer that beats mean-pool on
the *decision* metric, which is the earn-the-slot bar. (c) is out — no relatedness
signal.

## Acceptance criteria (written before build, per R2)

1. `adapt(composer=...)` produces a bundle with `composer.pt`; `load()` rebuilds
   it; a round-tripped composed decision is identical to the pre-save one
   (200/200, the bundle-roundtrip bar).
2. **E4 resolves**: with a trained composer, `full` (composed) vs `image_only` vs
   `text_only` on ScienceQA produces a *paired* readout where composition is
   either superior or honestly non-inferior with the composer carrying signal
   mean-pool cannot. The current "indistinguishable" must become a real verdict.
3. **Non-inferiority floor**: composed ≤ mean-pool accuracy by no more than the
   5-pt margin on every breadth dataset (the composer may not regress the shipped
   path). If it regresses, it does not ship — earn-the-slot.
4. **The loop carries structure**: a multi-turn state composed through the trained
   composer produces a decision that depends on turn *order* (swap two turns →
   different decision), which mean-pool cannot express. One test.
5. `composer=None` path unchanged: full suite green, byte-identical decisions.
6. Identity-at-init preserved *to the documented tolerance*: an untrained
   composer's output equals mean-pool up to the modality-embedding additive
   term (std 0.01) and the final RMSNorm — measured 3e-3 max drift after
   unit-norm on random 768-d input, NOT machine-zero. Criterion 3's
   non-inferiority floor is what actually guards regression; this invariant
   bounds the *untrained* perturbation so the baseline is meaningful.

## Where it runs

Torch-tier: composer training needs gradients, so the fit is Kaggle (ADR-0006 /
R1). The *factory function* is local-unit-testable with synthetic arrays and the
`StubEncoder`; the real fit is a kernel leg. `adapt()` already splits fit/cal/test
three ways — the composer fits on `fi` only, never sees cal/test.

## What it unblocks (the three payoffs)

- **E4** — a real composition arm, superior or honestly non-inferior.
- **Q6 / conversation** — the loop and dialogue carry structured history.
- **The retrieval proposer (T1)** — reads the composed state; Jev-Mem's
  "informed by observed behavior" needs structure to read.

## Risks

- **Objective coupling**: (b) ties the composer to the head; a composer trained
  for one head may not transfer. Mitigation: the bundle carries both together
  (`composer.pt` + head), and `_composer_id` is already folded into the cache key
  so a composed and pooled decision never collide.
- **Overfit at small n**: the composer is a 4-layer transformer; on a few hundred
  examples it can memorize. Mitigation: identity-at-init + the non-inferiority
  floor (criterion 3) — a memorizing composer regresses on test and doesn't ship.
- **Latency**: composition adds a forward pass per decide. Mitigation: it replaces
  the pool, doesn't add to it; measure in the Phase-8 leg, not assumed.
