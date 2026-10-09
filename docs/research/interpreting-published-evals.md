# Interpreting published evaluations on Hugging Face: a provenance-first method

**Date:** 2026-05-13
**Question:** before `/market-research` compares us to anything — how do we read
the numbers other models publish on Hugging Face, and what makes one of ours a
claim a reader can interpret?
**Depends on:** `docs/research/benchmarks-and-evaluation-readiness.md` (what we
measure); `src/decision/modelcard.py` (what we emit).

## Executive summary

A benchmark number on Hugging Face is a **claim with a contract attached**, not
a fact. The contract is the `model-index`/`EvalResult` schema: dataset id,
dataset git revision, config, split, metric id, metric args, the harness that
computed it, and a `verified` flag. Read any number missing those fields as a
vibe, not a result.

Three claim classes exist, in ascending order of what they can support:

1. **Self-reported** — a number the author typed into a card. Interpretable
   only with full provenance; comparable to nothing else by itself. This is the
   class our own output will live in — `verified=False` is honest, not a defect.
2. **Same-protocol baselines** — a number produced by running a different model
   through *our* harness on *our* splits. Directly comparable to us, comparable
   to nothing outside.
3. **Shared-harness** — a number produced by the same evaluation code that
   produced everyone else's (MTEB). The only class that supports ranking against
   the published field. For us this means `mteb.evaluate` on the frozen encoder.

Two numbers are comparable **iff** dataset id + dataset revision + config +
split + metric + protocol all match. Any single mismatch demotes a comparison
to "directional context" — and the discipline of saying so is itself a
differentiator, because most model marketing ignores it.

## 1. The contract: what a Hub eval row is

Hugging Face's structured eval metadata (`model-index` in card front matter;
`repocard_data.EvalResult` in `huggingface_hub`) fields each result as:

| field | what it pins |
|---|---|
| `task.type` / `task.name` | the task definition (e.g. `text-classification`) |
| `dataset.type` | the Hub dataset id (`mteb/banking77`) |
| `dataset.name` | human-readable name |
| `dataset.config` | which dataset config |
| `dataset.split` | which split the number was computed on |
| `dataset.revision` | the dataset's **git sha** — the version pin |
| `metrics[].type` | metric id (`accuracy`) |
| `metrics[].name`/`config`/`args` | metric identity and settings |
| `metrics[].value` | the scalar |
| `metrics[].verified` + `verifyToken` | Hub-side recompute proof |
| `source.name`/`source.url` | who/what produced the row |

**The decisive field is `verified`.** It is set only when Hugging Face's own
eval infrastructure recomputed the metric — the `verifyToken` exists so a fake
`verified: true` can be detected. Almost every model card on the Hub carries
self-reported numbers, including ours. A reader who treats `verified` as the
norm will systematically over-trust the ecosystem's least reliable numbers.

Sources: `huggingface_hub` `modelcard.md` spec + `repocard_data.py` `EvalResult`;
Hub docs "Model Cards" / "Evaluation results".

## 2. Claim classes, and what each can support

### Class A — self-reported (interpretable iff fully pinned)

Any number the model's own pipeline produced on the author's chosen split. The
vast majority of card rows. Interpretation rules:

- **Missing revision or split → treat as a marketing scalar.** It can motivate
  a follow-up, it cannot anchor a comparison.
- **Different split = different task.** "accuracy on CLINC150" means nothing
  without the split; our numbers always name it.
- **The number binds to the protocol, not the dataset.** Two self-reported rows
  on the *same* dataset id are still not comparable if one is zero-shot and the
  other is a fitted head.

### Class B — same-protocol baseline (the honest in-house comparator)

Running a published model through *our* harness on *our* splits. Comparable to
our rows and nothing else. We already do the cheap end of this: TF-IDF+LR runs
through the identical suite (`dataset_suite`) on every breadth dataset. The
strong version — a public SentenceTransformer baseline through the same
encoder→head→gate path — is the honest way to answer "is the encoder carrying
this?" and is a v25 candidate.

### Class C — shared-harness (the only ranking-grade number)

MTEB computes every embedding leaderboard row with one codebase: download a
pinned dataset revision, fit a **logistic regression on frozen train
embeddings**, score the test split — which is structurally our architecture
(encoder + shallow fitted head). `TaskResult` carries `dataset_revision` and
`mteb_version` for exactly this reason.

Because `StateEncoder.model` is a plain `SentenceTransformer`, our encoder can
go through `mteb.evaluate` unchanged — the v24 kernel does this for
`Banking77Classification`, `EmotionClassification`, `MassiveIntentClassification`
and `ImdbClassification`, three of which overlap our breadth suite. The result
is a number that sits on the same footing as every published embedding model:
**the anchor that says our foundation is a known quantity.**

### What each class licenses us to say

| claim | allowed form |
|---|---|
| "our decision layer scores X on banking77" | self-reported, pinned: dataset id + revision + split + arm + run source |
| "our encoder scores Y on MTEB Banking77" | shared-harness; comparable to leaderboard rows |
| "we beat model Z" | only when the comparison is same-harness same-data — Class C vs Class C, or Class B vs our rows. Otherwise: "directionally consistent with" |

## 3. The comparability checklist (before quoting any number)

For every external row we consume in `/market-research`, and every row we emit:

- [ ] dataset id **and** git revision recorded
- [ ] config and split named
- [ ] metric id + metric args named (`main_score` on MTEB classification = accuracy)
- [ ] who computed it: leaderboard harness version, or self-reported run
- [ ] `verified` flag read correctly (false ≠ untrustworthy; true = HF-side recompute)
- [ ] protocol class identified (zero-shot / fitted head / which harness)

**Any missing box demotes the row to context, not evidence.** A row we cannot
classify goes in the landscape as "reported, unverifiable as stated" — never in
a table implying comparability.

## 4. What we changed to be readable this way

- `modelcard.eval_row()` constructs a Hub-shaped row; `model_index()` now emits
  `dataset.config/split/revision`, `metrics[].name`, `verified` (always false —
  we have no Hub-side compute), and `source`.
- `datasets.dataset_revision(repo_id)` pins each breadth dataset's git sha at
  eval time — failing soft to "not recorded" rather than sinking a run.
- The kernel's `mteb_anchor` leg runs the raw encoder through MTEB tasks
  overlapping the breadth suite; rows land in the card labelled
  `main_score [encoder, MTEB harness]`.
- Qualify criterion **B3 (report)**: "a shared-harness anchor is recorded" —
  absent MTEB evidence is `deferred`, listed, never a silent pass.

## 5. What this does *not* establish

- **MTEB numbers measure the encoder + MTEB's own LR head**, not our TaskHead +
  gate. Conflating them in the card would be the precise dishonesty this
  document exists to prevent — the card labels the anchor rows as encoder-only.
- A verified-false row is not discredited; it is simply *our* computation. The
  obligation it creates is provenance, not third-party recompute.
- Comparability is per-row, not per-model: "beats X on banking77" can be true
  while "is a better decision model than X" is not (calibration, abstention and
  coverage are ours alone to show, since baselines don't emit them).

## Sources

1. `huggingface_hub` docs — `modelcard.md` spec; `repocard_data.py`
   (`EvalResult`, `verified`, `verifyToken`, `source` fields).
2. Hugging Face Hub docs — "Model Cards" / evaluation-results metadata.
3. `mteb` — task/evaluate API; `TaskResult` schema (`dataset_revision`,
   `mteb_version`, `main_score`); `descriptive_stats/Classification/*` task list
   (Banking77Classification, EmotionClassification, DBpediaClassification,
   ImdbClassification, MassiveIntentClassification).
4. `docs/research/benchmarks-and-evaluation-readiness.md` — the internal side:
   what our qualification actually measures.
