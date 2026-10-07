# s1_dispatch — routing and abstention on BFCL

**Kaggle:** `mauricew/nanocore-s1-dispatch`, version 2, T4.
**Status: COMPLETE.** First run of the dispatch benchmark (R9), and the first to use
`src/decision/protocol.py` for every measurement.

## Data

BFCL v3, loaded and filtered in-kernel: **1,252 routing queries** with 2–37 candidate
tools each, and **1,118 irrelevance queries** (no candidate is relevant), giving 2,300
irrelevant (query, tool) pairs.

## Headroom check failed — the benchmark is saturated too

| | zero-shot accuracy | headroom | usable |
|---|---:|---:|---|
| Banking77 (intent classification) | 92.92% | 6.9% | no |
| **BFCL tool routing** | **93.07%** | **6.93%** | **no** |

`protocol.headroom_check` returned `usable: false` with the note *"insufficient headroom —
a better head cannot show a better number"*. **This is the most important result in the
run.** The headroom problem is not a Banking77 quirk: EmbeddingGemma 2 zero-shot reaches
~93% on *realistic tool routing* as well, so a frozen encoder plus a temperature-scaled
cosine is already a complete router, and no trained head can demonstrate an improvement
because there is nothing left to win.

## Routing results

| method | accuracy | log score | Brier |
|---|---:|---:|---:|
| random (floor) | 32.00% | 1.202 | 0.676 |
| **zero-shot cosine** | **93.07%** | **0.176** | **0.099** |
| interaction head (mean of 3 seeds) | 90.84% | 0.253 | 0.137 |
| kNN-20 votes | 73.07% | 0.630 | 0.343 |
| kNN-5 votes | 71.47% | 0.689 | 0.368 |

`compare_to_best` returned **no winners on any metric**: the best baseline leads accuracy,
log score and Brier, and the trained head is worse than it on all three (seeds: 90.13%,
92.00%, 90.40%).

**Two things worth separating here.**

1. **The head loses because the task is saturated *and* the training set is small** — 877
   examples for a variable-option-set task. This is the documented low-data regime where a
   linear probe underperforms zero-shot (LP++: ~20 points *below* zero-shot in the 1-shot
   setting). Both causes point the same way: there is no accuracy to be won.
2. **kNN collapses on dispatch (71–73%) after winning on Banking77 (93.6%).** The
   query→tool mapping is not locally smooth in embedding space — similar queries need
   *different* tools. So "retrieval is the strong baseline" is task-dependent: retrieval
   wins on classification and loses badly on routing. That is a genuine, useful
   correction to the earlier framing.

## Order invariance holds on real variable option sets

| check | value |
|---|---|
| states checked | 80 |
| option cardinality | **2–37, varying per request** |
| max abs probability difference | **1.19e-07** |
| argmax flips | **0** |

Previously validated only on Banking77's fixed 77 labels. This is the real test, and it
passes. It remains the one design property with solid empirical support.

## Abstention — the unsaturated task, and the gate is weak

Relevance signal from the zero-shot raw score (not softmax: most irrelevance queries carry
a single candidate, where softmax is trivially 1.0):

| | value |
|---|---:|
| mean score, relevant pairs | 0.7401 |
| mean score, irrelevant pairs | 0.6428 |
| **separation** | **0.0973** |
| auto-handled at 90% precision | **3.46%** (96.5% escalated) |
| auto-handled at 95% precision | 2.36% |
| cost saved vs always-strong | **3.4%** |

**This is the finding that matters for the next step.** To hold 90% precision the gate must
escalate 96.5% of traffic — it saves 3.4% of cost, which is not a product. Meanwhile BFCL's
irrelevance set is *constructed to be hard* (tempting-but-wrong tools), so unlike routing
this task is **genuinely unsaturated** — and the routing literature names the quality
estimator as the critical factor for exactly this reason.

So: routing is saturated and uninteresting; **relevance/quality estimation is hard, real,
and unsaturated.** That is where the work belongs.

## A residual defect in the protocol I just wrote

Several fitted temperatures sit at **0.000333**, the lower edge of the *refinement* window.
The two-stage fit searches `[t/3, 3t]` around the coarse winner, so when the coarse winner
is the coarse grid's minimum (10^-3), the refinement's own lower bound can still bind. For
option counts 6, 7, 8, 10 and 37 the optimum lies below it.

This is the same class of bug as the original grid-edge failure, surviving one level down.
The fix is to extend the coarse grid downward (e.g. to 10^-5) or to detect an edge hit and
re-expand. It does not change the routing conclusion — the baseline wins by 2+ points of
accuracy regardless — but it means the reported log scores for those groups are not their
best, and it must be fixed before the numbers are used for a calibration claim.

## Artifacts

- `results.json` — full output
- `nanocore-s1-dispatch.log` — kernel log (v1 OOM'd on batch-64 encoding; fixed with
  clipped text and batch 16)
