# State of the Project — what the experiments established

Date: 2026-10-07. **Status: consolidation.** Six Kaggle runs, four on the decision model
itself. This document states what is established, what was withdrawn, what has never been
exercised, and what the evidence implies. It is written so that the next decision can be
made from a clean position.

Companions: `ARCHITECTURE-DECISION-MODEL.md` (components and validation matrix),
`RESEARCH-NOTES.md` (R1–R10), `ARCHITECTURE-SHARPENING.md` (proposals and outcomes),
`adr/` (decisions), `reports/runs/` (raw artifacts for every run below).

---

## 0. The one-paragraph answer

EmbeddingGemma 2 zero-shot reaches **~93%** on both benchmarks we tried — Banking77 intent
classification (92.92%) and BFCL tool routing (93.07%). A frozen encoder plus a
temperature-scaled cosine is therefore a **complete solution** for both tasks, and no trained
head can show an improvement because there is nothing left to win. The one design property
that is solidly validated is **order invariance**. The abstention gate is implemented but its
signal is weak (separation 0.097, 3.5% auto-handled at 90% precision). The composer and the
ordinal `Score` primitive have **never been exercised**. The binding constraint on every
architectural claim is **benchmark saturation**, not model quality.

---

## 1. Claim ledger — established, with evidence

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| 1 | EG2 zero-shot is ~93% on Banking77 and on BFCL tool routing | 92.92% / 93.07%; both flagged `usable: false` by `headroom_check` | **high** |
| 2 | Order invariance holds on real embeddings, fixed and variable option sets | Banking77: 2.38e-07, 0 flips / BFCL: 1.19e-07, 0 flips over 2–37 options | **high** |
| 3 | A temperature-scaled cosine matches the trained head on Banking77 | cosine+τ 0.380 / 0.117 / 0.045 vs head 0.377 / 0.116 / 0.046 (1 param vs 59,136) | **high** |
| 4 | The trained head is *worse* than zero-shot on dispatch routing | 90.84% vs 93.07%; no winners on any metric from `compare_to_best` | **high** |
| 5 | kNN wins on classification and collapses on routing | 93.6% on Banking77 vs 71.5–73.1% on BFCL | **high** |
| 6 | The prompt pairing materially affects results, and the incumbent was wrong | `SearchQuery`/`Document` gains +0.94 kNN, +0.50 head, +0.49 zero-shot; symmetric is *worst* | **high** |
| 7 | Head initialization from option embeddings has no effect | 93.12% vs 93.11% random, log 0.377 both | **high** |
| 8 | Normalizing fingerprints + training is harmful | accuracy 93.11% → **79.66%** | **high** |
| 9 | The abstention gate's relevance signal is weak | separation 0.097; 3.46% auto-handled at 90% precision | **medium** (single split) |
| 10 | The legacy decoder path is not viable | 20 steps at loss 6.7661 vs `ln(867)=6.7650`; inert `window_pattern`; decorative `head_dim`; synthetic decision accuracy; ~150 ms/token vs a claimed 15–35 ms | **high** |
| 11 | The fingerprint head ignores option text entirely | two option prompts produced byte-identical head results; `_scores_batch` never reads `option_embeddings` | **high** |

---

## 2. Claims made and withdrawn — the corrections

Recorded because the pattern matters more than the individual errors.

| Claim we published | What it actually was | Corrected to |
|---|---|---|
| "Naive retrieval calibration fails badly: cosine log 3.800, ECE 0.907" | **my temperature grid's lower bound** — three baselines fitted it | cosine+τ reaches log **0.380**, ECE 0.045 |
| "The head beats the baseline on Brier" | **verdict compared against the weakest baseline** of four | against the best baseline it loses on Brier, ECE and accuracy |
| "Headroom is a Banking77 problem" | assumed task-specific | BFCL routing is **also** saturated at 93.07% |
| "Initialize the head from option embeddings (highest expected value)" | never tested when proposed | tested: **no effect** |
| "Normalize fingerprints per the Zero-Bias paper" | paper's remedy presumes a norm-controlled parameterization | applied as a forward-pass transform: **harmful** |
| "The 90%-precision gate" | saturates when base accuracy exceeds 90% | gate must be set above the base error rate |

**Three of the six came from the measuring instrument, not the model.** That is why the
protocol now exists as tested code with regression tests rather than as repeated care.

---

## 3. Never exercised

| Capability | State |
|---|---|
| `EmbeddingComposer` | implemented (28,315,392 params), contracts tested, **never trained** |
| `Score` (ordinal) | schema tested, **never run on real data**; currently uses cross-entropy, the loss the ordinal literature says discards ordering |
| `Noul` | exercised only as the relevance signal in §1 row 9, not as a trained primitive |
| Multimodality | vision: 100-image smoke test. audio/video: **untested** |
| Latency | **never measured** at batch 1, despite "extremely lightweight" being the design's central claim |
| End-to-end deployment | never attempted; every number comes from an offline split |

---

## 4. The binding constraint: saturation

This is the strategic finding, and it is not about our code.

If a frozen 270M encoder already answers ~93% of a task, then:
- a trained head has **7% of headroom** to work with;
- differences of 0.2–0.5 points are the whole available signal;
- a null result is **uninformative** — it may mean "no head can help here", not "our head is bad".

Both benchmarks we selected hit this wall. The practical consequence for any future
experiment: **measure zero-shot first, and refuse to draw architectural conclusions without
headroom.** `protocol.headroom_check` now enforces that mechanically.

The one task we have found that is *not* saturated is relevance/quality estimation —
separation 0.097, and BFCL's irrelevance set is deliberately hard. That is also the task the
routing literature names as the critical factor.

---

## 5. What the evidence implies for the architecture

**Supported by evidence:**

```
frozen EmbeddingGemma 2  →  typed interface (Choice / Score / Noul)
                         →  temperature-scaled cosine scoring
                         →  abstention gate
```

No trained head. This is not a defeat: it is a 270M-parameter, training-free dispatcher that
routes at 93% with calibrated probabilities, and the whole cost argument (33–40 ms to decide
versus ~2 s to generate) survives intact.

**Not supported:** the trained head for `Choice` and for routing. Three experiments, no win
over the best baseline on any proper metric.

**Still open, and the only places a trained component could earn its place:**

1. **Ordinal `Score`** — temperature-scaled cosine has no notion of order at all. This is the
   strongest remaining candidate and the only capability that is categorically different.
2. **Composition** of multi-item states — unproven, prior leans negative for
   classification-shaped tasks, positive when relevance is uneven.

---

## 6. Decision options

| Option | What it means | Evidence position |
|---|---|---|
| **A. Ship the interface, not a model** | EG2 + typed interface + calibrated scoring + abstention. Drop the trained head for Choice/routing. | Directly supported; the only option with no unsupported claims |
| **B. Build the ordinal `Score` track** | CORN/CORAL + RPS + monotonicity, on a rubric dataset with ordinal labels | The one categorically different capability; needs new data |
| **C. Find harder tasks** | Deliberately seek domains where zero-shot is well below ceiling, where a head or composer could matter | Addresses the binding constraint; unbounded search |
| **D. Test the composer** | Composer vs mean pooling at equal information, both regimes | Decisive for ADR-0003; pre-register the null |

Options are not exclusive. A is available now; B and D are the only remaining paths to a
trained component; C is the only path that makes B or D *measurable*.

---

## 7. Instrument status

The measuring apparatus is now in better shape than the model.

| Artifact | State |
|---|---|
| `src/decision/protocol.py` | one temperature fitter (two-stage, edge-safe), one metric block, ragged support for variable option sets, headroom check, dispatch metrics, best-baseline verdicts |
| `tests/test_decision_protocol.py` | 26 tests, including regression tests for both published errors and the residual grid-edge defect |
| **87 tests** | pass on Kaggle in ~6 s |
| `notebooks/s1_tests` | reusable CPU fast-fail harness — no GPU, no internet, no quota |
| Execution boundary | ADR-0006; local is reading/writing/static checks only. Violated once, structurally fixed |

Every number in this document comes from a Kaggle run with artifacts in `reports/runs/`.
