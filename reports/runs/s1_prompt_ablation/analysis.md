# s1_prompt_ablation — was the state/option prompt asymmetry hurting the head?

**Kaggle:** `mauricew/nanocore-s1-prompt-ablation`, version 1, T4.
**Status: COMPLETE.** First kernel to re-encode from the encoder in-kernel — no cached
artifact, so this validates the pipeline end to end, not just the head.

## Why this ran

States were encoded with `prompt_name="Classification"`, options with `"Document"`.
EG2's task prefixes steer representations, and a dot product between two different
prompt-conditioned subspaces was never justified. This tested four pairings.

## Results (Banking77, 3,080-item test split)

| state / option prompt | zero-shot cosine | kNN-5 + τ | fingerprint head | head log | head Brier | head ECE |
|---|---:|---:|---:|---:|---:|---:|
| `Classification` / `Document` (incumbent) | 92.89% | 93.64% | 93.11% | 0.377 | 0.116 | 0.046 |
| **`SearchQuery` / `Document`** | **93.38%** | **94.58%** | **93.60%** | **0.299** | **0.103** | **0.026** |
| `Classification` / `Classification` | 92.05% | 93.64% | 93.11% | 0.377 | 0.116 | 0.046 |
| *(none)* / *(none)* | 92.89% | 94.25% | 93.51% | 0.316 | 0.104 | 0.033 |

Deltas against the incumbent, for `SearchQuery` / `Document`:

| metric | Δ |
|---|---:|
| zero-shot cosine accuracy | **+0.49** |
| kNN-5 accuracy | **+0.94** |
| head accuracy | **+0.50** |
| head log score | **−0.077** (better) |
| head Brier | **−0.012** (better) |

## What this establishes

**1. The incumbent prompt pairing was suboptimal, and the fix is the retrieval pairing —
not symmetry.** `SearchQuery`/`Document` is the canonical query→document pairing, and it
improves *every* candidate: zero-shot, kNN-5, the head, and the proper scores. The
asymmetry was not the defect; using the wrong prompt *family* for a matching task was.

**2. Symmetry is the worst option, which rules out the obvious hypothesis.** The
"principled" symmetric pairing `Classification`/`Classification` scored **92.05%**
zero-shot — *below* the asymmetric incumbent (92.89%) and 1.3 points below
`SearchQuery`/`Document`. "Make the prompts symmetric" was the intuitive fix and it is
wrong.

**3. The option prompt only affects cosine-scored retrieval.** `Classification`/`Document`
and `Classification`/`Classification` produce *byte-identical* kNN-5 and head numbers,
because neither uses option embeddings: kNN-5 scores against training states, and the
fingerprint head learns free per-label vectors.

**4. A significant architectural finding: the fingerprint head ignores option text
entirely.** In `fingerprint` mode, `_scores_batch` computes `states @ fingerprints.T` and
never reads `option_embeddings`. So the design's claim that "a new option needs no
retraining — it is just another embedding" (ADR-0002) is **true only for `interaction`
mode**. In fingerprint mode a new option has no fingerprint and must be registered and
learned. ADR-0002 has been corrected.

**5. The head and kNN-5 trade wins rather than one dominating.** In the best
configuration: the head wins the **log score** (0.299 vs 0.346) and **ECE** (0.026 vs
0.028); kNN-5 wins **accuracy** (94.58% vs 93.60%) and **Brier** (0.094 vs 0.103). Neither
model dominates, and any claim that one is simply better is unsupported.

## Caveats

- **My verdict threshold was marginally too strict.** The verdict block used
  `|Δaccuracy| > 0.005` and reported `prompt_choice_matters: false` for the head, whose
  delta was 0.00494 — under by 0.00006. For kNN-5 the delta was 0.0094, which exceeds the
  threshold. The threshold was arbitrary; the effect is real and larger for retrieval
  than for the head. This is a third instance of a verdict rule driving the conclusion
  rather than the measurement.
- **The temperature grid still binds.** kNN-5 fitted τ = 0.5012 in every configuration —
  the grid's lower bound again. Its log score and Brier are therefore not its best, so the
  head-vs-kNN-5 proper-score comparison remains provisional. Fixed in the sharpening
  kernel (grid widened to 10⁻³ … 10²).
- **Abstention at 90% precision saturates** in all four configurations (coverage 1.000),
  as expected when base accuracy exceeds 90%.
- One split, one dataset. No split-level variance.

## Consequence

The incumbent prompt pairing should be changed to `SearchQuery`/`Document` for this task
family. Every subsequent measurement should use it, and the earlier numbers produced
under `Classification`/`Document` are pessimistic by roughly half a point.

## Artifacts

- `results.json` — full per-configuration output
- `nanocore-s1-prompt-ablation.log` — kernel log
- `kernel_run_status.json` — Kaggle status record
