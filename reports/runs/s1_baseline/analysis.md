# NanoCore-S1 first decision-model baseline

**Kaggle:** `mauricew/nanocore-s1-baseline`, version 13, NvidiaTeslaT4 (two T4s listed by Kaggle). **Status:** complete. **Wall time:** 20m 31s. The full kernel output is under `/tmp/s1-out13/`; this directory preserves the compact durable result and interpretation.

## What ran

- Self-contained notebook: source and contract tests embedded in the notebook; `dataset_sources: []`. This removes the prior Kaggle mount failure (`ERRORED_MOUNTING_DATASET`) from the execution path.
- 55 contract tests passed on Kaggle.
- EmbeddingGemma 2 text+vision configuration loaded in bf16; 439M parameters, 768-d output.
- Banking77: 10,003 training / 3,080 test rows, 77 intents.
- CIFAR-10 smoke test: 10 fit and 10 test examples per class (100 each).
- The composer passed a finite-output shape smoke test only; no composer training or composition-quality comparison was run.

## Results

| Banking77 method | Accuracy | Log score | Brier | ECE |
|---|---:|---:|---:|---:|
| Majority | 1.30% | — | — | — |
| Zero-shot EG2 cosine | 92.92% | not measured | not measured | not measured |
| kNN-1 / kNN-5 | 93.60% | not measured | not measured | not measured |
| Fingerprint head | 93.15% | 0.402 | 0.117 | 0.049 |
| Interaction head | 92.73% | 0.387 | 0.123 | 0.046 |

CIFAR-10 smoke: zero-shot cosine 91%; fingerprint with 10 examples/class 93% on 100 test examples. Treat only as evidence that the image path executes, not as a reliable estimate of generalization.

Controls: shuffled labels scored 1.17%; zeroed head scored 1.30%, matching the 1/77 chance rate. This is a useful sanity check that the evaluation responds to broken inputs.

## Decision

- D1's accuracy success criterion **failed**: neither trained head beat zero-shot cosine, and kNN was stronger than both. Do not claim that head training improves Banking77 accuracy. Retrieval is the measured accuracy baseline.
- Among the heads, interaction had the lower log score (0.387 vs 0.402); fingerprint had better Brier (0.117 vs 0.123) and accuracy. Cosine/kNN proper scores were not computed, so this is only a head-to-head comparison.
- D2's +2-point result is underpowered; repeat on a larger predeclared split before treating it as a gain.
- The multimodal encoder and typed prediction interface are now implemented and Kaggle-executed. The transformer composer is implemented but untrained; D3 remains open. No end-to-end claim that the complete architecture beats a pooled baseline is established.

## Issues found and resolved during validation

1. Kaggle could not mount `mauricew/nanocore-s1-worktree` (`ERRORED_MOUNTING_DATASET`). The final notebook embeds the 13 required source/test files and attaches no dataset.
2. The first image attempt used batch size 32 and OOM'd. A retry with batch size 2 and 10 examples/class completed; measured vision peak allocation was 2.28 GiB.
3. Banking77's HF loading script is no longer accepted by `datasets` 4.x. The notebook reads the canonical GitHub CSVs and extracts the label ordering from the upstream dataset script.

## Reproducibility

Notebook: `notebooks/s1_baseline/s1_baseline.ipynb`

Metadata: `notebooks/s1_baseline/kernel-metadata.json`

Compact structured output: `results.json`
