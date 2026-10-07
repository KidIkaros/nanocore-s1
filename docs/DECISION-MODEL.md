# NanoCore-S1 Decision Model — Build Plan and First Results

Date: 2026-10-07. Status: **prototype implemented; D1/D2 Kaggle smoke baselines completed; D3 pending.**
This is the decision-model line of work alongside the historical decoder-only
plan in `ARCHITECTURE.md`; the reassessment evidence is in
`ARCHITECTURE-REASSESSMENT.md`.

## The model

```
state items  [text | {"image": p} | {"audio": p} | {"video": p} | code ...]
    │
    ▼  EmbeddingGemma 2 (FROZEN — 270M text / +170M vision / +300M audio)
item embeddings  v₁..vₙ  ∈ ℝ⁷⁶⁸   (one shared space; the multimodal shortcut)
    │
    ▼  EmbeddingComposer (TRAINABLE — karpathy-style bidirectional transformer)
state vector  s  ∈ ℝ⁷⁶⁸   (attention over item vectors; n=1 baseline is mean pooling)
    │
    ▼  DecisionHead (TRAINED — Jev-style, per-option isolated, zero-bias final)
score_j = φ(s, o_j)  →  softmax / τ
    │
    ▼  Prediction{probabilities, confidence, abstention}   Choice / Score / Noul
```

- **Karpathy's framework** is now a model component, not just a style reference:
  a 4-layer, bidirectional transformer over EG2 item vectors. EG2 acts as a
  multimodal tokenizer — each modality becomes a 768-d item vector. The
  composer lets multi-item states (text + screenshot + audio note) be composed.
  jev-stack's heads consume one pre-pooled vector; they do not compose a sequence.
- **Jev's decision behaviour** — `Choice`/`Score`/`Noul`, per-option isolated
  scoring (order invariance by construction), zero-bias final layer (open-set
  abstention), and log-score training. The head is trained/evaluated in the
  prototype; the composer has only passed a shape/finite-value smoke test and
  has **not yet been trained or shown to improve over mean pooling**.
- **EmbeddingGemma 2** — frozen encoder, modular loading (`config_kwargs`),
  MRL truncation with re-normalization, bf16/fp32 only. Implemented in
  `src/decision/encoder.py`.

## Demo schedule and decisions

| Demo | Content | Pass criterion | Result / decision |
|---|---|---|---|
| D1 — text Choice | Banking77, 77 intents; EG2 text embeddings | trained head > zero-shot cosine on held-out test | **Not met.** Zero-shot cosine 92.92%, kNN 93.60%; fingerprint 93.15%, interaction 92.73%. Keep retrieval as the accuracy baseline; a head may still be justified for typed calibrated distributions, but it has not earned an accuracy claim. |
| D2 — image Choice | CIFAR-10, 10 train + 10 test examples/class via EG2 vision | trained head > zero-shot cosine on held-out test | Fingerprint 93% vs zero-shot 91% on only 100 test images. Directionally positive, but far too small to claim a robust gain; repeat on a predeclared larger split before relying on it. |
| D3 — composer | multi-item states on a real composition task | composer + head > mean-pool + head on held-out | **Not run.** Composer is implemented (28.3M parameters) but only shape/finite-output tested. Do not claim it improves decisions yet. |

D1/D2 ran in Kaggle kernel `mauricew/nanocore-s1-baseline`, version 13.
The source and contract tests are embedded in the notebook, so the run does not
mount the worktree dataset. D1/D2 model loading and real-data work stay on Kaggle;
no EmbeddingGemma weights were loaded on the laptop.

## Baselines and evaluation

1. **Majority class** — floor: 1.30% on Banking77.
2. **Zero-shot cosine** `argmax_j cos(s, o_j)` — EG2 state vs intent-name
   embeddings; 92.92% on Banking77 and 91% on the tiny CIFAR smoke split.
3. **kNN** (k=1, 5) over training-state embeddings — both 93.60% on Banking77.
4. Candidates: `fingerprint` and `interaction` heads. Both are trained with
   one-hot real labels in this experiment; soft teacher distributions and RLCD
   remain supported by the API but were not evaluated here.

| Banking77 model | Accuracy | Log score | Brier | ECE |
|---|---:|---:|---:|---:|
| Zero-shot cosine | 92.92% | not measured | not measured | not measured |
| kNN-1 / kNN-5 | 93.60% | not measured | not measured | not measured |
| Fingerprint head | 93.15% | 0.402 | 0.117 | 0.049 |
| Interaction head | 92.73% | 0.387 | 0.123 | 0.046 |

The head calibration metrics are reported on the 3,080-item test split after
temperature fitting on a held-out 10% of the training partition. ECE is reported,
not gated; log score and Brier are the proper-score comparisons between heads.
The current run did not compute probabilistic metrics for the cosine/kNN
baselines, so the table does not imply those heads beat the baselines on every
metric. CIFAR's 100-item test subset is a smoke test, not a benchmark.

Honesty controls passed: shuffled labels scored 1.17% and a zeroed head scored
1.30%, both near the 1/77 chance rate. The full Kaggle run also passed all 55
contract tests. These validate code contracts and the evaluation harness, not
production capability.

## Fallback plan

- If a trained head does not beat zero-shot cosine on accuracy, do not claim a
  learned accuracy improvement. Prefer measured retrieval (kNN where it wins),
  and calibrate/validate abstention separately before deployment.
- Abstention is the human-in-the-loop path by design (`unseen` → escalate).
  Thresholds in the prototype are fitted on held-out data; they are not yet
  operational policy.
- If EG2's space separates labels poorly (fingerprint-distance diagnostic),
  improve option descriptions or choose another encoder — not merely a larger
  head.
- The multimodal choice result is preliminary until repeated on a larger,
  held-out image split with uncertainty intervals.

## Compute boundary

- Local: code authoring and contract tests on random/small tensors only; no
  EmbeddingGemma model loading.
- Kaggle only: EG2 any configuration, real-data encoding, composer training,
  and all capability claims. bf16 on GPU, never fp16. Re-normalize after MRL
  slicing.
- The final Kaggle run completed in about 20.5 minutes wall-clock. The
  text+vision encoder loaded 439M parameters in bf16. The image microbatch run
  peaked at 2.28 GiB allocated on the active GPU. An earlier batch-32 image
  attempt OOM'd; reducing image microbatches to 2 completed D2.

## Data and splits

- Banking77 canonical 13,083 examples / 77 intents; this run used 10,003 train
  and 3,080 test rows from the source CSVs. 90% of train fit the heads; 10% was
  held out for temperature and abstention-threshold fitting. Intent names were
  humanized for option text.
- CIFAR-10 via Hugging Face datasets; 10 examples/class for fit and 10/class
  for test in the smoke run.
- The shared training-example schema `{state, questions, gold}` remains
  compatible with jev-stack's data pipeline.

## Explicit non-goals / next experiment

- No autoregressive decoding in this decision path (the decoder remains an
  untouched Route-4 legacy path).
- No Laya dependency or comparison run — Laya is a category illustration only.
- No free-form generation or `[ANSWER]` text parsing.
- Next: choose a defensible multi-item task, then compare trained composer +
  head against mean-pooling + the same head on the same held-out split. Until
  that experiment passes, the composer is an implemented component, not a
  validated improvement.
