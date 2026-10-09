# Runbook — install, adapt, decide, serve

Four commands, in the order you meet them. Everything below runs on
`--backend stub` (no weights) so you can try the whole path before downloading
anything; the stub is a toy and must never be used for a real decision.

## 1. Get labeled data into a file

A CSV, TSV or JSONL with a text column and a label column. Nothing else is
required — no schema, no split, no preprocessing.

```csv
text,label
i need to cancel my flight,cancel
book me on the 8am,book
what is my balance,balance
```

For JSONL, name the columns explicitly:

```bash
--text-col utterance --label-col intent
```

Columns are **named, never guessed**. A file whose columns do not match fails
with the names it did find, because guessing which column is the label is how a
feature silently becomes a target and the resulting bundle is nonsense that
still trains.

## 2. `adapt` — data → a calibrated bundle

```bash
python -m src.decision.cli adapt \
    --data tickets.csv --text-col text --label-col intent \
    --out models/tickets \
    --backend st            # stub | st | llamacpp
```

Writes `models/tickets/bundle/` (the artifact you ship) and
`models/tickets/adapt_report.json` (the evidence for it).

| flag | meaning |
|---|---|
| `--alpha` | miscoverage target; set coverage ≈ `1 - alpha` (default 0.10) |
| `--min-cal` | calibration floor. `adapt` **fails below it** rather than shipping an under-calibrated gate |
| `--head-kind` | `linear` (default) or `mlp` |
| `--epochs` | head training epochs; the head's own default applies if unset |
| `--policy` | `full` (clarify enabled) / `escalate` / `answer` |
| `--glial` | attach the label-free slow state (adapts under shift) |

### Read the output, especially headroom

```
data      180 examples, 3 classes (fit 90 / cal 45 / test 45)
headroom  zero-shot cosine already gets 0.356 — below 0.85 there is room for a head to help (ADR-0011)
head      linear, 3 epochs
test      accuracy 0.244 | Brier 0.683 | coverage 1.000 | mean set 3.00 | resolved 0.178
bundle    models/tickets/bundle
```

- **headroom** — if zero-shot cosine already scores above ~0.85, the task is
  saturated and no trained head can be justified on it (ADR-0011). The honest
  outcome is "a frozen encoder already solves this", not a marginal win.
- **coverage** vs `1 - alpha` — it should sit *at* the target, not far above.
- **mean set** — the number that catches a degenerate gate. A set the size of
  the whole label space with coverage 1.000 means the conformal layer is
  vacuous; the fix is a fitted `t_set` (v19), not a higher `alpha`.
- **resolved** — the fraction the gate answered rather than clarifying or
  escalating. Low resolution with high accuracy means a conservative gate.

## 3. `decide` — one typed decision

```bash
python -m src.decision.cli decide "i need to cancel my flight" \
    --options "cancel,book flight,balance,transfer" \
    --backend llamacpp --model models/embeddinggemma-300M-Q8_0.gguf \
    --bundle models/tickets/bundle
```

Returns an action — `answer`, `clarify`, `escalate`, `abstain` — plus
calibrated probabilities and the conformal prediction set. **Without
`--bundle` it is ungated**: probabilities only, no action.

Batch mode loads the encoder once: `decide - --options "..." --inputs demos.txt --json`.

## 4. `serve` — the same thing over HTTP

```bash
python -m src.decision.cli serve --bundle models/tickets/bundle --port 8000
```

`POST /decide` with `{"text": ..., "options": [...]}`; `GET /healthz`, `GET /stats`
(latency percentiles, action counts). Every decision is appended to the
prediction log, which is what `monitor.py` and the Phase 6 release loop consume.

## Backends

| `--backend` | weights | use |
|---|---|---|
| `stub` | none | tests, dry runs. **Never for a real decision** |
| `st` | sentence-transformers EG2 | Kaggle / dev (needs a GPU for real data) |
| `llamacpp` | a GGUF file | the on-device deployment path (ADR-0007) |

## 5. Phase 8 — run it on your own machine, and measure it

The last definition-of-done item: **"it runs on the target device with measured
latency."** Four steps, no new code.

### 5.1 The runtime

```bash
pip install llama-cpp-python
```

Compiles from source unless a prebuilt wheel matches your platform. It is the only
optional dependency; everything else runs on numpy/torch.

### 5.2 The encoder, ~300 MB

```bash
python -c "from huggingface_hub import hf_hub_download as d; \
  print(d('ggml-org/embeddinggemma-300m-GGUF','embeddinggemma-300M-Q8_0.gguf',local_dir='models'))"
```

The CLI prints this exact command if the file is missing, rather than failing obscurely.

### 5.3 A bundle

Either pull one from a kernel run, or fit your own on your data:

```bash
nanocore adapt --data yours.csv --text-col text --label-col intent \
    --out models/mine --backend llamacpp --model models/embeddinggemma-300M-Q8_0.gguf
```

### 5.4 Run it, then measure it

```bash
# a decision, through the GGUF
nanocore decide "i need to cancel my flight" --options "cancel,book,balance" \
    --backend llamacpp --model models/embeddinggemma-300M-Q8_0.gguf \
    --bundle models/mine/bundle

# the acceptance run: parity against the PyTorch reference, plus latency percentiles
nanocore parity --inputs demos.txt --options "cancel,book,balance" \
    --bundle models/mine/bundle --backend llamacpp \
    --model models/embeddinggemma-300M-Q8_0.gguf
```

`parity` reports the two things the acceptance criterion names:

- **cosine agreement** between the two backends' embeddings — the *primary* metric,
  because argmax agreement hides a backend that shifted every vector while permuting
  no decisions (and that surfaces later as unexplained calibration drift)
- **latency percentiles** for both backends, and their p50 ratio

**Note:** `parity`'s `--reference` defaults to `st`, which loads PyTorch weights too.
Pass `--reference stub` to exercise the wiring without loading anything.

### What to expect

Prior measurements, so you know whether your number is sane:

| measurement | value | source |
|---|---|---|
| llama.cpp Q8_0 vs PyTorch, same CPU | **4.08× faster** (89.2 ms vs 363.5 ms) | ADR-0007 |
| quality after the swap | no measurable loss (93.41% vs 92.92% zero-shot) | ADR-0007 |
| throughput, 4-core Xeon | llama.cpp **9.9** vs PyTorch **2.4** texts/s (4.12×) | ADR-0007 |
| memory needed for the 300 MB GGUF | **0.51 GB free** (0.34 resident × 1.5 headroom) | `guard.py` |

**On parity, read this before interpreting the number.** The roadmap carried a row
reading *"llama.cpp GGUF argmax agreement 0.815 (QAT Q8_0) — below the ≥0.999 bar"*.
**That figure has no artifact behind it.** The `s1_llamacpp` kernel measured GGUF sizes,
thread scaling, load time, encode latency and Banking77 quality — it never implemented an
agreement measurement at all, and no file under `reports/runs/` contains the number.

So the honest position is: **GGUF parity is unmeasured, and `nanocore parity` is the first
implementation of that measurement.** Run it and you will be the first to know. Treat the
0.815 as a *rumour* about a quantised variant, not a result — and note that it is an
*argmax* agreement, which the roadmap's own reasoning says is the wrong metric anyway
(argmax hides a backend that shifted every embedding while permuting no decisions).

### Acceptance

1. **parity** — cosine agreement ≥ 0.99 mean, action agreement ≥ 0.95 (both are flags,
   `--min-cosine` / `--min-action-agreement`). These thresholds are ours, not a
   standard's: no published protocol defines backend-swap parity for this model shape.
2. **latency** — p50/p95/p99 on *your* hardware, recorded as an artifact
3. **memory gate passed** — the gate runs *before* any weights load, and refuses
   rather than taking the machine down

### Failure modes specific to this path

| message | cause |
|---|---|
| `refusing to load weights: N GB free < M GB needed` | the memory gate. Check what else is resident, or `--ignore-memory` (it warns) |
| `llama-cpp-python is required for the GGUF backend: pip install …` | step 5.1 |
| `model file not found: …` | step 5.2; the error prints the download command |
| `LlamaCppEncoder is text-only; 'image' items need StateEncoder` | the GGUF path is text-only by design; media states go through the PyTorch encoder |

## Failure modes, and what they mean

| message | cause |
|---|---|
| `column(s) [...] not in <file> — it has [...]` | column name typo; the file's real columns are listed |
| `calibration split too small: N < min_cal=M` | not enough data for the requested `--min-cal`; more data, or lower it deliberately |
| `rarest class has N examples — cannot split safely` | a class with fewer than 5 examples; the three-way split cannot cover it |
| `need at least 2 label classes` | single-class file |
| `model file not found: <path>` | GGUF missing; the error prints the exact download command |

## What "working" looks like

1. `adapt` writes a bundle **and** a report, and the report's coverage is near
   `1 - alpha` with a small mean set size.
2. `decide` against that bundle returns a real action, not `unevaluated`.
3. `serve` + `monitor.py` produce alerts you can name (the Phase 5 surface).
