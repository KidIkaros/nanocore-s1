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
