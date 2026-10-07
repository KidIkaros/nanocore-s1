# Audit run — validating the architecture re-assessment

**Result: every critical claim in the audit reproduced on Kaggle, and running the repository
itself surfaced two findings the static read had missed.**

| | |
|---|---|
| Kernel | `mauricew/nanocore-audit` (version 2) |
| Accelerator | 2× Tesla T4 (15360 MiB each) |
| Source under test | the **uploaded working tree**, `mauricew/nanocore-s1-worktree` — not a GitHub clone |
| Torch | 2.11.0+cu128 |
| Wall time | ~4.5 min, of which 150 s is the full quality gate |

## Why this run exists

`docs/ARCHITECTURE-REASSESSMENT.md` makes claims about *numbers the repository produces* —
that the evaluation suite measures nothing, and that the quality gate signs off on it. Those
are empirical claims, so they are settled by running the repository's own code. Nothing in
the notebook is reimplemented; every measurement calls the repo's own functions.

## What was settled

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| V0 | the transformer blocks are the identity at init | **confirmed** | `max｜TransformerBlock(x) − x｜ = 0.0`; `attn.c_proj` and `mlp.c_proj` both all-zero |
| V5 | the model is 135.3M params, not 157M | **confirmed** | counted `135,266,328`, by hand `135,266,328`; line 60 is off by 22M (16%) |
| V6 | `window_pattern` is inert | **confirmed** | identical logits for `SSSL` / `L` / `SL`, at init **and** de-zeroed |
| V1 | decision accuracy is not a function of `(model, data)` | **confirmed** | same model + same data → `[0.12, 0.16, 0.12, 0.10, 0.08, 0.16, 0.08, 0.16]` |
| V2 | ECE does not respond to the model | **confirmed** | 6 initialisations → ECE ∈ [0.002020, 0.002023]; zeroed model → 0.001953 = 1/vocab |
| V3 | "perplexity" uses unshifted targets | **confirmed** | repo 512.30 vs unshifted 512.57 vs shifted 512.72 |
| V8 | the shipped tokenizer is 1.7×, not nanochat's 4.8× | **confirmed** | vocab 867, 592 merges, 1.671× measured |
| V7 | "15–35 ms on CPU" is unreachable | **confirmed** | prefill 142 ms; 24 generated tokens 3601 ms (**150 ms/token**) |
| V9 | the cloud entry point is invalid JSON | **confirmed** | `JSONDecodeError: Expecting ',' delimiter: line 64 column 5` |
| V4 | the quality gate passes with checks skipped | **confirmed** | exit 0, `all_passed: true`, with Dry Run **and** Test Suite skipped |

## The finding the static read missed: the blocks are a no-op at initialization

`NanoCore._init_weights` calls `nn.init.zeros_` on `attn.c_proj` **and** `mlp.c_proj`. Both are
the *output* projection of their sublayer, so with them zero a block computes `x + 0 + 0` —
the identity function. Measured directly: `TransformerBlock(x) == x` exactly.

This is not cosmetic. The quality gate's architecture check runs a forward and backward pass
on a freshly initialised model and reports *"✓ Gradient flow verified (loss=10.3996)"* —
where `ln(32768) = 10.397`, i.e. exactly uniform-random. Because the blocks are inert, that
check exercises the `embedding → rms_norm → lm_head` path and essentially nothing else of
the architecture it claims to validate.

It also **confounded the first version of this notebook**: several tests were measuring a
no-op model rather than the architecture. Every affected test now runs twice — as shipped,
and with `c_proj` randomised so the blocks actually propagate. V6 and V1 hold in both
conditions, which is what makes them findings about the code rather than about the init.

## Two more findings, from the gate's own output

**The gate reports 48 tests; the repository has 52.** `check_test_suite` runs
`pytest tests/ -k "not DryRun"` and counts `" PASSED"` substrings in the output. The class
`TestTrainingDryRun` contains exactly four tests, and the `-k` filter excludes all four —
the ones that would assert the dry-run checkpoint's step count and loadability.

**The gate's evaluation step reports "ECE 0.0000" and calls it "✓ All metrics valid."**
A perfectly calibrated score, from an untrained model, computed by the function V2 shows
cannot respond to the model at all.

## What this run does *not* settle

- **V3 is structural, not quantitative.** The repo computes unshifted targets — that is
  visible in the code and reproduced here. But on random data with an untrained model the
  three numbers all sit near the vocab size (512), so the *magnitude* of the error is not
  demonstrated. A trained model on real text would show it properly.
- **The full gate genuinely passes.** 52 tests' worth of pipeline runs green in 150 s, and
  the dry run completes in 15.5 s. The gate is not *only* failing open — it also passes for
  real. The defect is that it cannot tell the two apart.
- **No capability was measured in either direction.** There is still no trained model. This
  run validates the *harness*, not the model.
- **Latency is one CPU.** Kaggle's 4-vCPU container. The claim is about CPU, so this is the
  right environment, but a different CPU will give a different number — the ~100× gap to
  15–35 ms is not a close call at any CPU speed.

## Artifacts

- `audit_results.json` — every measurement, machine-readable
- `gate_a.json` — the gate with `--skip-tests --skip-dry-run` (the fail-open case)
- `gate_b.json` — the full gate (the genuine pass)
- `nanocore-audit.log` — the complete kernel log

Reproduce with:

```bash
python scripts/kaggle_run.py push notebooks/nanocore_audit
```
