# ADR-0007: Serve the encoder through llama.cpp GGUFs, not PyTorch

**Date**: 2026-10-07
**Status**: accepted
**Deciders**: project owner, agent

## Context

The latency run measured our PyTorch/sentence-transformers path at **416 ms per decision on
CPU** — **7.7× above its own memory-bandwidth floor** and, more tellingly, **18× above** the
floor once mmap is accounted for (half the model's 271M parameters are a 262,144-entry
vocabulary table that a short input never fully reads).

The design's central claim is "extremely lightweight". It had never been measured, and when it
was, a *larger* competitor was faster on the same hardware: Laya (421M ModernBERT) does typed
decisions in 33–40 ms on a T4 while our 270M EG2 took 56.9 ms. That prompted an efficiency
investigation whose first conclusion — that EG2 lacked modern attention features — was **wrong**
(see `docs/ENCODER-EFFICIENCY.md` §0; EG2 already uses 5:1 alternating sliding/full attention
with GQA and a 512 hidden size).

What remained was a runtime gap, and llama.cpp is the runtime production actually uses.

## Decision

**Serve the frozen encoder through `llama.cpp` with the official `ggml-org` GGUFs**, via
`llama-server`, as the primary inference path. Keep the sentence-transformers backend as a
reference implementation for parity checks.

Measured on the same 4-core Xeon (2.20 GHz, AVX2 only), same task, Q8_0 vs PyTorch fp32:

| | PyTorch fp32 | llama.cpp Q8_0 | speedup |
|---|---:|---:|---:|
| short state, batch 1 | 363.5 ms | **89.2 ms** (p95 97.4) | **4.08×** |
| long state, batch 1 | 6,595 ms | **1,754.8 ms** | **3.76×** |
| throughput | 2.4 texts/s | **9.9 texts/s** | **4.12×** |

**Quality is preserved:** Banking77 zero-shot **93.41%** under Q8_0 against **92.92%** for
PyTorch fp32 (+0.49, within noise). Thread scaling is real but saturating: 1→2 threads gives
1.84×, 2→4 only 1.16× more.

## Alternatives Considered

### Alternative 1: ONNX Runtime with dynamic int8 quantization
- **Pros**: framework-native for sentence-transformers (`export_dynamic_quantized_onnx_model`);
  mature tooling; CPU-optimized execution providers.
- **Cons**: still a Python-hosted runtime; the ONNX graph would have to be exported from a model
  whose architecture is one day old; it optimizes a path nobody deploys.
- **Why not**: llama.cpp is what production runs, so it measures the path a deployment would
  actually take. Also chosen by the project owner.

### Alternative 2: ExecuTorch / Core ML export
- **Pros**: the genuine on-device answer — ANE/GPU dispatch, ahead-of-time graph compilation,
  8-bit and 4-bit quantization, scales down to microcontrollers.
- **Cons**: requires a `.pte` export per target backend; Core ML quantization needs calibration
  and iOS 17+; a large amount of tooling for a case we can already measure.
- **Why not**: deferred, not rejected. llama.cpp *is* an ahead-of-time-compiled runtime and it
  already produced the 4.08×. Revisit if the target becomes a specific phone.

### Alternative 3: Stay on PyTorch and optimize threads only
- **Pros**: no new dependency; no format risk.
- **Cons**: measured **27–30 GFLOPS at every batch size**, i.e. no parallelism benefit at all —
  the CPU path is effectively thread-starved and would need work inside PyTorch to fix.
- **Why not**: it addresses one of the two causes. llama.cpp addresses both, and the mmap
  behaviour (never reading the 134M-parameter embedding table) is not something we get for free
  in PyTorch.

### Alternative 4: fp16 weights to halve traffic
- **Pros**: simplest possible speedup.
- **Cons**: **forbidden by the model card.** EG2's activation range exceeds fp16's dynamic
  range and the model returns NaN.
- **Why not**: correctness. Q8_0 (integer weights, fp32 activations) is the safe precision, and
  it costs nothing measurable in quality.

## Consequences

### Positive
- 4.08× faster on CPU with no measurable quality loss, and a 4.12× throughput gain.
- mmap means the 134M-parameter embedding table is never fully read — a structural saving that
  a dense PyTorch forward cannot match.
- `llama-server` is the production serving pattern: one resident process, HTTP requests, no
  Python in the hot path.
- On CPU we now **beat Jev's published cloud p50 of 236–276 ms by ~2.9×**.

### Negative
- A new external dependency with its own release cadence and GGUF format churn.
- Serving via HTTP adds a process boundary; in-process bindings are not currently viable (below).
- We still **lose to Laya's 33–40 ms on a T4** and to our own PyTorch-on-T4 (56.9 ms).

### Risks
- **The Python bindings cannot load these GGUFs.** `llama-cpp-python` — PyPI 0.3.36 *and* its
  git master, which still reports 0.3.36 — fails with `Failed to load model from file`. It
  vendors a llama.cpp revision that predates `embedding_gemma2`. **Use `llama-server` built from
  `ggml-org/llama.cpp` master (verified at commit `bd4eeaa`).** This is a deployment trap worth
  recording; the GGUF's own `convert.log` confirms it was produced by llama.cpp's standard
  `convert_hf_to_gguf.py`, so the architecture *is* supported.
- **The residual 3.9× gap is clock speed and instruction set, not architecture.** The
  measurement host is a 2.20 GHz Xeon with **AVX2 only**. A modern CPU with AVX-512 at 4+ GHz
  would plausibly be 2–3× faster → ~30–45 ms, matching Laya. This should be stated as the
  honest explanation rather than chased as a design defect.
- **The bandwidth-floor analysis is retracted.** I predicted Q8_0's 1.8× smaller weights would
  give a proportional speedup; measured, Q8_0 (0.31 GB) is 89.2 ms against BF16 (0.56 GB) at
  92.3 ms — 3% apart. With mmap the runtime is **compute-bound**, not bandwidth-bound.
- Pin the GGUF repo and the llama.cpp commit when this moves into code (Stage 1).

## Sources

- `reports/runs/s1_latency/` — PyTorch CPU and T4 measurements
- `reports/runs/s1_llamacpp/` — llama.cpp Q8_0/BF16 measurements, thread scaling, quality parity
- `docs/ENCODER-EFFICIENCY.md` — the investigation, including two retractions
- `ggml-org/embeddinggemma-2-GGUF` and its `convert.log`
