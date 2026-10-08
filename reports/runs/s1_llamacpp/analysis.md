# s1_llamacpp — the llama.cpp path

**Kaggle:** `mauricew/nanocore-s1-llamacpp-server`, version 3, **CPU-only** (no GPU quota).
Hardware: **Intel Xeon @ 2.20 GHz, 4 cores, AVX2 only** (no AVX-512).

**Result: llama.cpp works, it is 4.1× faster than our PyTorch path on the same CPU, and the
quantized model is not degraded.** It still loses to Laya's T4 figure, and the remaining gap is
hardware-class.

## llama.cpp supports EmbeddingGemma 2 — confirmed, and the bindings do not

| | outcome |
|---|---|
| `llama-cpp-python` 0.3.36 (PyPI) | `Failed to load model from file` |
| `llama-cpp-python` **git master** (also reports 0.3.36) | same failure |
| **`ggml-org/llama.cpp` master** (`bd4eeaa`), built from source | **loads and embeds** |

The GGUF's own `convert.log` shows it was produced by llama.cpp's standard
`convert_hf_to_gguf.py`, so the architecture is supported — **the Python bindings simply vendor
an older llama.cpp revision.** For anyone deploying this model today: use `llama-server` from
source, not the pip bindings. That is a real, actionable finding independent of our project.

## Latency, same CPU, same task

| | PyTorch fp32 | llama.cpp Q8_0 | speedup |
|---|---:|---:|---:|
| short state, batch 1 | 363.5 ms | **89.2 ms** (p95 97.4) | **4.08×** |
| long state, batch 1 | 6,595 ms | **1,754.8 ms** | **3.76×** |
| throughput | 2.4 texts/s | **9.9 texts/s** | **4.12×** |

Thread scaling on Q8_0, short state:

| threads | latency | speedup |
|---:|---:|---:|
| 1 | 191.7 ms | 1.00× |
| 2 | 104.1 ms | 1.84× |
| 4 | 89.6 ms | **2.14×** |

Threading is real but **saturating**: 1→2 threads buys 1.84×, 2→4 buys only 1.16× more.

## Quality is preserved — quantization is safe here

Banking77 zero-shot, same task and the same task prefixes:

| | accuracy | log | Brier | ECE |
|---|---:|---:|---:|---:|
| PyTorch fp32 reference | 92.92% | — | — | — |
| **llama.cpp Q8_0** | **93.41%** | see log | see log | see log |
| delta | **+0.49** | | | |

So int8 is a **safe** precision for this model — which matters because the model card forbids
fp16 (EG2's activations exceed fp16's dynamic range). Q8_0 is integer weights with fp32
activations, and it costs nothing measurable.

## Self-correction: weight traffic is NOT the binding constraint

My bandwidth-floor analysis predicted that Q8_0's 1.8× smaller weights (0.31 GB vs 0.56 GB)
would give a proportional speedup. It does not:

| | weights | latency |
|---|---:|---:|
| Q8_0 | 0.31 GB | 89.2 ms |
| BF16 | 0.56 GB | 92.3 ms |

**1.8× less traffic, 3% faster.** Once mmap is in play the embedding table is never fully read,
and the runtime is **compute-bound on an AVX2-only 2.2 GHz Xeon**, not bandwidth-bound. The
22.6 ms floor I computed is real but irrelevant at this clock speed and ISA. **Retracted as the
primary explanation.**

## Against the bars

| bar | value | llama.cpp Q8_0 CPU |
|---|---:|---|
| Jev (cloud p50) | 236–276 ms | **wins by ~2.9×** |
| our own PyTorch on a T4 | 56.9 ms | loses (1.6× slower) |
| Laya (T4) | 33–40 ms | loses (2.2–2.7×) |

Remaining gap to the compute floor: **3.9×**.

## What this establishes

1. **llama.cpp is a viable deployment path for EG2**, 4.1× faster than PyTorch on CPU, with
   quality intact. The `ggml-org` GGUFs are standard conversions.
2. **The bindings lag the engine** — a deployment trap worth recording.
3. **Sequence length is still the top lever.** Long states cost 19.7× short ones (1,754.8 vs
   89.2 ms) for ~8× the tokens. Capping input length is free and worth more than everything
   else combined on long inputs.
4. **Quantization is a quality decision, not a speed one** at batch 1 on this hardware.
5. **The residual gap is hardware-class.** A modern CPU with AVX-512 at 4+ GHz would plausibly
   be 2–3× faster → ~30–45 ms, which would match Laya. The honest statement is that we now lose
   to Laya by a factor explicable by clock speed and instruction set, not by architecture.

## Caveats

- Kaggle CPU only: Xeon 2.2 GHz, 4 cores, AVX2. Not a phone, not a desktop.
- HTTP overhead is included (localhost, ~sub-ms); this is the production serving pattern.
- The run died on a **typo in my own verdict cell** (`q8_short` vs `q8_0_short`) *after* every
  measurement completed, so `results.json` was not written and the Q8-vs-BF16 embedding drift
  was computed but not saved. All numbers above are read from the kernel log. The typo is fixed
  in the notebook.
- One earlier attempt cloned llama.cpp into `/kaggle/working`, which made the source tree a
  kernel artifact and turned `kaggle kernels output` into a 350 MB download. Moved to `/tmp`.

## Artifacts

- `kernel.log` — full log with every measurement
