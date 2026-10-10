# Encoder Efficiency — investigation and decision record

Date: 2026-10-07. Triggered by the latency result: a *larger* competitor (Laya, 421M
ModernBERT) does typed decisions in 33–40 ms on a T4 while our 270M EG2 takes 56.9 ms.

**Status: this investigation is now decided.** Its conclusions are recorded as
[ADR-0007](adr/0007-llamacpp-runtime.md) (serve through llama.cpp GGUFs) and
[ADR-0012](adr/0012-input-length-cap.md) (hard input-length cap). The document is kept as the
evidence trail — including the two claims it had to retract.

**Headline: the encoder is not the problem — our deployment path was.** Measured outcome:
llama.cpp Q8_0 is **4.08× faster** than the PyTorch path on the same CPU (89.2 ms vs 363.5 ms)
with **no measurable quality loss** (Banking77 zero-shot 93.41% vs 92.92%). One claim I made in
`RESEARCH-NOTES.md` (R11/R12) about EG2's architecture was **wrong**; §0 corrects it, and a
second claim (the bandwidth-floor analysis) was retracted in §4.

---

## 0. Correction: EG2 already has the efficiency features I said it lacked

I wrote that EG2 "descends from a decoder stack: full attention at every layer, quadratic".
The model's own `config.json` says otherwise:

```
text_config:
  hidden_size: 512          intermediate_size: 2048        head_dim: 256
  layer_types: [sliding_attention ×5, full_attention, sliding_attention ×5, ...]
```

| property | EG2 text encoder |
|---|---|
| attention pattern | **alternating local/global, 5:1** — the same idea as ModernBERT |
| local window | 1024 tokens |
| attention | GQA / MQA — 4 query heads, 2 local / 1 global KV heads |
| hidden size | **512** (not 768 — there is a 512→768 projection at the end) |
| FFN | gated FFN with GELU, intermediate 2048 |
| layers | 24 |

So EG2 is **not** a naive full-attention decoder. It has sliding-window attention, grouped-query
attention and a small hidden size — the modern efficiency playbook. My explanation for the
latency gap was incorrect and is retracted.

**But there is a real reason it does not help us.** The sliding window is **1024 tokens**, and
our states are far below it. Within a single window, every token still attends to every
preceding token up to the window bound, so cost is **quadratic in practice** for short and
medium inputs. That is exactly what our data shows: a ~500-token state cost 18× a 20-token one
(6,595 ms vs 363 ms), which is super-linear for ~8× the tokens.

**The sliding window only pays off beyond 1024 tokens — and our inputs never get there.**

---

## 1. Where the time actually goes

Batch-1 inference of an embedding model is dominated by **weight traffic**, not FLOPs. One
forward pass must read every weight:

| precision | weight traffic | CPU floor @20 GB/s | T4 floor @320 GB/s |
|---|---:|---:|---:|
| fp32 | 1.08 GB | 54 ms | 3.4 ms |
| bf16 | 0.54 GB | 27 ms | 1.7 ms |
| **int8** | **0.27 GB** | **13.6 ms** | **0.8 ms** |

Measured against those floors:

| | measured | floor | ratio |
|---|---:|---:|---:|
| CPU batch 1 (fp32) | 416.1 ms | 54.2 ms | **7.7× above** |
| T4 batch 1 (bf16) | 55.2 ms | 1.7 ms | **32.6× above** |

**We are 8–33× slower than our own memory-bandwidth floor.** That is the whole story: the
runtime is leaving 90%+ on the table, and none of it is architectural.

Two supporting observations:

- **CPU: ~27–30 GFLOPS at every batch size**, batch 1 through 32 — identical per-text cost.
  No parallelism benefit whatsoever. A single modern x86 core does 50–100 GFLOPS fp32, so the
  CPU path is running at roughly single-core throughput and is likely thread-starved.
- **GPU: 197 → 3,669 GFLOPS** from batch 1 to batch 32. Batching is what unlocks the T4, which
  means batch-1 GPU latency is dominated by launch and interpreter overhead — the classic
  target for export and CUDA graphs.

---

## 2. What the vendor publishes for on-device

From Google's own launch material:

> "With quantization, on a Google Pixel 11 Pro, EmbeddingGemma 2 requires as little as
> **~191 MB active RAM for text-only weights** and ~567 MB for the full multimodal model."

**Google ships this model as an on-device model.** Our 416 ms on a shared cloud vCPU is not
evidence that the model is unfit for a phone; it is evidence that we ran an unoptimized
PyTorch path with no quantization and no export.

Quality, for reference (model card): **MTEB (eng, v2) = 68.46**, MTEB multilingual v2 61.36,
MTEB code 78.68, MIEB lite 64.64, MSEB retrieval 69.54. Careful: MTEB-eng-v2 is not the same
aggregation as the "Average (56 tasks)" figures quoted for text-only encoders, so cross-model
comparison needs the same harness.

---

## 3. llama.cpp instead of ONNX — official GGUFs already exist

Asked whether llama.cpp is usable here rather than ONNX. It is, and it is the better target:
llama.cpp is what production actually runs, so it measures the path a deployment would take
rather than a framework-specific optimization.

**Official GGUF builds of EmbeddingGemma 2 already exist**, published the same day as the model:

| repo | files | note |
|---|---|---|
| **`ggml-org/embeddinggemma-2-GGUF`** | BF16 **558 MB**, Q8_0 **310 MB**, plus `mmproj-*` | **the llama.cpp project's own org** |
| `unsloth/embeddinggemma-2-GGUF` | BF16, F16, Q8_0, **UD-Q4_K_XL / Q5_K_XL / Q6_K_XL**, mmproj | dynamic quants, 11,470 downloads |

Weight traffic, which is the binding constraint at batch 1:

| precision | weights | CPU floor @20 GB/s | vs our measured fp32 path |
|---|---:|---:|---:|
| our PyTorch fp32 | 1.08 GB | 54.2 ms | — |
| GGUF **BF16** | 0.56 GB | 27.9 ms | 1.9× less traffic |
| GGUF **Q8_0** | 0.31 GB | 15.5 ms | **3.5× less traffic** |
| unsloth **Q4_K_XL** | ~0.18 GB | ~9 ms | ~6× less traffic |

So llama.cpp attacks **both** bottlenecks at once: 3.5× less weight traffic (Q8_0) *and* a
SIMD-optimized, properly threaded C++ runtime with no Python or interpreter overhead — against
our measured 7.7× CPU gap above the floor.

**Precision caution.** The model card forbids fp16 because EG2's activation range exceeds fp16's
dynamic range. That constraint maps onto GGUF as: use **BF16** (same exponent range as fp32) or
an **integer quantization (Q8_0 / Q4_K_XL)**, and **avoid the F16 build**. Note the constraint is
about activations; llama.cpp's integer quants compute activations in fp32, which is the safe
configuration.

**Multimodality is covered too.** The `mmproj-embeddinggemma-2-*` files (554–982 MB) carry the
vision and audio encoders, so the full multimodal path is available under llama.cpp — this is not
a text-only escape hatch.

**Kaggle feasibility.** Yes: `pip install llama-cpp-python` builds from source with cmake (a few
minutes on a Kaggle CPU image), or llama.cpp itself can be cloned and built directly. A CUDA
build is possible on a T4 but slower to compile; the CPU path is where our 7.7× gap lives, so
that is where to start. The GGUF download is 310 MB for Q8_0.

**Revised experiment plan** (steps 1–2 unchanged, step 3 replaced, step 4 subsumed):

1. Parallelism baseline on the PyTorch path (CPU) — is the gap thread starvation?
2. Sequence-length cap — latency/quality curve at 128/256/512 tokens.
3. **llama.cpp with Q8_0 (and BF16 as a control)** — measure load, batch-1 latency, throughput,
   and end-to-end decision. Then measure **quality drift**: cosine between llama.cpp and PyTorch
   fp32 embeddings on a fixed sample, and re-run Banking77 zero-shot accuracy on the GGUF
   embeddings to confirm the quantized model is still the same model.
4. ~~AOT export~~ — subsumed: llama.cpp *is* the ahead-of-time-compiled runtime.
5. Two-encoder fork — now cheaper to test, since llama.cpp has GGUFs for the text-only
   alternatives (`nomic-embed-text-v1.5`, `bge-m3`, …) as well.

---

## 4. The levers, ranked by expected gain — with measured outcomes

Measured on Kaggle CPU (Xeon 2.20 GHz, 4 cores, **AVX2 only**), llama.cpp Q8_0 vs our PyTorch
fp32 path, same task:

| # | Lever | Expected | **Measured** | Verdict |
|---|---|---|---|---|
| 1 | **Cap sequence length** | up to 18× | not yet applied | **still the top lever** — long states cost 19.7× short ones (1,754.8 vs 89.2 ms) |
| 2 | **Threading** | 2–4× | **2.14×** (1→2 threads 1.84×, 2→4 only 1.16×) | real but saturating |
| 3 | **llama.cpp + int8 GGUF** | 2–4× | **4.08× short, 3.76× long, 4.12× throughput** | **the biggest single win so far** |
| 4 | AOT export | 2–5× GPU | subsumed by 3 | llama.cpp *is* the AOT runtime |
| 5 | Batching | 137× throughput GPU | not re-measured | unchanged |
| 6 | Flash attention / padding-free | 1.5–3× padded | not measured | untested |
| 7 | Switch encoder (text-only) | ~2× | not measured | still a fork |
| 8 | Distill | 2–4× | not attempted | last resort |

### Two corrections the measurement forced

**Weight traffic is not the binding constraint.** I predicted Q8_0's 1.8× smaller weights
(0.31 GB vs 0.56 GB) would give a proportional speedup. Measured: Q8_0 **89.2 ms** vs BF16
**92.3 ms** — 3% apart. With mmap the embedding table is never fully read, and the runtime is
**compute-bound** on an AVX2-only 2.2 GHz Xeon. The 22.6 ms bandwidth floor is real but
irrelevant at this clock speed and instruction set. **The bandwidth-floor analysis is retracted
as the primary explanation** — though the deeper point stands: we were 18× above a floor, and
4.1× of that gap was runtime, not silicon.

**Quantization is a quality decision, not a speed one** at batch 1. It costs nothing measurable
in quality (Banking77 zero-shot **93.41%** under Q8_0 against 92.92% for PyTorch fp32), and it
is *required* safety-wise anyway, since the model card forbids fp16.

### What the residual gap is

After 4.1×, short-state latency is **89.2 ms** against a compute floor of ~22.6 ms — 3.9× left.
Against the published bars: it **beats Jev's cloud p50 (236–276 ms) by ~2.9×**, and still
**loses to Laya on a T4 (33–40 ms) and to our own PyTorch-on-T4 (56.9 ms)**.

A modern CPU with AVX-512 at 4+ GHz would plausibly be 2–3× faster → **~30–45 ms**, which would
match Laya. The honest summary is that the remaining gap is explicable by **clock speed and
instruction set**, not by architecture — which is a very different statement from the one this
investigation started with.

### A deployment trap worth recording

`llama-cpp-python` — PyPI 0.3.36 *and* its git master, which still reports 0.3.36 — **cannot
load the EmbeddingGemma 2 GGUFs** (`Failed to load model from file`). It vendors a llama.cpp
revision that predates the architecture. `ggml-org/llama.cpp` master (`bd4eeaa`) loads them
fine, and the GGUF's own `convert.log` confirms it was produced by llama.cpp's standard
`convert_hf_to_gguf.py`. **Use `llama-server` from source, not the pip bindings.**

---

## 4. What not to do

- **Do not use fp16.** EG2's activation range exceeds fp16's dynamic range; it returns NaN.
  bf16 on accelerators, fp32 on CPU. int8/4-bit weight quantization is a different thing and is
  supported.
- **Do not use Matryoshka for latency.** Scoring 77 options costs 0.94 ms against a 420 ms
  decision — 0.2%. MRL is a storage feature. Already established, now quantified.
- **Do not optimize the head.** It is 0.2% (CPU) to 3% (GPU) of the decision.
- **Do not expect the sliding window to help.** It only engages beyond 1024 tokens; our states
  are far below that, so attention is effectively quadratic for us.
- **Do not switch encoders casually.** Text-only alternatives are faster, but multimodality is
  the differentiator that survives the competitive analysis (R12).

---

## 5. Sub-1-bit candidates — the LittleBit recipe (Samsung Labs, NeurIPS'25 / ICML'26)

`github.com/SamsungLabs/LittleBit` compresses weights to 1.0–0.1 **bits per weight**:
factorize each matrix into low-rank latent factors → binarize the factors → restore
magnitude with learned scales (optional residual low-rank path). LittleBit-2's fix is
geometric: Joint-ITQ aligns the SVD factors to the binary hypercube *before* QAT —
**the geometry, not the rounding, is where compression succeeds or fails.**

What to take — and what to leave:

- **Binarized prototypes for the scorer (Phase-8 candidate).** `CosineScorer`
  prototypes are the perfect LittleBit substrate: binarize to sign vectors →
  scoring becomes Hamming distance (XOR+popcount — no float matmul on-device),
  learned per-prototype scales restore magnitude. No QAT needed — prototypes can
  be binarized post-hoc *with* the geometry-alignment trick (align to the
  hypercube first, then round). Probe first: paired-CI the binarized scorer vs
  the float scorer on held-out embeddings; gate on non-inferiority, same
  discipline as every artifact that earns the slot.
- **Field compression path.** If the competence field becomes parametric
  (GMM/prototype centroids per glial-regulator §queue), binarized centroids ship
  as kilobytes and query as Hamming lookups.
- **Compression belongs to the artifact layer, never the architecture** —
  LittleBit's "no inference-time change" rule matches the bundle contract:
  compress packaging, keep contracts.

The guard, explicit: **do not sub-1-bit the encoder without a geometry probe.**
EG2's embedding geometry is the substrate — gate calibration, the field's
spatial structure, `state_hash` linkage all key off it. LittleBit's supported
list is decoder LLMs (OPT/Llama); EG2 is an embedding encoder outside that
envelope, and embedding degradation poisons downstream in ways accuracy legs
don't see. Q8_0 GGUF remains the proven floor; sub-1-bit EG2 is a research bet
that needs its own evidence before it's a deployment option.

## 6. Experiment plan

Each step is cheap and independently verifiable, in dependency order.

1. **Establish the parallelism baseline** (CPU, minutes). Record `torch.get_num_threads()`, then
   sweep `torch.set_num_threads(1..N)`. If throughput scales, lever 2 is real and free. If it
   does not, the CPU path is bandwidth-bound and lever 3 dominates.
2. **Sequence-length cap** (CPU+GPU, minutes). Re-measure the long state at caps of 128/256/512
   tokens and record the accuracy cost on a labelled set. Publish the latency/quality curve.
3. **int8 ONNX** (CPU, ~30 min). `export_dynamic_quantized_onnx_model` on EG2 text-only, then
   re-run the same harness. Report latency and the embedding-space drift (cosine between fp32
   and int8 vectors on a fixed sample) — quality first, speed second.
4. **AOT export on GPU** (T4, ~30 min). `torch.compile` first (cheapest), then ExecuTorch if
   promising. Report batch-1 latency against the 1.7 ms floor.
5. **Two-encoder fork** (GPU, ~1 h). Measure `gte-modernbert-base` on the same Banking77 and
   BFCL tasks as EG2, for latency *and* accuracy. That decides whether the multimodal encoder is
   worth its cost on the text-only path.

Steps 1–3 are CPU-only and cost no GPU quota.

---

## 7. Sources

- EG2 `config.json` (layer types, hidden size, GQA) and model card (MTEB figures, precision
  constraint, quantization RAM) — `google/embeddinggemma-2`.
- Google blog, *EmbeddingGemma 2 is a best-in-class open model for natively multimodal
  embeddings* — the ~191 MB on-device figure.
- Warner et al., *Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder* (ModernBERT),
  ACL 2025 — alternating attention, unpadding and sequence packing, hardware-aware design.
- Hugging Face, *Finally, a Replacement for BERT* — ModernBERT's three efficiency components.
- Sentence Transformers efficiency docs — ONNX backends, `export_dynamic_quantized_onnx_model`,
  int8 configs for avx2/avx512/avx512_vnni.
- PyTorch, *ExecuTorch* (MLSys 2026) and the ExecuTorch Core ML backend + quantization docs.
- `Alibaba-NLP/gte-modernbert-base`, `nomic-ai/modernbert-embed-base` model cards.
