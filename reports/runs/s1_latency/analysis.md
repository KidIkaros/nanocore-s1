# s1_latency — is "extremely lightweight" a fact or an adjective?

**Kaggle:** `mauricew/nanocore-s1-latency` (CPU, v1) and `mauricew/nanocore-s1-latency-gpu`
(T4, v1). Same harness, one variable: the device.

The project has claimed "extremely lightweight multimodality" since its first commit and had
never produced a latency number. This is that number, and the answer is **both**: true on an
accelerator, false on CPU.

## Results

| measurement | CPU (fp32) | T4 (bf16) | ratio |
|---|---:|---:|---:|
| cold load | 52.9 s | 49.9 s | — |
| encode, short state, batch 1 | **363.5 ms** | **55.2 ms** | 6.6× |
| encode, long state, batch 1 | **6,595 ms** | **64.6 ms** | **102×** |
| throughput, batch 1 (ms/text) | 416.1 | 53.6 | 7.8× |
| throughput, batch 8 | 401.2 | 7.8 | 51.6× |
| throughput, batch 32 | 406.2 | **2.95** | 137.5× |
| **end-to-end decision** | **420.4 ms** (p95 441.5) | **56.9 ms** (p95 62.3) | 7.4× |
| encoder share of the decision | 86.5% | **97.0%** | — |

## Against the published bars

| bar | value | CPU | T4 |
|---|---|---|---|
| Jev (cloud, typed decisions) | 236–276 ms p50 | ✗ 420 ms | **✓ 56.9 ms** |
| within 100 ms | — | ✗ | **✓** |
| Laya (same T4, typed decisions) | 33–40 ms | — | ✗ 56.9 ms |

**The claim holds on a T4 and fails on CPU.** On an accelerator a decision costs 56.9 ms —
comfortably inside Jev's cloud p50 of 236–276 ms, and within the 100 ms target. On CPU it
costs 420 ms and fails every bar.

## Four findings worth keeping

**1. Batching transforms GPU throughput and does nothing on CPU.** Per-text cost falls
53.6 → 7.8 → 2.95 ms on the T4 (137× at batch 32) and stays flat at ~400 ms on CPU. CPU
inference is compute-bound and effectively unparallelized, so *no amount of batching rescues
it*.

**2. Long states are catastrophic on CPU and cheap on GPU** — 6,595 ms versus 64.6 ms, a
102× gap. Attention over a ~500-token state is the CPU failure mode. If this ever runs
without an accelerator, state length must be tightly bounded, and the current design does not
bound it.

**3. The encoder is 86.5–97% of the decision, so the head is latency-irrelevant.** Scoring 77
options through the interaction head costs 0.94 ms; through a 256-d cosine, 0.066 ms. Against
a 420 ms decision that is 0.2%. **Matryoshka truncation buys nothing for decision latency** —
it reduces storage and scoring cost, both of which are already noise. S7 should be recorded
as a storage argument only, not a latency one.

**4. A *larger* competitor is faster on the same hardware.** Laya is a 421M ModernBERT
encoder and does typed decisions in 33–40 ms on a T4; our 270M EG2 takes 56.9 ms. The
smaller model is ~1.5× slower.

**Correction (see `docs/ENCODER-EFFICIENCY.md`).** This was first attributed to EG2 lacking
modern efficiency features. That was wrong: EG2's own config shows **alternating
local/global attention at 5:1, GQA/MQA, and a 512 hidden size** — the same playbook as
ModernBERT. The real explanation is two-fold. First, EG2's sliding window is **1024 tokens**
and our states are far below it, so attention is *effectively quadratic* for us — which is
exactly what the 18× long-state penalty shows. Second, we run the **unoptimized reference
path**: no quantization, no export, no thread tuning, no sequence cap. We are **7.7× (CPU) to
32.6× (GPU) above our own memory-bandwidth floor**, and Google publishes the same model at
~191 MB quantized on a phone. The gap is our deployment path, not the encoder.

## Caveats

- The absolute numbers are Kaggle hardware. The **breakdown** (encoder vs scoring) is
  portable; the encoder dominates at every batch size on both devices.
- `platform.processor()` reported only `x86_64`, so the CPU model is unidentified. The CPU
  number should be treated as "a shared cloud vCPU", not a phone or a desktop.
- The comparison against Laya's 33–40 ms is approximate: ours is measured end-to-end
  including tokenization and Python overhead at batch 1, and the published figure may be
  measured differently. The *direction* — a larger encoder doing the same job faster — is the
  robust part.
- MiniCPM's <2 s TTFT / >17 tok/s figures are for a 4.1B **generative** model on a phone.
  That is a different regime and is not a like-for-like comparison; it is included only to
  show what the market publishes.

## Consequences

1. **"Extremely lightweight" is now a measured claim with a condition attached**: it holds
   with an accelerator, not on CPU. The design is accelerator-requiring, which weakens the
   on-device story unless the device has an NPU.
2. **Encoder efficiency becomes a first-class design axis** — currently unexamined, and
   apparently costing ~1.5× against a larger competitor.
3. **State length is a hard constraint on CPU**, not a soft one.
4. **Matryoshka is a storage feature, not a latency feature** — S7 restated.
5. The head's latency contribution is nil, so any future argument for the head must be about
   *quality*, never speed.

## Artifacts

- `results_cpu.json`, `results_gpu.json` — full measurements
- `kernel_cpu.log`, `kernel_gpu.log` — kernel logs
