# NanoCore-S1 — Architecture Design

> **Status:** Approved for Implementation  
> **Goal:** A ~140M-parameter model that makes fast, type-safe, calibrated decisions (Jev-style System One) while remaining trainable at $0 cost.

## Core Philosophy

Derived from Karpathy's "Small LLMs Setup" (Oct 2026): the interesting frontier in language models is downward. Current frontier models are far larger than the capability they deliver — the gap is a **data quality problem**, not architecture.

A cognitive core should:
- Sacrifice encyclopedic knowledge for capability
- Look up facts externally (facts live outside the model)
- Be always-on, tool-using, and make fast structured decisions
- Be software-native — slot into code as a function call

Jev's System One approach maps perfectly: **unstructured state in, typed probabilistic decisions out**. NanoCore-S1 is the open, free, tiny model version.

## Model Configuration

Inherits from nanochat's `GPTConfig` with depth=12 as the single complexity dial:

| Parameter | Value | Derivation |
|---|---|---|
| `sequence_len` | 2048 | nanochat default |
| `vocab_size` | 32768 | nanochat default (padded to 64-mult) |
| `n_layer` | **12** | Depth dial — ~$5 training cost, 157M params |
| `n_head` | **6** | `model_dim / head_dim` (128) |
| `n_kv_head` | **6** | Same as n_head (no GQA — unnecessary at scale) |
| `n_embd` | **768** | `depth * 64` = 768, rounded to head_dim multiple |
| `window_pattern` | **"SSSL"** | Sliding window attention, final layer full context |

### Parameter Count Breakdown

| Component | Params |
|---|---|
| Token embedding (wte) | 32768 × 768 = 25.2M |
| LM head (untied) | 32768 × 768 = 25.2M |
| 12 transformer layers (attn + MLP) | ~87M |
| Per-layer scalars (resid_lambdas, x0_lambdas) | 24 |
| **Total** | **~137M params** |

### Decision Primitives (Jev Integration)

After training stages 1-2 on general text and decision format, we extend the vocabulary with 512 decision tokens and train stages 3-4 on structured decision data:

```python
@dataclass
class Choice:
    value: str          # Selected option
    confidence: float   # 0.0-1.0, calibrated
    probabilities: dict # Per-option probabilities

@dataclass
class Noul:
    value: float        # P(true), 0.0-1.0
    confidence: float   # Calibrated confidence in the judgment

@dataclass
class Score:
    value: str          # Selected level (e.g. "critical")
    confidence: float
    probabilities: dict # Per-level probabilities
```

### Decision Prompt Format

```
[STATE] {arbitrary text or JSON context} [/STATE]
[CHOICE options="billing technical_support security_fraud human_review"] [/CHOICE]
[NOUL statement="This requires immediate attention"] [/NOUL]
[SCORE options="low medium high critical"] [/SCORE]
[ANSWER] route=security_fraud, requires_alert=0.92, urgency=critical [/ANSWER]
```

Each option is scored in isolation (no competitors in context) — guaranteeing 100% order invariance.

## Architecture Components (from nanochat gpt.py)

### What We Keep (Proven, Efficient)

1. **Rotary Positional Embeddings (RoPE)**: No positional embeddings, applied to Q and K — efficient, no learnable position parameters
2. **QK Layer Normalization**: `F.rms_norm` on Q and K before attention — stabilizes training
3. **Untied Embeddings**: Separate token embedding and lm_head — better gradient flow
4. **ReLU² Activation**: `F.relu(x).square()` — simpler than GELU
5. **Per-Layer Residual Scalars**: Init at depth-derived values via μP
6. **Sliding Window Attention**: "SSSL" pattern — 3 short-window layers, final layer full context
7. **Flash Attention 3**: With SDPA fallback for CPU/non-CUDA
8. **Smear Gate**: Previous-token embedding mixing — cheap bigram info

### What We Remove (for simplicity at d12)

| nanochat Feature | Our Choice | Reason |
|---|---|---|
| Value embeddings | Remove | Saves ~8M params, adds complexity not needed for decision tasks |
| Backout lambda | Remove | Minor efficiency gain, complexity not justified at 140M scale |
| FP8 training | Not supported | Requires H100, irrelevant at $0 cost target |
| MuonAdamW | Use AdamW only | Muon requires distributed; single-node/CPU uses AdamW |
| Full FP8 pipeline | Skip | Only relevant for production-scale training |

### Single-Dial Scaling

All hyperparameters derived from `depth` via μP:

```python
# From nanochat base_train.py:
base_dim = depth * 64  # aspect_ratio = 64
model_dim = ceil(base_dim / 128) * 128  # head_dim = 128
num_heads = model_dim // 128

target_tokens = 12 * scaling_params  # 8:1 ratio (compute-optimal, not Chinchilla 20:1)
batch_size_ratio = target_tokens / D_REF
predicted_batch_size = 524288 * batch_size_ratio^0.383  # Power Lines paper
total_batch_size = 2^round(log2(predicted_batch_size))  # nearest power of 2
```

## Training Pipeline (Four Stages)

### Stage 1: Base Pretraining (~$0, 15 min)
- Data: FineWeb-EDU subset (first 200M tokens via HF datasets)
- Objective: Next-token prediction
- Compute: Colab free tier (T4) or Kaggle Kernel
- Output: `nanocore-s1-base.pt`

### Stage 2: Midtraining (~$0, 5 min)
- Data: SmolTalk subset + Decision traces (100K + 200K synthetic)
- Objective: Format adaptation — teaches `[STATE]...[CHOICE]...` format
- Output: `nanocore-s1-mid.pt`

### Stage 3: Supervised Finetuning (~$0, 2 min)
- Data: 10K curated decision examples
- Objective: Quality of structured decisions
- Output: `nanocore-s1-sft.pt`

### Stage 4: RLCD (~$0, 2 min)
- Data: GSM8K + Custom Decision Benchmark (verifiable answers)
- Reward: Binary correctness (decision was right or wrong)
- Objective: Calibrated probabilities (RLCD-style)
- Output: `nanocore-s1-rlcd.pt`

## Memory & Speed

| Component | fp16 | int4 |
|---|---|---|
| Model weights (~140M params) | ~280 MB | ~70 MB |
| KV cache (2048 tokens) | ~64 MB | ~64 MB |
| Activations + overhead | ~200 MB | ~200 MB |
| **Total (CPU inference)** | **~544 MB** | **~334 MB** |
| **Total (GPU inference)** | **~280 MB** | **~70 MB** |

**Latency**: 15-35ms on CPU (single batch), 2-5ms on GPU.
**Training cost**: $0-$5 on free GPU tiers.

## Jev Integration: System One Decision Layer

The lm_head is trained to output decision tokens that enforce typed, order-invariant decisions:

1. **Each option scored independently** — no competitor tokens in context during evaluation
2. **Softcap on logits** — `softcap * torch.tanh(logits / softcap)` (softcap=15) for stable probabilities
3. **No free-form text generation** — outputs are strictly typed (Choice, Score, Noul)
4. **Confidence calibration** — RLCD trains on verifiable tasks where outcome = binary correct/wrong

## Limitations (Explicitly Acknowleded)

- No encyclopedic knowledge — facts must be in state/context
- No general reasoning — excels at bounded decisions
- Not a competitive general assistant — designed as a cognitive core
- No production serving infrastructure included
- Quantization damage must be validated on your specific task

## References

- Karpathy, A. "Small LLMs Setup" (Oct 2026). In *Your Library*.
- Karpathy, A. nanochat repository (`github.com/karpathy/nanochat`). GPT architecture: `gpt.py`.
- Karpathy, A. "Introducing System One Models & Jev" (Sep 2026, TypeSafe AI Blog).
- Kaplan, J. et al. "Scaling Laws for Neural Language Models" (2020).
- Yang, L. "Power Lines: Batch Size Scaling Laws" (2025). Formula B_opt ∝ D^0.383.
- Sanh, V. et al. "DistilBERT" (2019). Reference for distillation 40% smaller, 60% faster, ~97% GLUE.
