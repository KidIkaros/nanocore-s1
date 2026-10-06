# NanoCore-S1 — Implementation Plan

> **For:** subagent-driven-development workflow (TDD enforced)  
> **Goal:** Train NanoCore-S1 (140M param System One model) at $0 cost  
> **Prerequisites:** Python 3.11+, PyTorch, HuggingFace `datasets` library

---

## A. Setup Track

### Task 1: Create project structure
```bash
mkdir -p src tests data scripts docs
touch src/__init__.py tests/__init__.py
python -m venv .venv && source .venv/bin/activate
pip install torch transformers datasets numpy pytest
git init && git remote add origin https://github.com/yourname/nanocore-s1.git
```

### Task 2: Write failing test for model config
```python
# tests/test_model.py
from src.model import NanoCoreConfig, NanoCore

def test_config_defaults():
    config = NanoCoreConfig()
    assert config.n_layer == 12
    assert config.n_embd == 768
    assert config.n_head == 6

def test_model_forward():
    model = NanoCore(NanoCoreConfig(vocab_size=100, sequence_len=128))
    tokens = torch.randint(0, 100, (1, 128))
    logits = model(tokens)
    assert logits.shape == (1, 128, 100)
```

### Task 3: Implement minimal NanoCore model (copy from microgpt)
- 140-line pure Python transformer (no Flash Attention needed for CPU)
- Forward pass only
- Parameter count: verify ~140M at d12

---

## B. Learning Track (Reproducibility)

### Task 4: Write failing test for config builder
```python
# tests/test_config.py
from src.config import build_config

def test_depth_scales_everything():
    config = build_config(depth=12)
    assert config.n_embd == 768
    assert config.n_head == 6

def test_aspect_ratio_default():
    config = build_config(depth=12, aspect_ratio=64)
    assert config.n_embd == 768  # 12 * 64
```

### Task 5: Implement config builder
```python
# src/config.py
def build_config(depth, aspect_ratio=64, head_dim=128, max_seq_len=2048, vocab_size=32768):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return NanoCoreConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_embd=model_dim,
    )
```

### Task 6: ✅ Write failing test for tokenizer [COMPLETE]
```python
# tests/test_tokenizer.py — 13 tests: training, encode/decode roundtrip,
# compression ratio, special tokens, save/load
```

### Task 7: ✅ Implement BPE tokenizer (COMPLETE)
- `src/tokenizer.py`: Byte-level BPE tokenizer (tiktoken-style) with:
  - 256 byte base + 20 decision special tokens (`[STATE]`, `[CHOICE]`, `[ANSWER]`, `[Noul]`, `[SCORE]`, etc.)
  - `NanoCoreTokenizer` wrapper with 32768 vocab target
  - Encode/decode with special token preservation
  - JSON save/load serialization
  - Compression ratio calculation
- `scripts/train_tokenizer.py`: Training script with dry-run + FineWeb-EDU streaming
- `scripts/train_base.py`: Integrated tokenizer (was byte-level, now uses trained BPE)
- `notebooks/train_colab.ipynb`: Added tokenizer as Step 0
- **All 52 tests pass (39 existing + 13 new)**
- **Quality gate passes: Architecture, Memory, Dry Run, Evaluation all green**

### Task 8: Write failing test for training loop
```python
# tests/test_train.py
def test_single_step_loss_decreases():
    model = NanoCore(build_config(depth=4))  # tiny for test
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(10):
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
```

### Task 9: Implement training loop
- micro-batch → gradient accumulation → optimizer step
- μP learning rate scaling
- Weight scheduling

---

## C. Data Track (Free Sources)

### Task 10: Write failing test for data loader
```python
# tests/test_data.py
def test_dataloader_shapes():
    loader = DecisionDataLoader("test_state: [STATE]{test_state}[/STATE]", batch_size=4, seq_len=128)
    batch = next(loader)
    assert batch['inputs'].shape == (4, 128)
    assert batch['targets'].shape == (4, 128)
```

### Task 11: Create free data pipeline
- FineWeb-EDU via HF datasets (200M tokens)
- SmolTalk conversations (100K)
- **Synthetic decision traces** (200K examples) — `python scripts/gen_decision_data.py`
- Decision benchmark with verifiable answers (10K cases)

### Task 12: Write failing test for decision parser
```python
# tests/test_decisions.py
def test_parse_choice():
    output = "[ANSWER] route=security_fraud, P(billing)=0.05, P(tech_support)=0.02, P(security_fraud)=0.92, P(human_review)=0.01, confidence=0.95 [/ANSWER]"
    result = parse_decision(output)
    assert result.value == "security_fraud"
    assert abs(result.probabilities["security_fraud"] - 0.92) < 0.01
    assert result.confidence == 0.95
```

### Task 13: Implement decision parser
- Parse `[ANSWER]...[/ANSWER]` block
- Extract Choice, Noul, Score with probabilities/confidence
- Handle malformed input gracefully

---

## D. Training Stages (All $0)

### Stage 1: Base ($0, 15 min on Colab T4)
```bash
# Task 14: Implement train_base.py
python scripts/train_base.py --depth=12 --device=cpu --tokens=200M
```
**Test**: After 10 iterations, loss should be decreasing.
**Verify**: Save checkpoint `models/nanocore-s1-base.pt`

### Stage 2: Midtraining ($0, 5 min)
```bash
# Task 15: Implement train_mid.py
python scripts/train_mid.py --depth=12 --epochs=3 --data="smoltalk+decision_traces"
```
**Test**: Model correctly parses `[STATE]...[CHOICE options]...` format 90%+ of the time.
**Verify**: Evaluate on 100 held-out decision format examples.

### Stage 3: SFT ($0, 2 min)
```bash
# Task 16: Implement train_sft.py
python scripts/train_sft.py --depth=12 --epochs=10 --data="curated_decisions_10k"
```
**Test**: Model produces valid `[ANSWER]` blocks on 95% of inputs.
**Verify**: Run on 10 test decisions, verify output format.

### Stage 4: RLCD ($0, 2 min)
```bash
# Task 17: Implement train_rlcd.py
python scripts/train_rlcd.py --depth=12 --tasks="gsm8k+decision_bench" --episodes=50
```
**Test**: Confidence correlates with accuracy ≥ 0.7 correlation coefficient.
**Verify**: Calibration curve shows ECE < 0.1.

---

## E. Shipping Track (Speedrun Script)

### Task 18: Write speedrun script (one-command pipeline)
```bash
# Task 19: Test speedrun end-to-end
python speedrun.py  # all 4 stages, reports to report.md
```
**Verify**:
```
✓ Stage 1 (Pretrain): 15m, loss 4.2 → 2.1
✓ Stage 2 (Midtrain): 5m, format accuracy 92%
✓ Stage 3 (SFT): 2m, answer validity 96%
✓ Stage 4 (RLCD): 2m, ECE 0.08
```

---

## F. Verification Track

### Task 20: Implement CORE-like evaluation
- 22 datasets from nanochat CORE metric
- Plus custom "Decision Accuracy" on 1K held-out decision examples

### Task 21: Write inference CLI
```bash
# Task 22: Test inference
echo "Unauthorized login from 192.168.1.45" | python -m nanocore_s1.cli \
  --choices "billing,technical_support,security_fraud,human_review"
# Expected output: security_fraud, confidence=0.95
```

### Task 23: Quantization test (int4)
```bash
# Task 24: Verify int4 inference quality
python -m nanocore_s1.quantize --model models/nanocore-s1-rlcd.pt
python -m nanocore_s1.cli "test state"  # verify still works
```

---

## Execution Order

1. **Week 1:** Setup Track (Tasks 1-3) — project structure, minimal model
2. **Week 1-2:** Learning Track (Tasks 4-9) — config scaling, tokenizer, training loop
3. **Week 2:** Data Track (Tasks 10-13) — free data pipeline, decision parser
4. **Week 3:** Training Stages (Tasks 14-17) — all 4 stages on free GPU
5. **Week 4:** Shipping Track (Tasks 18-22) — speedrun, CLI, evaluation
6. **Week 4:** Quantization (Tasks 23-24) — int4 deployment

Each task follows strict TDD (Red → Green → Refactor → Commit).

---

## Dependencies (All Free)

| Dependency | Purpose | Cost |
|---|---|---|
| PyTorch | Model + training | Free (pip install) |
| HuggingFace datasets | FineWeb-EDU, SmolTalk | Free (public datasets) |
| Kaggle/Colab | GPU compute | Free tier (15-30h/week) |
| `uv` | Python env management | Free (astral.sh) |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| Colab rate limits | Use Kaggle as backup; split across accounts |
| Free GPU OOM | Start with depth=4 for testing, scale up |
| Data quality | FineWeb-EDU filtering is already done |
| Quantization loss | Measure on decision task, not MMLU |
| Token limit | 32768 vocab is sufficient for decision tokens |

---

## Acceptance Criteria

Before each stage is marked done:
1. **All tests pass:** `pytest tests/ -q` — 0 failures
2. **No warnings:** Clean output (no deprecation warnings)
3. **Verified:** Manual test with real input
4. **Committed:** `git add -A && git commit -m "feat: ..."`
5. **Report card:** Update `report.md` with metrics

**Final success threshold:** Can classify a "security fraud" support ticket with ≥90% accuracy and <100ms latency on CPU.
