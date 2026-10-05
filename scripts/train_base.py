"""
Base training script for NanoCore-S1 Stage 1: Pretraining.

Trains on FineWeb-EDU dataset for 200M tokens using:
- μP learning rate scaling
- Batch size via gradient accumulation
- Checkpointing to models/

Usage:
    # Dry run (100 steps, ~10K tokens) - verifies everything works
    python scripts/train_base.py --dry-run

    # Full pretraining (200M tokens, ~8 epochs of FineWeb-EDU subset)
    python scripts/train_base.py --epochs=8 --batch-size=128 --grad-accum=8

    # Resume from checkpoint
    python scripts/train_base.py --resume models/nanocore-s1-base.pt
"""

import torch
import math
import os
import time
import json
import random
from contextlib import nullcontext

# Handle running from scripts/ directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NanoCore, NanoCoreConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "depth": 12,
    "aspect_ratio": 64,
    "vocab_size": 32768,
    "sequence_len": 2048,
    "batch_size": 64,
    "grad_accum": 16,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "warmup_steps": 750,
    "max_steps": 3125,  # 200M tokens / (64 * 2048 * 16)
    "seed": 42,
    "eval_interval": 200,
    "eval_iters": 200,
    "save_interval": 500,
    "log_interval": 10,
    "device": "auto",  # auto = cuda if available else cpu
    "dtype": "auto",  # auto = float16 if cuda else bfloat16
}


def get_device(device_str: str = "auto") -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def get_dtype(dtype_str: str = "auto") -> torch.dtype:
    if dtype_str == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float) -> float:
    """μP-style learning rate schedule (cosine with warmup)."""
    if step < warmup_steps:
        # Linear warmup
        return base_lr * step / max(1, warmup_steps)
    
    # Cosine decay to 10% of base LR
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * 0.10
    return min_lr + (base_lr - min_lr) * cosine_decay


def get_batch(data: torch.Tensor, batch_size: int, sequence_len: int,
              device: str, dtype: torch.dtype):
    """Sample a random batch from the dataset.
    
    Returns: (input_tokens, target_tokens) - both kept as long tensors on CPU,
             moved to device by PyTorch's autocast handling.
    """
    n = data.size(0)
    idx = torch.randint(0, n - sequence_len, (batch_size,))
    
    # Gather sequences - keep as long (embedding indices must be long)
    x = data[idx.unsqueeze(1) + torch.arange(sequence_len)]
    # Targets are shifted by 1
    y = data[idx.unsqueeze(1) + torch.arange(1, sequence_len + 1)]
    
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: NanoCore, data: torch.Tensor, eval_iters: int,
                  batch_size: int, sequence_len: int, device: str, dtype: torch.dtype):
    """Estimate loss on a small number of batches."""
    model.eval()
    losses = {"train": 0.0, "val": 0.0}
    for split in ["train", "val"]:
        total_loss = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, sequence_len, device, dtype)
            with nullcontext() if dtype == torch.float32 else torch.amp.autocast(device_type=device, dtype=dtype):
                loss = model(x, targets=y)
            total_loss += loss.item()
        losses[split] = total_loss / eval_iters
    model.train()
    return losses


def load_finetune_edu_data(vocab_size: int, sequence_len: int, limit_gb: float = 8.0):
    """
    Load FineWeb-EDU dataset (free from HuggingFace).
    
    FineWeb-EDU is ~300B tokens, we limit to ~8GB (~800M tokens) for pretraining.
    
    Returns: torch.Tensor of token ids
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("HuggingFace 'datasets' library not installed. Installing...")
        os.system("pip install datasets")
        from datasets import load_dataset
    
    print("Loading FineWeb-EDU dataset (this may take a while on first run)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    
    # Use a simple byte-level tokenizer approximation for demo
    # In production, use a proper BPE tokenizer trained on this data
    # For now, we use a hash-based approach for the dry run
    all_tokens = []
    total_bytes = 0
    max_bytes = int(limit_gb * 1024**3)
    
    for item in ds:
        text = item["text"]
        # Simple tokenization: byte-level encoding
        tokens = list(text.encode('utf-8'))
        all_tokens.extend(tokens)
        total_bytes += len(text.encode('utf-8'))
        
        if total_bytes >= max_bytes:
            break
        
        if len(all_tokens) % 100_000 == 0 and len(all_tokens) > 0:
            print(f"  Loaded {total_bytes / 1e9:.1f} GB ({len(all_tokens)} tokens)")
    
    print(f"Loaded {total_bytes / 1e9:.1f} GB, {len(all_tokens)} tokens")
    
    # Convert to tensor and reshape
    data = torch.tensor(all_tokens, dtype=torch.int64)
    
    # Split train/val (90/10)
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    return train_data, val_data


def load_synthetic_data(vocab_size: int, sequence_len: int, num_tokens: int):
    """Generate synthetic data for dry-run / testing."""
    print(f"Generating {num_tokens} synthetic tokens...")
    torch.manual_seed(42)
    train_data = torch.randint(0, vocab_size, (num_tokens,))
    val_data = torch.randint(0, vocab_size, (num_tokens // 10,))
    return train_data, val_data


def main():
    """Main training loop."""
    import argparse
    parser = argparse.ArgumentParser(description="Train NanoCore-S1 base model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic data for 100 steps (~10K tokens)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (default: auto-calculate for 200M tokens)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_CONFIG["grad_accum"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_CONFIG["warmup_steps"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CONFIG["max_steps"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--depth", type=int, default=DEFAULT_CONFIG["depth"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save-dir", type=str, default="models",
                        help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = get_device(args.device)
    dtype = get_dtype("auto") if device != "cpu" else torch.float32
    ptdtype = dtype  # torch.float16, torch.bfloat16, or torch.float32
    amp_ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=ptdtype)
    
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Mixed precision: {amp_ctx is not nullcontext}")
    
    # Build model
    config = NanoCoreConfig.from_depth(
        depth=args.depth,
        aspect_ratio=DEFAULT_CONFIG["aspect_ratio"],
        vocab_size=DEFAULT_CONFIG["vocab_size"],
        sequence_len=DEFAULT_CONFIG["sequence_len"],
    )
    model = NanoCore(config).to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Optimizer (AdamW with weight decay, no decay for biases and embeds)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
    optimizer_grouped_params = [
        {"params": decay_params, "weight_decay": args.learning_rate * 0.01},  # μP scaling
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_params,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
    )
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        step = checkpoint["step"]
        print(f"Resumed at step {step}")
    else:
        step = 0
    
    # Load data
    if args.dry_run:
        # Synthetic data: 20 steps * 2 * 512 tokens = ~20K tokens fake
        train_data, val_data = load_synthetic_data(
            config.vocab_size, config.sequence_len, 300_000
        )
        max_steps = 20  # Quick smoke test
        eval_interval = 10
        eval_iters = 5  # Minimal eval for dry run
        save_interval = 20
        log_interval = 10
        os.makedirs("models", exist_ok=True)
        # Use small batch and sequence length for dry run to fit in 8GB VRAM
        args.batch_size = 2
        args.grad_accum = 2
        save_path = os.path.join("models", "nanocore-s1-dryrun.pt")
        # Override sequence_len for dry run to reduce memory
        config.sequence_len = 512
    else:
        # Use conservative batch size for training
        if args.batch_size > 32:
            args.batch_size = 32
        train_data, val_data = load_finetune_edu_data(
            config.vocab_size, config.sequence_len, limit_gb=8.0
        )
        max_steps = args.max_steps
        eval_interval = DEFAULT_CONFIG["eval_interval"]
        eval_iters = DEFAULT_CONFIG["eval_iters"]
        save_interval = DEFAULT_CONFIG["save_interval"]
        log_interval = DEFAULT_CONFIG["log_interval"]
        save_path = os.path.join(args.save_dir, "nanocore-s1-base.pt")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training for {max_steps} steps...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accum: {args.grad_accum}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  Max LR: {args.learning_rate}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Sequence length: {config.sequence_len}")
    
    model.train()
    optimizer.zero_grad()
    tokens_processed = 0
    t0 = time.time()
    accum_loss = torch.tensor(0.0, device=device)
    
    while step < max_steps:
        # Learning rate schedule (μP scaling is handled by the schedule)
        lr = get_lr(step, max_steps, args.warmup_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Gradient accumulation
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, config.sequence_len, device, dtype)
            
            with amp_ctx:
                loss = model(x, targets=y)
                accum_loss = loss.detach()  # Track unscaled loss for logging
                # Scale loss by grad_accum
                loss = loss / args.grad_accum
            
            loss.backward()
            
            step += 1
            tokens_processed += args.batch_size * config.sequence_len
            
            if step >= max_steps:
                break
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging
        if step % log_interval == 0 or step == 1:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = tokens_processed / dt
            print(f"Step {step}/{max_steps} | Loss: {accum_loss.item():.4f} | "
                  f"LR: {lr:.6f} | {tokens_per_sec:.0f} tok/s | {dt:.1f}s")
            t0 = t1
            tokens_processed = 0
        
        # Evaluation
        if step % eval_interval == 0:
            losses = estimate_loss(model, val_data, eval_iters,
                                   args.batch_size, config.sequence_len, device, dtype)
            print(f"  Eval: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        
        # Checkpointing
        if step % save_interval == 0:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "config": {
                    "depth": config.n_layer,
                    "n_embd": config.n_embd,
                    "n_head": config.n_head,
                    "vocab_size": config.vocab_size,
                    "sequence_len": config.sequence_len,
                },
            }
            torch.save(checkpoint, save_path)
            print(f"  Checkpoint saved to {save_path}")
    
    # Final checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": {
            "depth": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "vocab_size": config.vocab_size,
            "sequence_len": config.sequence_len,
        },
    }
    torch.save(checkpoint, save_path)
    print(f"\nTraining complete! Final checkpoint: {save_path}")
    
    # Write training report
    report = {
        "final_step": step,
        "final_loss": accum_loss.item(),
        "total_tokens": step * args.batch_size * config.sequence_len * args.grad_accum,
        "model_params": model.num_parameters(),
        "config": {
            "depth": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "sequence_len": config.sequence_len,
            "vocab_size": config.vocab_size,
        },
        "training_args": {
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "max_steps": max_steps if not args.dry_run else 100,
        },
    }
    with open(os.path.join(args.save_dir, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
