"""
NanoCore-S1 Quality Gate — Pre-Flight Check

Runs before ANY cloud GPU training session to verify:
1. Model architecture is valid
2. Training script imports and runs (dry-run smoke test)
3. Evaluation suite produces valid results
4. Memory budget fits the target hardware
5. All tests pass

Usage:
    python scripts/quality_gate.py --hardware rtx_2070_8gb --model-depth=12

Exit codes:
    0 = All checks passed, safe to train
    1 = Quality gate failed, DO NOT launch training
"""

import torch
import json
import os
import sys
import time
import subprocess
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import NanoCore, NanoCoreConfig
from scripts.evaluate import Evaluator, EvalResult


HARDWARE_PROFILES = {
    "rtx_2070_8gb": {
        "vram_mb": 8192,
        "safe_batch": 4,  # Conservative for 8GB
        "max_seq_len": 2048,
        "fp16": True,
    },
    "t4_colab": {
        "vram_mb": 15360,
        "safe_batch": 16,
        "max_seq_len": 2048,
        "fp16": True,
    },
    "a100_40gb": {
        "vram_mb": 40960,
        "safe_batch": 64,
        "max_seq_len": 8192,
        "fp16": True,
    },
    "cpu_only": {
        "vram_mb": 0,
        "safe_batch": 4,
        "max_seq_len": 1024,
        "fp16": False,
    },
}


def check_imports():
    """Verify all required packages can be imported."""
    print("[1/6] Checking imports...")
    required = ["torch", "torch.nn", "torch.nn.functional"]
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError as e:
            print(f"  ✗ {pkg}: {e}")
            return False

    # Optional but recommended
    for pkg in ["datasets"]:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ! {pkg} not installed (needed for FineWeb-EDU)")

    return True


def check_model_architecture(depth):
    """Verify the model architecture is valid and matches design spec."""
    print(f"\n[2/6] Checking model architecture (d{depth})...")
    
    try:
        config = NanoCoreConfig.from_depth(depth)
        model = NanoCore(config)
        
        params = model.num_parameters()
        n_layers = config.n_layer
        n_dim = config.n_embd
        n_heads = config.n_head
        head_dim = config.head_dim
        
        print(f"  ✓ Model created successfully")
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  Layers: {n_layers}, Dim: {n_dim}, Heads: {n_heads}, Head dim: {head_dim}")
        
        # Verify architecture consistency
        assert config.n_embd == config.n_head * config.head_dim, "n_embd != n_head * head_dim"
        print(f"  ✓ Architecture consistent (n_embd = n_head * head_dim)")
        
        # Verify forward pass works
        test_tokens = torch.randint(0, config.vocab_size, (1, 128))
        with torch.no_grad():
            logits = model(test_tokens)
        
        expected_shape = (1, 128, config.vocab_size)
        assert logits.shape == expected_shape, f"Output shape mismatch: {logits.shape} vs {expected_shape}"
        print(f"  ✓ Forward pass: {test_tokens.shape} -> {logits.shape}")
        
        # Verify gradient flow
        loss = model(test_tokens, targets=test_tokens)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients computed"
        print(f"  ✓ Gradient flow verified (loss={loss.item():.4f})")
        
        # Expected parameter count for d12
        if depth == 12:
            expected_min = 110_000_000
            expected_max = 180_000_000
            assert expected_min <= params <= expected_max, \
                f"Parameter count {params} outside expected range for d12"
            print(f"  ✓ Parameter count in expected range ({expected_min//1_000_000}M-{expected_max//1_000_000}M)")
        
        return True, model, config
        
    except Exception as e:
        print(f"  ✗ Architecture check failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def check_memory_budget(config, hardware, batch_size):
    """Verify model fits in target hardware memory."""
    print(f"\n[3/6] Checking memory budget ({hardware})...")
    
    profile = HARDWARE_PROFILES[hardware]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model and check size
    model = NanoCore(config).to(device)
    param_count = model.num_parameters()
    
    weights_mb = param_count * 2 / 1e6  # fp16
    
    # KV cache for batch_size and sequence_len
    seq_len = config.sequence_len
    kv_cache_mb = (config.n_layer * 2 * config.n_head * config.head_dim * 
                   seq_len * batch_size * 2) / 1e6
    
    # Activations: simplified estimate based on actual measurements
    # nanochat uses ~80MB per sample at seq_len=2048 for d12
    # Formula: weights * 0.6 * batch_size * seq_len / max_seq_len
    activations_mb = weights_mb * 0.6 * batch_size * seq_len / 2048
    
    # Gradients  
    gradients_mb = weights_mb
    
    # Optimizer (AdamW: 2x model size)
    optimizer_mb = 2 * weights_mb
    
    # Add PyTorch overhead (~1GB for context, CUDA libs)
    cuda_overhead_mb = 1024
    
    total_mb = weights_mb + kv_cache_mb + activations_mb + gradients_mb + optimizer_mb + cuda_overhead_mb
    
    print(f"  Weights:     {weights_mb:.0f} MB")
    print(f"  KV cache:    {kv_cache_mb:.0f} MB (batch={batch_size}, seq={seq_len})")
    print(f"  Activations: {activations_mb:.0f} MB")
    print(f"  Gradients:   {gradients_mb:.0f} MB")
    print(f"  Optimizer:   {optimizer_mb:.0f} MB")
    print(f"  ─────────────────────────")
    print(f"  Total:       {total_mb:.0f} MB")
    print(f"  Available:   {profile['vram_mb']} MB ({hardware})")
    
    if profile["vram_mb"] > 0:
        if total_mb < profile["vram_mb"] * 0.8:  # 20% headroom
            print(f"  ✓ Memory budget OK ({total_mb/profile['vram_mb']*100:.0f}% utilization)")
            return True
        else:
            print(f"  ✗ Memory EXCEEDS safe limit ({total_mb/profile['vram_mb']*100:.0f}% utilization)")
            print(f"    Reduce batch_size or use gradient accumulation")
            return False
    else:
        print(f"  ⚠ No VRAM limit (CPU mode) — training will be slow")
        return True


def check_dry_run():
    """Run dry-run to verify pipeline end-to-end."""
    print(f"\n[4/6] Running dry-run smoke test...")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/train_base.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=180,  # Dry run takes ~50s including CUDA init
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    t1 = time.time()
    
    if result.returncode != 0:
        print(f"  ✗ Dry run FAILED after {t1-t0:.1f}s")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    
    print(f"  ✓ Dry run completed in {t1-t0:.1f}s")
    
    # Check output
    if "Training complete!" in result.stdout:
        print(f"  ✓ Training loop executed successfully")
    else:
        print(f"  ✗ Training loop may not have completed")
        return False
    
    # Check checkpoint was saved
    if os.path.exists("models/nanocore-s1-dryrun.pt"):
        print(f"  ✓ Checkpoint saved")
    else:
        print(f"  ✗ Checkpoint NOT saved")
        return False
    
    return True


def check_evaluation(model, config):
    """Run evaluation suite to verify quality metrics."""
    print(f"\n[5/6] Running evaluation suite...")
    
    # Use GPU if available for faster evaluation
    eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Move model to eval device if needed
        eval_model = NanoCore(config)
        eval_model.load_state_dict(model.state_dict())
        evaluator = Evaluator(eval_model, config, device=eval_device)
        
        # Use small datasets for speed
        datasets = {
            "perplexity": evaluator._create_dummy_dataset("perplexity", 10),
            "calibration": evaluator._create_dummy_dataset("calibration", 10),
        }
        
        results = evaluator.evaluate(datasets=datasets)
        
        for r in results:
            print(f"  {r.metric_name}: {r.value:.4f} ({r.num_samples} samples)")
        
        # Sanity checks
        for r in results:
            if not (isinstance(r.value, (int, float)) and r.value >= 0):
                print(f"  ✗ Invalid metric value for {r.metric_name}: {r.value}")
                return False
        
        print(f"  ✓ All metrics valid")
        return True
        
    except Exception as e:
        print(f"  ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_test_suite():
    """Run the test suite to verify everything works."""
    print(f"\n[6/6] Running test suite (excluding dry-run tests)...")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "-x", "--tb=short", "-k", "not DryRun"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    t1 = time.time()
    
    if result.returncode != 0:
        print(f"  ✌ Tests FAILED after {t1-t0:.1f}s")
        # Print last 20 lines of output
        lines = result.stdout.split('\n')
        for line in lines[-20:]:
            print(f"  {line}")
        return False
    
    # Count passed tests
    passed = result.stdout.count(" PASSED")
    print(f"  ✓ All tests passed ({passed} tests, {t1-t0:.1f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="NanoCore-S1 Quality Gate")
    parser.add_argument("--hardware", type=str, default="rtx_2070_8gb",
                       choices=list(HARDWARE_PROFILES.keys()),
                       help="Target hardware profile")
    parser.add_argument("--model-depth", type=int, default=12,
                       help="Model depth (default: 12)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size to check (default: hardware profile)")
    parser.add_argument("--skip-dry-run", action="store_true",
                       help="Skip dry-run check (faster but less thorough)")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip full test suite")
    parser.add_argument("--output", type=str, default="models/quality_gate_report.json",
                       help="Output report path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NanoCore-S1 Quality Gate — Pre-Flight Check")
    print("=" * 60)
    print(f"Hardware: {args.hardware}")
    print(f"Model depth: d{args.model_depth}")
    print(f"Batch size: {args.batch_size or HARDWARE_PROFILES[args.hardware]['safe_batch']}")
    print()
    
    checks = []
    
    # 1. Imports
    ok = check_imports()
    checks.append(("Imports", ok))
    if not ok:
        print("\n❌ QUALITY GATE FAILED — fix imports before training")
        return 1
    
    # 2. Model architecture
    ok, model, config = check_model_architecture(args.model_depth)
    checks.append(("Architecture", ok))
    if not ok:
        print("\n❌ QUALITY GATE FAILED — fix model architecture")
        return 1
    
    # 3. Memory budget
    batch_size = args.batch_size or HARDWARE_PROFILES[args.hardware]["safe_batch"]
    ok = check_memory_budget(config, args.hardware, batch_size)
    checks.append(("Memory", ok))
    if not ok:
        print("\n⚠️  WARNING — Memory may be tight. Consider smaller batch.")
        # Don't fail on memory, just warn
    
    # 4. Dry run
    if not args.skip_dry_run:
        ok = check_dry_run()
        checks.append(("Dry Run", ok))
        if not ok:
            print("\n❌ QUALITY GATE FAILED — dry-run verification failed")
            return 1
    else:
        print("\n[4/6] Dry-run check SKIPPED (--skip-dry-run)")
        checks.append(("Dry Run", "skipped"))
    
    # 5. Evaluation
    ok = check_evaluation(model, config)
    checks.append(("Evaluation", ok))
    if not ok:
        print("\n❌ QUALITY GATE FAILED — evaluation suite error")
        return 1
    
    # 6. Test suite
    if not args.skip_tests:
        ok = check_test_suite()
        checks.append(("Test Suite", ok))
        if not ok:
            print("\n❌ QUALITY GATE FAILED — tests failing")
            return 1
    else:
        print("\n[6/6] Test suite SKIPPED (--skip-tests)")
        checks.append(("Test Suite", "skipped"))
    
    # Summary
    print("\n" + "=" * 60)
    print("QUALITY GATE REPORT")
    print("=" * 60)
    for name, ok in checks:
        status = "✅ PASS" if ok is True else ("⚠️ WARN" if ok is False else "⏭️ SKIP" if ok == "skipped" else "❌ FAIL")
        print(f"  {name:20s} {status}")
    
    all_passed = all(ok is True or ok == "skipped" for _, ok in checks)
    
    # Save report
    report = {
        "hardware": args.hardware,
        "model_depth": args.model_depth,
        "batch_size": batch_size,
        "checks": [{"name": name, "status": str(ok)} for name, ok in checks],
        "all_passed": all_passed,
    }
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output}")
    
    if all_passed:
        print("\n✅ ALL CHECKS PASSED — Safe to launch training!")
        print("   Proceed with: python scripts/train_base.py --batch-size", batch_size)
    else:
        print("\n❌ QUALITY GATE FAILED — DO NOT launch training until checks pass")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
