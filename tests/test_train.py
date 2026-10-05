"""
Test the training script's dry-run mode.

Verifies:
- Dry run completes without errors
- Checkpoint is saved
- Loss is reasonable (not NaN, finite)
- Parameter count is correct
"""

import os
import torch
import pytest
import subprocess
import sys

from src.model import NanoCore, NanoCoreConfig


@pytest.fixture(scope="module")
def dry_run_output():
    """Run dry-run once and cache the result."""
    # Remove old checkpoint if exists
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "models", "nanocore-s1-dryrun.pt")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    result = subprocess.run(
        [sys.executable, "scripts/train_base.py", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"
    return result


class TestTrainingDryRun:
    """Tests for the training script dry-run mode."""

    def test_dry_run_completes(self, dry_run_output):
        """Dry run should complete without errors and save a checkpoint."""
        # Check loss is finite
        assert "Loss:" in dry_run_output.stdout, "No loss logged"
        loss_line = [line for line in dry_run_output.stdout.split('\n') if 'Loss:' in line][-1]
        loss_value = float(loss_line.split('Loss:')[1].split('|')[0].strip())
        assert torch.isfinite(torch.tensor(loss_value)), f"Loss is not finite: {loss_value}"

        # Check checkpoint was saved
        assert os.path.exists("models/nanocore-s1-dryrun.pt"), "Checkpoint not saved"

    def test_checkpoint_loadable(self):
        """Saved checkpoint should be loadable and contain correct keys."""
        checkpoint = torch.load("models/nanocore-s1-dryrun.pt", weights_only=False)
        assert "model_state" in checkpoint
        assert "optimizer_state" in checkpoint
        assert "step" in checkpoint
        assert "config" in checkpoint

        # Verify config matches
        config = NanoCoreConfig.from_depth(12)
        model = NanoCore(config)

        # Should be able to load state dict without missing keys
        result = model.load_state_dict(checkpoint["model_state"], strict=False)
        assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
        assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"

    def test_throughput_reasonable(self, dry_run_output):
        """Throughput should be >500 tokens/sec on GPU (dry run uses small batch)."""
        assert "tok/s" in dry_run_output.stdout, "No throughput logged"
        throughput_line = [line for line in dry_run_output.stdout.split('\n') if 'tok/s' in line][-1]
        throughput = float(throughput_line.split('tok/s')[0].strip().split('|')[-1].strip())
        # Dry run uses batch_size=2 (low GPU utilization); real training uses 32+
        # Expect >500 tok/s on RTX 2070 at batch=2
        assert throughput > 500, f"Throughput too low: {throughput} tok/s"

    def test_checkpoint_step_count(self, dry_run_output):
        """Checkpoint should show the correct number of steps."""
        checkpoint = torch.load("models/nanocore-s1-dryrun.pt", weights_only=False)
        assert checkpoint["step"] == 20, f"Expected 20 steps, got {checkpoint['step']}"


class TestModelMemoryEstimate:
    """Verify model fits in expected memory budgets."""

    def test_model_size_estimate(self):
        """Model should be ~270MB in fp16 (weights only)."""
        config = NanoCoreConfig.from_depth(12)
        model = NanoCore(config)
        param_count = model.num_parameters()

        # fp16: 2 bytes per param
        size_fp16 = param_count * 2 / 1e6
        print(f"d12 model size (fp16): {size_fp16:.1f} MB")

        assert 200 < size_fp16 < 350, f"Unexpected model size: {size_fp16} MB"

    def test_training_memory_estimate_8gb(self):
        """Training should fit in 8GB GPU with conservative batch settings."""
        config = NanoCoreConfig.from_depth(12)
        model = NanoCore(config)

        # Estimate memory for batch_size=2, seq_len=512 (dry run config)
        batch_size = 2
        seq_len = 512

        params = model.num_parameters()
        weights_mb = params * 2 / 1e6  # fp16

        # KV cache: 12 layers * 2 (K+V) * 6 heads * 64 dim * seq_len * batch_size * 2 bytes
        kv_cache_mb = 12 * 2 * 6 * 64 * seq_len * batch_size * 2 / 1e6

        # Activations: roughly 4x model size per sample
        activations_mb = 4 * weights_mb * batch_size * seq_len / (2048)

        # Gradients
        gradients_mb = weights_mb

        # Optimizer (AdamW: 2x model size)
        optimizer_mb = 2 * weights_mb

        total_mb = weights_mb + kv_cache_mb + activations_mb + gradients_mb + optimizer_mb
        print(f"Estimated training memory (batch={batch_size}, seq={seq_len}): {total_mb:.1f} MB")
        print(f"  Weights: {weights_mb:.1f} MB")
        print(f"  KV cache: {kv_cache_mb:.1f} MB")
        print(f"  Activations: {activations_mb:.1f} MB")
        print(f"  Gradients: {gradients_mb:.1f} MB")
        print(f"  Optimizer: {optimizer_mb:.1f} MB")

        # Should fit in 8GB with room to spare
        assert total_mb < 6000, f"Estimated memory {total_mb} MB exceeds safe limit for 8GB GPU"
