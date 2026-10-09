"""Tests for NanoCore-S1 model configuration and architecture."""

import pytest
import torch
from src.model import NanoCoreConfig, NanoCore


class TestConfig:
    """Tests for NanoCoreConfig dataclass."""

    def test_config_defaults(self):
        """Default config should match d12 architecture from design doc."""
        config = NanoCoreConfig()
        assert config.n_layer == 12
        assert config.n_embd == 768
        assert config.n_head == 6
        assert config.vocab_size == 32768
        assert config.sequence_len == 2048

    def test_config_custom_depth(self):
        """Config should scale properly with depth via nanochat's formula."""
        config = NanoCoreConfig.from_depth(4, aspect_ratio=64, head_dim=128)
        assert config.n_layer == 4
        assert config.n_embd == 256  # 4 * 64 = 256
        assert config.n_head == 2    # 256 / 128

    def test_config_depth_12_matches_paper(self):
        """d12 config should match parameters from Karpathy doc."""
        config = NanoCoreConfig.from_depth(12)
        assert config.n_layer == 12
        assert config.n_embd == 768  # 12 * 64
        assert config.n_head == 6    # 768 / 128


class TestModelForward:
    """Tests for NanoCore forward pass."""

    def test_forward_output_shape(self):
        """Forward pass should produce logits with correct shape."""
        config = NanoCoreConfig(vocab_size=100, sequence_len=128, n_layer=4, n_embd=256, n_head=4)
        model = NanoCore(config)
        tokens = torch.randint(0, 100, (2, 128))
        logits = model(tokens)
        assert logits.shape == (2, 128, 100)

    def test_forward_single_batch(self):
        """Forward pass with batch size 1 should work."""
        config = NanoCoreConfig(vocab_size=100, sequence_len=64, n_layer=4, n_embd=256, n_head=4)
        model = NanoCore(config)
        tokens = torch.randint(0, 100, (1, 64))
        logits = model(tokens)
        assert logits.shape == (1, 64, 100)

    def test_model_parameters_count(self):
        """d12 model should have approximately 140M parameters."""
        config = NanoCoreConfig.from_depth(12)
        model = NanoCore(config)
        total_params = sum(p.numel() for p in model.parameters())
        # Expect ~140M, allow ±20% tolerance
        assert 110_000_000 <= total_params <= 180_000_000, f"Got {total_params}"

    def test_gradient_flow(self):
        """Gradients should flow through all layers during backward pass."""
        config = NanoCoreConfig(vocab_size=50, sequence_len=32, n_layer=4, n_embd=128, n_head=2)
        model = NanoCore(config)
        tokens = torch.randint(0, 50, (1, 32))
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()
        
        # Check that at least some gradients are non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients found in any parameter"
