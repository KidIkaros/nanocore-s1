"""Tests for NanoCore-S1 configuration builder.

Tests the μP-style depth scaling formula from Karpathy's nanochat:
  model_dim = depth * aspect_ratio, rounded to head_dim multiple
  num_heads = model_dim // head_dim
"""

import pytest
from src.model import NanoCoreConfig


class TestConfigBuilder:
    """Tests for NanoCoreConfig.from_depth - the muP scaling formula."""

    def test_depth_4_scales_correctly(self):
        """d4 with aspect_ratio=64, head_dim=128 → 256 dim, 2 heads."""
        config = NanoCoreConfig.from_depth(4)
        assert config.n_layer == 4
        assert config.n_embd == 256    # 4 * 64 = 256
        assert config.n_head == 2      # 256 / 128
        assert config.head_dim == 128

    def test_depth_8_scales_correctly(self):
        """d8 with aspect_ratio=64, head_dim=128 → 512 dim, 4 heads."""
        config = NanoCoreConfig.from_depth(8)
        assert config.n_layer == 8
        assert config.n_embd == 512    # 8 * 64 = 512
        assert config.n_head == 4      # 512 / 128

    def test_depth_12_scales_correctly(self):
        """d12 with aspect_ratio=64, head_dim=128 → 768 dim, 6 heads."""
        config = NanoCoreConfig.from_depth(12)
        assert config.n_layer == 12
        assert config.n_embd == 768    # 12 * 64 = 768
        assert config.n_head == 6      # 768 / 128

    def test_depth_16_scales_correctly(self):
        """d16 with aspect_ratio=64, head_dim=128 → 1024 dim, 8 heads."""
        config = NanoCoreConfig.from_depth(16)
        assert config.n_layer == 16
        assert config.n_embd == 1024   # 16 * 64 = 1024
        assert config.n_head == 8      # 1024 / 128

    def test_depth_24_scales_correctly(self):
        """d24 with aspect_ratio=64, head_dim=128 → 1536 dim, 12 heads."""
        config = NanoCoreConfig.from_depth(24)
        assert config.n_layer == 24
        assert config.n_embd == 1536   # 24 * 64 = 1536
        assert config.n_head == 12     # 1536 / 128

    def test_aspect_ratio_80(self):
        """aspect_ratio=80 with head_dim=128 rounds up to nearest multiple.
        
        12 * 80 = 960, but 960 isn't divisible by 128.
        Formula rounds up: ceil(960/128)*128 = 8*128 = 1024
        heads = 1024 / 128 = 8
        """
        config = NanoCoreConfig.from_depth(12, aspect_ratio=80, head_dim=128)
        assert config.n_embd == 1024  # 960 rounded up to 1024 (nearest 128 multiple)
        assert config.n_head == 8     # 1024 / 128

    def test_head_dim_64(self):
        """head_dim=64 gives more heads for same dimension."""
        config = NanoCoreConfig.from_depth(12, aspect_ratio=64, head_dim=64)
        assert config.n_embd == 768
        assert config.n_head == 12     # 768 / 64

    def test_head_dim_256(self):
        """head_dim=256 gives fewer heads."""
        config = NanoCoreConfig.from_depth(12, aspect_ratio=64, head_dim=256)
        assert config.n_embd == 768
        assert config.n_head == 3      # 768 / 256

    def test_custom_vocab_size(self):
        """Custom vocab_size is respected."""
        config = NanoCoreConfig.from_depth(12, vocab_size=50257)
        assert config.vocab_size == 50257

    def test_custom_sequence_len(self):
        """Custom sequence length is respected."""
        config = NanoCoreConfig.from_depth(12, sequence_len=4096)
        assert config.sequence_len == 4096

    def test_default_config_values(self):
        """Direct config construction has correct defaults."""
        config = NanoCoreConfig()
        assert config.sequence_len == 2048
        assert config.vocab_size == 32768
        assert config.n_layer == 12
        assert config.n_head == 6
        assert config.n_embd == 768

    def test_from_depth_preserves_defaults(self):
        """from_depth should preserve sequence_len and vocab_size defaults."""
        config = NanoCoreConfig.from_depth(12)
        assert config.sequence_len == 2048
        assert config.vocab_size == 32768

    def test_muir_scaling_linear_in_depth(self):
        """Parameter count should scale roughly linearly with depth."""
        small = NanoCoreConfig.from_depth(6)
        medium = NanoCoreConfig.from_depth(12)
        # d12 has 2x the layers and 2x the dim → ~4x params
        # (params ~ n_layer * n_embd^2, so d12/d6 = 2 * 4 = 8x for transformer layers)
        # But embeddings also scale with n_embd. Total should be roughly proportional.
        total_small = small.n_layer * small.n_embd ** 2
        total_medium = medium.n_layer * medium.n_embd ** 2
        # d12/d6 ratio should be exactly 2 * 4 = 8 for layer params
        assert total_medium == total_small * 8


class TestConfigValidation:
    """Tests for config validation and edge cases."""

    def test_head_dim_must_divide_model_dim(self):
        """head_dim must divide model_dim evenly."""
        config = NanoCoreConfig.from_depth(12)
        assert config.n_embd % config.head_dim == 0

    def test_min_depth(self):
        """Even depth=1 should work."""
        config = NanoCoreConfig.from_depth(1)
        assert config.n_layer == 1
        assert config.n_head >= 1

    def test_large_depth(self):
        """Large depths should scale without error."""
        config = NanoCoreConfig.from_depth(100)
        assert config.n_layer == 100
        assert config.n_embd == 6400  # 100 * 64
