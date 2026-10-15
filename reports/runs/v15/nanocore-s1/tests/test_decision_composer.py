"""Contract tests for EmbeddingComposer — the karpathy component.

Covers the properties the audit taught us to check explicitly: identity-at-init
is *measured* (not assumed), attention actually mixes items once the
projections move, padding is honored, and pooling modes behave.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.decision.composer import (
    ComposerConfig, EmbeddingComposer, MODALITY_IDS, rms_norm,
)

DIM = 64  # small: contract tests, not capacity
torch.manual_seed(0)


def _composer(**kw):
    cfg = ComposerConfig(n_layer=2, n_head=4, n_embd=DIM, embd_dim=DIM,
                         max_items=16, **kw)
    torch.manual_seed(0)
    return EmbeddingComposer(cfg)


def _dezero(model):
    """Move the zero-init output projections off zero, as training would."""
    with torch.no_grad():
        for b in model.blocks:
            b.attn.c_proj.weight.normal_(0, 0.02)
            b.mlp.c_proj.weight.normal_(0, 0.02)


def _batch(n=3, dim=DIM, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, n, dim, generator=g)


class TestContract:
    def test_output_shape(self):
        out = _composer()(_batch(5))
        assert out.shape == (1, DIM)

    def test_batch_dimension(self):
        out = _composer()(torch.randn(4, 3, DIM))
        assert out.shape == (4, DIM)

    def test_max_items_enforced(self):
        with pytest.raises(ValueError):
            _composer()(torch.randn(1, 17, DIM))

    def test_bad_pooling_rejected(self):
        with pytest.raises(ValueError):
            _composer(pooling="last")

    def test_modality_ids_shape(self):
        m = _composer()
        ids = torch.tensor([[MODALITY_IDS["text"], MODALITY_IDS["image"], 0]])
        assert m(_batch(3), modality_ids=ids).shape == (1, DIM)


class TestIdentityAtInit:
    """Documented behavior: zero-init projections ⇒ blocks are the identity."""

    def test_zero_blocks_mean_rms_norm_of_inputs(self):
        m = _composer()
        torch.nn.init.zeros_(m.modality_embed.weight)
        emb = _batch(4)
        expected = F.rms_norm(emb.float(), (DIM,)).mean(dim=1)   # per-position norm → mean
        out = m(emb)
        assert torch.allclose(out, expected[0], atol=1e-5)

    def test_training_would_depart_from_identity(self):
        """After projections move, output is NOT just pooled input — the
        composer must actually compose, which is what we're paying for."""
        m = _composer()
        torch.nn.init.zeros_(m.modality_embed.weight)
        emb = _batch(4)
        before = m(emb).clone()
        _dezero(m)
        after = m(emb)
        assert not torch.allclose(before, after, atol=1e-4)


class TestAttentionActuallyMixes:
    def test_perturbing_one_item_changes_output(self):
        m = _composer()
        _dezero(m)
        emb = _batch(4)
        out_a = m(emb)
        emb2 = emb.clone()
        emb2[0, 0] += 1.0                      # perturb first item
        out_b = m(emb2)
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_item_order_matters(self):
        """Rotary positions mean [a,b,c] and [c,b,a] compose differently —
        order-sensitive by design (documented), not a bug."""
        m = _composer()
        _dezero(m)
        emb = _batch(3)
        fwd, rev = m(emb), m(emb.flip(1))
        assert not torch.allclose(fwd, rev, atol=1e-5)


class TestMaskingAndPooling:
    def test_padding_ignored(self):
        m = _composer()
        emb = _batch(3)
        padded = torch.cat([emb, torch.randn(1, 2, DIM)], dim=1)
        mask = torch.tensor([[True, True, True, False, False]])
        out_padded = m(padded, pad_mask=mask)
        out_clean = m(emb)
        assert torch.allclose(out_padded, out_clean, atol=1e-5)

    def test_cls_pooling(self):
        m = _composer(pooling="cls")
        out = m(_batch(4))
        assert out.shape == (1, DIM)

    def test_gradients_flow_to_inputs_path(self):
        """The composer must be trainable end-to-end: a loss on its output
        must produce nonzero grads in the attention projections."""
        m = _composer()
        _dezero(m)
        out = m(_batch(3)).sum()
        out.backward()
        grads = [b.attn.c_proj.weight.grad for b in m.blocks]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_parameter_count_sane():
    m = _composer()
    n = m.n_parameters()
    # 2 layers × (4·d² attn + 8·d² mlp) + embeddings ≈ 12·d²·L ≈ 98k at d=64
    assert 50_000 < n < 200_000
