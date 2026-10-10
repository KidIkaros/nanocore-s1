"""
NanoCore-S1: A minimal GPT-style transformer model for System One decisions.

Architecture derived from Karpathy's nanochat with these principles:
- Single "depth" dial controls architecture (muP scaling)
- Modern improvements: RoPE, QK norm, ReLU^2, untied embeddings
- Pure PyTorch, no external inference deps required for training
- Decision primitives: Choice, Noul, Score (Jev System One integration)

Based on nanochat/gpt.py (github.com/karpathy/nanochat)
"""

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoCoreConfig:
    """Configuration for NanoCore-S1 model.

    Follows Karpathy's nanochat single-dial philosophy:
    all dimensions derive from depth * aspect_ratio.
    """
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    head_dim: int = 128
    window_pattern: str = "SSSL"

    @classmethod
    def from_depth(cls, depth: int, aspect_ratio: int = 64, head_dim: int = 128,
                   vocab_size: int = 32768, sequence_len: int = 2048):
        """Build config from depth via nanochat's muP-style scaling.

        model_dim = depth * aspect_ratio, rounded to head_dim multiple
        num_heads = model_dim // head_dim
        """
        base_dim = depth * aspect_ratio
        model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
        num_heads = model_dim // head_dim
        return cls(
            sequence_len=sequence_len,
            vocab_size=vocab_size,
            n_layer=depth,
            n_head=num_heads,
            head_dim=head_dim,
            n_embd=model_dim,
        )


def rms_norm(x, dim=-1):
    """RMSNorm without learnable parameters (per nanochat)."""
    return F.rms_norm(x, (x.size(dim),))


class CausalSelfAttention(nn.Module):
    """Multi-head attention with RoPE and QK normalization.

    Following nanochat's simplified approach:
    - Standard multi-head attention (no GQA)
    - RoPE for positional encoding
    - QK normalization for stable gradients
    - Shifted scale (q * 1.2, k * 1.2)
    """

    def __init__(self, config: NanoCoreConfig, layer_idx: int = 0):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Standard QKV projections (no bias, per nanochat)
        self.c_q = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Precompute rotary embeddings (10x sequence length for safety)
        self._build_rotary_embeddings(config.sequence_len * 10, self.head_dim)

    def _build_rotary_embeddings(self, seq_len: int, head_dim: int):
        """Precompute RoPE cos/sin matrices.

        Stored as (1, seq_len, 1, head_dim/2) — will be reshaped during apply
        to (1, 1, seq_len, head_dim/2) for broadcasting with (B, n_head, T, head_dim/2).
        """
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos", cos[None, :, None, :])
        self.register_buffer("sin", sin[None, :, None, :])

    def _apply_rope(self, q, k):
        """Apply rotary positional embeddings to queries and keys.

        q, k shapes: (B, n_head, T, head_dim)
        cos/sin reshaped from (1, T, 1, head_dim/2) to (1, 1, T, head_dim/2)
        for broadcasting across head dimension.
        """
        seq_len = q.size(2)
        cos = self.cos[:, :seq_len, :, :].squeeze(2).unsqueeze(1)  # (1, 1, T, head_dim/2)
        sin = self.sin[:, :seq_len, :, :].squeeze(2).unsqueeze(1)

        # Split last dim in half for rotation
        q1, q2 = q[..., :self.head_dim // 2], q[..., self.head_dim // 2:]
        k1, k2 = k[..., :self.head_dim // 2], k[..., self.head_dim // 2:]

        # Rotate: (x1, x2) -> (x1*cos + x2*sin, -x1*sin + x2*cos)
        q_rot = torch.cat([q1 * cos + q2 * sin, -q1 * sin + q2 * cos], dim=-1)
        k_rot = torch.cat([k1 * cos + k2 * sin, -k1 * sin + k2 * cos], dim=-1)
        return q_rot, k_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, T, C) -> (B, T, C)."""
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self._apply_rope(q, k)

        # QK normalization (rms_norm, per nanochat)
        q = rms_norm(q)
        k = rms_norm(k)

        # Shifted attention scale (q * 1.2, k * 1.2 per nanochat)
        q = q * 1.2
        k = k * 1.2

        # Causal attention with PyTorch's efficient SDPA
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True
        )

        # Re-assemble heads
        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(attn)


class MLP(nn.Module):
    """Feed-forward network with ReLU^2 activation (per nanochat).

    nanochat uses: Linear -> ReLU^2 -> Linear
    where ReLU^2 = (max(0, x))^2
    """

    def __init__(self, config: NanoCoreConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 as in nanochat
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (per nanochat).

    Layer structure:
        x = x + attn(rms_norm(x))
        x = x + mlp(rms_norm(x))
    """

    def __init__(self, config: NanoCoreConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(rms_norm(x))
        x = x + self.mlp(rms_norm(x))
        return x


class NanoCore(nn.Module):
    """NanoCore-S1: A minimal GPT-style transformer for System One decisions.

    Architecture inspired by Karpathy's nanochat (github.com/karpathy/nanochat).
    Modified with Jev-style decision primitives for type-safe, calibrated decisions.

    Key features:
    - Untied token embeddings and LM head
    - Per-layer residual scaling (muP-style lambdas)
    - Mid-layer backout (subtracts mid-layer features for sharper representations)
    - Softcapped logits for stable training
    """

    def __init__(self, config: NanoCoreConfig):
        super().__init__()
        self.config = config

        # Token embedding and lm head (untied, per nanochat)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layer)
        ])

        # Per-layer residual scaling lambdas (muP-style, per nanochat)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))

        self._init_weights()

    def _init_weights(self):
        """Weight initialization following nanochat's scheme."""
        # Embeddings: std=0.8 for tokens, std=0.001 for lm_head
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Uniform init for layers (bound matching std)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5

        for block in self.transformer:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer residual lambdas: stronger residual at early layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                loss_reduction: str = 'mean') -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: Token ids of shape (batch, sequence)
            targets: Optional target ids for loss computation
            loss_reduction: 'mean' or 'sum' for cross-entropy

        Returns:
            Logits of shape (batch, sequence, vocab_size)
            OR loss scalar if targets provided
        """
        B, T = idx.size()

        # Embed tokens
        x = self.wte(idx)
        x0 = x  # Save for x0 residual

        # Forward through transformer blocks with per-layer lambdas
        backout_layer = self.config.n_layer // 2
        x_backout = None

        for i, block in enumerate(self.transformer):
            x = x * self.resid_lambdas[i] + self.x0_lambdas[i] * x0
            x = block(x)
            if i == backout_layer:
                x_backout = x

        # Backout: subtract mid-layer features (per nanochat)
        if x_backout is not None:
            x = x - 0.2 * x_backout

        # Final norm and projection
        x = rms_norm(x)
        logits = self.lm_head(x)

        # Softcap for stable logits (per nanochat)
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss

        return logits

    @torch.no_grad()
    def generate(self, tokens: list, max_tokens: int, temperature: float = 1.0,
                 top_k: Optional[int] = None, seed: int = 42):
        """Autoregressive generation."""
        rng = torch.Generator(device=self.wte.weight.device) if temperature > 0 else None
        if rng is not None:
            rng.manual_seed(seed)

        ids = torch.tensor([tokens], dtype=torch.long, device=self.wte.weight.device)

        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)

            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()

    def num_parameters(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
