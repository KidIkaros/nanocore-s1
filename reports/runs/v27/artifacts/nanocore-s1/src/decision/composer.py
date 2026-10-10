"""EmbeddingComposer — the karpathy component of NanoCore-S1.

EmbeddingGemma 2 turns every state item — a text chunk, an image, an audio
clip — into a 768-d vector in one shared space. A multi-item state is therefore
a *sequence of embedding tokens*. This module is the small bidirectional
transformer that composes that sequence into one state vector.

It is deliberately **not** autoregressive: the state is composed, not
generated. Attention is full bidirectional (encoder-style), which matches the
System One lineage — every production decision model (Jev, Laya) is
encoder-only.

Style follows `src/model.py` (nanochat): parameter-free RMSNorm, QK-norm,
ReLU^2 MLP, rotary positions, and zero-initialized output projections.

**Identity-at-init, by design.** The attention and MLP output projections start
at zero, so each block — and therefore the whole composer — is the identity map
at initialization: early training fits the head against pooled encoder output,
and the composer learns to *depart* from that baseline rather than having to
unlearn noise. (This confused the repo audit into thinking the blocks were
broken; they are not — a composed state only differs from pooled input once
training moves the projections.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.decision.items import modality_of


#: Modality tags produced by ``items.modality_of``; index = type-embedding row.
MODALITY_IDS = {"text": 0, "code": 1, "image": 2, "video": 3, "audio": 4}
N_MODALITIES = len(MODALITY_IDS)


def compose_items(items, item_embs: "torch.Tensor",
                  composer: Optional["EmbeddingComposer"] = None) -> "torch.Tensor":
    """Item embeddings → one ``(dim,)`` state vector — the single composition path.

    ``None`` composer is masked mean-pooling, the baseline every composer must
    beat. This is the *only* implementation of that choice: the shipped
    ``DecisionModel`` and the legacy ``NanoCoreS1`` both call here, so the two
    paths cannot drift into different composition semantics (the failure mode
    the E6 twin-vs-shipped gap demonstrated).
    """
    if composer is None:
        return item_embs.float().mean(dim=0)
    modality_ids = torch.tensor(
        [MODALITY_IDS[modality_of(i)] for i in items],
        dtype=torch.long, device=item_embs.device)
    return composer(item_embs.float().unsqueeze(0), modality_ids.unsqueeze(0))[0]


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free RMSNorm (per nanochat)."""
    return F.rms_norm(x, (x.size(-1),))


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (B, n_head, T, head_dim)."""
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Rotary(nn.Module):
    """Precomputed rotary frequencies for sequences up to ``max_items``."""

    def __init__(self, head_dim: int, max_items: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2)))
        t = torch.arange(max_items, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                       # (T, d/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-2)
        return _apply_rotary(x, self.cos[:T], self.sin[:T])


class Attention(nn.Module):
    """Bidirectional multi-head self-attention over item embeddings."""

    def __init__(self, config: "ComposerConfig"):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim, config.max_items)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = rms_norm(q), rms_norm(k)          # QK norm (per nanochat)
        q, k = self.rotary(q), self.rotary(k)

        attn_mask = None
        if pad_mask is not None:                  # pad_mask: (B, T) True = real item
            attn_mask = pad_mask[:, None, None, :]  # key-padding mask, bidirectional

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    """ReLU^2 MLP (per nanochat), 4x expansion."""

    def __init__(self, config: "ComposerConfig"):
        super().__init__()
        hidden = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    """Pre-norm encoder block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

    def __init__(self, config: "ComposerConfig"):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), pad_mask)
        return x + self.mlp(rms_norm(x))


@dataclass
class ComposerConfig:
    """Single-dial config in the nanochat spirit; defaults suit 768-d EG2 items."""
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 768
    embd_dim: int = 768          # EG2 output dim (projected if != n_embd)
    max_items: int = 64
    pooling: str = "mean"        # "mean" (masked) or "cls" (prepended token)
    resid_zero_init: bool = True # zero-init attn/mlp output projections

    @classmethod
    def from_depth(cls, depth: int, **kw) -> "ComposerConfig":
        return cls(n_layer=depth, **kw)


class EmbeddingComposer(nn.Module):
    """Sequence of item embeddings → one state vector.

    Args:
        config: ComposerConfig. ``pooling="mean"`` masked-averages real items;
            ``"cls"`` prepends a learned token and reads it out.
    """

    def __init__(self, config: Optional[ComposerConfig] = None):
        super().__init__()
        self.config = config or ComposerConfig()
        c = self.config
        if c.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {c.pooling!r}")

        self.in_proj = (nn.Linear(c.embd_dim, c.n_embd, bias=False)
                        if c.embd_dim != c.n_embd else nn.Identity())
        self.modality_embed = nn.Embedding(N_MODALITIES, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, c.n_embd)) if c.pooling == "cls" else None
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_embed.weight, std=0.01)
        if self.config.resid_zero_init:
            # nanochat convention: residual paths start at zero, so the stack is
            # the identity at init (see module docstring).
            for block in self.blocks:
                nn.init.zeros_(block.attn.c_proj.weight)
                nn.init.zeros_(block.mlp.c_proj.weight)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        embeddings: torch.Tensor,
        modality_ids: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compose item embeddings into a state vector.

        Args:
            embeddings: ``(B, T, embd_dim)`` frozen encoder output.
            modality_ids: ``(B, T)`` longs into :data:`MODALITY_IDS`; defaults to
                all-text (0).
            pad_mask: ``(B, T)`` bool, True = real item. Defaults to all real.

        Returns:
            ``(B, n_embd)`` — one composed state vector per state.
        """
        B, T, _ = embeddings.shape
        if T > self.config.max_items:
            raise ValueError(f"{T} items exceeds max_items={self.config.max_items}")
        if pad_mask is None:
            pad_mask = torch.ones(B, T, dtype=torch.bool, device=embeddings.device)
        if modality_ids is None:
            modality_ids = torch.zeros(B, T, dtype=torch.long, device=embeddings.device)

        x = self.in_proj(embeddings.float()) + self.modality_embed(modality_ids)

        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
            pad_mask = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=pad_mask.device), pad_mask], dim=1
            )
            modality_ids = torch.cat(
                [torch.zeros(B, 1, dtype=torch.long, device=modality_ids.device), modality_ids],
                dim=1,
            )

        for block in self.blocks:
            x = block(x, pad_mask)

        x = rms_norm(x)
        if self.cls_token is not None:
            return x[:, 0]
        # masked mean over real items
        mask = pad_mask[..., None].float()
        return (x * mask).sum(1) / mask.sum(1).clamp_min(1.0)
