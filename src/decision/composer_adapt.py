"""Composer factory — train an EmbeddingComposer end-to-end with a head.

Component 38. Everything except the trainer exists (DecisionModel has the
slot, save/load round-trips composer.pt, compose_items is the shared path);
this module is the missing piece: given per-item embeddings for multi-item
training states, train a composer whose composed state vector improves the
DECISION (objective b — joint fusion, the Multimodal DL taxonomy's name for
end-to-end composition), and report the earn-the-slot margin vs mean-pool.

No encoder runs here — the caller passes item embeddings (the kernel encodes
with the real EG2; unit tests pass synthetic arrays). Torch-tier: the fit is
Kaggle's job (ADR-0006); this file is local-unit-testable with small
synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.decision.composer import ComposerConfig, EmbeddingComposer

#: The non-inferiority floor (spec criterion 3): a composer that regresses
#: the pooled baseline by more than this does not ship — earn-the-slot.
NON_INFERIORITY_MARGIN = 0.05


@dataclass
class ComposerAdaptConfig:
    """Everything needed to reproduce a composer-adaptation run."""
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64                # composed-vector width (small for local tests)
    epochs: int = 30
    lr: float = 5e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    seed: int = 0
    fit_frac: float = 0.6           # rest splits evenly into cal / test
    cal_frac: float = 0.2


@dataclass
class ComposerAdaptResult:
    composer: EmbeddingComposer
    report: Dict
    classes: List = field(default_factory=list)


def _compose(composer: EmbeddingComposer, item_embs: Sequence[np.ndarray]):
    """Compose a batch of item-embedding lists -> (B, n_embd) state vectors.

    Delegates each row to ``composer.compose_items`` — the single canonical
    composition path shared with ``DecisionModel`` — so the composer factory
    cannot drift into a second composition semantics (the E6 twin-vs-shipped
    failure mode). Placeholder text items of the right length drive the
    all-text modality tags; padding is handled per-row, not batched, so a
    short state never sees a padded longer one.
    """
    from src.decision.composer import compose_items
    rows = []
    for e in item_embs:
        e = np.asarray(e, dtype=np.float32)
        items = ["x"] * int(e.shape[0])          # all-text modality tags
        rows.append(compose_items(items, torch.as_tensor(e), composer))
    return torch.stack(rows)


def _pool(item_embs: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-pool baseline — the honest baseline the composer must beat."""
    return np.stack([np.asarray(e, dtype=np.float64).mean(axis=0)
                     for e in item_embs])


def _fit_head(z_train: np.ndarray, y_train: np.ndarray, n_classes: int,
              epochs: int, lr: float, batch_size: int, seed: int,
              weight_decay: float):
    """A linear head fit on fixed vectors (the pooled baseline's head)."""
    torch.manual_seed(seed)
    head = nn.Linear(int(z_train.shape[1]), n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr,
                           weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    Z = torch.as_tensor(z_train, dtype=torch.float32)
    Y = torch.as_tensor(y_train, dtype=torch.long)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(Z))
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            loss = lossf(head(Z[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def adapt_composer(item_embs: Sequence[np.ndarray], y: Sequence,
                   cfg: Optional[ComposerAdaptConfig] = None
                   ) -> ComposerAdaptResult:
    """Train a composer end-to-end with a head; report the earn-the-slot margin.

    Args:
        item_embs: per-state item embeddings — a list of ``(T_i, D)`` arrays,
            one per training state. ``T_i`` may vary (padded internally).
        y: labels per state (any hashable; sorted-unique becomes the classes).
        cfg: ``ComposerAdaptConfig``.

    Returns:
        ``ComposerAdaptResult(composer, report, classes)``. The report carries
        the composed and pooled baselines' train/test accuracy and the
        ``margin_test`` — positive means the composer earned its slot,
        within ``[-NON_INFERIORITY_MARGIN, 0]`` means non-inferior, below the
        floor means it does not ship.

    Objective (b): the composer and the decision head train together —
    compose → head → cross-entropy — so the composer learns the composition
    that improves the decision, not one that merely preserves the pooled
    signal. Identity-at-init (the composer's zero-init output projections)
    means epoch 0 *is* the mean-pool baseline; training departs from it.
    """
    cfg = cfg or ComposerAdaptConfig()
    if len(item_embs) == 0:
        raise ValueError("adapt_composer needs at least one state")
    y_arr = np.asarray(y)
    if len(y_arr) != len(item_embs):
        raise ValueError("item_embs and y must be the same length")

    classes = sorted(set(y_arr.tolist()))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([cls_idx[v] for v in y_arr], dtype=np.int64)
    K = len(classes)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(item_embs))
    n_fit = int(len(item_embs) * cfg.fit_frac)
    n_cal = int(len(item_embs) * cfg.cal_frac)
    fit_idx = perm[:n_fit]
    cal_idx = perm[n_fit:n_fit + n_cal]
    test_idx = perm[n_fit + n_cal:]

    D = int(item_embs[0].shape[1])
    torch.manual_seed(cfg.seed)
    composer = EmbeddingComposer(ComposerConfig(
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
        embd_dim=D, max_items=max(int(e.shape[0]) for e in item_embs)))
    head = nn.Linear(cfg.n_embd, K)
    params = list(composer.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.CrossEntropyLoss()

    def composed_logits(idx):
        return head(_compose(composer, [item_embs[i] for i in idx]))

    for _ in range(cfg.epochs):
        composer.train()
        head.train()
        order = rng.permutation(len(fit_idx))
        for b in range(0, len(order), cfg.batch_size):
            sel = fit_idx[order[b:b + cfg.batch_size]]
            loss = lossf(composed_logits(sel),
                         torch.as_tensor(y_idx[sel], dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()

    composer.eval()
    head.eval()

    def composed_acc(idx):
        if len(idx) == 0:
            return None
        with torch.no_grad():
            pred = composed_logits(idx).argmax(1).numpy()
        return float((pred == y_idx[idx]).mean())

    # pooled baseline: a head of the same capacity fit on mean-pooled vectors
    pooled = _pool(item_embs)
    pooled_head = _fit_head(pooled[fit_idx], y_idx[fit_idx], K,
                            cfg.epochs, cfg.lr, cfg.batch_size, cfg.seed,
                            cfg.weight_decay)

    def pooled_acc(idx):
        if len(idx) == 0:
            return None
        with torch.no_grad():
            Z = torch.as_tensor(pooled[idx], dtype=torch.float32)
            pred = pooled_head(Z).argmax(1).numpy()
        return float((pred == y_idx[idx]).mean())

    c_test, p_test = composed_acc(test_idx), pooled_acc(test_idx)
    margin = (None if c_test is None or p_test is None else c_test - p_test)
    report = {
        "n_total": len(item_embs), "n_classes": K,
        "splits": {"fit": len(fit_idx), "cal": len(cal_idx),
                   "test": len(test_idx), "seed": cfg.seed},
        "composer": {"n_layer": cfg.n_layer, "n_head": cfg.n_head,
                     "n_embd": cfg.n_embd, "embd_dim": D,
                     "max_items": composer.config.max_items,
                     "params": composer.n_parameters()},
        "composed": {"train_acc": composed_acc(fit_idx),
                     "test_acc": c_test},
        "pooled": {"train_acc": pooled_acc(fit_idx),
                   "test_acc": p_test},
        "margin_test": margin,
        "non_inferior": (margin is not None
                         and margin >= -NON_INFERIORITY_MARGIN),
        "earns_slot": margin is not None and margin > 0,
    }
    return ComposerAdaptResult(composer=composer, report=report,
                               classes=classes)
