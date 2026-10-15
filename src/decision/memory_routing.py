"""Type-routed memory plane (MemoType mechanism) — a frontier candidate for the
weak-core anchor.

MemoType's claim (from the owner's library): a single shared-encoder similarity
metric has a provable ceiling 1 − (C−1)δ/2C that tightens as the type space
grows; partitioning the memory by the type the core already predicts and scoring
each partition with a strategy matched to it beats a single pooled metric. The
router needs no generation — it's pure arithmetic over the frozen encoder.

This module implements the memory plane as a **batched, memory-bounded**
retrieval: query rows are processed in blocks so the similarity matrix never
materializes at full size (a 5427×43410 float64 matrix is ~1.9 GB — enough to
freeze a workstation). Full-dataset runs belong on Kaggle; local use caps the
batch/ref sizes.

Two modes compared by the frontier leg:
- ``plain``: pooled kNN over the whole memory.
- ``type_routed``: partition the memory by the core's top-T predicted types,
  retrieve within each partition, weight votes by the type's predicted rank.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class MemoryPlane:
    """The reference memory: unit-normalized embeddings + their labels."""
    ref_vecs: np.ndarray      # (n, d) unit-normalized
    ref_y: np.ndarray         # (n,) int labels

    @classmethod
    def build(cls, emb: np.ndarray, y: Sequence[int]) -> "MemoryPlane":
        v = np.asarray(emb, dtype=np.float32)
        v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        return cls(ref_vecs=v, ref_y=np.asarray(y, dtype=int))

    def _partition(self, label: int) -> np.ndarray:
        return np.where(self.ref_y == label)[0]


def _batched_topk(sims: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices per row of a (b, n) similarity block."""
    k = min(k, sims.shape[1])
    if k <= 0:
        return np.empty((sims.shape[0], 0), dtype=int)
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    # order the k by similarity (argpartition doesn't sort)
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return part[rows, order]


def plain_knn(plane: MemoryPlane, query_vecs: np.ndarray, k: int = 10,
              n_classes: Optional[int] = None,
              batch: int = 256) -> np.ndarray:
    """Pooled kNN over the whole memory. Batched to bound memory."""
    q = np.asarray(query_vecs, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    if n_classes is None:
        n_classes = int(plane.ref_y.max()) + 1
    out = np.empty(len(q), dtype=int)
    for s in range(0, len(q), batch):
        qb = q[s:s + batch]
        sims = qb @ plane.ref_vecs.T
        top = _batched_topk(sims, k)
        for i in range(len(qb)):
            out[s + i] = np.bincount(plane.ref_y[top[i]],
                                     minlength=n_classes).argmax()
    return out


def type_routed_knn(plane: MemoryPlane, query_vecs: np.ndarray,
                    head_probs: np.ndarray, top_t: int = 3, k: int = 10,
                    n_classes: Optional[int] = None,
                    batch: int = 256) -> np.ndarray:
    """MemoType-style retrieval: partition memory by the core's top-T predicted
    types, retrieve within each partition, weight votes by the type's rank.

    Votes are weighted ``1/(rank+1)`` so the core's top prediction counts most,
    but lower-ranked partitions still contribute — the type-routing signal.
    """
    q = np.asarray(query_vecs, dtype=np.float32)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    if n_classes is None:
        n_classes = int(plane.ref_y.max()) + 1
    out = np.empty(len(q), dtype=int)
    # precompute partitions once (indices per label), not per query
    parts = {t: plane._partition(t) for t in range(n_classes)}
    for s in range(0, len(q), batch):
        qb = q[s:s + batch]
        sims = qb @ plane.ref_vecs.T
        for i in range(len(qb)):
            votes = np.zeros(n_classes, dtype=np.float32)
            types = np.argsort(-head_probs[s + i])[:top_t]
            for rank, t in enumerate(types):
                pidx = parts[int(t)]
                if len(pidx) == 0:
                    continue
                ps = sims[i, pidx]
                kk = min(k, len(pidx))
                top = np.argpartition(-ps, kk - 1)[:kk]
                w = 1.0 / (rank + 1)
                np.add.at(votes, plane.ref_y[pidx[top]], w)
            out[s + i] = votes.argmax() if votes.sum() > 0 else \
                int(head_probs[s + i].argmax())
    return out


def blend_with_head(knn_pred: np.ndarray, head_pred: np.ndarray,
                    head_probs: np.ndarray, trust: float = 0.5) -> np.ndarray:
    """Additive combination: take the kNN vote when the head is unsure
    (below ``trust`` confidence), else the head's own answer. Strictly
    additive — never overrides a confident head."""
    conf = head_probs.max(axis=1)
    return np.where(conf >= trust, head_pred, knn_pred)


def run_memory_frontier(emb_ref: np.ndarray, y_ref: np.ndarray,
                        emb_q: np.ndarray, y_q: np.ndarray,
                        head_logits_q: np.ndarray, temperature: float,
                        n_classes: int, top_t: int = 3, k: int = 10,
                        trust: float = 0.5, batch: int = 256,
                        hard_quantile: float = 0.25) -> Dict[str, Any]:
    """The frontier experiment: does a type-routed memory plane help a WEAK core?

    Compares, on the same query set: the head alone, plain pooled kNN,
    type-routed kNN, and the additive blend — overall and on the hard quartile
    (lowest head confidence), which is where the frontier diagnostic said the
    bottleneck lives. Returns a dict the notebook leg prints.

    ``emb_ref``/``y_ref`` build the memory plane; ``emb_q``/``y_q`` are held-out
    queries; ``head_logits_q``/``temperature`` give the head's confidences (the
    type router's input and the difficulty axis).
    """
    from src.decision.frontier_diag import diagnose_frontier

    plane = MemoryPlane.build(emb_ref, y_ref)
    P = _softmax(np.asarray(head_logits_q, dtype=np.float64), temperature)
    head_pred = P.argmax(1)
    conf = P.max(1)
    correct = head_pred == y_q

    plain = plain_knn(plane, emb_q, k=k, n_classes=n_classes, batch=batch)
    routed = type_routed_knn(plane, emb_q, P, top_t=top_t, k=k,
                             n_classes=n_classes, batch=batch)
    blend = blend_with_head(routed, head_pred, P, trust=trust)

    # hard quartile: lowest head confidence (the #17 bottleneck population)
    thr = np.quantile(conf, hard_quantile)
    hard = conf <= thr

    def _acc(pred):
        return {"overall": round(float((pred == y_q).mean()), 4),
                "hard_quartile": round(float((pred[hard] == y_q[hard]).mean()), 4)
                if hard.any() else float("nan")}

    # depth structure of each variant (the #17 signature)
    def _slope(pred):
        rec = {"top1": list(pred == y_q), "top_prob": list(conf),
               "intent": [int(v) for v in y_q],
               "action": ["answer"] * len(y_q), "covered": [True] * len(y_q)}
        return round(diagnose_frontier(rec, P, n_buckets=4).depth_slope, 4)

    return {
        "n": int(len(y_q)), "n_hard": int(hard.sum()), "k": k, "top_t": top_t,
        "head": {**_acc(head_pred), "depth_slope": _slope(head_pred)},
        "plain_knn": {**_acc(plain), "depth_slope": _slope(plain)},
        "type_routed": {**_acc(routed), "depth_slope": _slope(routed)},
        "blend": {**_acc(blend), "depth_slope": _slope(blend)},
        "hard_quartile_gain_vs_head": round(
            float((blend[hard] == y_q[hard]).mean()
                  - (head_pred[hard] == y_q[hard]).mean()), 4),
    }


def _softmax(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / max(t, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
