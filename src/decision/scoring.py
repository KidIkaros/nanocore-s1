"""Scorers — the three-mode scorer layer.

- ``CosineScorer``: zero-shot universal scorer. Cosine similarity between the
  state vector and option embeddings, divided by a fitted temperature. One
  parameter; the right choice on saturated tasks (ADR-0008).
- ``TaskHead``: label-schema-bound fitted head (linear or MLP on the state
  embedding). The per-deployment upgrade when the headroom gate admits the
  task and labeled data exist (ADR-0011; measured: +0.168 macro-F1 on
  GoEmotions, 0.970 on 150-way CLINC150).

Both expose ``scores`` (pre-softmax) — the gate owns temperature and set
construction, so a scorer must never ship probabilities it sharpened itself.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np


def softmax_rows(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of a ``(n, k)`` or ``(k,)`` score matrix."""
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    z = z / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out = e / e.sum(axis=1, keepdims=True)
    return out


def aps_members(P: np.ndarray, qhat: float) -> list:
    """Adaptive prediction sets: smallest set whose cumulative mass >= qhat.

    ``P`` is ``(n, k)`` probabilities built at T_set (not T_prob — the
    temperature that reports probabilities is not the one that builds sets).
    """
    P = np.asarray(P, dtype=np.float64)
    srt = np.argsort(-P, axis=1)
    cum = np.cumsum(-np.sort(-P, axis=1), axis=1)
    sizes = (cum < qhat).sum(axis=1) + 1
    return [srt[i, : sizes[i]].tolist() for i in range(len(P))]


def mass_needed(P: np.ndarray, true_idx: np.ndarray) -> np.ndarray:
    """APS nonconformity: cumulative sorted mass needed to reach the true class."""
    P = np.asarray(P, dtype=np.float64)
    srt = -np.sort(-P, axis=1)
    cum = np.cumsum(srt, axis=1)
    i = np.arange(len(P))
    rank = (srt > P[i, true_idx][:, None]).sum(axis=1)
    return cum[i, np.minimum(rank, srt.shape[1] - 1)]


def qhat_from_scores(scores: np.ndarray, true_idx: np.ndarray,
                     t_set: float, alpha: float) -> float:
    """Split-conformal quantile of the APS score at ``t_set``.

    One definition, used by both the gate's calibration and the set-temperature
    search: this formula *is* the coverage guarantee, so two copies of it would
    be two chances to disagree about what the gate promises.
    """
    m = mass_needed(softmax_rows(scores, t_set), true_idx)
    n = len(m)
    return float(np.quantile(m, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)))


def fit_set_temperature(scores: np.ndarray, true_idx: np.ndarray, alpha: float,
                        grid: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                        seed: int = 0) -> float:
    """APS temperature fitted to shrink sets while keeping coverage ≥ 1 − alpha.

    ``t_set`` was fixed at 1.0 because 1.0 "measured sane" — true for a cosine
    scorer, whose similarities are already spread across the label set. It is
    *not* sane for a trained head: measured on the v18 head path, the conformal
    score (cumulative mass at the true label) sits in [0.99, 1.0] for ~75% of
    items, so the 0.90 quantile saturates at 0.9999 and every set becomes the
    whole label space — mean 31 of 150 labels at coverage 1.0000 against a 0.90
    target, which makes the guarantee vacuous and ``k_clarify`` unreachable.

    Softening spreads the mass, which spreads the score, which makes the
    quantile discriminative again (measured: t=8 → mean set 2.4, coverage
    0.9907). The grid therefore runs *upward* only — sharpening compresses the
    score further and worsens the degeneracy.

    Selection is out-of-sample by construction: the passed data is halved, the
    quantile is fitted on one half and the coverage/size trade-off measured on
    the other, so the choice cannot be bought with an in-sample quantile.

    The returned temperature only affects *efficiency*. Coverage comes from the
    quantile at whatever temperature is chosen, so falling back to ``grid[0]``
    when nothing clears the bar cannot weaken the guarantee — it just declines
    to optimise set size.
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=int).reshape(-1)
    perm = np.random.default_rng(seed).permutation(len(scores))
    half = len(perm) // 2
    q_idx, c_idx = perm[:half], perm[half:]

    best_t, best_size = float(grid[0]), np.inf
    for t in grid:
        q = qhat_from_scores(scores[q_idx], true_idx[q_idx], t, alpha)
        members = aps_members(softmax_rows(scores[c_idx], t), q)
        sizes = np.array([len(m) for m in members])
        covered = np.mean([true_idx[c_idx][i] in members[i]
                           for i in range(len(members))])
        if covered >= 1 - alpha and sizes.mean() < best_size:
            best_t, best_size = float(t), float(sizes.mean())
    return best_t


def fit_temperature(scores: np.ndarray, true_idx: np.ndarray,
                    floor: float = 0.25, ceiling: float = 4.0,
                    n_grid: int = 32) -> float:
    """Log-loss temperature on held-out data, floored.

    The floor exists because log-loss fitting otherwise sharpens to degeneracy
    on mostly-correct data — measured: T=0.02 on CLINC150 destroyed every APS
    set (qhat≈1.0, mean set 56.7). Reported probabilities get a floored T;
    set construction uses T_set separately (ADR-0009, s1_policy_v2).
    """
    scores = np.asarray(scores, dtype=np.float64)
    true_idx = np.asarray(true_idx, dtype=int).reshape(-1)
    best_t, best_ll = 1.0, np.inf
    for t in np.linspace(floor, ceiling, n_grid):
        P = softmax_rows(scores, t)
        ll = float(-np.log(np.clip(P[np.arange(len(P)), true_idx], 1e-9, 1)).mean())
        if ll < best_ll:
            best_ll, best_t = ll, float(t)
    if best_t == ceiling:
        # The floor is a deliberate degeneracy guard (see docstring), but a
        # ceiling hit means log loss wanted more smoothing than the grid
        # allows — a bound-limited fit is exactly how a scorer once looked
        # catastrophically miscalibrated (3.800 vs 0.380; protocol.py).
        import warnings
        warnings.warn(f"fit_temperature bound-limited at the ceiling "
                      f"(t={best_t}); widen `ceiling` before trusting the fit")
    return best_t


def _fit_torch(X: np.ndarray, y: np.ndarray, *, build_net, target_fn, lossf,
               epochs: int, lr: float, batch_size: int, seed: int,
               val_fraction: float, patience: int, weight_decay: float,
               verbose: bool = False):
    """The one SGD loop every fitted scorer shares.

    Standardized inputs come in already scaled; ``target_fn`` maps the carved
    split's labels to loss targets (identity for CE, cumulative thresholds for
    CORN); ``build_net`` constructs the module *after* ``torch.manual_seed`` so
    weight init is seeded. Early stopping restores the best validation state.
    Returns ``(net, record)``.
    """
    import torch

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_fraction))
    vi, ti = perm[:n_val], perm[n_val:]
    Xt = torch.as_tensor(X[ti]); Xv = torch.as_tensor(X[vi])
    Yt = torch.as_tensor(target_fn(y[ti])); Yv = torch.as_tensor(target_fn(y[vi]))

    torch.manual_seed(seed)
    net = build_net()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, bad = np.inf, None, 0
    for epoch in range(epochs):
        order = torch.as_tensor(rng.permutation(len(Xt)))
        for start in range(0, len(Xt), batch_size):
            idx = order[start:start + batch_size]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(lossf(net(Xv), Yv))
        if verbose and (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:4d}  val_loss {vl:.4f}")
        if vl < best_val - 1e-5:
            best_val, bad = vl, 0
            best_state = {n: p.detach().clone() for n, p in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, {"best_val_loss": best_val, "epochs_run": epoch + 1,
                 "n_train": int(len(Xt))}


class CosineScorer:
    """Zero-shot scorer: cosine between the state vector and option vectors.

    Args:
        temperature: Divisor applied at probability time. ``fit_temperature``
            sets it on held-out data; the gate may override with its own
            floored fit.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def scores(self, state_vec: np.ndarray, option_vecs: np.ndarray) -> np.ndarray:
        """State ``(d,)`` and options ``(k, d)`` → ``(k,)`` cosine scores."""
        s = np.asarray(state_vec, dtype=np.float64).reshape(-1)
        o = np.asarray(option_vecs, dtype=np.float64)
        if o.ndim == 1:
            o = o[None, :]
        s = s / max(np.linalg.norm(s), 1e-12)
        o = o / np.clip(np.linalg.norm(o, axis=1, keepdims=True), 1e-12, None)
        return o @ s

    def probabilities(self, state_vec, option_vecs, temperature=None) -> np.ndarray:
        t = self.temperature if temperature is None else temperature
        return softmax_rows(self.scores(state_vec, option_vecs), t)[0]


class OrdinalScorer:
    """CORN ordinal head for ``score`` questions (ordered rubric levels).

    Ordinal regression treats the k options as thresholds, not classes: for
    levels 0..k-1 the head emits k-1 cumulative logits ``logit(P(y > j))``.
    Per-level mass is recovered by differencing sigmoids — monotone by
    construction, unlike a softmax that ignores the ordering (ADR-0010).

    ``scores`` returns log level-probabilities so the gate's
    ``softmax(x/T)`` remains a well-defined temperature operation on them.
    """

    def __init__(self):
        self._torch = None
        self.labels_: Optional[list] = None
        self.temperature = 1.0
        self._mu = self._sd = None

    def fit(self, X: np.ndarray, y_levels: np.ndarray,
            labels: Optional[Sequence[str]] = None,
            epochs: int = 40, lr: float = 1e-3, batch_size: int = 256,
            seed: int = 0, val_fraction: float = 0.1, patience: int = 8) -> Dict:
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_levels, dtype=np.int64).reshape(-1)
        k = int(y.max()) + 1
        if k < 2:
            raise ValueError("ordinal scoring needs at least two levels")
        self.labels_ = list(labels) if labels is not None else [str(i) for i in range(k)]

        self._mu = X.mean(0, keepdims=True)
        self._sd = X.std(0, keepdims=True) + 1e-6
        X = (X - self._mu) / self._sd

        def cum_targets(yy):
            j = torch.arange(k - 1)
            return (torch.as_tensor(yy)[:, None] > j[None, :]).float()

        net, rec = _fit_torch(
            X, y, build_net=lambda: nn.Linear(X.shape[1], k - 1),
            target_fn=cum_targets, lossf=nn.BCEWithLogitsLoss(),
            epochs=epochs, lr=lr, batch_size=batch_size, seed=seed,
            val_fraction=val_fraction, patience=patience, weight_decay=1e-4)
        self._torch = net
        return {**rec, "n_levels": k}

    def level_probs(self, X: np.ndarray) -> np.ndarray:
        """CORN reconstruction: cumulative sigmoids → per-level mass ``(n, k)``."""
        import torch
        if self._torch is None:
            raise RuntimeError("OrdinalScorer is not fitted")
        X = (np.asarray(X, dtype=np.float32) - self._mu) / self._sd
        with torch.no_grad():
            cum = torch.sigmoid(self._torch(torch.as_tensor(X))).numpy()  # P(y > j)
        # CORN logits aren't guaranteed monotone — project to non-increasing so
        # differencing can't produce negative mass, then renormalize.
        cum = np.minimum.accumulate(cum, axis=1)
        k = cum.shape[1] + 1
        P = np.zeros((len(cum), k))
        P[:, 0] = 1.0 - cum[:, 0]
        for j in range(1, k - 1):
            P[:, j] = cum[:, j - 1] - cum[:, j]
        P[:, k - 1] = cum[:, k - 2]
        P = np.clip(P, 1e-12, 1.0)
        return P / P.sum(axis=1, keepdims=True)

    def scores(self, state_vec: np.ndarray, option_vecs=None) -> np.ndarray:
        """Log level-probabilities — softmax(log p / T) is proper tempering."""
        P = self.level_probs(np.asarray(state_vec, dtype=np.float32).reshape(1, -1))
        return np.log(P[0])

    def expected(self, state_vec: np.ndarray, scale=None) -> float:
        """Expected rubric position (0..k-1, or caller's scale)."""
        P = self.level_probs(np.asarray(state_vec, dtype=np.float32).reshape(1, -1))[0]
        k = len(P)
        w = np.asarray(scale, dtype=np.float64) if scale is not None else np.arange(k)
        return float((P * w).sum())

    def save(self, path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"labels": self.labels_, "temperature": self.temperature,
                    "mu": self._mu, "sd": self._sd,
                    "state_dict": self._torch.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "OrdinalScorer":
        import torch
        import torch.nn as nn
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        s = cls()
        s.labels_, s.temperature = d["labels"], d.get("temperature", 1.0)
        s._mu, s._sd = d.get("mu"), d.get("sd")
        k = len(s.labels_)
        net = nn.Linear(d["state_dict"]["weight"].shape[1], k - 1)
        net.load_state_dict(d["state_dict"]); net.eval()
        s._torch = net
        return s


class TaskHead:
    """Fitted head on state embeddings — linear or one-hidden-layer MLP.

    Args:
        kind: ``"linear"`` or ``"mlp"``. Measured sweet spot: linear below
            ~16k examples, mlp at scale (s1_goemotions_scaling).
        hidden: MLP hidden width.
    """

    KINDS = ("linear", "mlp")

    def __init__(self, kind: str = "linear", hidden: int = 256):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.kind, self.hidden = kind, hidden
        self._torch = None          # fitted module (torch.nn.Module)
        self.labels_: Optional[list] = None
        self.temperature = 1.0      # fitted by the gate's calibrate step
        self._mu = self._sd = None  # input standardization, fitted on train

    # ── fitting ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_idx: np.ndarray,
            labels: Optional[Sequence[str]] = None,
            epochs: int = 40, lr: float = 1e-3, batch_size: int = 256,
            l2: float = 1e-4, seed: int = 0,
            val_fraction: float = 0.1, patience: int = 8,
            verbose: bool = False) -> Dict:
        """Cross-entropy fit with early stopping on a carved validation split.

        Returns the training record (best val loss, epochs run). Labels are
        stored so ``scores`` always returns columns in a fixed order.
        """
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_idx, dtype=np.int64).reshape(-1)
        k = int(y.max()) + 1
        self.labels_ = list(labels) if labels is not None else [str(i) for i in range(k)]
        if len(self.labels_) != k:
            raise ValueError("labels must have one entry per class index")

        self._mu = X.mean(0, keepdims=True)
        self._sd = X.std(0, keepdims=True) + 1e-6
        X = (X - self._mu) / self._sd

        if self.kind == "linear":
            def build_net():
                return nn.Linear(X.shape[1], k)
        else:
            def build_net():
                return nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(),
                                     nn.Linear(self.hidden, k))

        net, rec = _fit_torch(
            X, y, build_net=build_net, target_fn=lambda a: a,
            lossf=nn.CrossEntropyLoss(), epochs=epochs, lr=lr,
            batch_size=batch_size, seed=seed, val_fraction=val_fraction,
            patience=patience, weight_decay=l2, verbose=verbose)
        self._torch = net
        return {**rec, "n_classes": k}

    # ── inference ────────────────────────────────────────────────────────

    def logits(self, X: np.ndarray) -> np.ndarray:
        """``(n, k)`` pre-softmax logits — the gate owns temperature."""
        import torch
        if self._torch is None:
            raise RuntimeError("TaskHead is not fitted")
        X = (np.asarray(X, dtype=np.float32) - self._mu) / self._sd
        with torch.no_grad():
            return self._torch(torch.as_tensor(X)).numpy()

    def scores(self, state_vec: np.ndarray, option_vecs=None) -> np.ndarray:
        """Per-label logits for one state; ``option_vecs`` unused (schema-bound)."""
        return self.logits(np.asarray(state_vec, dtype=np.float32).reshape(1, -1))[0]

    def probabilities(self, state_vec, option_vecs=None, temperature=None) -> np.ndarray:
        t = self.temperature if temperature is None else temperature
        return softmax_rows(self.scores(state_vec), t)[0]

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": self.kind, "hidden": self.hidden,
                    "labels": self.labels_, "temperature": self.temperature,
                    "mu": self._mu, "sd": self._sd,
                    "state_dict": self._torch.state_dict()}, path)

    @classmethod
    def load(cls, path) -> "TaskHead":
        import torch
        import torch.nn as nn
        d = torch.load(Path(path), map_location="cpu", weights_only=False)
        head = cls(kind=d["kind"], hidden=d["hidden"])
        head.labels_ = d["labels"]
        head.temperature = d.get("temperature", 1.0)
        head._mu, head._sd = d.get("mu"), d.get("sd")
        k = len(head.labels_)
        if head.kind == "linear":
            net = nn.Linear(list(d["state_dict"].values())[0].shape[1], k)
        else:
            net = nn.Sequential(nn.Linear(list(d["state_dict"].values())[0].shape[1],
                                          head.hidden), nn.ReLU(),
                                nn.Linear(head.hidden, k))
        net.load_state_dict(d["state_dict"])
        net.eval()
        head._torch = net
        return head
