"""DecisionHead — the Jev-style typed output layer.

Ported from jev-stack's Phase B heads (`src/head.py`, `src/interaction_head.py`)
into a single class with two scoring modes:

- ``interaction`` — dual-encoder scorer. Each option is embedded once by the
  frozen encoder and scored against the state through
  ``phi(s, o) = [s, o, s * o]`` → hidden → **zero-bias** output. A *new* option
  at request time needs no retraining — it is just another ``o``.
- ``fingerprint`` — per-option learned vector, ``score = <s, f_i>``, zero-bias.
  Cheapest (options needn't be embedded at request time) but the state never
  sees the option, and labels are fixed at fit time.

Design commitments (from the papers via jev-stack ADR-002):

- **Per-option isolated scoring** — order invariance by construction:
  permuting options can only permute the scores.
- **Zero-bias final layer** — no learned class prior leaks into confidence;
  ``max_score`` below threshold means *unseen*, so the model can abstain
  instead of guessing.
- **Log-score training** — a strictly proper scoring rule, whose unique minimum
  is the teacher's true conditional distribution (RLCD's objective in
  differentiable form). Temperature is fitted post-hoc on held-out data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.decision import metrics
from src.decision.schema import DecisionExample, Prediction, normalized_entropy


class _Adam:
    """Minimal Adam over a flat parameter dict (no nn.Module plumbing)."""

    def __init__(self, params: Dict[str, torch.Tensor], lr: float,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params, self.lr, (self.b1, self.b2), self.eps = params, lr, betas, eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    def step(self):
        self.t += 1
        for k, p in self.params.items():
            if p.grad is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * p.grad
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * p.grad * p.grad
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


def interaction_features(state: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
    """``phi(s, o) = [s, o, s * o]`` — state (n, d) or (n,1,d), options (n,k,d) → (n,k,3d)."""
    if state.dim() == 2:
        state = state.unsqueeze(1)
    return torch.cat([state.expand(-1, options.shape[1], -1), options, state * options], dim=-1)


class DecisionHead:
    """Zero-bias typed decision head over state + option embeddings.

    Args:
        dim: State/option embedding dimension.
        mode: ``"interaction"`` or ``"fingerprint"``.
        hidden: Hidden width of the interaction MLP (interaction mode).
        qtypes: Question types served; each gets its own scorer.
        seed: Initialization seed.
        normalize_fingerprints: L2-normalize fingerprints when scoring, so
            ``score_j`` is an exact cosine. Zero-Bias Corollary 1 warns that
            magnitude variance biases the layer toward specific classes; this is
            that paper's second remedy (the first is magnitude regularization).
    """

    MODES = ("interaction", "fingerprint")

    def __init__(self, dim: int = 768, mode: str = "interaction", hidden: int = 64,
                 qtypes: Iterable[str] = ("choice", "score", "noul"), seed: int = 0,
                 normalize_fingerprints: bool = False):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.dim, self.mode, self.hidden = dim, mode, hidden
        self.normalize_fingerprints = bool(normalize_fingerprints)
        self.qtypes = tuple(qtypes)
        self.temperature: Dict[str, float] = {q: 1.0 for q in self.qtypes}
        self.abstention_threshold: Dict[str, float] = {}
        self._mlp: Dict[str, Dict[str, torch.Tensor]] = {}       # interaction
        self._fp: Dict[Tuple[str, str], torch.Tensor] = {}       # fingerprint
        if mode == "interaction":
            for q in self.qtypes:
                self._init_mlp(q, seed)

    # ── parameters ───────────────────────────────────────────────────────

    def _init_mlp(self, qtype: str, seed: int = 0) -> None:
        if qtype in self._mlp:
            return
        g = torch.Generator().manual_seed(seed + len(self._mlp))
        self._mlp[qtype] = {
            "W1": (torch.randn(self.hidden, 3 * self.dim, generator=g)
                   * (3 * self.dim) ** -0.5).requires_grad_(True),
            "b1": torch.zeros(self.hidden, requires_grad=True),
            # Deliberately no b2: the final layer is zero-bias.
            "W2": (torch.randn(1, self.hidden, generator=g)
                   * self.hidden ** -0.5).requires_grad_(True),
        }

    def _ensure_fp(self, qtype: str, label: str, seed: Optional[int] = None) -> torch.Tensor:
        key = (qtype, label)
        if key not in self._fp:
            g = torch.Generator().manual_seed(seed) if seed is not None else None
            self._fp[key] = (torch.randn(self.dim, generator=g) * 0.02).requires_grad_(True)
        return self._fp[key]

    def _fp_rows(self, qtype: str, labels: Sequence[str],
                 ensure: bool = True) -> torch.Tensor:
        """Stack the fingerprints for ``labels``, optionally normalized to unit length.

        Normalization makes ``score_j`` an exact cosine, which is the Zero-Bias
        paper's second remedy for magnitude bias (Corollary 1).
        """
        getter = (lambda l: self._ensure_fp(qtype, l)) if ensure else (lambda l: self._fp[(qtype, l)])
        rows = torch.stack([getter(label) for label in labels])
        return F.normalize(rows, dim=1) if self.normalize_fingerprints else rows

    @torch.no_grad()
    def initialize_fingerprints(self, qtype: str, labels: Sequence[str],
                                option_embeddings: Sequence, normalize: bool = True) -> None:
        """Start each option's fingerprint at its own encoder embedding (ZS-LP).

        Zero-shot cosine is already a strong baseline, so random initialization
        forces training to rediscover it from noise. Initializing at the option
        prototype makes the head a *learned correction* to that baseline instead
        of a from-scratch classifier. See docs/ARCHITECTURE-SHARPENING.md S3.
        """
        if len(labels) != len(option_embeddings):
            raise ValueError("labels and option_embeddings must be the same length")
        for label, vec in zip(labels, option_embeddings):
            v = self._as_vec(vec)
            if v.numel() != self.dim:
                raise ValueError(f"option embedding has {v.numel()} dims, head expects {self.dim}")
            if normalize:
                v = F.normalize(v, dim=0)
            self._fp[(qtype, label)] = v.clone().requires_grad_(True)

    @property
    def final_layer_has_bias(self) -> bool:
        """Always False — the property the design is named for."""
        return any("b2" in p for p in self._mlp.values())

    def parameters(self) -> Dict[str, torch.Tensor]:
        out = {f"{q}.{k}": v for q, p in self._mlp.items() for k, v in p.items()}
        out.update({f"fp.{q}.{l}": v for (q, l), v in self._fp.items()})
        return out

    def n_parameters(self) -> int:
        return sum(v.numel() for v in self.parameters().values())

    def labels_for(self, qtype: str) -> List[str]:
        return sorted(l for (q, l) in self._fp if q == qtype)

    # ── inference ────────────────────────────────────────────────────────

    @staticmethod
    def _as_vec(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1))

    @torch.no_grad()
    def raw_scores(self, state_embedding, option_embeddings=None, *,
                   qtype: str, labels: Sequence[str]) -> torch.Tensor:
        """One raw score per option — no bias in the final layer.

        ``option_embeddings`` may be None in fingerprint mode (fingerprints are
        the learned option vectors); interaction mode requires them.
        """
        s = self._as_vec(state_embedding)
        if s.numel() != self.dim:
            raise ValueError(f"state embedding has {s.numel()} dims, head expects {self.dim}")
        if self.mode == "fingerprint":
            return self._fp_rows(qtype, labels) @ s
        if option_embeddings is None:
            raise ValueError("interaction mode requires option_embeddings")
        o = torch.as_tensor(np.stack([np.asarray(x, dtype=np.float32).reshape(-1)
                                      for x in option_embeddings]))
        if o.shape[1] != self.dim:
            raise ValueError(f"option embeddings have {o.shape[1]} dims, head expects {self.dim}")
        self._init_mlp(qtype)
        feats = interaction_features(s.unsqueeze(0), o.unsqueeze(0))[0]
        p = self._mlp[qtype]
        return (torch.relu(feats @ p["W1"].T + p["b1"]) @ p["W2"].T).squeeze(-1)

    @torch.no_grad()
    def predict(self, state_embedding, option_embeddings=None, *,
                qtype: str, labels: Sequence[str],
                threshold: Optional[float] = None) -> Prediction:
        """Score one state against one question's options."""
        if not labels:
            raise ValueError(f"No labels for qtype {qtype!r}")
        scores = self.raw_scores(state_embedding, option_embeddings,
                                 qtype=qtype, labels=labels)
        probs = F.softmax(scores / self.temperature.get(qtype, 1.0), dim=0)
        probs_list = [float(p) for p in probs]
        max_score = float(scores.max())

        tau = threshold if threshold is not None else self.abstention_threshold.get(qtype)
        abstention = ("unevaluated" if tau is None
                      else "abstained" if max_score < tau else "passed")
        return Prediction(
            qtype=qtype,
            labels=list(labels),
            probabilities={l: p for l, p in zip(labels, probs_list)},
            answer_confidence=float(probs.max()),
            entropy_confidence=normalized_entropy(probs_list),
            abstention=abstention,
            abstention_threshold=tau,
            max_score=max_score,
        )

    # ── training ─────────────────────────────────────────────────────────

    def _grouped(self, examples: Sequence[DecisionExample]):
        """Group by (qtype, label-tuple) so each group is one batched matmul.

        Questions differ in option count; grouping also pins the option column
        order inside a group.
        """
        groups: Dict[Tuple[str, Tuple[str, ...]], List[DecisionExample]] = {}
        for ex in examples:
            groups.setdefault((ex.qtype, tuple(ex.labels)), []).append(ex)
        return groups

    def _scores_batch(self, qtype, labels, states, options):
        """Differentiable scores: states (b,d), options (b,k,d)|None → (b,k)."""
        if self.mode == "fingerprint":
            return states @ self._fp_rows(qtype, labels, ensure=False).T
        feats = interaction_features(states, options)
        p = self._mlp[qtype]
        return (torch.relu(feats @ p["W1"].T + p["b1"]) @ p["W2"].T).squeeze(-1)

    def fit(self, examples: Sequence[DecisionExample], epochs: int = 300,
            lr: float = 0.01, batch_size: int = 32, l2: float = 0.0,
            seed: int = 0, verbose: bool = False) -> Dict[str, Any]:
        """Fit by minimizing the log score against soft targets.

        Returns a dict with the loss history — the strictly proper objective
        means lower is unambiguously better.
        """
        if not examples:
            raise ValueError("No training examples supplied")

        prepared = []
        for (qtype, labels), group in self._grouped(examples).items():
            states = torch.as_tensor(np.stack([ex.state_embedding for ex in group]))
            options = None
            if self.mode == "interaction":
                if any(ex.option_embeddings is None for ex in group):
                    raise ValueError("interaction mode requires option_embeddings")
                options = torch.as_tensor(np.stack([
                    np.stack(ex.option_embeddings) for ex in group]))
            else:
                for l in labels:
                    self._ensure_fp(qtype, l, seed=seed)
            targets = torch.as_tensor(np.stack([np.asarray(ex.target, dtype=np.float32)
                                                for ex in group]))
            if self.mode == "interaction":
                self._init_mlp(qtype, seed)
            prepared.append((qtype, labels, states, options, targets))

        optim = _Adam(self.parameters(), lr=lr)
        history: List[float] = []
        for epoch in range(epochs):
            rng = np.random.default_rng(seed + epoch)
            epoch_loss, seen = 0.0, 0
            for qtype, labels, states, options, targets in prepared:
                order = rng.permutation(len(states))
                for start in range(0, len(states), batch_size):
                    idx = torch.as_tensor(order[start:start + batch_size])
                    optim.zero_grad()
                    scores = self._scores_batch(qtype, labels, states[idx],
                                                options[idx] if options is not None else None)
                    log_probs = F.log_softmax(scores / self.temperature.get(qtype, 1.0), dim=1)
                    loss = -(targets[idx] * log_probs).sum(1).mean()
                    if l2:
                        loss = loss + l2 * sum(float((v ** 2).sum())
                                               for v in self.parameters().values())
                    loss.backward()
                    optim.step()
                    epoch_loss += float(loss.detach()) * len(idx)
                    seen += len(idx)
            history.append(epoch_loss / max(1, seen))
            if verbose and (epoch + 1) % 25 == 0:
                print(f"epoch {epoch + 1:4d}  loss {history[-1]:.4f}")
        return {"loss_history": history, "final_loss": history[-1],
                "n_examples": len(examples), "n_parameters": self.n_parameters()}

    def fit_temperature(self, examples: Sequence[DecisionExample],
                        grid: Optional[Sequence[float]] = None) -> Dict[str, float]:
        """One temperature per qtype on held-out data; argmax untouched."""
        grid = list(grid) if grid is not None else [
            round(10 ** e, 3) for e in np.linspace(-0.3, 0.7, 21)]
        fitted: Dict[str, float] = {}
        for qtype in sorted({ex.qtype for ex in examples}):
            subset = [ex for ex in examples if ex.qtype == qtype]
            best_t, best_value = 1.0, float("inf")
            for t in grid:
                probs, targets = [], []
                for ex in subset:
                    scores = self.raw_scores(ex.state_embedding, ex.option_embeddings,
                                             qtype=ex.qtype, labels=ex.labels)
                    probs.append(F.softmax(scores / t, dim=0).numpy())
                    targets.append(np.asarray(ex.target, dtype=np.float64))
                value = metrics.weighted_metric(probs, targets, metrics.log_score)
                if value < best_value:
                    best_t, best_value = float(t), value
            self.temperature[qtype] = best_t
            fitted[qtype] = best_t
        return fitted

    def fit_abstention_threshold(self, examples: Sequence[DecisionExample],
                                 qtype: str, target_precision: float = 0.9,
                                 min_coverage: float = 0.1) -> float:
        """Threshold on raw max score, chosen by *achieved* precision on kept items."""
        subset = [ex for ex in examples if ex.qtype == qtype]
        if not subset:
            raise ValueError(f"No held-out examples for qtype {qtype!r}")
        scores, correct = [], []
        for ex in subset:
            pred = self.predict(ex.state_embedding, ex.option_embeddings,
                                qtype=ex.qtype, labels=ex.labels)
            scores.append(pred.max_score)
            correct.append(float(pred.choice == ex.labels[int(np.argmax(ex.target))]))
        scores_arr, correct_arr = np.asarray(scores), np.asarray(correct)
        best = float(scores_arr.min())
        for tau in np.unique(scores_arr):
            mask = scores_arr >= tau
            if float(mask.mean()) < min_coverage:
                continue
            best = float(tau)
            if float(correct_arr[mask].mean()) >= target_precision:
                break
        self.abstention_threshold[qtype] = best
        return best

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": "decision_head",
            "dim": self.dim, "mode": self.mode, "hidden": self.hidden,
            "qtypes": list(self.qtypes),
            "normalize_fingerprints": self.normalize_fingerprints,
            "temperature": self.temperature,
            "abstention_threshold": self.abstention_threshold,
            "mlp": {q: {k: v.detach().tolist() for k, v in p.items()}
                    for q, p in self._mlp.items()},
            "fp": {f"{q}||{l}": v.detach().tolist() for (q, l), v in self._fp.items()},
        }))

    @classmethod
    def load(cls, path) -> "DecisionHead":
        d = json.loads(Path(path).read_text())
        head = cls(dim=d["dim"], mode=d["mode"], hidden=d["hidden"],
                   qtypes=d.get("qtypes", ("choice", "score", "noul")),
                   normalize_fingerprints=d.get("normalize_fingerprints", False))
        head._mlp = {q: {k: torch.tensor(v, dtype=torch.float32).requires_grad_(True)
                         for k, v in p.items()} for q, p in d["mlp"].items()}
        head._fp = {tuple(k.split("||", 1)): torch.tensor(v, dtype=torch.float32).requires_grad_(True)
                    for k, v in d["fp"].items()}
        head.temperature = dict(d.get("temperature", {}))
        head.abstention_threshold = dict(d.get("abstention_threshold", {}))
        return head
