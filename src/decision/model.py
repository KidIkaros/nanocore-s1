"""DecisionModel — the assembled decision model.

    state items → encoder (frozen EG2) → composer (or mean-pool)
                → scorer → gate → Prediction

The encoder is injectable: contract tests pass any object with
``encode_state``/``encode_options`` returning tensors, so the package is
testable without model weights. The real encoder is only constructed on
Kaggle-class hardware (see docs/DECISION-MODEL.md, compute boundary).

This module deliberately imports no torch at top level: the shipped bundle
must load under a torch-free runtime (llama.cpp/GGUF, ADR-0007). Torch-dependent
pieces — the optional composer and torch scorer weights — are imported lazily,
only when a bundle actually carries them. The deprecated ``NanoCoreS1`` path
lives in ``legacy.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from src.decision.items import to_numpy
from src.decision.schema import Prediction, Question, State, StateItem
from src.decision.scoring import softmax_rows


def bundle_digest(bundle_dir) -> str:
    """Content-addressed SHA-256 of a saved bundle.

    This is what makes "the artifact we qualified" and "the artifact we ship"
    provably the same object — a qualification that cannot name its subject is
    not a qualification. Hashes *contents* in sorted path order, so a copy at a
    different path, or a touched mtime, does not change the digest, while any
    change to the scorer or the gate calibration does.
    """
    root = Path(bundle_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"no bundle at {root}")
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"bundle at {root} has no files")
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()


class DecisionModel:
    """The decided pipeline: encoder → scorer → gate → cache (ADR-0008/0009/0013).

    Unlike ``legacy.NanoCoreS1`` (the deprecated head path), this wires the
    measured architecture: a frozen encoder produces the state, a scorer
    produces raw scores, and the gate owns calibration, prediction sets, and
    the action.

    Args:
        encoder: ``encode_state(items) -> (n, dim)`` and
            ``encode_options(texts) -> (k, dim)`` — ``StateEncoder`` in
            production, any stub in tests.
        scorer: ``scores(state_vec, option_vecs) -> (k,)`` — ``CosineScorer``
            or a fitted ``TaskHead``/``OrdinalScorer`` (per the headroom gate,
            ADR-0011).
        gate: ``ConformalGate`` or ``None`` (ungated — probs only).
        cache: ``DecisionCache`` or ``None``.
        composer: ``EmbeddingComposer`` or ``None`` — ``None`` is masked
            mean-pooling, the honest baseline the composer must beat.
        encoder_id: Model identity string folded into cache keys.
    """

    def __init__(self, encoder=None, scorer=None, gate=None, cache=None,
                 encoder_id: str = "", composer=None):
        from src.decision.scoring import CosineScorer
        self.encoder = encoder
        self.scorer = scorer or CosineScorer()
        self.gate = gate
        self.cache = cache
        self.composer = composer
        self.encoder_id = encoder_id or getattr(encoder, "model_name", "stub")
        self._option_cache: dict = {}

    def _items(self, state) -> List:
        """State → flat item list. Bare str/dict is one item; a list or State is
        its items. Anything else is a caller error, not a silently-empty state."""
        if isinstance(state, State):
            return list(state.items)
        if isinstance(state, (str, dict)):
            return [state]
        return list(state)

    def _state_vec(self, state) -> np.ndarray:
        items = self._items(state)
        e = to_numpy(self.encoder.encode_state(items))
        if self.composer is None:
            s = e.mean(axis=0)
        else:
            from src.decision.composer import compose_items
            import torch
            s = to_numpy(compose_items(
                items, torch.as_tensor(e, dtype=torch.float32), self.composer))
        return s / max(np.linalg.norm(s), 1e-12)

    def _composer_id(self) -> str:
        """Fingerprint of the composer — the cache key must know which state
        function produced the decision, or a composed and a pooled result for
        the same state would collide."""
        if self.composer is None:
            return "meanpool"
        cached = getattr(self, "_composer_key", None)
        if cached is None:
            h = hashlib.sha256(
                json.dumps(vars(self.composer.config), sort_keys=True).encode())
            for _, p in sorted(self.composer.state_dict().items()):
                h.update(p.detach().cpu().numpy().tobytes())
            cached = "composer:" + h.hexdigest()[:12]
            self._composer_key = cached
        return cached

    def _options(self, question: Question) -> np.ndarray:
        key = tuple(question.options)
        if key not in self._option_cache:
            v = self.encoder.encode_options(list(question.options))
            self._option_cache[key] = to_numpy(v)
        return self._option_cache[key]

    def _align_scores(self, scores: np.ndarray, options) -> np.ndarray:
        """Reorder a schema-bound scorer's output to the question's order.

        ``TaskHead`` and ``OrdinalScorer`` emit logits in their fitted label
        order — ``np.unique`` order, not necessarily the question's — while the
        gate pairs ``scores[i]`` with ``options[i]`` positionally. A cosine
        scorer scores the option vectors directly and is already aligned.
        Options outside the fitted space are a hard error, not a zero column.
        """
        labels_ = getattr(self.scorer, "labels_", None)
        if labels_ is None or list(labels_) == list(options):
            return scores
        pos = {lab: i for i, lab in enumerate(labels_)}
        missing = [o for o in options if o not in pos]
        if missing:
            raise ValueError(
                f"question options outside the scorer's label space: "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
        return scores[[pos[o] for o in options]]

    def decide(self, state: Union[State, Sequence[StateItem], str],
               question: Question, policy: Optional[str] = None) -> Prediction:
        """One typed, gated decision. Cache-hit returns the stored result."""
        from src.decision.cache import decision_key
        from src.decision.schema import normalized_entropy

        key = decision_key(state, question, self.encoder_id,
                           {"policy": policy or getattr(self.gate, "policy", None),
                            "alpha": getattr(self.gate, "alpha", None),
                            "composer": self._composer_id()})
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return Prediction(**hit)

        s = self._state_vec(state)
        opts = self._options(question)
        scores = self._align_scores(self.scorer.scores(s, opts), question.options)
        g = self.gate.decide(scores, question.options) if self.gate is not None else None

        if g is not None:
            probs, action = g.probabilities, g.action
            pred_set, alpha, top, mx = g.prediction_set, g.alpha, g.top_prob, g.max_score
        else:
            t = getattr(self.scorer, "temperature", 1.0)
            p = softmax_rows(scores[None], t)[0]
            probs = {l: float(x) for l, x in zip(question.options, p)}
            action, pred_set, alpha = "unevaluated", [], None
            top, mx = float(p.max()), float(scores.max())

        pred = Prediction(
            qtype=question.qtype, labels=list(question.options),
            probabilities=probs, answer_confidence=top,
            entropy_confidence=normalized_entropy(list(probs.values())),
            max_score=mx, prediction_set=pred_set, action=action, alpha=alpha)
        if self.cache is not None:
            self.cache.put(key, {
                "qtype": pred.qtype, "labels": pred.labels,
                "probabilities": pred.probabilities,
                "answer_confidence": pred.answer_confidence,
                "entropy_confidence": pred.entropy_confidence,
                "abstention": pred.abstention,
                "abstention_threshold": pred.abstention_threshold,
                "max_score": pred.max_score,
                "prediction_set": pred.prediction_set,
                "action": pred.action, "alpha": pred.alpha})
        return pred

    # ── deployable bundle ────────────────────────────────────────────────

    def save(self, dir) -> "Path":
        """Write a portable calibrated bundle: manifest + scorer + gate.

        ``dir/manifest.json`` describes the encoder and scorer so
        ``DecisionModel.load`` can rebuild the pipeline. The encoder itself is
        not serialized — it is identified by ``model_name`` and constructed by
        the caller's backend (StateEncoder, LlamaCppEncoder, ...).
        """
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {"encoder_id": self.encoder_id, "scorer": None, "gate": None}

        if self.scorer is not None:
            if hasattr(self.scorer, "save"):
                self.scorer.save(d / "scorer.pt")
                manifest["scorer"] = {"class": self.scorer.__class__.__name__,
                                      "path": "scorer.pt"}
            else:
                manifest["scorer"] = {"class": self.scorer.__class__.__name__,
                                      "temperature": getattr(self.scorer, "temperature", 1.0)}
        if self.gate is not None:
            self.gate.save(d / "gate.json")
            manifest["gate"] = "gate.json"
        if self.composer is not None:
            import torch
            torch.save({"config": vars(self.composer.config),
                        "state_dict": self.composer.state_dict()},
                       d / "composer.pt")
            manifest["composer"] = "composer.pt"
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return d

    @classmethod
    def load(cls, dir, encoder=None, cache=None) -> "DecisionModel":
        """Rebuild a bundle. ``encoder`` is the caller's backend instance."""
        from src.decision.gate import ConformalGate
        from src.decision.scoring import CosineScorer, OrdinalScorer, TaskHead

        d = Path(dir)
        manifest = json.loads((d / "manifest.json").read_text())
        sc = manifest.get("scorer") or {}
        if sc.get("path"):
            loaders = {"TaskHead": TaskHead.load, "OrdinalScorer": OrdinalScorer.load}
            loader = loaders.get(sc.get("class"))
            if loader is None:
                raise ValueError(
                    f"bundle carries scorer class {sc.get('class')!r} with a "
                    f"persisted artifact, but no loader is registered for it")
            scorer = loader(d / sc["path"])
        elif sc.get("class") in (None, "CosineScorer") or "temperature" in sc:
            scorer = CosineScorer(temperature=sc.get("temperature", 1.0))
        else:
            raise ValueError(
                f"bundle declares scorer class {sc.get('class')!r} with no "
                f"persisted artifact — loading a CosineScorer instead would "
                f"silently change the model")
        gate = ConformalGate.load(d / manifest["gate"]) if manifest.get("gate") else None
        composer = None
        if manifest.get("composer"):
            from src.decision.composer import ComposerConfig, EmbeddingComposer
            import torch
            blob = torch.load(d / manifest["composer"], map_location="cpu",
                              weights_only=False)
            composer = EmbeddingComposer(ComposerConfig(**blob["config"]))
            composer.load_state_dict(blob["state_dict"])
        return cls(encoder=encoder, scorer=scorer, gate=gate, cache=cache,
                   encoder_id=manifest.get("encoder_id", ""), composer=composer)
