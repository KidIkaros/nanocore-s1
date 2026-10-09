"""NanoCoreS1 — the assembled decision model.

    state items → StateEncoder (frozen EG2) → EmbeddingComposer (or mean-pool)
                → DecisionHead → Prediction

The encoder is injectable: contract tests pass any object with
``encode_state``/``encode_options`` returning tensors, so the package is
testable without model weights. The real encoder is only constructed on
Kaggle-class hardware (see docs/DECISION-MODEL.md, compute boundary).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.decision.composer import EmbeddingComposer, MODALITY_IDS
from src.decision.encoder import modality_of, to_numpy


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
from src.decision.head import DecisionHead
from src.decision.schema import Prediction, Question, State, StateItem
from src.decision.scoring import softmax_rows


class NanoCoreS1:
    """Typed decision model over frozen multimodal embeddings.

    Args:
        encoder: Object with ``encode_state(items) -> (n, dim)`` and
            ``encode_options(texts) -> (k, dim)``. ``StateEncoder`` in
            production; any stub in tests.
        head: ``DecisionHead``. Its ``dim`` must equal the state-vector dim.
        composer: ``EmbeddingComposer`` or ``None`` — ``None`` means masked
            mean-pooling, which is the honest baseline the composer must beat.
        normalize_state: L2-normalize the composed state vector. Keeps the
            fingerprint head's ``<s, f>`` on a comparable scale.
    """

    def __init__(self, encoder=None, composer: Optional[EmbeddingComposer] = None,
                 head: Optional[DecisionHead] = None, normalize_state: bool = True):
        self.encoder = encoder
        self.composer = composer
        self.head = head or DecisionHead()
        self.normalize_state = normalize_state

    # ── encoding ─────────────────────────────────────────────────────────

    def encode_state_items(self, items: Sequence[StateItem]) -> torch.Tensor:
        """Items → ``(n, dim)`` item embeddings via the frozen encoder."""
        return self.encoder.encode_state(list(items))

    @torch.no_grad()
    def state_vector(self, state: Union[State, Sequence[StateItem]]) -> torch.Tensor:
        """State → one ``(dim,)`` state vector.

        Without a composer this is the mean over item embeddings — the
        baseline path. With one, the composed output of the item sequence.
        """
        items = state.items if isinstance(state, State) else list(state)
        item_embs = self.encode_state_items(items)
        if item_embs.dim() == 1:
            item_embs = item_embs.unsqueeze(0)

        if self.composer is None:
            s = item_embs.mean(dim=0)
        else:
            modality_ids = torch.tensor(
                [MODALITY_IDS[modality_of(i)] for i in items],
                dtype=torch.long, device=item_embs.device)
            s = self.composer(item_embs.unsqueeze(0), modality_ids.unsqueeze(0))[0]

        return F.normalize(s.float(), dim=-1) if self.normalize_state else s.float()

    def option_vectors(self, question: Question) -> np.ndarray:
        """Option label texts → ``(k, dim)`` numpy matrix (cached by caller ideally)."""
        vecs = self.encoder.encode_options(list(question.options))
        return to_numpy(vecs)

    # ── the typed interface ──────────────────────────────────────────────

    def decide(self, state: Union[State, Sequence[StateItem], str],
               question: Question,
               option_embeddings: Optional[np.ndarray] = None) -> Prediction:
        """One typed decision.

        Args:
            state: ``State``, item list, or a bare string (one text item).
            question: ``Question`` — choice/score/noul.
            option_embeddings: Precomputed ``(k, dim)`` option vectors. Pass a
                cache when the same options recur; required in interaction mode
                unless the head is fingerprint.
        """
        if isinstance(state, str):
            state = State.text(state)
        s = self.state_vector(state).numpy()
        opts = option_embeddings
        if opts is None and self.head.mode == "interaction":
            opts = self.option_vectors(question)
        return self.head.predict(s, opts, qtype=question.qtype, labels=question.options)

    def decide_batch(self, state: Union[State, Sequence[StateItem], str],
                     questions: Sequence[Question],
                     option_cache: Optional[dict] = None) -> List[Prediction]:
        """Several questions over one encoded state (the Jev calling pattern)."""
        s = self.state_vector(State.text(state) if isinstance(state, str) else state).numpy()
        cache = option_cache if option_cache is not None else {}
        preds = []
        for q in questions:
            key = (q.qtype, tuple(q.options))
            if key not in cache and self.head.mode == "interaction":
                cache[key] = self.option_vectors(q)
            preds.append(self.head.predict(s, cache.get(key),
                                         qtype=q.qtype, labels=q.options))
        return preds


class DecisionModel:
    """The decided pipeline: encoder → scorer → gate → cache (ADR-0008/0009/0013).

    Unlike ``NanoCoreS1`` (the deprecated head path), this wires the measured
    architecture: a frozen encoder produces the state, a scorer produces raw
    scores, and the gate owns calibration, prediction sets, and the action.

    Args:
        encoder: ``encode_state(items) -> (n, dim)`` and
            ``encode_options(texts) -> (k, dim)`` — ``StateEncoder`` in
            production, any stub in tests.
        scorer: ``scores(state_vec, option_vecs) -> (k,)`` — ``CosineScorer``
            or a fitted ``TaskHead`` (per the headroom gate, ADR-0011).
        gate: ``ConformalGate`` or ``None`` (ungated — probs only).
        cache: ``DecisionCache`` or ``None``.
        encoder_id: Model identity string folded into cache keys.
    """

    def __init__(self, encoder=None, scorer=None, gate=None, cache=None,
                 encoder_id: str = ""):
        from src.decision.scoring import CosineScorer
        self.encoder = encoder
        self.scorer = scorer or CosineScorer()
        self.gate = gate
        self.cache = cache
        self.encoder_id = encoder_id or getattr(encoder, "model_name", "stub")
        self._option_cache: dict = {}

    def _state_vec(self, state) -> np.ndarray:
        items = state.items if isinstance(state, State) else [state]
        embs = self.encoder.encode_state(list(items))
        e = to_numpy(embs)
        s = e.mean(axis=0)
        return s / max(np.linalg.norm(s), 1e-12)

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
                            "alpha": getattr(self.gate, "alpha", None)})
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
        import json
        from pathlib import Path
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        manifest = {"encoder_id": self.encoder_id, "scorer": None, "gate": None}

        if self.scorer is not None:
            if hasattr(self.scorer, "save") and self.scorer.__class__.__name__ == "TaskHead":
                self.scorer.save(d / "scorer.pt")
                manifest["scorer"] = {"class": "TaskHead", "path": "scorer.pt"}
            else:
                manifest["scorer"] = {"class": self.scorer.__class__.__name__,
                                      "temperature": getattr(self.scorer, "temperature", 1.0)}
        if self.gate is not None:
            self.gate.save(d / "gate.json")
            manifest["gate"] = "gate.json"
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return d

    @classmethod
    def load(cls, dir, encoder=None, cache=None) -> "DecisionModel":
        """Rebuild a bundle. ``encoder`` is the caller's backend instance."""
        import json
        from pathlib import Path
        from src.decision.gate import ConformalGate
        from src.decision.scoring import CosineScorer, TaskHead

        d = Path(dir)
        manifest = json.loads((d / "manifest.json").read_text())
        sc = manifest.get("scorer") or {}
        if sc.get("class") == "TaskHead":
            scorer = TaskHead.load(d / sc["path"])
        else:
            scorer = CosineScorer(temperature=sc.get("temperature", 1.0))
        gate = ConformalGate.load(d / manifest["gate"]) if manifest.get("gate") else None
        return cls(encoder=encoder, scorer=scorer, gate=gate, cache=cache,
                   encoder_id=manifest.get("encoder_id", ""))
