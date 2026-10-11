"""NanoCoreS1 — the deprecated assembled-model path, kept for the record.

This is the decoder-era wiring: ``state items → encoder → composer (or
mean-pool) → DecisionHead → Prediction``. It was superseded by
``DecisionModel`` (encoder → scorer → gate → cache), which is the measured
architecture. It lives here — importable, tested, documented — rather than in
``model.py`` so the shipped path carries no torch dependency at import time
(the llama.cpp deployment target does not ship PyTorch).

The composer slot is shared with the shipped path via
:func:`composer.compose_items`, so the two cannot drift into different
composition semantics.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.decision.composer import EmbeddingComposer, compose_items
from src.decision.encoder import to_numpy
from src.decision.head import DecisionHead
from src.decision.schema import Prediction, Question, State, StateItem


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
        s = compose_items(items, item_embs, self.composer)
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
