"""DecisionCache — content-addressed store for embeddings and decisions.

The cache key binds *everything* that can change the answer: the state
content, the question (qtype + option texts + scale), the encoder identity,
and the caller knobs that alter the math (temperature, alpha, policy, dims).
A hit returns the stored result verbatim; a miss returns None.

Files are ``<dir>/<key>.json`` — one small file per decision, cheap to
inspect, safe to delete individually.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from src.decision.schema import Question, State, StateItem


def _item_repr(item: StateItem) -> Any:
    """Hashable, JSON-stable rendering of a state item (paths, not bytes)."""
    if isinstance(item, str):
        return item
    return {k: str(v) for k, v in sorted(item.items())}


def decision_key(state: Union[State, Sequence[StateItem], str],
                 question: Question, encoder_id: str = "",
                 knobs: Optional[Dict[str, Any]] = None) -> str:
    """SHA-256 over the full decision context."""
    if isinstance(state, str):
        items = [state]
    elif isinstance(state, State):
        items = state.items
    else:
        items = list(state)
    payload = {
        "items": [_item_repr(i) for i in items],
        "qtype": question.qtype,
        "options": list(question.options),
        "scale": question.scale,
        "encoder": encoder_id,
        "knobs": knobs or {},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class DecisionCache:
    """Filesystem decision cache. ``None`` dir disables it."""

    def __init__(self, dir=None):
        self.dir = Path(dir) if dir is not None else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[Dict]:
        if self.dir is None:
            return None
        p = self._path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def put(self, key: str, result: Dict) -> None:
        if self.dir is None:
            return
        self._path(key).write_text(json.dumps(result))

    def __contains__(self, key: str) -> bool:
        return self.dir is not None and self._path(key).exists()
