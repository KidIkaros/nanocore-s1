"""State-item utilities — the torch-free core every path shares.

``DecisionModel`` must import without torch (the llama.cpp/GGUF runtime,
ADR-0007), but these helpers were stranded in ``encoder.py``, which is
torch-bound through ``StateEncoder``. They are pure — duck-typed, numpy-only —
so they live here and are re-exported by ``encoder`` for compatibility.
"""
from __future__ import annotations

import numpy as np

#: Media keys a state-item dict may carry (item → encoder dispatch).
MEDIA_KEYS = ("image", "audio", "video")


def modality_of(item) -> str:
    """The modality tag for one state item — used for composer type embeddings."""
    if isinstance(item, str):
        return "text"
    if isinstance(item, dict):
        for key in MEDIA_KEYS:
            if key in item:
                return "audio" if key == "audio" else ("video" if key == "video" else "image")
        return "text"
    raise TypeError(f"state items must be str or dict, got {type(item)}")


def to_numpy(t) -> np.ndarray:
    """GPU/bf16-safe conversion: `.numpy()` rejects bfloat16.

    Accepts a plain array as well as a tensor. This idiom was spelled out inline
    in eight places across four modules, each needing both cases (the stub
    backend returns numpy), so the shared helper handles both rather than each
    caller guarding for itself.
    """
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
