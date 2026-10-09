"""Deployment backends — encoders that don't need PyTorch/Transformers.

`LlamaCppEncoder` runs a GGUF-quantized EmbeddingGemma via llama.cpp — the
on-device path (ADR-0007). Measured: Q8_0 ≈ 4× faster than PyTorch fp32 on the
same CPU (~89 ms/short state on a 4-core Kaggle box), ~300MB on disk.

Text-only: media dicts raise — multimodal stays on `StateEncoder`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch

MEDIA_KEYS = ("image", "audio", "video")

# Task prefixes mirroring encoder.TASK_PROMPTS — llama.cpp gets raw strings,
# so the instruction prefix is folded into the text itself.
TASK_PREFIXES = {
    "SearchQuery": "task: search query | query: ",
    "Document": "task: search result | query: ",
    "Classification": "task: classification | query: ",
}


class LlamaCppEncoder:
    """GGUF embedding backend via llama-cpp-python.

    Args:
        model_path: Path to a `*.gguf` (e.g. embeddinggemma-300m Q8_0).
        n_ctx: Context window. The input-length cap (ADR-0012) is enforced by
            truncation to ``max_tokens`` before embedding.
        max_tokens: Hard cap per item — long states truncate, matching the
            measured design (ADR-0012).
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, max_tokens: int = 512):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "llama-cpp-python is required for the GGUF backend: "
                "pip install llama-cpp-python") from e
        if not Path(model_path).exists():
            raise FileNotFoundError(model_path)
        self.model_path = str(model_path)
        self.model_name = f"gguf:{Path(model_path).name}"
        self.max_tokens = int(max_tokens)
        self.llm = Llama(model_path=str(model_path), embedding=True,
                         n_ctx=n_ctx, n_gpu_layers=0, verbose=False)

    def _embed_text(self, text: str) -> np.ndarray:
        tokens = self.llm.tokenize(text.encode("utf-8"))
        if len(tokens) > self.max_tokens:           # ADR-0012: cap, don't error
            tokens = tokens[: self.max_tokens]
            text = self.llm.detokenize(tokens).decode("utf-8", errors="ignore")
        v = np.asarray(self.llm.embed(text), dtype=np.float32).reshape(-1)
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def _prep(self, item, prompt_name=None) -> str:
        if isinstance(item, str):
            return TASK_PREFIXES.get(prompt_name or "", "") + item
        if isinstance(item, dict):
            for k in MEDIA_KEYS:
                if k in item:
                    raise ValueError(
                        f"LlamaCppEncoder is text-only; {k!r} items need StateEncoder")
            return TASK_PREFIXES.get(prompt_name or "", "") + str(item.get("text", ""))
        raise TypeError(f"state items must be str or dict, got {type(item)}")

    def encode_state(self, items: Sequence[Union[str, dict]],
                     prompt_name: str | None = "Classification") -> torch.Tensor:
        rows = [self._embed_text(self._prep(i, prompt_name)) for i in items]
        return torch.from_numpy(np.stack(rows))

    def encode_options(self, option_texts: Sequence[str],
                       prompt_name: str | None = "Document") -> torch.Tensor:
        rows = [self._embed_text(self._prep(t, prompt_name)) for t in option_texts]
        return torch.from_numpy(np.stack(rows))


class StubEncoder:
    """Deterministic weightless encoder for dry-runs and tests.

    The last whitespace token seeds a class center; the full text adds small
    jitter — texts sharing a last token cluster together, so the entire
    CLI/serve/adapt path can be exercised locally (no weights, no torch at
    import) with a *learnable* signal rather than noise.

    Never use for a real decision: ``model_name`` is ``"stub"`` so bundles
    produced through it cannot masquerade as encoder-backed.
    """
    model_name = "stub"

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        key = str(text).split()[-1] if str(text).split() else "empty"
        rng = np.random.default_rng(abs(hash(key)) % (2**31))
        center = rng.standard_normal(self.dim)
        center /= np.linalg.norm(center)
        jitter = 0.05 * np.random.default_rng(
            abs(hash(text)) % (2**31)).standard_normal(self.dim)
        v = center + jitter
        return v / np.linalg.norm(v)

    def encode(self, texts: Sequence[str], **kw) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_state(self, items, prompt_name=None):
        return self.encode(
            [i if isinstance(i, str) else str(i.get("text", "")) for i in items])

    def encode_options(self, option_texts, prompt_name=None):
        return self.encode(list(option_texts))
