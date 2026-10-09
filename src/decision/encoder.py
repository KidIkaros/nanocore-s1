"""EmbeddingGemma 2 state encoder — the frozen multimodal front-end.

Adapted from jev-stack `src/gemma_embed.py`. Wraps `google/embeddinggemma-2`
(270M text / +170M vision / +300M audio / 740M full), which maps text, code,
images, video and audio into one shared 768-dimensional space.

Requirements (Kaggle): ``sentence-transformers>=6.1.0`` and
``transformers>=5.19.0`` — earlier transformers raises ``KeyError:
'embedding_gemma2'``.

Notes that matter:

- **Precision.** EmbeddingGemma 2 activations are incompatible with
  ``float16`` — it produces NaN or degraded embeddings. bfloat16 on
  accelerators, float32 on CPU. fp16 is rejected here.
- **Modular loading.** Unselected encoders never enter memory:
  ``config_kwargs={"vision_config": None, "audio_config": None}`` gives the
  270M text-only setup from the same checkpoint.
- **Task prefixes.** Text is encoded with short instruction prefixes
  (``prompt_name``); media inputs take none.
- **MRL truncation.** 768 → 512/256/128 is supported; re-normalize after
  slicing — slicing a unit vector silently breaks ranking.

Reference: https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

# Re-exported: the torch-free home is items.py (DecisionModel must not need torch).
from src.decision.items import MEDIA_KEYS, modality_of, to_numpy

#: Task instruction prefixes EmbeddingGemma 2 was trained with. Text only.
TASK_PROMPTS = {
    "search_query": "SearchQuery",
    "document": "Document",
    "question_answering": "QuestionAnswering",
    "fact_checking": "FactChecking",
    "code_retrieval": "CodeRetrieval",
    "classification": "Classification",
    "clustering": "Clustering",
    "sentence_similarity": "SentenceSimilarity",
}

#: Supported MRL output dimensions.
VALID_DIMS = (128, 256, 512, 768)

#: Supported modality names (encoder configs that can be loaded).
VALID_MODALITIES = ("text", "vision", "audio")

#: Media dict keys recognised as non-text items (defined in items.py).

EmbeddingInput = Union[str, dict, Sequence[Union[str, dict]]]


def _sentence_transformer():
    """Lazy import so this module loads without the dependency installed.

    Contract tests exercise the package without model weights; the encoder is
    only needed on Kaggle where sentence-transformers is present.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "sentence-transformers>=6.1.0 and transformers>=5.19.0 are required "
            "to load EmbeddingGemma 2. Run this on Kaggle, not the laptop."
        ) from e
    return SentenceTransformer


class StateEncoder:
    """Encoder over EmbeddingGemma 2 with modular loading and MRL truncation.

    Args:
        model_name: HuggingFace model ID.
        device: Torch device string; defaults to CUDA if available.
        modalities: Subset of ``("text", "vision", "audio")``. Text-only
            (default) loads 270M parameters; adding vision/audio loads 440M/570M.
        dtype: Defaults to bfloat16 on accelerators, float32 on CPU.
            ``float16`` is rejected.
        truncate_dim: Matryoshka output dimension at load time.
        normalize_embeddings: Unit-normalize outputs (default).
    """

    DEFAULT_MODEL = "google/embeddinggemma-2"
    DEFAULT_DIM = 768

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        modalities: Iterable[str] = ("text",),
        dtype: torch.dtype | None = None,
        truncate_dim: int | None = None,
        normalize_embeddings: bool = True,
    ):
        modalities = tuple(modalities)
        unknown = set(modalities) - set(VALID_MODALITIES)
        if unknown:
            raise ValueError(f"Unknown modalities {sorted(unknown)}; expected {VALID_MODALITIES}")
        if truncate_dim is not None and truncate_dim not in VALID_DIMS:
            raise ValueError(f"truncate_dim must be one of {VALID_DIMS}, got {truncate_dim}")

        self.model_name = model_name
        self.modalities = modalities
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = truncate_dim or self.DEFAULT_DIM

        if dtype is None:
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        if dtype == torch.float16:
            raise ValueError(
                "EmbeddingGemma 2 activations are incompatible with float16; "
                "use bfloat16 (accelerators) or float32 (CPU)."
            )
        self.dtype = dtype

        config_kwargs: dict = {}
        if "vision" not in modalities:
            config_kwargs["vision_config"] = None
        if "audio" not in modalities:
            config_kwargs["audio_config"] = None
        model_kwargs = {} if dtype == torch.float32 else {"torch_dtype": dtype}

        self.model = _sentence_transformer()(
            model_name,
            device=self.device,
            config_kwargs=config_kwargs or None,
            model_kwargs=model_kwargs or None,
            truncate_dim=truncate_dim,
        )

    @torch.no_grad()
    def encode(
        self,
        inputs: EmbeddingInput,
        dim: int | None = None,
        batch_size: int = 32,
        prompt_name: str | None = None,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Encode items to embeddings.

        Args:
            inputs: A string, a media dict (``{"image": path}``), or a sequence
                of either. Media inputs take no task prefix.
            dim: Output dimension via Matryoshka truncation; re-normalized.
            batch_size: Batch size for multiple inputs.
            prompt_name: Task prefix for text (a TASK_PROMPTS value).

        Returns:
            Single input: ``(dim,)``. Sequence: ``(n, dim)``. On ``self.device``.
        """
        if dim is not None and dim not in VALID_DIMS:
            raise ValueError(f"dim must be one of {VALID_DIMS}, got {dim}")

        single = isinstance(inputs, (str, dict))
        items = [inputs] if single else list(inputs)
        normalize = self.normalize_embeddings if normalize is None else normalize

        embeddings = self.model.encode(
            items,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            prompt_name=prompt_name,
        )

        if dim is not None and dim < self.embedding_dim:
            embeddings = embeddings[:, :dim]
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings[0] if single else embeddings

    def encode_state(self, items: Sequence[Union[str, dict]],
                     prompt_name: str | None = "Classification") -> torch.Tensor:
        """Encode a state's item list → ``(n_items, dim)``.

        Text items get the symmetric ``Classification`` prefix (the right
        family for decision states). Batches that mix text and media are
        encoded per-item, since media take no prefix.
        """
        if all(isinstance(i, str) for i in items):
            return self.encode(list(items), prompt_name=prompt_name)
        rows = [
            self.encode(i, prompt_name=prompt_name if isinstance(i, str) else None)
            for i in items
        ]
        return torch.stack([r if r.dim() == 1 else r[0] for r in rows])

    def encode_options(self, option_texts: Sequence[str],
                       prompt_name: str | None = "Document") -> torch.Tensor:
        """Encode option label texts → ``(k, dim)``. Options are always text."""
        return self.encode(list(option_texts), prompt_name=prompt_name)

    def __call__(self, inputs: EmbeddingInput, **kwargs) -> torch.Tensor:
        return self.encode(inputs, **kwargs)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between embedding tensors."""
        return self.model.similarity(a, b)



