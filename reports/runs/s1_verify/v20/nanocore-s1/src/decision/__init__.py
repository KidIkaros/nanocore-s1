"""NanoCore-S1 decision model — typed decisions over frozen multimodal embeddings.

See docs/DECISION-MODEL.md for the build plan and compute boundary.
"""
from src.decision.schema import (
    QTYPES,
    DecisionExample,
    Prediction,
    Question,
    State,
    normalized_entropy,
)
from src.decision.composer import ComposerConfig, EmbeddingComposer, MODALITY_IDS
from src.decision.head import DecisionHead
from src.decision.model import NanoCoreS1

__all__ = [
    "QTYPES",
    "DecisionExample",
    "Prediction",
    "Question",
    "State",
    "normalized_entropy",
    "ComposerConfig",
    "EmbeddingComposer",
    "MODALITY_IDS",
    "DecisionHead",
    "NanoCoreS1",
]
