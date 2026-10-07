"""Typed decision schema — the Jev System One interface.

A *state* is a sequence of input items (text, media dicts); a *question* is one
of three types over a label set; a *prediction* is a distribution plus
confidence and abstention metadata. No generated text is ever parsed.

The training schema ``{state, questions, gold}`` matches jev-stack, so one
harvested file feeds either repo's head trainer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

#: The three typed decision primitives.
QTYPES = ("choice", "score", "noul")

#: A state item is text or a media dict passed to the encoder
#: ({"image": path}, {"audio": path}, {"video": path}, or an interleaved
#: {"text": str, "image": path, ...} dict).
StateItem = Union[str, Dict[str, Any]]


@dataclass
class State:
    """The input being evaluated: an ordered list of items.

    Single-item states (one utterance, one image) are the common case; the
    composer exists for multi-item states.
    """
    items: List[StateItem]

    @classmethod
    def text(cls, text: str) -> "State":
        return cls(items=[text])

    @classmethod
    def media(cls, **kwargs) -> "State":
        return cls(items=[dict(kwargs)])

    def __post_init__(self):
        if not self.items:
            raise ValueError("State must contain at least one item")
        for item in self.items:
            if not isinstance(item, (str, dict)):
                raise TypeError(f"state items must be str or dict, got {type(item)}")


@dataclass
class Question:
    """One typed question over a label set.

    - ``choice``: pick one of ``options``.
    - ``score``: distribution over ordered ``options`` (a rubric); ``scale``
      gives numeric values per level, defaulting to ``0..k-1``.
    - ``noul``: binary judgment; ``options`` defaults to ``["yes", "no"]`` and
      the prediction reports ``P(yes)`` under the "yes" label.
    """
    qtype: str
    options: List[str] = field(default_factory=list)
    scale: Optional[List[float]] = None

    def __post_init__(self):
        if self.qtype not in QTYPES:
            raise ValueError(f"qtype must be one of {QTYPES}, got {self.qtype!r}")
        if self.qtype == "noul" and not self.options:
            self.options = ["yes", "no"]
        if not self.options:
            raise ValueError(f"{self.qtype} question requires options")
        if self.qtype == "noul" and len(self.options) != 2:
            raise ValueError("noul is binary: exactly two options")
        if self.scale is not None and len(self.scale) != len(self.options):
            raise ValueError("scale must have one value per option")


@dataclass
class Prediction:
    """One typed decision.

    ``max_score`` is the highest raw (pre-softmax) match — the open-set signal:
    a low max means no option matches, not that the model is unsure which.
    """
    qtype: str
    labels: List[str]
    probabilities: Dict[str, float]
    answer_confidence: float
    entropy_confidence: float
    abstention: str = "unevaluated"
    abstention_threshold: Optional[float] = None
    max_score: float = 0.0

    @property
    def choice(self) -> Optional[str]:
        """Selected label for choice/score questions."""
        if self.qtype == "noul" or not self.probabilities:
            return None
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def noul(self) -> Optional[float]:
        """P(yes) for noul questions."""
        if self.qtype != "noul":
            return None
        return self.probabilities.get("yes")

    @property
    def score(self) -> Optional[float]:
        """Expected ordinal position for score questions (0..k-1 scale)."""
        if self.qtype != "score":
            return None
        from src.decision import metrics
        return metrics.expected_value([self.probabilities[l] for l in self.labels])

    def __repr__(self) -> str:
        return (
            f"Prediction(qtype={self.qtype!r}, choice={self.choice!r}, "
            f"noul={self.noul!r}, score={self.noul if self.qtype == 'noul' else self.score!r}, "
            f"answer_confidence={self.answer_confidence:.4f}, "
            f"abstention={self.abstention!r})"
        )


def normalized_entropy(probs: Sequence[float]) -> float:
    """Normalized entropy in [0, 1]; 1 is uniform, 0 is certain."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if p.size <= 1:
        return 0.0
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(p.size))


@dataclass
class DecisionExample:
    """One training example: a state vector, option vectors, labels, soft target.

    ``target`` is a *distribution*, not a hard label — the strictly proper
    scoring rules train against teacher distributions.
    """
    state_embedding: np.ndarray
    qtype: str
    labels: List[str]
    target: List[float]
    option_embeddings: Optional[List[np.ndarray]] = None
    source: str = ""

    def __post_init__(self):
        self.state_embedding = np.asarray(self.state_embedding, dtype=np.float32).reshape(-1)
        if len(self.labels) != len(self.target):
            raise ValueError("labels and target must be the same length")
        total = float(sum(self.target))
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"target distribution must sum to 1, got {total}")
        if self.qtype not in QTYPES:
            raise ValueError(f"qtype must be one of {QTYPES}, got {self.qtype!r}")
        if self.option_embeddings is not None:
            self.option_embeddings = [
                np.asarray(o, dtype=np.float32).reshape(-1) for o in self.option_embeddings
            ]
            if len(self.option_embeddings) != len(self.labels):
                raise ValueError("option_embeddings must have one vector per label")
