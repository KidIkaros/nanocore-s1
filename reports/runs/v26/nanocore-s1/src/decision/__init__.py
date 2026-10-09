"""NanoCore-S1 — typed decisions over frozen multimodal embeddings.

**What the model is.** Two things, and neither is a program:

1. **A fitted artifact** — the bundle: ``manifest.json`` (encoder identity),
   ``scorer.pt`` (the fitted head), ``gate.json`` (the calibration). Identified
   by content via :func:`bundle_digest`.
2. **One contract** — :meth:`DecisionModel.decide`:

       DecisionModel.load(bundle, encoder=...).decide(state, question) -> Prediction

   ``Prediction`` carries the calibrated probabilities, the conformal prediction
   set, and the action (``answer`` / ``clarify`` / ``escalate`` / ``abstain``).
   The encoder is frozen and injected by the caller; :func:`adapt` fits the
   scorer and the gate from labeled data.

**What the model is not.** ``cli`` and ``serve`` are a **reference application** —
one way to call the model, not part of it. ``guard``, ``parity``, ``registry``,
``monitor``, ``qualify``, ``modelcard`` and ``rubric`` are release and operations
tooling. They live in this package for now; the boundary is not yet enforced by
the directory layout, which is why this module states it instead.

**Not exported, deliberately.** ``NanoCoreS1``, ``DecisionHead`` and
``EmbeddingComposer`` are the earlier decoder-era path. They remain importable
from their own modules for the records they hold, but the package front door
should not offer them: a public API that leads with the abandoned architecture
does not describe the model.
"""
from src.decision.adapt import AdaptConfig, AdaptResult, adapt
from src.decision.encoder import StateEncoder, to_numpy
from src.decision.gate import ACTIONS, ConformalGate, action_for
from src.decision.model import DecisionModel, bundle_digest
from src.decision.scoring import CosineScorer, OrdinalScorer, TaskHead
from src.decision.schema import (
    QTYPES,
    DecisionExample,
    Prediction,
    Question,
    State,
    answer_labels,
    normalized_entropy,
)

__all__ = [
    # the contract
    "DecisionModel",
    "Prediction",
    "Question",
    "State",
    "QTYPES",
    "answer_labels",
    "normalized_entropy",
    "ACTIONS",
    "action_for",
    # fitting a bundle
    "adapt",
    "AdaptConfig",
    "AdaptResult",
    # what a bundle is made of
    "ConformalGate",
    "CosineScorer",
    "TaskHead",
    "OrdinalScorer",
    "StateEncoder",
    "bundle_digest",
    "to_numpy",
    # kept for the audit trail of the earlier path
    "DecisionExample",
]
