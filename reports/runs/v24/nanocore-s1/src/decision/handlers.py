"""Action handlers — the door that ``escalate`` and ``clarify`` point at.

The gate emits an action; something must *do* it. This module makes every
action concrete and pluggable:

- ``answer``  → the prediction itself (already resolved).
- ``clarify`` → a ``ClarifyPayload`` the caller can render as a question.
- ``escalate``→ an ``EscalationHandler``: callable (System Two / LLM),
              ``QueuedEscalation`` (JSONL file → human/queue consumer).
- ``abstain`` → explicit no-answer with reason (never silent).

Handlers are callables injected by the deployment — nothing here knows about
any particular System Two. ``handle()`` is the single dispatch entry point.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

from src.decision.schema import Prediction, Question


@runtime_checkable
class EscalationHandler(Protocol):
    """Anything that can receive an escalation: a callable wrapper, a queue,
    a human handoff. Contract: ``resolved_by`` attr + ``handle()``."""
    resolved_by: str

    def handle(self, state, question: Question, pred: Prediction) -> dict: ...


# ── payloads ─────────────────────────────────────────────────────────────────

@dataclass
class ClarifyPayload:
    """What to ask the user. ``options`` are the prediction-set candidates."""
    text: str
    options: list
    qtype: str = "choice"


@dataclass
class DispatchResult:
    """The outcome after handling an action."""
    action: str                       # answer | clarify | escalate | abstain
    resolved_by: str                  # system_one | system_two | queue | caller | nobody
    prediction: Prediction
    payload: Optional[dict] = None    # clarify payload / escalation record / abstain reason


# ── clarify ──────────────────────────────────────────────────────────────────

def clarify_payload(pred: Prediction, question: Question,
                    max_options: int = 3) -> ClarifyPayload:
    """Turn a prediction set into a question the caller can ask."""
    opts = pred.prediction_set[:max_options] or [
        l for l, _ in sorted(pred.probabilities.items(), key=lambda kv: -kv[1])[:max_options]]
    text = "I need to disambiguate — which of these did you mean?"
    if opts:
        text = f"{text} ({', '.join(opts)})"
    return ClarifyPayload(text=text, options=list(opts), qtype=question.qtype)


# ── escalation ───────────────────────────────────────────────────────────────

class CallableEscalation:
    """Escalate to a callable — a System Two model, an API, a rule — anything
    with the signature ``fn(state, question, prediction) -> dict | str``.
    The callable's return becomes ``payload``; failures escalate to the
    ``fallback`` (a queue) rather than dying.
    """

    resolved_by = "system_two"

    def __init__(self, fn: Callable, fallback: Optional["QueuedEscalation"] = None):
        self.fn = fn
        self.fallback = fallback

    def handle(self, state, question: Question, pred: Prediction) -> dict:
        try:
            out = self.fn(state, question, pred)
            return {"resolved_by": self.resolved_by,
                    "response": out}
        except Exception as e:
            if self.fallback is None:
                raise
            fb = self.fallback.handle(state, question, pred)
            fb["system_two_error"] = repr(e)[:200]
            return fb


class QueuedEscalation:
    """Append escalation records to a JSONL queue — a human or a downstream
    consumer drains it. Always succeeds (fail-open for the *caller*, not the
    answer)."""

    resolved_by = "queue"

    def __init__(self, path):
        self.path = Path(path)

    def handle(self, state, question: Question, pred: Prediction) -> dict:
        rec = {"ts": time.time(), "qtype": question.qtype,
               "labels": question.options,
               "top_prob": pred.answer_confidence,
               "prediction_set": pred.prediction_set,
               "state": state if isinstance(state, str) else repr(state)[:500]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return {"resolved_by": self.resolved_by, "queued_at": str(self.path),
                "record": rec}


# ── dispatch ─────────────────────────────────────────────────────────────────

def handle(pred: Prediction, state, question: Question,
           escalate_to: Optional[EscalationHandler] = None,
           max_clarify_options: int = 3) -> DispatchResult:
    """Execute the gated action. ``escalate_to`` is required for ``escalate``
    to reach a real destination; without one, escalation falls back to an
    explicit unresolved record rather than pretending to be handled.
    """
    if pred.action == "answer":
        return DispatchResult(action="answer", resolved_by="system_one",
                              prediction=pred,
                              payload={"probabilities": pred.probabilities,
                                       "prediction_set": pred.prediction_set})

    if pred.action == "clarify":
        payload = clarify_payload(pred, question, max_clarify_options)
        return DispatchResult(action="clarify", resolved_by="caller",
                              prediction=pred,
                              payload={"text": payload.text,
                                       "options": payload.options})

    if pred.action == "escalate":
        if escalate_to is None:
            return DispatchResult(action="escalate", resolved_by="nobody",
                                  prediction=pred,
                                  payload={"reason": "no escalation handler configured"})
        return DispatchResult(action="escalate",
                              resolved_by=escalate_to.resolved_by,
                              prediction=pred,
                              payload=escalate_to.handle(state, question, pred))

    if pred.action == "abstain":
        return DispatchResult(action="abstain", resolved_by="nobody",
                              prediction=pred,
                              payload={"reason": "out of schema or below abstention threshold",
                                       "max_score": pred.max_score,
                                       "alpha": pred.alpha})

    return DispatchResult(action=pred.action, resolved_by="nobody",
                          prediction=pred,
                          payload={"reason": f"unknown action {pred.action!r}"})
