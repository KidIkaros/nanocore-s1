"""Action handlers — every action reaches a concrete destination."""
import json

import numpy as np
import pytest

from src.decision.handlers import (CallableEscalation, QueuedEscalation,
                                   clarify_payload, handle)
from src.decision.schema import Prediction, Question


def _pred(action, probs=None, pred_set=None):
    probs = probs or {"a": 0.6, "b": 0.3, "c": 0.1}
    return Prediction(qtype="choice", labels=list(probs), probabilities=probs,
                      answer_confidence=max(probs.values()),
                      entropy_confidence=0.5, max_score=0.8,
                      prediction_set=pred_set or [], action=action, alpha=0.1)


Q = Question(qtype="choice", options=["a", "b", "c"])


def test_answer_resolves_as_system_one():
    r = handle(_pred("answer"), "state", Q)
    assert r.resolved_by == "system_one" and r.payload["probabilities"]


def test_clarify_returns_question_payload():
    p = _pred("clarify", pred_set=["a", "b"])
    r = handle(p, "state", Q)
    assert r.action == "clarify" and r.resolved_by == "caller"
    assert r.payload["options"] == ["a", "b"]
    assert "a" in r.payload["text"]


def test_escalate_without_handler_is_explicit_not_silent():
    r = handle(_pred("escalate"), "state", Q)
    assert r.resolved_by == "nobody"
    assert "no escalation handler" in r.payload["reason"]


def test_escalate_to_callable():
    esc = CallableEscalation(lambda s, q, p: {"answer": "system-two said b"})
    r = handle(_pred("escalate"), "state", Q, escalate_to=esc)
    assert r.resolved_by == "system_two"
    assert r.payload["response"]["answer"] == "system-two said b"


def test_escalate_callable_failure_falls_back_to_queue(tmp_path):
    q_path = tmp_path / "esc.jsonl"
    esc = CallableEscalation(lambda s, q, p: 1 / 0,
                             fallback=QueuedEscalation(q_path))
    r = handle(_pred("escalate"), "state", Q, escalate_to=esc)
    assert r.payload["resolved_by"] == "queue"
    assert "system_two_error" in r.payload
    assert len(q_path.read_text().strip().splitlines()) == 1


def test_queued_escalation_writes_jsonl(tmp_path):
    q_path = tmp_path / "q" / "esc.jsonl"
    handle(_pred("escalate"), "state", Q,
           escalate_to=QueuedEscalation(q_path))
    rec = json.loads(q_path.read_text().strip())
    assert rec["labels"] == ["a", "b", "c"] and rec["top_prob"] == 0.6


def test_abstain_returns_reason():
    r = handle(_pred("abstain"), "state", Q)
    assert r.action == "abstain" and r.resolved_by == "nobody"
    assert "reason" in r.payload


def test_clarify_payload_falls_back_to_top_probs():
    q = Question(qtype="choice", options=["x", "y"])
    p = _pred("clarify", probs={"x": 0.55, "y": 0.45}, pred_set=[])
    cp = clarify_payload(p, q)
    assert cp.options == ["x", "y"] and "x" in cp.text
