"""Serving layer — the decision model callable over HTTP.

Prediction mode decision (roadmap Phase 4): **online** — decisions are
interactive (route/verify/escalate mid-conversation). Batch scoring already
exists via ``cli decide --inputs``; this module serves single decisions.

    POST /decide   {"text": "...", "options": ["a","b"], "qtype": "choice",
                    "policy": "full"}
                -> Prediction fields (action, probabilities, prediction_set, ...)

    GET  /healthz  -> liveness
    GET  /stats    -> request counters, latency percentiles, cost summary

Every decision is appended to a JSONL prediction log: input, output, model
id, latency. The log is what Phase 5 monitoring and Phase 6 shadow runs
consume — a deployment without it is unobservable.

Stdlib only (http.server) — a decision layer does not need a web framework.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from src.decision.schema import Question


# ── prediction log ───────────────────────────────────────────────────────────

class PredictionLogger:
    """Append-only JSONL log of every decision — inputs, outputs, model id,
    latency. Rotation/retention is the deployer's concern; correctness here is
    that every served decision leaves a record.

    Component 42 fields make each row a trajectory unit, not just a decision:
    ``decision_id`` is the join key for later outcomes; ``stream_id``/``seq``/
    ``prev_id`` link consecutive calls; ``obs`` and ``thresholds`` record what
    the gate saw and what ruled it, so a logged row replays under the same (or
    any alternate) policy without re-encoding; ``raw_scores`` is the
    counterfactual substrate; ``state_hash`` fingerprints the input state
    without storing the vector."""

    _MAX_STREAMS = 10_000

    def __init__(self, path, model_id: str = "", log_scores: bool = True):
        self.path = Path(path)
        self.model_id = model_id
        self.log_scores = log_scores
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._streams: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def record(self, *, text: str, question: Question, pred, latency_ms: float,
               policy: Optional[str], stream_id: Optional[str] = None,
               gold=None, thresholds=None) -> dict:
        decision_id = uuid.uuid4().hex[:16]
        prev_id, seq = None, 0
        if stream_id is not None:
            with self._lock:
                last = self._streams.pop(stream_id, (None, -1))
                prev_id, seq = last[0], last[1] + 1
                self._streams[stream_id] = (decision_id, seq)
                if len(self._streams) > self._MAX_STREAMS:
                    self._streams.popitem(last=False)
        probs = sorted(pred.probabilities.values(), reverse=True)
        margin = probs[0] - probs[1] if len(probs) > 1 else (probs[0] if probs else 0.0)
        rec = {"ts": time.time(), "model_id": self.model_id,
               "decision_id": decision_id,
               "stream_id": stream_id, "seq": seq, "prev_id": prev_id,
               "input": text, "qtype": pred.qtype, "labels": pred.labels,
               "policy": policy, "action": pred.action,
               "top_prob": pred.answer_confidence,
               "prediction_set": pred.prediction_set,
               "alpha": pred.alpha, "latency_ms": round(latency_ms, 2),
               "state_hash": pred.state_hash,
               "obs": {"top_prob": pred.answer_confidence,
                       "set_size": len(pred.prediction_set),
                       "entropy_norm": pred.entropy_confidence,
                       "max_score": pred.max_score,
                       "margin": round(margin, 6)},
               "thresholds": (None if thresholds is None else
                              {"tau_answer": thresholds.tau_answer,
                               "k_clarify": thresholds.k_clarify,
                               "tau_in_schema": thresholds.tau_in_schema})}
        if self.log_scores and pred.raw_scores is not None:
            rec["raw_scores"] = pred.raw_scores
        if gold is not None:
            rec["gold"] = gold
        line = json.dumps(rec) + "\n"
        with self._lock, self.path.open("a") as f:
            f.write(line)
        return rec


class OutcomeLogger:
    """Append-only side log joining outcomes back to decisions by ``decision_id``.

    Lives beside the prediction log as ``<log>.outcomes.jsonl`` — the prediction
    row is written at decision time (outcome unknowable then); the outcome row
    is written by whoever learns it: a benchmark harness (``gold``), the
    clarification resolver (``clarification_resolved``), the escalation
    consumer (``escalation_confirmed``/``overruled``), or a human reviewer
    (``user_corrected``).
    """

    OUTCOME_TYPES = ("gold", "clarification_resolved", "escalation_confirmed",
                     "escalation_overruled", "user_corrected")

    def __init__(self, log_path):
        self.path = Path(str(log_path) + ".outcomes.jsonl")
        self._lock = threading.Lock()

    def record(self, *, decision_id: str, outcome: str, source: str = "",
               payload=None) -> dict:
        if outcome not in self.OUTCOME_TYPES:
            raise ValueError(f"outcome must be one of {self.OUTCOME_TYPES}")
        rec = {"ts": time.time(), "decision_id": decision_id,
               "outcome": outcome, "source": source, "payload": payload}
        line = json.dumps(rec) + "\n"
        with self._lock, self.path.open("a") as f:
            f.write(line)
        return rec


# ── server ───────────────────────────────────────────────────────────────────

class _Stats:
    """Request counters. Latencies are bounded: percentiles over a long-running
    server should describe *recent* service, and an unbounded list is a leak."""

    def __init__(self, window: int = 10_000):
        from collections import deque
        self.n = 0
        self.latencies = deque(maxlen=window)
        self.actions = {}
        self.t0 = time.time()

    def observe(self, action: str, latency_ms: float):
        self.n += 1
        self.latencies.append(latency_ms)
        self.actions[action] = self.actions.get(action, 0) + 1

    def summary(self) -> dict:
        lat = np.asarray(list(self.latencies)) if self.latencies else np.zeros(1)
        return {"requests": self.n, "uptime_s": round(time.time() - self.t0, 1),
                "actions": dict(sorted(self.actions.items())),
                "latency_ms": {"p50": float(np.percentile(lat, 50)),
                               "p95": float(np.percentile(lat, 95)),
                               "p99": float(np.percentile(lat, 99)),
                               "mean": float(lat.mean())},
                "throughput_rps": round(self.n / max(time.time() - self.t0, 1e-9), 3)}


def make_handler(model, logger: PredictionLogger, stats: _Stats,
                 default_policy: Optional[str] = None,
                 outcomes: Optional[OutcomeLogger] = None):
    """Build a request handler bound to ``model``. Pure function so tests can
    construct a server on an ephemeral port without the CLI."""
    if outcomes is None:
        outcomes = OutcomeLogger(logger.path)

    class DecisionHandler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/healthz":
                self._send(200, {"ok": True, "model_id": logger.model_id})
            elif self.path == "/stats":
                self._send(200, stats.summary())
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path == "/outcome":
                try:
                    body = json.loads(self.rfile.read(
                        int(self.headers.get("Content-Length", 0))))
                    rec = outcomes.record(decision_id=body["decision_id"],
                                          outcome=body["outcome"],
                                          source=body.get("source", "http"),
                                          payload=body.get("payload"))
                    self._send(200, {"ok": True, "recorded": rec["outcome"]})
                except (KeyError, ValueError, json.JSONDecodeError) as e:
                    self._send(400, {"error": str(e)[:300]})
                return
            if self.path != "/decide":
                self._send(404, {"error": "unknown path"})
                return
            try:
                body = json.loads(self.rfile.read(
                    int(self.headers.get("Content-Length", 0))))
                q = Question(qtype=body.get("qtype", "choice"),
                             options=body.get("options", []),
                             scale=body.get("scale"))
                t0 = time.perf_counter()
                pred = model.decide(body["text"], q,
                                    policy=body.get("policy", default_policy))
                ms = (time.perf_counter() - t0) * 1000
                thresholds = (model.gate.active_thresholds()
                              if getattr(model, "gate", None) is not None else None)
                rec = logger.record(text=body["text"], question=q, pred=pred,
                                    latency_ms=ms,
                                    policy=body.get("policy", default_policy),
                                    stream_id=body.get("stream_id"),
                                    gold=body.get("gold"),
                                    thresholds=thresholds)
                stats.observe(pred.action, ms)
                self._send(200, {"action": pred.action, "qtype": pred.qtype,
                                 "decision_id": rec["decision_id"],
                                 "top_prob": pred.answer_confidence,
                                 "prediction_set": pred.prediction_set,
                                 "probabilities": pred.probabilities,
                                 "alpha": pred.alpha,
                                 "latency_ms": round(ms, 2)})
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": str(e)[:300]})

        def log_message(self, *a):  # quiet — the prediction log is the record
            pass

    return DecisionHandler


def serve(model, host: str = "127.0.0.1", port: int = 8000,
          log_path=".nanocore-predictions.jsonl",
          model_id: str = "", policy: Optional[str] = None) -> ThreadingHTTPServer:
    """Build the server (caller runs ``serve_forever``). Returns the instance
    so tests can shut it down."""
    logger = PredictionLogger(log_path, model_id=model_id)
    stats = _Stats()
    outcomes = OutcomeLogger(log_path)
    httpd = ThreadingHTTPServer(
        (host, port), make_handler(model, logger, stats, policy, outcomes))
    httpd.logger, httpd.stats, httpd.outcomes = logger, stats, outcomes
    return httpd
