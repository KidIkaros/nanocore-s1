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
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from src.decision.schema import Question


# ── prediction log ───────────────────────────────────────────────────────────

class PredictionLogger:
    """Append-only JSONL log of every decision — inputs, outputs, model id,
    latency. Rotation/retention is the deployer's concern; correctness here is
    that every served decision leaves a record."""

    def __init__(self, path, model_id: str = ""):
        self.path = Path(path)
        self.model_id = model_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *, text: str, question: Question, pred, latency_ms: float,
               policy: Optional[str]) -> dict:
        rec = {"ts": time.time(), "model_id": self.model_id,
               "input": text, "qtype": pred.qtype, "labels": pred.labels,
               "policy": policy, "action": pred.action,
               "top_prob": pred.answer_confidence,
               "prediction_set": pred.prediction_set,
               "alpha": pred.alpha, "latency_ms": round(latency_ms, 2)}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
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
                 default_policy: Optional[str] = None):
    """Build a request handler bound to ``model``. Pure function so tests can
    construct a server on an ephemeral port without the CLI."""

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
                logger.record(text=body["text"], question=q, pred=pred,
                              latency_ms=ms,
                              policy=body.get("policy", default_policy))
                stats.observe(pred.action, ms)
                self._send(200, {"action": pred.action, "qtype": pred.qtype,
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
    httpd = ThreadingHTTPServer((host, port),
                                make_handler(model, logger, stats, policy))
    httpd.logger, httpd.stats = logger, stats
    return httpd
