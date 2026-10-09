"""Serving layer — real HTTP roundtrip on 127.0.0.1, no weights."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from src.decision.gate import ConformalGate
from src.decision.model import DecisionModel
from src.decision.scoring import CosineScorer
from src.decision.serve import PredictionLogger, serve


class StubEncoder:
    model_name = "stub"

    def _vec(self, text):
        center = np.random.default_rng(
            abs(hash(text.split()[-1])) % (2**31)).standard_normal(64)
        return center / np.linalg.norm(center)

    def encode_state(self, items):
        return np.stack([self._vec(t if isinstance(t, str) else t.text)
                         for t in items])

    def encode_options(self, texts):
        return np.stack([self._vec(t) for t in texts])


def _calibrated_gate():
    rng = np.random.default_rng(0)
    k, n = 3, 300
    y = rng.integers(0, k, n)
    scores = rng.standard_normal((n, k)) * 0.3
    scores[np.arange(n), y] += 4.0
    g = ConformalGate(alpha=0.10, min_n=40)
    g.calibrate(scores, y)
    return g


@pytest.fixture()
def server(tmp_path):
    model = DecisionModel(encoder=StubEncoder(), scorer=CosineScorer(),
                          gate=_calibrated_gate())
    httpd = serve(model, port=0, log_path=str(tmp_path / "preds.jsonl"),
                  model_id="test-bundle")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown(); t.join()


def _post(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/decide", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def test_decide_roundtrip(server):
    out = _post(server.server_port,
                {"text": "route to alpha please alpha", "options": ["alpha", "beta", "gamma"]})
    assert out["action"] in ("answer", "clarify", "escalate", "abstain")
    assert set(out["probabilities"]) == {"alpha", "beta", "gamma"}
    assert out["latency_ms"] >= 0


def test_decision_logged_with_latency(server, tmp_path):
    _post(server.server_port, {"text": "pick beta beta", "options": ["alpha", "beta"]})
    lines = (tmp_path / "preds.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["model_id"] == "test-bundle"
    assert rec["latency_ms"] >= 0 and rec["action"]


def test_healthz_and_stats(server):
    _post(server.server_port, {"text": "a gamma gamma", "options": ["alpha", "gamma"]})
    h = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{server.server_port}/healthz").read())
    assert h["ok"] and h["model_id"] == "test-bundle"
    s = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{server.server_port}/stats").read())
    assert s["requests"] == 1 and "p95" in s["latency_ms"]


def test_bad_request_400(server):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/decide",
        data=json.dumps({"options": ["a"]}).encode(),  # no text
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400
