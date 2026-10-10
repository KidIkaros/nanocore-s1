"""Component 42 — trajectory logging + outcome joins. No weights."""
import json
import threading
import urllib.request

import numpy as np
import pytest

from src.decision.gate import ConformalGate
from src.decision.model import DecisionModel
from src.decision.monitor import join_outcomes
from src.decision.schema import Question
from src.decision.scoring import CosineScorer
from src.decision.serve import OutcomeLogger, PredictionLogger, serve


def _q():
    return Question(qtype="choice", options=["a", "b", "c"])


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


def _model():
    rng = np.random.default_rng(0)
    k, n = 3, 300
    y = rng.integers(0, k, n)
    scores = rng.standard_normal((n, k)) * 0.3
    scores[np.arange(n), y] += 4.0
    g = ConformalGate(alpha=0.10, min_n=40)
    g.calibrate(scores, y)
    return DecisionModel(encoder=StubEncoder(), scorer=CosineScorer(), gate=g)


def test_prediction_carries_state_hash_and_raw_scores():
    pred = _model().decide("hello world", question=_q())
    assert pred.state_hash and len(pred.state_hash) == 16
    assert pred.raw_scores is not None and len(pred.raw_scores) == 3
    # same state → same hash (frozen encoder determinism)
    pred2 = _model().decide("hello world", question=_q())
    assert pred2.state_hash == pred.state_hash


def test_log_row_is_a_trajectory_unit(tmp_path):
    log = PredictionLogger(tmp_path / "p.jsonl", model_id="m")
    pred = _model().decide("one", question=_q())
    r1 = log.record(text="one", question=_q(), pred=pred, latency_ms=1.0,
                    policy="full", stream_id="s1",
                    thresholds=_model().gate.active_thresholds())
    r2 = log.record(text="two", question=_q(), pred=pred, latency_ms=1.0,
                    policy="full", stream_id="s1")
    assert r1["decision_id"] and r2["prev_id"] == r1["decision_id"]
    assert r2["seq"] == 1 and r1["stream_id"] == "s1"
    assert r1["obs"]["set_size"] == len(pred.prediction_set)
    assert "margin" in r1["obs"] and "entropy_norm" in r1["obs"]
    assert r1["thresholds"]["tau_answer"] is not None
    assert r1["raw_scores"] == pred.raw_scores


def test_logged_row_replays_the_logged_action(tmp_path):
    """The corpus invariant: row + thresholds → same action, no encoder."""
    from src.decision.slow import PolicyThresholds
    from src.decision.gate import action_for
    log = PredictionLogger(tmp_path / "p.jsonl")
    pred = _model().decide("x", question=_q())
    rec = log.record(text="x", question=_q(), pred=pred, latency_ms=0,
                     policy="full",
                     thresholds=_model().gate.active_thresholds())
    t = PolicyThresholds(**rec["thresholds"])
    assert action_for(rec["obs"]["top_prob"], rec["obs"]["max_score"],
                    rec["obs"]["set_size"], t, policy="full") == rec["action"]


def test_outcomes_join_by_decision_id(tmp_path):
    log = PredictionLogger(tmp_path / "p.jsonl")
    out = OutcomeLogger(tmp_path / "p.jsonl")
    pred = _model().decide("y", question=_q())
    rec = log.record(text="y", question=_q(), pred=pred, latency_ms=0,
                     policy="full")
    out.record(decision_id=rec["decision_id"], outcome="gold",
               source="bench", payload="b")
    j = join_outcomes(tmp_path / "p.jsonl")
    assert j["n"] == 1 and j["n_with_outcome"] == 1
    assert j["rows"][0]["outcomes"][0]["payload"] == "b"
    assert j["coverage_by_action"][pred.action]["with_outcome"] == 1


def test_outcome_types_are_enforced(tmp_path):
    out = OutcomeLogger(tmp_path / "p.jsonl")
    with pytest.raises(ValueError):
        out.record(decision_id="x", outcome="vibes")


@pytest.fixture()
def server(tmp_path):
    httpd = serve(_model(), port=0, log_path=str(tmp_path / "preds.jsonl"),
                  model_id="test-bundle")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown(); t.join()


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def test_http_stream_linkage_and_outcome_roundtrip(server, tmp_path):
    port = server.server_address[1]
    r1 = _post(port, "/decide", {"text": "hi there", "options": ["a", "b", "c"],
                               "stream_id": "conv-1"})
    r2 = _post(port, "/decide", {"text": "and again", "options": ["a", "b", "c"],
                               "stream_id": "conv-1"})
    assert r1["decision_id"] and r2["decision_id"] != r1["decision_id"]
    ok = _post(port, "/outcome", {"decision_id": r1["decision_id"],
                                 "outcome": "clarification_resolved"})
    assert ok["ok"]
    j = join_outcomes(tmp_path / "preds.jsonl")
    by_id = {r["decision_id"]: r for r in j["rows"]}
    assert by_id[r2["decision_id"]]["prev_id"] == r1["decision_id"]
    assert len(by_id[r1["decision_id"]]["outcomes"]) == 1
    assert j["coverage_by_action"][r1["action"]]["with_outcome"] == 1


def test_scores_can_be_dropped_for_constrained_deploys(tmp_path):
    log = PredictionLogger(tmp_path / "p.jsonl", log_scores=False)
    pred = _model().decide("z", question=_q())
    rec = log.record(text="z", question=_q(), pred=pred, latency_ms=0,
                     policy="full")
    assert "raw_scores" not in rec
