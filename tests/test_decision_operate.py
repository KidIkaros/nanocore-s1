"""The operate flow — log, monitor, handlers, registry, shadow — end to end.

These are shipped surfaces (serve's logger, the monitor, the handlers, the
registry) and nothing else in the suite exercises them together: each module has
unit tests, but the *loop* an operator actually runs had no coverage. Stubs
stand in for the model, so this runs in a second and needs no weights.
"""
import json

import numpy as np
import pytest

from src.decision.adapt import AdaptConfig, adapt
from src.decision.handlers import QueuedEscalation, handle
from src.decision.monitor import Monitor, read_log, shadow_compare
from src.decision.registry import Registry, data_fingerprint
from src.decision.schema import Prediction, Question
from src.decision.serve import PredictionLogger

N_CLASSES, PER_CLASS, N_LOG = 6, 220, 120
LABELS = [f"intent{i}" for i in range(N_CLASSES)]


class StubEncoder:
    """Class-separable embeddings — no weights, no framework."""

    def encode(self, texts, **kw):
        return np.array([[1.0 if f"intent{i}" in t else 0.05 for i in range(N_CLASSES)]
                         for t in texts], dtype=np.float32)

    def encode_state(self, items):
        return self.encode([str(i) for i in items])

    def encode_options(self, texts):
        return np.array([[1.0 if t == f"intent{i}" else 0.05
                          for i in range(N_CLASSES)] for t in texts], dtype=np.float32)


class StubModel:
    def decide(self, text, question):
        return stub_decision(text, question)


def corpus() -> tuple:
    texts = [f"intent{i} sample{j}" for i in range(N_CLASSES) for j in range(PER_CLASS)]
    labels = [f"intent{i}" for i in range(N_CLASSES) for _ in range(PER_CLASS)]
    return texts, labels


class OrderedScorer:
    """Schema-bound scorer: logits live in ``labels_`` order, not option order."""

    labels_ = ["beta", "alpha"]
    temperature = 1.0

    def scores(self, state_vec, option_vecs=None):
        return np.array([5.0, -5.0])   # strong logit for "beta"


def test_decide_realigns_a_schema_bound_scorer_to_option_order():
    """The v16 failure's next layer: a TaskHead fitted via np.unique returns
    logits in sorted-label order while questions carry CLINC's order. Without
    realignment every prediction silently maps to the wrong intent — no
    exception, just garbage."""
    from src.decision.model import DecisionModel
    model = DecisionModel(encoder=StubEncoder(), scorer=OrderedScorer())
    pred = model.decide("anything", Question(qtype="choice", options=["alpha", "beta"]))
    assert pred.probabilities["beta"] > 0.99
    assert pred.max_score == 5.0


def test_decide_fails_loudly_on_options_outside_the_label_space():
    from src.decision.model import DecisionModel
    model = DecisionModel(encoder=StubEncoder(), scorer=OrderedScorer())
    with pytest.raises(ValueError, match="outside the scorer's label space"):
        model.decide("anything",
                     Question(qtype="choice", options=["alpha", "gamma"]))


def stub_decision(text: str, question: Question) -> Prediction:
    """A Prediction shaped like the model's, with a mix of actions."""
    hit = text.split()[0] if text.split() else "intent0"
    escalating = hash(text) % 3 == 0
    return Prediction(qtype=question.qtype, labels=list(question.options),
                      action="escalate" if escalating else "answer",
                      probabilities={label: 0.1 for label in question.options},
                      prediction_set=[hit],
                      answer_confidence=0.4 if escalating else 0.9,
                      entropy_confidence=0.2, max_score=0.6, alpha=0.1)


def log_traffic(path, question, texts) -> list:
    """Write real-shaped decisions through the logger the server uses."""
    logger = PredictionLogger(path, model_id="test")
    for text in texts[:N_LOG]:
        logger.record(text=text, question=question,
                      pred=stub_decision(text, question), latency_ms=1.0,
                      policy=None)
    return read_log(path)


def retrain(root, span) -> tuple:
    texts, labels = corpus()
    result = adapt(texts[span], labels[span], StubEncoder(),
                   cfg=AdaptConfig(min_cal=100), out_dir=root / f"rt{span.start}")
    return result.bundle_dir, result.report, data_fingerprint(texts[span], labels[span])


def test_prediction_log_is_readable_by_the_monitor(tmp_path):
    question = Question(qtype="choice", options=LABELS)
    texts, _ = corpus()
    records = log_traffic(tmp_path / "preds.jsonl", question, texts)
    assert len(records) == N_LOG

    status = Monitor(tmp_path / "preds.jsonl").check()
    assert status["n"] == N_LOG
    assert status["operational"]["latency_ms"]["p99"] > 0
    assert 0.0 <= status["ml"]["aggregate"]["escalate_rate"] <= 1.0


def test_every_action_reaches_a_destination(tmp_path):
    question = Question(qtype="choice", options=LABELS)
    texts, _ = corpus()
    queue = tmp_path / "escalations.jsonl"
    dispatcher = QueuedEscalation(queue)
    counts: dict = {}
    for text in texts[:N_LOG]:
        result = handle(stub_decision(text, question), state=text,
                        question=question, escalate_to=dispatcher)
        counts[result.action] = counts.get(result.action, 0) + 1

    assert set(counts) <= {"answer", "clarify", "escalate"}
    assert counts.get("escalate", 0) > 0
    assert sum(1 for _ in queue.open()) == counts["escalate"]


def test_retrain_registers_with_lineage_and_rolls_back(tmp_path):
    registry = Registry(tmp_path / "registry")
    v1 = registry.register("clinc", *retrain(tmp_path, slice(0, 700)))
    registry.promote("clinc", v1)
    v2 = registry.register("clinc", *retrain(tmp_path, slice(700, 1400)))

    assert registry.versions("clinc") == ["v1", "v2"]
    assert registry.lineage("clinc", v2)["parent"] == v1

    assert registry.promote("clinc", v2) == v2
    assert registry.current("clinc") == v2
    assert registry.rollback("clinc") == v1
    assert registry.current("clinc") == v1


def test_shadow_run_compares_a_candidate_on_logged_traffic(tmp_path):
    """The evidence a promotion decision needs before it touches ``current``."""
    question = Question(qtype="choice", options=LABELS)
    texts, _ = corpus()
    records = log_traffic(tmp_path / "preds.jsonl", question, texts)

    shadow = shadow_compare(records, StubModel())
    assert shadow["n"] == N_LOG
    assert 0.0 <= shadow["agreement"] <= 1.0
    assert sum(shadow["action_transitions"].values()) == N_LOG


def test_shadow_run_counts_a_candidate_that_cannot_decide(tmp_path):
    """A broken candidate must not look like an empty log.

    Measured in the v15 kernel: a retrained bundle whose option labels were raw
    class ids raised on every call, and the silent ``except`` reported
    ``n = 0`` — indistinguishable from having no traffic to replay.
    """
    class BrokenModel:
        def decide(self, text, question):
            raise ValueError("labels not in the option space")

    question = Question(qtype="choice", options=LABELS)
    texts, _ = corpus()
    records = log_traffic(tmp_path / "preds.jsonl", question, texts)

    shadow = shadow_compare(records, BrokenModel())
    assert shadow["n"] == 0
    assert shadow["n_failed"] == N_LOG
    assert "option space" in shadow["first_error"]


def test_shadow_per_record_carries_the_log_index(tmp_path):
    """A caller pairing shadow results back to the log must not rely on failed
    records being absent — one failure would shift every later index."""
    class FlakyModel:
        def __init__(self):
            self.n = 0

        def decide(self, text, question):
            self.n += 1
            if self.n == 3:
                raise ValueError("boom")
            return stub_decision(text, question)

    question = Question(qtype="choice", options=LABELS)
    texts, _ = corpus()
    records = log_traffic(tmp_path / "preds.jsonl", question, texts)

    shadow = shadow_compare(records, FlakyModel())
    assert shadow["n_failed"] == 1
    indices = [r["i"] for r in shadow["per_record"]]
    assert 2 not in indices
    assert indices == sorted(indices)
    assert shadow["per_record"][0]["input"] == records[0]["input"]
    assert shadow["per_record"][2]["input"] == records[3]["input"]


def test_retrain_labels_are_texts_not_raw_class_ids(tmp_path):
    """The v15 failure: ``adapt`` was handed raw class ids, so the bundle's
    option labels were "11"/"42" and nothing that knows the task could serve it.

    The end-to-end check for this is the kernel's ``shadow_ran`` verdict; this
    asserts the shape of the contract at the boundary we control.
    """
    texts, labels = corpus()
    result = adapt(texts, labels, StubEncoder(), cfg=AdaptConfig(min_cal=100),
                   out_dir=tmp_path / "bundle")
    assert result.report["n_classes"] == N_CLASSES
    assert all(not str(label).isdigit() for label in set(labels))


def test_registry_refuses_to_promote_a_version_with_no_bundle(tmp_path):
    registry = Registry(tmp_path / "registry")
    with pytest.raises(ValueError):
        registry.promote("clinc", "v9")
