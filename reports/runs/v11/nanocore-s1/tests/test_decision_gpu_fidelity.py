"""GPU-fidelity guard: every real decision path must survive an encoder
that returns CUDA-like tensor output.

The recurring failure (v3/v28b: prompt-name, then cuda-tensor) was always the
same structure — the local stub returned plain numpy, so GPU-path bugs only
surfaced on Kaggle. This module makes the tensor_like stub the DEFAULT encoder
for the real paths, so the blind spot is closed at the source rather than
patched crash-by-crash. Any future raw `.numpy()` on encoder output fails here.
"""
import numpy as np
import pytest

from src.decision.backends import StubEncoder
from src.decision.model import DecisionModel
from src.decision.scoring import CosineScorer, TaskHead
from src.decision.schema import Question
from src.decision.pair import PairScorer
from src.decision.loop import run_loop
from src.decision.fusion import late_fusion_decide


@pytest.fixture
def gpu_encoder():
    """An encoder that behaves like the real CUDA one: .numpy() raises."""
    return StubEncoder(dim=32, tensor_like=True)


def test_decide_survives_gpu_encoder(gpu_encoder):
    m = DecisionModel(encoder=gpu_encoder, scorer=CosineScorer(), gate=None,
                      cache=None)
    p = m.decide("hello world", Question(qtype="choice", options=["a", "b"]))
    assert p.probabilities  # produced a decision


def test_taskhead_fit_and_score_survives_gpu_encoder(gpu_encoder):
    from src.decision.items import to_numpy
    texts = [f"class {i % 3} text {i}" for i in range(60)]
    y = [i % 3 for i in range(60)]
    X = to_numpy(gpu_encoder.encode(texts))   # tensor-like -> routed through to_numpy
    head = TaskHead()
    head.fit(X, y, labels=[0, 1, 2])
    logits = head.logits(X[:5])
    assert logits.shape == (5, 3)


def test_pair_scorer_survives_gpu_encoder(gpu_encoder):
    ps = PairScorer(encoder=gpu_encoder)
    s = ps.similarity("alpha beta", "alpha beta")
    # identical texts -> cosine ~1.0; float arithmetic can overshoot by an ulp
    assert isinstance(s, float) and abs(s - 1.0) < 1e-6


def test_loop_survives_gpu_encoder(gpu_encoder):
    m = DecisionModel(encoder=gpu_encoder, scorer=CosineScorer(), gate=None,
                      cache=None)

    def propose(steps):
        return "candidate"

    result = run_loop(m, propose, Question(qtype="choice",
                                           options=["a", "b"]),
                      max_iterations=3)
    assert result.steps  # ran at least one step to a terminal state


def test_late_fusion_survives_gpu_encoder(gpu_encoder):
    # late_fusion_decide takes (model, state_groups, question) and must stay
    # pure regardless of encoder tensor-ness.
    m = DecisionModel(encoder=gpu_encoder, scorer=CosineScorer(), gate=None,
                      cache=None)
    probs, per_modality = late_fusion_decide(m, [["img text a"], ["q text a"]],
                                             Question(qtype="choice",
                                                      options=["a", "b"]),
                                             rule="mean")
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert len(per_modality) == 2  # auditable per-modality
