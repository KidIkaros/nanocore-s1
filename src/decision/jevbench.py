"""jev-benchmarking adapter — NanoCore answers in the reference harness's format.

The harness (``AppliedMachineLearning-Lab/jev-benchmarking``, MIT) builds one
request per example — ``state`` + ``questions`` — and scores answers in Jev's
response format (``evaluate.compute(task, exs, answers)``). This module maps
each question onto ``DecisionModel.decide`` and emits that format, so every
metric, bootstrap CI, and per-dataset result in the harness applies unchanged
— a same-protocol comparison against Jev, Qwen3.8-27B and Gemma-4-E4B on
identical requests.

Mapping choices, all deliberate:

- **Instructions fold into the state text.** ``Question`` carries no
  instructions field (roadmap component 34); the question text is prepended to
  the state — the same prompt content Jev reads, different plumbing.
- **Choice option text mirrors ``openmodel.prompts.render``:** anonymous keys
  (A, B, 1, 2, …) score their description; named keys score ``key (desc)``, or
  the bare key when no description exists. The scorer sees the same option
  content the open-model arm reads.
- **Noul uses the prop-in-options contract** — the proposition is an option
  and ``P(proposition)`` is the noul probability (measured .97–1.0 AUROC vs
  .60–.69 for prop-in-state on CLINC150; the proposition does not survive
  mean-pooling into a state vector).
- **Score** treats each level description as an option; the reported score is
  the expected level index under the returned probabilities.

The adapter never imports ``jev_benchmarking`` — it takes duck-typed
``state``/``questions``/``gold`` so the kernel can install the package and
drive it without this package gaining a dependency.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.decision.schema import Question

_ANONYMOUS_KEY = re.compile(r"^(?:[A-Z]{1,2}|\d+)$")

#: The generic alternative paired against a proposition in the noul arm — the
#: measured contract is proposition-vs-generic-negative, not prop-vs-prop.
NOUL_NEGATIVE = "the statement does not apply"


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, indent=1)


def option_label(key: Any, desc: Any) -> str:
    """The option text a scorer should see — mirrors the harness's render."""
    if _ANONYMOUS_KEY.match(str(key)):
        return _fmt(desc) if desc is not None else str(key)
    return str(key) if desc is None else f"{key} ({_fmt(desc)})"


def confidence(probs: list) -> float:
    """The harness's confidence formula: (K * p_max - 1) / (K - 1), clipped."""
    k = len(probs)
    return max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0


def _choice_answer(model, state_text: str, q: dict) -> dict:
    keys = list(q["criteria"])
    labels = [option_label(k, q["criteria"][k]) for k in keys]
    pred = model.decide(state_text, Question(qtype="choice", options=labels))
    probs = {k: pred.probabilities.get(l, 0.0) for k, l in zip(keys, labels)}
    z = sum(probs.values())
    probs = {k: v / z for k, v in probs.items()} if z > 0 else {
        k: 1.0 / len(keys) for k in keys}
    return {"type": "choice", "choice": max(probs, key=probs.get),
            "probabilities": probs, "confidence": confidence(list(probs.values()))}


def _noul_answer(model, state_text: str, q: dict) -> dict:
    prop = _fmt(q["instructions"])
    pred = model.decide(state_text, Question(
        qtype="noul", options=[prop, NOUL_NEGATIVE]))
    return {"type": "noul", "noul": float(pred.probabilities.get(prop, 0.0))}


def _score_answer(model, state_text: str, q: dict) -> dict:
    levels = list(q["criteria"])
    labels = [_fmt(l) for l in levels]
    pred = model.decide(state_text, Question(qtype="score", options=labels))
    probs = [pred.probabilities.get(l, 0.0) for l in labels]
    z = sum(probs)
    probs = [p / z for p in probs] if z > 0 else [1.0 / len(levels)] * len(levels)
    return {"type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): l for i, l in enumerate(levels)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": confidence(probs)}


def nanocore_answers(model, state: Any, questions: Dict[str, dict]) -> Dict[str, dict]:
    """One harness request → Jev-format answers for every question it carries.

    ``model`` is a ``DecisionModel`` (or anything with a compatible
    ``decide(state, question)``). The returned dict keys are the harness's
    question ids; values are the answer shapes ``evaluate.compute`` consumes.
    """
    out = {}
    for qid, q in questions.items():
        kind = q["type"]
        # Instructions fold into the state for choice/score — the question text
        # is context. For noul the instructions *are* the proposition, and the
        # prop-in-options contract already places them in an option; putting
        # the same text in the state would inflate P(prop) through literal
        # overlap — the surface-cue failure G4 exists to catch.
        instr = (_fmt(q["instructions"])
                 if kind != "noul" and q.get("instructions") is not None else "")
        state_text = f"{instr}\n\n{_fmt(state)}" if instr else _fmt(state)
        if kind == "choice":
            out[qid] = _choice_answer(model, state_text, q)
        elif kind == "noul":
            out[qid] = _noul_answer(model, state_text, q)
        elif kind == "score":
            out[qid] = _score_answer(model, state_text, q)
        else:
            raise ValueError(f"unknown question type {kind!r}")
    return out


def run_task(model, task, split: str = "eval", limit: int | None = None) -> dict:
    """Run one harness task with NanoCore as the model; return its metrics.

    Uses the harness's own ``examples``/``compute`` — identical requests,
    identical scoring. The only variable is whose probabilities fill the
    answer slots.
    """
    from jev_benchmarking.evaluate import compute
    exs = task.examples(split, limit=limit)
    answers = [nanocore_answers(model, e.state, e.questions) for e in exs]
    metrics = compute(task, exs, answers)
    return {"task": task.name, "n_examples": len(exs), **metrics}
