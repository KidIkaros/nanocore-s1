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
  mean-pooling into a state vector). The proposition is *not* folded into the
  state — verbatim overlap would inflate P(prop) through surface matching, the
  failure G4's withheld-state probe exists to catch.
- **Score** treats each level description as an option; the reported score is
  the expected level index under the returned probabilities.
- **The gated arm abstains by not answering.** A head whose decision action is
  not ``answer`` withholds the whole example: the harness counts answered
  examples, so coverage drops and ``sel_acc``/``aurc`` show whether the
  abstention was well-aimed — the differentiator made legible on their
  metrics, not ours.

The adapter never imports ``jev_benchmarking`` — it takes duck-typed
``state``/``questions``/``gold`` so the kernel can install the package and
drive it without this package gaining a dependency.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

from src.decision.schema import Question

#: A pair state is two text items; the pair scorer reads them directly.
#: The scorer is injected by the caller (the kernel builds it with the
#: real encoder); None means no pair path and the pooled-state scoring
#: stands — the adapter stays weight-free and unit-testable.

_ANONYMOUS_KEY = re.compile(r"^(?:[A-Z]{1,2}|\d+)$")

#: The generic alternative paired against a proposition in the noul arm — the
#: measured contract is proposition-vs-generic-negative, not prop-vs-prop.
NOUL_NEGATIVE = "the statement does not apply"

#: Actions that answer. ``unevaluated`` (no gate attached) counts as answered —
#: abstention is a gate decision, and a gate that is not there cannot make one.
_ANSWERED = {"answer", "unevaluated"}


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, indent=1)


#: Terse rubric levels ("Very negative.") are near-identical in embedding
#: space, so cosine cannot rank them — v27's sst5 anchor came in at .177,
#: indistinguishable from 5-way chance. Rich levels (stsb's full sentences)
#: pass through unchanged; expansion is idempotent by word count.
_TERSE_LEVEL_WORDS = 4


def _level_text(level: Any) -> str:
    """The text a scorer sees for one rubric level.

    A terse level is expanded into a full sentence so the encoder can rank
    it; an already-rich level is returned untouched. The harness scores
    against the ORIGINAL level indices — this only changes what the scorer
    reads, never the answer shape.
    """
    text = _fmt(level)
    if len(text.split()) > _TERSE_LEVEL_WORDS:
        return text
    return f"The text expresses this sentiment: {text.rstrip('.')}."


def option_label(key: Any, desc: Any) -> str:
    """The option text a scorer should see — mirrors the harness's render."""
    if _ANONYMOUS_KEY.match(str(key)):
        return _fmt(desc) if desc is not None else str(key)
    return str(key) if desc is None else f"{key} ({_fmt(desc)})"


def confidence(probs: List[float]) -> float:
    """The harness's confidence formula: (K * p_max - 1) / (K - 1), clipped."""
    k = len(probs)
    return max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0


def _choice_answer(model, state, q: dict) -> Optional[dict]:
    keys = list(q["criteria"])
    labels = [option_label(k, q["criteria"][k]) for k in keys]
    pred = model.decide(state, Question(qtype="choice", options=labels))
    if pred.action not in _ANSWERED:
        return None
    probs = {k: pred.probabilities.get(l, 0.0) for k, l in zip(keys, labels)}
    z = sum(probs.values())
    probs = {k: v / z for k, v in probs.items()} if z > 0 else {
        k: 1.0 / len(keys) for k in keys}
    return {"type": "choice", "choice": max(probs, key=probs.get),
            "probabilities": probs, "confidence": confidence(list(probs.values()))}


def _pair_items(state) -> Optional[list]:
    """The two text items of a pair state, or ``None`` if not a pair.

    A pair is a state with exactly two string items — the two sentences
    a pair question relates (paws/stsb sentence_1+sentence_2, boolq
    passage+question). The pair scorer reads them directly; any other
    shape (single item, media, three-plus items) is not a pair and
    scores through the pooled-state path.
    """
    items = getattr(state, "items", state)
    if isinstance(items, (str, dict)) or not isinstance(items, (list, tuple)):
        return None
    items = [i for i in items if isinstance(i, str)]
    # A pair is two text items. The choice path folds the instruction in
    # as a leading item (3 total); the pair is the two NON-instruction
    # items, so take the last two when exactly 2 or 3 string items exist.
    if len(items) in (2, 3):
        return items[-2:]
    return None


def _noul_answer(model, state, q: dict, pair=None) -> Optional[dict]:
    prop = _fmt(q["instructions"])
    items = _pair_items(state)
    if items is not None and pair is not None:
        # Pair noul: P(the two items match) via cosine(u,v). The
        # proposition is the relationship, not an option — prop-in-options
        # would score the proposition text against a pooled state, the
        # failure the pair path exists to replace.
        return {"type": "noul", "noul": float(pair.noul(items[0], items[1]))}
    pred = model.decide(state, Question(
        qtype="noul", options=[prop, NOUL_NEGATIVE]))
    if pred.action not in _ANSWERED:
        return None
    return {"type": "noul", "noul": float(pred.probabilities.get(prop, 0.0))}


def _score_answer(model, state, q: dict, pair=None) -> Optional[dict]:
    levels = list(q["criteria"])
    labels = [_level_text(l) for l in levels]
    items = _pair_items(state)
    if items is not None and pair is not None and len(levels) >= 2:
        # Pair score: the rubric rates the SIMILARITY of the two items.
        # Map the cosine onto the level scale so the expected index
        # varies with the pair instead of sitting at the middle.
        cos = pair.similarity(items[0], items[1])
        # cosine [-1,1] -> level index [0, k-1]
        frac = (cos + 1.0) / 2.0
        expected = frac * (len(levels) - 1)
        # a peaked distribution at the expected level keeps the harness's
        # argmax and spearman well-defined without a second forward pass
        w = np.exp(-0.5 * ((np.arange(len(levels)) - expected) / 0.6) ** 2)
        w /= w.sum()
        return {"type": "score",
                "score": float(expected),
                "legend": {str(i): _fmt(l) for i, l in enumerate(levels)},
                "probabilities": {str(i): float(p) for i, p in enumerate(w)},
                "confidence": confidence(list(w))}
    pred = model.decide(state, Question(qtype="score", options=labels))
    if pred.action not in _ANSWERED:
        return None
    probs = [pred.probabilities.get(l, 0.0) for l in labels]
    z = sum(probs)
    probs = [p / z for p in probs] if z > 0 else [1.0 / len(levels)] * len(levels)
    return {"type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "legend": {str(i): _fmt(l) for i, l in enumerate(levels)},
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "confidence": confidence(probs)}


def _state_items(state: Any) -> list:
    """State → item list for ``decide``.

    **The v27 pair fix.** A state dict used to be JSON-rendered into ONE
    string, so a two-sentence pair was encoded as a single item: the
    composer saw T=1, mean-pooling averaged the sentences together, and
    cosine(option, pooled_pair) could not express "are these two similar?"
    — paws .440, stsb .215 and boolq .583 are all pair tasks, and the pair
    signal was destroyed before the scorer ever ran.

    Splitting the dict into one item per field gives the composer (or the
    pooler) per-sentence vectors — the substrate a pair-aware scorer reads.
    A single-field state stays one item; a bare string stays one item.
    """
    if isinstance(state, str):
        return [state]
    if isinstance(state, dict):
        return [str(v) for v in state.values()]
    return [str(s) for s in state]


def nanocore_answers(model, state: Any, questions: Dict[str, dict],
                     pair=None) -> Optional[Dict[str, dict]]:
    """One harness request → Jev-format answers, or ``None`` on abstention.

    ``model`` is a ``DecisionModel`` (or anything with a compatible
    ``decide(state, question)``). The returned dict keys are the harness's
    question ids; values are the answer shapes ``evaluate.compute`` consumes.

    ``pair`` is an optional ``PairScorer``. When supplied AND the state is
    a two-text-item pair, noul and score questions route through
    ``cosine(u, v)`` — the native similarity primitive — instead of
    scoring a pooled state against option text. ``None`` (the default)
    keeps the pooled-state path, so the adapter stays weight-free and
    unit-testable.

    Abstention is example-level: a request bundles several heads (one state,
    many questions — GoEmotions fans out to ~28 nouls), so a single withheld
    head withholds the request. Finer-grained per-head abstention would need
    the harness to score partial answers, which its ``compute`` does not do.
    """
    from src.decision.schema import State

    out = {}
    for qid, q in questions.items():
        kind = q["type"]
        instr = (_fmt(q["instructions"])
                 if kind != "noul" and q.get("instructions") is not None else "")
        items = _state_items(state)
        if len(items) > 1:
            # The pair case: the instruction is part of the comparison
            # prompt and gets its own item, alongside the split fields —
            # per-item vectors are the whole point of the v27 fix.
            state_obj = State(items=([instr] + items if instr else items))
        elif instr:
            # Single-field: preserve the measured shape exactly — the
            # prompt-ablation numbers came from the instruction prepended
            # to the state text as ONE item. Splitting here would change
            # the pooler's input for every single-field anchor task.
            state_obj = State(items=[f"{instr}\n\n{items[0]}"])
        else:
            state_obj = State(items=items)
        if kind == "choice":
            a = _choice_answer(model, state_obj, q)
        elif kind == "noul":
            a = _noul_answer(model, state_obj, q, pair=pair)
        elif kind == "score":
            a = _score_answer(model, state_obj, q, pair=pair)
        else:
            raise ValueError(f"unknown question type {kind!r}")
        if a is None:
            return None
        out[qid] = a
    return out


def run_task(model, task, split: str = "eval", limit: int | None = None,
             pair=None) -> dict:
    """Run one harness task with NanoCore as the model; return its metrics.

    Uses the harness's own ``examples``/``compute`` — identical requests,
    identical scoring. The only variable is whose probabilities fill the
    answer slots. Abstained examples count in ``n_examples``/``coverage`` but
    not in the metrics — the same accounting the harness applies to timed-out
    or rejected requests.

    ``pair`` is an optional ``PairScorer`` forwarded to
    ``nanocore_answers``; see its docstring.
    """
    from jev_benchmarking.evaluate import compute
    exs = task.examples(split, limit=limit)
    answered = [(e, nanocore_answers(model, e.state, e.questions, pair=pair))
                for e in exs]
    kept = [(e, a) for e, a in answered if a is not None]
    metrics = compute(task, [e for e, _ in kept], [a for _, a in kept]) if kept else {}
    return {"task": task.name, "n_examples": len(exs), "n_answered": len(kept),
            "coverage": len(kept) / len(exs) if exs else 0.0, **metrics}
