"""Trajectory-corpus generation (#19) — drive the confirmation loop to produce
the (state, action → outcome) records the predictive organ (#18) learns from.

The loop (`src/decision/loop.py`) already does the hard part: each iteration the
state is `candidate + prior verdicts` so the head sees loop history, and
`on_step` logs each step. This module wraps it over a *set* of questions with a
System-Two proposer, stratifies by difficulty so the corpus contains the hard
cases where the depth bottleneck lives, and emits self-describing per-question
traces.

Pure and stub-testable: it takes a ``model`` with ``.decide(state, question)``
and a ``propose`` callable, so a stub model drives it locally; the real encoder
runs it on Kaggle (ADR-0006).
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.decision.loop import LoopResult, LoopStep, run_loop


@dataclass
class QuestionTrace:
    """One question's full loop trace — a trajectory unit for the predictor."""
    true_intent: Optional[int]
    steps: List[Dict[str, Any]]       # per-iteration records
    resolved: bool                     # loop confirmed?
    stop_reason: str
    n_iterations: int
    difficulty: float = 0.0            # 1 - head top_prob (the depth axis)

    def to_dict(self) -> dict:
        return {"true_intent": self.true_intent, "steps": self.steps,
                "resolved": self.resolved, "stop_reason": self.stop_reason,
                "n_iterations": self.n_iterations,
                "difficulty": round(self.difficulty, 4)}


def make_proposer(candidates: Sequence[Any],
                  reindex: Optional[Callable[[List[int]], Any]] = None
                  ) -> Callable[[List[LoopStep]], Any]:
    """A minimal System-Two proposer over a fixed candidate set.

    Iteration 1: return the first candidate. On a non-answer verdict, re-propose
    from the remaining candidates (the DP4 "informed by observed behavior" move —
    narrow on what the gate rejected). ``reindex`` optionally maps the remaining
    candidate indices to whatever ``decide`` accepts as a state.

    This is deliberately simple: no new model. The loop's value is producing
    genuine multi-step traces, not a clever proposer.
    """
    def propose(steps: List[LoopStep]) -> Any:
        # candidate index = iteration position (advance through the set on
        # non-answer verdicts); wrap so the loop is bounded by max_iterations.
        idx = len(steps) % len(candidates)
        cand = candidates[idx]
        if reindex is not None:
            return reindex([idx])
        return cand

    return propose


def make_narrowing_proposer(
        candidates: Sequence[Any],
        head_scorer: Any,
        question_options: Sequence[str],
        encoder: Any = None) -> Callable[[List[LoopStep]], Any]:
    """A DP4 proposer that narrows on the gate's surviving prediction set.

    On a non-answer verdict, the loop's ``LoopStep`` now carries the APS
    ``prediction_set`` — the options the conformal gate kept. Instead of
    advancing blindly through the candidate list (which escalates forever, as
    the v10 corpus showed), this proposer re-proposes the **highest-scored
    surviving option** from that set. That is the "informed by observed
    behavior" move: use what the gate just told us survived, not the next
    arbitrary candidate. Produces genuine recovery traces (iteration 2 can
    resolve) — the two-sided signal the consequence predictor needs.

    ``head_scorer.scores(state_vec, option_vecs)`` takes EMBEDDING vectors, so
    ``encoder`` (with ``.encode_state``/``.encode_options``) encodes the state
    and survivors first. Falls back to the linear proposer on iteration 1 (no
    prediction set yet) or when no encoder is given (scores as zeros → first
    survivor).
    """
    offered: set = set()

    def _rank(survivors: List[str]) -> List[str]:
        if encoder is None or head_scorer is None:
            return list(survivors)
        try:
            state_vec = _encode_one(encoder, steps_state[0])
            opt_vecs = encoder.encode_options(list(survivors))
            sc = head_scorer.scores(state_vec, opt_vecs)
            sc = sc.tolist() if hasattr(sc, "tolist") else list(sc)
            return [survivors[i] for i in sorted(
                range(len(survivors)), key=lambda i: sc[i], reverse=True)]
        except Exception:
            return list(survivors)

    steps_state: List[Any] = []

    def propose(steps: List[LoopStep]) -> Any:
        if not steps:
            offered.clear()
            steps_state.clear()
            return candidates[0]
        # remember the current state (candidate) so _rank can encode it
        steps_state[:] = [steps[-1].candidate]
        last = steps[-1]
        survivors = last.prediction_set
        if not survivors:
            idx = len(steps) % len(candidates)
            return candidates[idx]
        ranked = _rank(list(survivors))
        for opt in ranked:
            if opt not in offered:
                offered.add(opt)
                return opt
        return ranked[0]

    return propose


def _encode_one(encoder: Any, state: Any) -> np.ndarray:
    """Encode a single state item to a vector via the encoder's state encoder."""
    import numpy as _np
    emb = encoder.encode_state([state] if not isinstance(state, (list, tuple))
                               else list(state))
    return _np.asarray(emb, dtype=float).reshape(-1)


def generate_traces(candidate_sets: Sequence[Sequence[Any]],
                    model: Any,
                    question: Any,
                    true_intents: Optional[Sequence[int]] = None,
                    difficulties: Optional[Sequence[float]] = None,
                    max_iterations: int = 3,
                    proposer_factory: Optional[Callable[
                        [Sequence[Any]], Callable[[List[LoopStep]], Any]]
                    ] = None) -> List[QuestionTrace]:
    """Drive the confirmation loop over ``candidate_sets`` and collect traces.

    Each entry in ``candidate_sets`` is one question's ordered candidate list;
    the proposer advances through it on non-answer verdicts. ``question`` is the
    shared ``Question`` (the option set) passed to ``decide`` every iteration —
    the loop's docstring signature. ``proposer_factory`` builds the per-question
    proposer (default: the linear ``make_proposer``); pass
    ``make_narrowing_proposer``'s factory for DP4 recovery traces.
    ``true_intents``/``difficulties`` ride along for the corpus labels and depth
    stratification. Returns one QuestionTrace per candidate set, in input order.
    """
    traces: List[QuestionTrace] = []
    for qi, candidates in enumerate(candidate_sets):
        if proposer_factory is not None:
            propose = proposer_factory(list(candidates))
        else:
            propose = make_proposer(list(candidates))
        result: LoopResult = run_loop(model, propose, question,
                                      max_iterations=max_iterations)
        steps = [{"iteration": s.iteration, "candidate": s.candidate,
                  "action": s.action, "top_prob": s.top_prob,
                  "state_items": s.state_items,
                  "state_emb": (s.state_emb.tolist()
                                if hasattr(s.state_emb, "tolist")
                                else s.state_emb)}
                 for s in result.steps]
        traces.append(QuestionTrace(
            true_intent=(int(true_intents[qi]) if true_intents is not None
                         else None),
            steps=steps,
            resolved=result.confirmed,
            stop_reason=result.stop_reason,
            n_iterations=result.n_iterations,
            difficulty=(float(difficulties[qi]) if difficulties is not None
                        else 0.0)))
    return traces


def corpus_stats(traces: Sequence[QuestionTrace]) -> Dict[str, Any]:
    """Summary the ticket requires: how many questions resolved per iteration
    count (the depth structure), plus the multi-step count."""
    n = len(traces)
    if n == 0:
        return {"n": 0}
    by_iter: Dict[int, Dict[str, int]] = {}
    for t in traces:
        b = by_iter.setdefault(t.n_iterations, {"n": 0, "resolved": 0})
        b["n"] += 1
        b["resolved"] += int(t.resolved)
    multi = sum(1 for t in traces if t.n_iterations > 1)
    return {
        "n": n,
        "resolved": sum(t.resolved for t in traces),
        "resolve_rate": round(sum(t.resolved for t in traces) / n, 4),
        "multi_step": multi,
        "multi_step_rate": round(multi / n, 4),
        "by_iterations": {str(k): v for k, v in sorted(by_iter.items())},
    }
