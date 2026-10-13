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
    tried: set = set()

    def propose(steps: List[LoopStep]) -> Any:
        if not steps:
            tried.clear()
        # candidate index = iteration position (advance through the set on
        # non-answer verdicts); wrap so the loop is bounded by max_iterations.
        idx = len(steps) % len(candidates)
        cand = candidates[idx]
        if reindex is not None:
            return reindex([idx])
        return cand

    return propose


def generate_traces(candidate_sets: Sequence[Sequence[Any]],
                    model: Any,
                    question: Any,
                    true_intents: Optional[Sequence[int]] = None,
                    difficulties: Optional[Sequence[float]] = None,
                    max_iterations: int = 3) -> List[QuestionTrace]:
    """Drive the confirmation loop over ``candidate_sets`` and collect traces.

    Each entry in ``candidate_sets`` is one question's ordered candidate list;
    the proposer advances through it on non-answer verdicts. ``question`` is the
    shared ``Question`` (the option set) passed to ``decide`` every iteration —
    the loop's docstring signature. ``true_intents``/``difficulties`` ride along
    for the corpus labels and depth stratification. Returns one QuestionTrace
    per candidate set, in input order.
    """
    traces: List[QuestionTrace] = []
    for qi, candidates in enumerate(candidate_sets):
        propose = make_proposer(list(candidates))
        result: LoopResult = run_loop(model, propose, question,
                                      max_iterations=max_iterations)
        steps = [{"iteration": s.iteration, "candidate": s.candidate,
                  "action": s.action, "top_prob": s.top_prob,
                  "state_items": s.state_items} for s in result.steps]
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
