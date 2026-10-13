"""Decision loop — System One as a component inside an iteration, not the
thing being iterated.

The DP4 confirmation-loop pattern (arXiv:2609.28940, "Calibrated Decision
Models for Autonomous Penetration-Testing Harnesses"): a code-driven loop
alternates between a generative proposer (System Two — articulate/render,
the ``converse`` ability, still MISSING) and a typed decision head (System
One — ``decide``, BUILT). The proposer offers a candidate; the decision
head adjudicates it; a non-``answer`` verdict feeds back into the next
proposal. The loop runs until confirmation (``answer``) or exhaustion.

The design intent this encodes: the decision head *serves* the loop — it
is the calibrated adjudicator each step consults, never the thing being
looped. The generative model proposes; System One decides whether the
proposal stands. Separating them is the paper's whole thesis: delegating
adjudication to the same model that proposes is the calibration failure
mode (inflated confidence, false positives accepted on plausibility).

The gate's action is the loop's control signal, with no new vocabulary:

- ``answer``      → confirmed; stop, return the accepted candidate.
- ``clarify``     → ambiguous; re-propose with the clarification in state.
- ``escalate``    → out-of-schema or low-confidence; re-propose narrowed.
- ``abstain``     → nothing fits; stop, report exhaustion.

Each iteration is one trajectory row (component 42: ``z_t, action,
z_{t+1}``), so the loop *is* the trajectory the glial regulator learns
over — the same object, not a parallel one.

No model weights run here beyond the ``decide`` the caller supplies; the
proposer is a callable so a stub proves the control flow locally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

#: Actions that end the loop. ``answer`` accepts the candidate; the rest
#: are non-terminal — the loop re-proposes with the verdict in state.
#: (``abstain`` is terminal-but-empty: nothing fit, report exhaustion.)
TERMINAL_ACCEPT = "answer"
NON_TERMINAL = ("clarify", "escalate")
ABSTAIN = "abstain"


@dataclass
class LoopStep:
    """One iteration's record — a trajectory unit (component 42)."""
    iteration: int
    candidate: Any                 # what the proposer offered
    action: str                    # the gate's verdict
    top_prob: float = 0.0
    state_items: int = 0           # how much history the state carried
    prediction_set: Optional[List[str]] = None  # APS survivors at this verdict


@dataclass
class LoopResult:
    """The loop's outcome plus its full trace.

    ``accepted`` is the confirmed candidate (or ``None`` on exhaustion);
    ``steps`` is the trajectory the regulator learns over; ``stop_reason``
    names why it ended (``confirmed`` / ``max_iterations`` / ``abstained``).
    """
    accepted: Optional[Any]
    steps: List[LoopStep] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def n_iterations(self) -> int:
        return len(self.steps)

    @property
    def confirmed(self) -> bool:
        return self.accepted is not None


def run_loop(model, propose: Callable[[List[LoopStep]], Any],
             question, max_iterations: int = 8,
             on_step: Optional[Callable[[LoopStep], None]] = None
             ) -> LoopResult:
    """The confirmation loop: propose → decide → feed the verdict back.

    Args:
        model: a ``DecisionModel`` (or anything with ``decide(state,
            question)``). Its gate action drives the loop.
        propose: ``(prior_steps) -> candidate``. The System-Two proposer.
            Receives the steps so far so it can narrow on a non-answer
            verdict (the DP4 "informed by observed behavior" move). A
            candidate is anything ``decide`` accepts as a state.
        question: the shared ``Question`` over the option set.
        max_iterations: exhaustion bound — the loop cannot run away.
        on_step: optional callback per iteration (trajectory logging hook).

    Returns:
        ``LoopResult`` with the accepted candidate, the full step trace,
        and why it stopped. The state passed to ``decide`` each iteration
        is the candidate plus the prior verdicts, so the decision head
        sees the loop's history — the composer (component 38) is what
        will later compose that history into one vector instead of a
        pooled list; the loop is correct without it, just pooled.
    """
    steps: List[LoopStep] = []
    for it in range(1, max_iterations + 1):
        candidate = propose(steps)
        # state = the candidate plus every prior verdict, so decide() sees
        # the loop's history. A list state is composed (or pooled) by the
        # model; the verdict text is what makes the history legible.
        history = [f"previous verdict: {s.action}" for s in steps]
        state = [candidate] + history if history else candidate
        pred = model.decide(state, question)
        step = LoopStep(iteration=it, candidate=candidate,
                        action=pred.action,
                        top_prob=float(max(pred.probabilities.values()))
                        if pred.probabilities else 0.0,
                        state_items=len(history) + 1,
                        prediction_set=list(pred.prediction_set)
                        if getattr(pred, "prediction_set", None) else None)
        steps.append(step)
        if on_step is not None:
            on_step(step)
        # ``unevaluated`` (no gate attached) is a committed answer with no
        # abstention signal to act on — treat it as confirmed, or an
        # ungated model loops forever on a verdict that never varies.
        if step.action in (TERMINAL_ACCEPT, "unevaluated"):
            return LoopResult(accepted=candidate, steps=steps,
                              stop_reason="confirmed")
        if step.action == ABSTAIN:
            return LoopResult(accepted=None, steps=steps,
                              stop_reason="abstained")
        # clarify / escalate → non-terminal, re-propose with the verdict
    return LoopResult(accepted=None, steps=steps, stop_reason="max_iterations")
