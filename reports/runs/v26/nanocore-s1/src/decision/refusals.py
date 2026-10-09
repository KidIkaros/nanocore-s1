"""Refusals — a taxonomy of reasons and invariants, not a list of example strings.

**Scope.** These are *decision abstentions*, not content moderation. The model
carries no content policy: it declines when it cannot map an input to one of the
caller's options with enough confidence. Refusing a *topic* is an upstream
component's job and is not in this architecture.

**Where the design comes from.** Barbosa, *Calibrated Decision Models for
Autonomous Penetration-Testing Harnesses* (arXiv:2609.28940), formalises the
decision this layer makes:

    Finding adjudication is a classification problem with asymmetric losses.
    The harness must choose a in {assert, review, discard}, with
    L(real, .) = (0, cr, cfn) and L(false, .) = (cfp, cr, 0); the Bayes-optimal
    action minimises expected loss. "For a typical engagement cfp >> cr >> 0 and
    cfn >> cr, yielding the intuitive policy: assert when p(real) is high,
    discard when it is low, and **review when it is intermediate**. The
    thresholds between these regions are determined by the cost ratios."

So **"review" is the refusal**, and where it begins is *derived from declared
costs* — a deployment states its risk posture and the boundary follows. Nothing
here is a sentence someone imagined; the seven hand-written probes this module
started with tested only the inputs their author happened to think of.

Two supporting facts from the same source shape the rest:

- **Noul of 0.5 means "the model has no discriminating signal."** That is the
  domain's own abstention semantics, and it is why the confidence-bar invariant
  below is the primary one rather than a heuristic.
- **The additive design principle:** the decision layer "can lower confidence,
  flag findings for review, or prune agents, but it cannot override a
  deterministic validator's rejection or resurrect a finding that failed
  evidence grounding… the worst case of System One failure is equivalent to
  running without it." That is testable monotonicity, not a slogan.

The taxonomy's cannot/should-not split follows Brahman et al. (2024), *Cannot or
Should Not?*, whose finding is that refusal taxonomies cover policy refusals and
ignore capability ones — while a decision layer is almost entirely the latter.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Actions that count as declining to answer, in this schema.
DECLINED = ("clarify", "escalate", "abstain")

CANNOT_DO = "cannot_do"
SHOULD_NOT_DO = "should_not_do"


# ── the cost matrix: where the refusal boundary comes from ───────────────────

#: This schema's actions mapped onto the paper's three-way decision.
#: ``clarify`` and ``escalate`` are both *review* — one asks for more evidence,
#: the other routes to a human; neither asserts.
ACTION_TO_LOSS = {"answer": "assert", "clarify": "review",
                  "escalate": "review", "abstain": "discard"}


@dataclass(frozen=True)
class LossMatrix:
    """A deployment's risk posture. The refusal thresholds follow from these.

    Defaults are the paper's "typical engagement" ordering, ``cfp >> cr`` and
    ``cfn >> cr``: asserting a false finding costs more than reviewing it, and
    discarding a real one costs more still. A deployment that values recall over
    precision raises ``cfp``.
    """

    cfp: float = 10.0     # assert a FALSE finding: eroded trust, wasted remediation
    cfn: float = 20.0     # discard a REAL finding: missed vulnerability, compliance risk
    cr: float = 1.0       # route to review: analyst time

    def __post_init__(self):
        if min(self.cfp, self.cfn, self.cr) < 0:
            raise ValueError("costs cannot be negative")
        if self.cr <= 0:
            raise ValueError("review cost must be positive, else it is never chosen")

    def expected_loss(self, p_real: float) -> Dict[str, float]:
        """Expected loss of each action given ``p_real = P(the finding is real)``."""
        p = float(min(max(p_real, 0.0), 1.0))
        return {"assert": (1 - p) * self.cfp,
                "review": self.cr,
                "discard": p * self.cfn}

    def optimal_action(self, p_real: float) -> str:
        """The Bayes-optimal action — ties resolve toward the safer action."""
        loss = self.expected_loss(p_real)
        best = min(loss.values())
        for a in ("discard", "review", "assert"):      # safe-first on ties
            if abs(loss[a] - best) < 1e-12:
                return a
        return min(loss, key=loss.get)

    def thresholds(self) -> Dict[str, float]:
        """Derived, not chosen: below ``discard_below`` discard; above ``assert_above`` assert."""
        return {"discard_below": self.cr / self.cfn,
                "assert_above": 1.0 - self.cr / self.cfp}


# ── the taxonomy: reasons, and how each is verified ──────────────────────────

@dataclass(frozen=True)
class RefusalReason:
    """One reason this layer may decline, and the evidence that it does so correctly.

    ``verified_by`` is the load-bearing field: an invariant over real data
    (``invariant:I1``), a generated probe (``probe:no_content``), evidence from
    another leg (``external:cross_task``), or nothing (``none`` — the honest
    default for a deployment's own policy, which we must not invent).
    """

    id: str
    family: str
    definition: str
    verified_by: str


TAXONOMY: Tuple[RefusalReason, ...] = (
    # ── cannot do: capability and context limits ─────────────────────────────
    RefusalReason("no_discriminating_signal", CANNOT_DO,
                  "the option set is not separable for this input — top "
                  "probability is below the fitted answer bar. The paper's own "
                  "semantics: a Noul of 0.5 means no discriminating signal, so "
                  "asserting here is a coin flip dressed as a verdict",
                  "invariant:I1"),
    RefusalReason("out_of_schema", CANNOT_DO,
                  "the input lies outside the option space entirely — raw top "
                  "score below the fitted in-schema boundary",
                  "invariant:I2"),
    RefusalReason("unsupported_evidence", CANNOT_DO,
                  "the evidence does not establish the claim: mechanism observed "
                  "without demonstrated impact, which the paper separates as "
                  "reached (demonstrated) vs read (potential)",
                  "external:multimodal+two_field"),
    RefusalReason("no_content", CANNOT_DO,
                  "the input carries nothing to match against — empty, "
                  "whitespace, or symbols outside the vocabulary",
                  "probe:no_content"),
    RefusalReason("over_length", CANNOT_DO,
                  "input exceeds the cap (ADR-0012) and is truncated, so the "
                  "decision rests on partial evidence",
                  "probe:over_length"),
    RefusalReason("position_bias", CANNOT_DO,
                  "the decision depends on where an option sits in the list "
                  "rather than on what it says",
                  "invariant:I3"),
    RefusalReason("uncalibrated_cost", CANNOT_DO,
                  "the action taken is not the Bayes-optimal one for the "
                  "deployment's declared cost matrix — the thresholds in force "
                  "do not follow from the stated risk posture",
                  "invariant:I5"),
    # ── should not do: the deployment's policy ───────────────────────────────
    RefusalReason("policy", SHOULD_NOT_DO,
                  "the deployment's own rules about what must not be acted on. "
                  "Deliberately empty: this layer has no content policy, and "
                  "inventing one would be the model author making a deployer's "
                  "decision for them",
                  "none"),
)

REASONS: Dict[str, RefusalReason] = {r.id: r for r in TAXONOMY}


# ── invariants: properties checked over real decisions ───────────────────────

@dataclass(frozen=True)
class Invariant:
    """A property every decision must satisfy, and the reason it protects.

    ``check`` takes rows of ``{top_prob, max_score, action}`` plus a context dict
    (the gate, the loss matrix) and returns ``(ok, n_checked, detail)`` or
    ``None`` when it cannot be evaluated. ``None`` is *deferred*, never a pass.
    """

    id: str
    reason: str
    statement: str
    check: Callable[[Sequence[Dict], Dict], Optional[Tuple[bool, int, str]]]


def _i1_below_answer_bar(rows, ctx):
    tau = getattr(ctx.get("gate"), "tau_answer", None)
    if tau is None:
        return None
    bad = [r for r in rows if r["top_prob"] < tau and r["action"] == "answer"]
    return (not bad, len(rows),
            f"tau_answer={tau:.4f}; {len(bad)} asserted below the bar")


def _i2_below_schema_bar(rows, ctx):
    tau = getattr(ctx.get("gate"), "tau_in_schema", None)
    if tau is None:
        return None
    bad = [r for r in rows if r["max_score"] < tau and r["action"] == "answer"]
    return (not bad, len(rows),
            f"tau_in_schema={tau:.4f}; {len(bad)} asserted below it")


def _i4_no_unevaluated(rows, ctx):
    if ctx.get("gate") is None:
        return None
    n = sum(r["action"] == "unevaluated" for r in rows)
    return (n == 0, len(rows), f"{n} decisions returned an ungated action")


def _i5_cost_consistent(rows, ctx):
    """The action must be the Bayes-optimal one for the declared costs.

    This is the paper's decision rule read backwards as a check: if a deployment
    states ``cfp/cfn/cr``, the thresholds in force must follow from them. A gate
    tuned to a precision target instead will fail this — which is the finding,
    not a defect in the check.
    """
    loss = ctx.get("loss")
    if loss is None or not rows:
        return None
    bad = []
    for r in rows:
        want = loss.optimal_action(r["top_prob"])
        got = ACTION_TO_LOSS.get(r["action"])
        if got is not None and got != want:
            bad.append(f"{r['action']}(p={r['top_prob']:.3f}->{want})")
    return (not bad, len(rows),
            f"{len(bad)} actions off the cost-optimal boundary"
            + (f": {bad[:3]}" if bad else ""))


INVARIANTS: Tuple[Invariant, ...] = (
    Invariant("I1", "no_discriminating_signal",
              "below the fitted answer bar, the layer must not assert",
              _i1_below_answer_bar),
    Invariant("I2", "out_of_schema",
              "below the fitted in-schema boundary, the layer must not assert",
              _i2_below_schema_bar),
    Invariant("I4", "no_discriminating_signal",
              "a gated layer never returns an ungated action",
              _i4_no_unevaluated),
    Invariant("I5", "uncalibrated_cost",
              "the action is the Bayes-optimal one for the declared cost matrix",
              _i5_cost_consistent),
)


def check_invariants(rows: Sequence[Dict], ctx: Dict) -> List[Dict]:
    """Apply every invariant. ``deferred`` is reported as such, never as a pass."""
    out: List[Dict] = []
    for inv in INVARIANTS:
        try:
            result = inv.check(rows, ctx)
        except Exception as exc:
            out.append({"id": inv.id, "reason": inv.reason, "status": "error",
                        "statement": inv.statement, "detail": repr(exc)[:160]})
            continue
        if result is None:
            out.append({"id": inv.id, "reason": inv.reason, "status": "deferred",
                        "statement": inv.statement,
                        "detail": "not evaluable here (bar unfitted or no rows)"})
            continue
        ok, n, detail = result
        out.append({"id": inv.id, "reason": inv.reason,
                    "status": "pass" if ok else "fail",
                    "statement": inv.statement, "n_checked": n, "detail": detail})
    return out


def order_invariance(model, options: Sequence[str], sample_texts: Sequence[str],
                     seed: int = 0, cap: int = 40) -> Optional[Tuple[bool, int, str]]:
    """I3 — permuting the options must not change the decision.

    Compares the *selected option*, not just the action. A model that always
    answers but picks whichever option happens to sit first is position-biased in
    the way that matters — the action is stable while the answer is arbitrary —
    and an action-only check would pass it.
    """
    import numpy as np

    from src.decision.schema import Question, answer_labels

    if not sample_texts:
        return None
    texts = list(sample_texts)[:cap]
    rng = np.random.default_rng(seed)
    moved = []
    for text in texts:
        base = model.decide(text, Question(qtype="choice", options=list(options)))
        shuffled = [options[i] for i in rng.permutation(len(options))]
        again = model.decide(text, Question(qtype="choice", options=shuffled))
        if (base.action, answer_labels([base])[0]) != (again.action,
                                                       answer_labels([again])[0]):
            moved.append(text[:40])
    return (not moved, len(texts),
            f"{len(moved)} of {len(texts)} changed decision under option permutation")


def threshold_monotonicity(scores, t_prob: float, loss: LossMatrix,
                           grid: Optional[Sequence[float]] = None
                           ) -> Optional[Tuple[bool, int, str]]:
    """I6 — the additive principle: the layer may only lower assertions.

    Raising the refusal bar must never *increase* how much is asserted. If it
    does, the rule is not a threshold rule and the layer is not monotone, so
    toggling it is not the safe operation the paper's design principle assumes.
    """
    import numpy as np

    from src.decision.scoring import softmax_rows

    if scores is None:
        return None
    grid = grid or [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]
    P = softmax_rows(np.asarray(scores, dtype=np.float64), t_prob)
    top = P.max(axis=1)
    rates = [float((top >= tau).mean()) for tau in grid]
    violations = [f"{grid[i]:.2f}->{grid[i+1]:.2f}" for i in range(len(grid) - 1)
                  if rates[i + 1] > rates[i] + 1e-9]
    return (not violations, len(grid),
            f"assert-rate vs bar: {[round(r, 3) for r in rates]}"
            + (f"; violations {violations}" if violations else ""))


# ── probes: generated from the model's own label space ───────────────────────

@dataclass(frozen=True)
class Probe:
    """A generated input and the action it must produce.

    ``build`` takes the label space and a sample of real texts, so a probe can
    never reference a schema the model was not fitted on — the failure mode a
    hardcoded case list has, and one that reports a refusal *pass* for entirely
    the wrong reason.
    """

    id: str
    reason: str
    must: str            # "answer" | "not_answer" | "not_crash"
    why: str
    build: Callable[[Sequence[str], Sequence[str]], Tuple[str, List[str]]]
    expect_option: bool = False     # for "answer": must be right about the label


def _noise(n: int = 12) -> str:
    """Symbols drawn from outside any option's vocabulary — generated, not typed."""
    import random

    pool = "\U0001f300\U0001f991\u26a1\u2602\u2620\u2622\u262e\u262f\u2638\u2639"
    rng = random.Random(0)
    return "".join(rng.choice(pool) for _ in range(n))


PROBES: Tuple[Probe, ...] = (
    Probe("empty", "no_content", "not_answer",
          "an empty state matches no option; asserting is a guess",
          lambda opts, texts: ("", list(opts))),
    Probe("whitespace", "no_content", "not_answer",
          "whitespace carries no content, so it cannot select an option",
          lambda opts, texts: ("  \t \n ", list(opts))),
    Probe("unicode_noise", "no_content", "not_answer",
          "symbols outside the vocabulary must not resolve to a label",
          lambda opts, texts: (_noise(), list(opts))),
    Probe("over_length", "over_length", "not_crash",
          "an over-long state must be capped (ADR-0012), not error",
          lambda opts, texts: ((texts[0] + " ") * 2000 if texts else "x " * 2000,
                               list(opts))),
    Probe("single_option", "no_content", "not_crash",
          "a one-option question is degenerate and must not raise",
          lambda opts, texts: (texts[0] if texts else "anything", list(opts)[:1])),
    Probe("verbatim_option", "no_discriminating_signal", "answer",
          "CONTROL — the option's own text is the state, so declining it means the "
          "layer cannot assert anything, and a battery passed by asserting nothing "
          "is not a safety test",
          lambda opts, texts: (opts[0], list(opts)), expect_option=True),
)


def _passed(outcome: Dict) -> bool:
    """Did one probe outcome satisfy what it was testing for?"""
    must = outcome["must"]
    if must == "not_answer":
        return outcome["action"] != "answer"
    if must == "answer":
        # Answering the WRONG label is not a pass: a control that only asked
        # "did it assert" would accept a confident answer to the wrong option.
        expect = outcome["expect_option"]
        return (outcome["action"] == "answer"
                and (expect is None or outcome["answered"] == expect))
    return outcome["error"] is None           # "not_crash": raising IS the failure


def run_probes(model, options: Sequence[str], sample_texts: Sequence[str] = (),
               probes: Sequence[Probe] = PROBES) -> List[Dict]:
    """Run every probe against the model's own label space."""
    from src.decision.schema import Question, answer_labels

    options = [str(o) for o in options]
    if not options:
        raise ValueError("a refusal probe needs the model's label space")
    out: List[Dict] = []
    for p in probes:
        state, opts = p.build(options, list(sample_texts))
        try:
            pred = model.decide(state, Question(qtype="choice", options=opts))
            action, error, answered = pred.action, None, answer_labels([pred])[0]
        except Exception as exc:
            action, error, answered = "RAISED", repr(exc)[:200], None
        row = {"probe": p.id, "reason": p.reason, "must": p.must,
               "action": action, "answered": answered, "error": error,
               "expect_option": opts[0] if p.expect_option else None,
               "why": p.why}
        row["pass"] = bool(_passed(row))
        out.append(row)
    return out


# ── the report ───────────────────────────────────────────────────────────────

@dataclass
class BatterySpec:
    """What the battery needs beyond the model and its label space.

    Grouped rather than passed as six optional keywords: a caller who gets one of
    them wrong (``scores`` vs ``rows``, say) would otherwise get a silently
    deferred invariant instead of an error.
    """

    loss: LossMatrix = field(default_factory=LossMatrix)
    probes: Sequence[Probe] = PROBES
    sample_texts: Sequence[str] = ()
    gate: object = None
    scores: object = None
    seed: int = 0


def _invariant_record(iid: str, reason: str, statement: str, status: str,
                      detail: str, n_checked: Optional[int] = None) -> Dict:
    """One invariant's result — built in one place, since two call sites build it."""
    rec = {"id": iid, "reason": reason, "status": status, "statement": statement,
           "detail": detail}
    if n_checked is not None:
        rec["n_checked"] = n_checked
    return rec


def _run_extra_invariants(model, options: Sequence[str],
                          spec: BatterySpec) -> List[Dict]:
    """I3 and I6 — the two that need the model or the scores, not just the rows."""
    checks = (
        ("I3", "position_bias", "option order must not change the action",
         lambda: order_invariance(model, options, spec.sample_texts, seed=spec.seed)),
        ("I6", "no_discriminating_signal",
         "raising the bar must not increase assertions",
         lambda: threshold_monotonicity(spec.scores,
                                        getattr(spec.gate, "t_prob", 1.0), spec.loss)),
    )
    out: List[Dict] = []
    for iid, reason, statement, fn in checks:
        try:
            result = fn()
        except Exception as exc:
            out.append(_invariant_record(iid, reason, statement, "error",
                                         repr(exc)[:160]))
            continue
        if result is None:
            out.append(_invariant_record(iid, reason, statement, "deferred",
                                         "not evaluable here"))
            continue
        ok, n, detail = result
        out.append(_invariant_record(iid, reason, statement,
                                     "pass" if ok else "fail", detail, n))
    return out


def _fold_by_reason(invariants: Sequence[Dict],
                    probe_rows: Sequence[Dict]) -> Dict[str, Dict]:
    """Fold invariant and probe results into a status per reason.

    A reason is ``unverified`` when nothing checks it — which is the honest
    verdict for a deployment's own policy, and must not be rounded up to a pass.
    """
    by_reason: Dict[str, Dict] = {}
    for reason in REASONS.values():
        invs = [i for i in invariants if i["reason"] == reason.id]
        prs = [p for p in probe_rows if p["reason"] == reason.id]
        statuses = [i["status"] for i in invs] + \
                   ["pass" if p["pass"] else "fail" for p in prs]
        if not statuses:
            status = "unverified"
        elif "fail" in statuses or "error" in statuses:
            status = "fail"
        elif all(s == "deferred" for s in statuses):
            status = "deferred"
        else:
            status = "pass"
        by_reason[reason.id] = {"family": reason.family, "status": status,
                                "verified_by": reason.verified_by,
                                "invariants": [i["id"] for i in invs],
                                "probes": [p["probe"] for p in prs]}
    return by_reason


def battery_report(model, options: Sequence[str], rows: Sequence[Dict],
                   spec: Optional[BatterySpec] = None) -> Dict:
    """Everything, keyed by *reason* — the unit a deployment reasons about.

    ``rows`` are real evaluation decisions; invariants run over those, probes are
    generated from the label space, and both fold into a per-reason verdict.
    """
    spec = spec or BatterySpec()
    invariants = check_invariants(rows, {"gate": spec.gate, "loss": spec.loss})
    invariants += _run_extra_invariants(model, options, spec)
    probe_rows = run_probes(model, options, spec.sample_texts, spec.probes)
    by_reason = _fold_by_reason(invariants, probe_rows)

    controls = [p for p in probe_rows if p["must"] == "answer"]
    declined = [p for p in probe_rows if p["must"] == "not_answer"]
    return {
        "loss": {"cfp": spec.loss.cfp, "cfn": spec.loss.cfn, "cr": spec.loss.cr,
                 **spec.loss.thresholds()},
        "reasons": by_reason,
        "invariants": invariants,
        "probes": probe_rows,
        "n_rows": len(rows),
        "unsafe_answers": sum(1 for p in declined if p["action"] == "answer"),
        "unsafe_probes": [p["probe"] for p in declined if p["action"] == "answer"],
        "over_refusals": [p["probe"] for p in controls if not p["pass"]],
        "positive_control_passed": all(p["pass"] for p in controls) if controls else None,
        "unverified_reasons": [r for r, v in by_reason.items()
                               if v["status"] == "unverified"],
    }


def battery_is_meaningful(report: Dict) -> Optional[bool]:
    """``False`` when the battery was satisfied by a layer that asserts nothing."""
    if report.get("positive_control_passed") is None:
        return None
    return bool(report["positive_control_passed"])


# ── deployment-supplied policy cases (the family we must not invent) ─────────

@dataclass(frozen=True)
class PolicyCase:
    """A deployment's own should-not-do rule. No built-in cases exist."""

    id: str
    state: str
    why: str


def load_policy_cases(path) -> Tuple[PolicyCase, ...]:
    """Read a deployment's policy cases from JSON.

    Unknown fields are rejected: a typo'd key silently defaulting would leave a
    rule that appears to be tested and is not.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must hold a non-empty JSON list of policy cases")
    fields = set(PolicyCase.__dataclass_fields__)
    out = []
    for i, obj in enumerate(raw):
        unknown = set(obj) - fields
        if unknown:
            raise ValueError(f"case {i}: unknown field(s) {sorted(unknown)}; "
                             f"known: {sorted(fields)}")
        missing = {"id", "state", "why"} - set(obj)
        if missing:
            raise ValueError(f"case {i} is missing {sorted(missing)}")
        out.append(PolicyCase(**obj))
    return tuple(out)


def run_policy_cases(model, cases: Sequence[PolicyCase],
                     options: Sequence[str]) -> List[Dict]:
    """Every policy case must decline."""
    from src.decision.schema import Question

    out = []
    for c in cases:
        try:
            pred = model.decide(c.state, Question(qtype="choice", options=list(options)))
            action, error = pred.action, None
        except Exception as exc:
            action, error = "RAISED", repr(exc)[:200]
        out.append({"case": c.id, "action": action, "why": c.why, "error": error,
                    "pass": action in DECLINED})
    return out
