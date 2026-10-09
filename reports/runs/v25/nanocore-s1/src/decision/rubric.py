"""Coverage against Google's ML Test Score — an external view of the same system.

Our own gate (`qualify.py`) asks *"may this build ship?"*. This asks a different
and complementary question: *"how much of what practitioners consider
production-readiness testing do we actually do?"* Keeping both is deliberate —
a self-authored gate can only measure what it thought to include, and the point
of an external rubric is the items we did not think of.

Honest scope: this is a coverage map over the rubric's **four published
categories** using the tests named in the paper and its public summaries, not a
reproduction of all 28 items. Where an item is partially covered it says so
rather than rounding up — the value of the exercise is the holes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

COVERED, PARTIAL, MISSING = "covered", "partial", "missing"


@dataclass(frozen=True)
class RubricItem:
    """One practice, and where (if anywhere) we implement it."""

    category: str
    practice: str
    status: str
    where: str


ITEMS: Tuple[RubricItem, ...] = (
    # ── data ────────────────────────────────────────────────────────────────
    RubricItem("data", "feature/target distributions match expectation",
               COVERED, "monitor.detect_drift — KS on top_prob and set size, named shift types"),
    RubricItem("data", "the relationship between each feature and the target is understood",
               PARTIAL, "ADR-0011 headroom check measures zero-shot separability; no per-feature analysis"),
    RubricItem("data", "data invariants are asserted",
               COVERED, "datasets.py label-space guards; adapt._validate fails closed on <5/class"),
    RubricItem("data", "private data is handled appropriately",
               COVERED, "privacy tested — no socket in the decision path; local-only GGUF backend"),
    # ── model ───────────────────────────────────────────────────────────────
    RubricItem("model", "model spec undergoes review and is checked in",
               COVERED, "ADRs 0001-0013; every claim traceable to reports/runs/<run>/"),
    RubricItem("model", "offline proxy metrics correlate with online impact",
               MISSING, "no deployment exists — unclaimable offline; recorded as deferred"),
    RubricItem("model", "hyperparameters are tuned",
               PARTIAL, "head epochs/lr fixed by hand; temperature and t_set *are* fitted"),
    RubricItem("model", "impact of model staleness is known",
               COVERED, "cadence.py (scheduled vs drift-triggered, floored) + v18 causal readout"),
    RubricItem("model", "a simpler model is used as a baseline",
               COVERED, "TF-IDF+LR on every breadth dataset; head must beat it on both proper metrics"),
    RubricItem("model", "quality on important slices is tested",
               PARTIAL, "language slices measured (v20); demographic/attribute slices not"),
    RubricItem("model", "the model is calibrated",
               COVERED, "log score + Brier gated; ECE15 reported; conformal coverage gated"),
    RubricItem("model", "prediction sets are actionable, not vacuous",
               COVERED, "mean set size gated on both paths — the v19 finding, now a criterion"),
    RubricItem("model", "the model declines rather than guessing",
               COVERED, "refusals.py two-sided battery incl. the over-refusal control"),
    # ── infrastructure ──────────────────────────────────────────────────────
    RubricItem("infrastructure", "training is reproducible",
               PARTIAL, "seeded splits + bundle round-trip identical; no train-twice comparison"),
    RubricItem("infrastructure", "an integration test runs across the whole pipeline",
               COVERED, "s1_verify runs protocol tests, adapt, decide, serve and the CLI in-kernel"),
    RubricItem("infrastructure", "model artifacts are versioned with lineage",
               COVERED, "registry.py — data hash, config, parent, promote/rollback pointers"),
    RubricItem("infrastructure", "the shipped artifact is identifiable",
               COVERED, "model.bundle_digest — content-addressed SHA-256 over the bundle"),
    # ── monitoring ──────────────────────────────────────────────────────────
    RubricItem("monitoring", "upstream instability is detected",
               COVERED, "monitor.py alert thresholds on latency, escalate rate, KS drift"),
    RubricItem("monitoring", "training/serving skew is detected",
               PARTIAL, "one scoring path in-kernel by construction; not asserted as an invariant"),
    RubricItem("monitoring", "prediction drift is monitored",
               COVERED, "action-rate deltas + top_prob KS; v18 fired on a real logged alert"),
)


def coverage(items: Tuple[RubricItem, ...] = ITEMS) -> Dict:
    """Counts and lists by status, overall and per category."""
    def tally(rows: List[RubricItem]) -> Dict:
        return {"covered": sum(r.status == COVERED for r in rows),
                "partial": sum(r.status == PARTIAL for r in rows),
                "missing": sum(r.status == MISSING for r in rows),
                "n": len(rows)}

    by_cat: Dict[str, Dict] = {}
    for cat in sorted({r.category for r in items}):
        by_cat[cat] = tally([r for r in items if r.category == cat])
    overall = tally(list(items))
    overall["fraction_covered"] = round(
        (overall["covered"] + 0.5 * overall["partial"]) / max(overall["n"], 1), 3)
    return {"overall": overall, "by_category": by_cat,
            "gaps": [{"category": r.category, "practice": r.practice, "status": r.status}
                     for r in items if r.status != COVERED]}


def render(rep: Optional[Dict] = None) -> str:
    """A compact table for the qualification report."""
    rep = rep or coverage()
    lines = [f"ML Test Score coverage: {rep['overall']['covered']} covered / "
             f"{rep['overall']['partial']} partial / {rep['overall']['missing']} missing "
             f"({rep['overall']['fraction_covered']:.0%} weighted)", ""]
    for cat, c in rep["by_category"].items():
        lines.append(f"  {cat:16} covered {c['covered']}/{c['n']} "
                     f"(partial {c['partial']}, missing {c['missing']})")
    if rep["gaps"]:
        lines += ["", "  gaps:"]
        lines += [f"    [{g['status']}] {g['category']}: {g['practice']}"
                  for g in rep["gaps"]]
    return "\n".join(lines)
