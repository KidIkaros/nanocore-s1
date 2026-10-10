"""Slice-conditional and deferral evaluation — what a marginal number can hide.

From ``docs/research/benchmarks-and-evaluation-readiness.md``. Four checks, each
aimed at a failure mode that reports *fine* on an aggregate while being wrong on
a slice or on the team:

- :func:`group_coverage` — conformal coverage is *marginal*; a hard slice can sit
  under-covered while the average holds (MAPIE conditional-CP / Mondrian).
- :func:`deferral_quality` — escalation only pays if it targets the items the
  model would get wrong (learning-to-defer; Mozannar et al. 2023).
- :func:`rejection_by_group` — abstention must not concentrate on one slice
  (minority over-rejection; Pugnana & Ruggieri).
- :func:`memorization_verdict` — the reference evaluation's two probes adapted
  to a fitted head: candidate-order invariance and withheld-state collapse
  (arXiv:2609.37647).

Every function is pure array logic: the kernel gathers per-example records, these
turn them into verdicts, so the same code runs in the unit suite on synthetic
arrays and on Kaggle on the real ones.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

#: Actions that route the item away from an answer — clarify/escalate/abstain all
#: defer resolution to a downstream step, so all three count as "not answered".
NON_ANSWER = ("clarify", "escalate", "abstain")


def _iter_groups(groups):
    """Yield ``(label, boolean mask)`` per distinct group, sorted.

    Group labels are coerced to ``str`` so an intent index and a language code
    can both serve as a slice.
    """
    g = np.asarray([str(x) for x in groups])
    for label in sorted(set(g.tolist())):
        yield label, g == label


def confidence_bands(top_prob, n_bands: int = 3) -> np.ndarray:
    """Difficulty tertiles by calibrated confidence.

    The low-confidence band is precisely the slice a marginal coverage figure can
    hide — this is the grouping the conditional-coverage check wants. Bands are
    labelled ``low`` / ``mid`` / ``high`` (then ``band<i>`` for n_bands > 3).
    """
    p = np.asarray(top_prob, dtype=float).reshape(-1)
    if p.size == 0:
        return np.array([])
    edges = np.quantile(p, np.linspace(0, 1, n_bands + 1))[1:-1]
    idx = np.clip(np.digitize(p, edges), 0, n_bands - 1)
    names = (["low", "mid", "high"] if n_bands == 3
             else [f"band{i}" for i in range(n_bands)])
    return np.array([names[i] for i in idx])


def group_coverage(covered, groups, alpha: float,
                   min_group_n: int = 200) -> Dict:
    """Per-group conformal coverage against the marginal target.

    A single global cutoff achieves ``1 - alpha`` on average while a hard or rare
    slice sits under-covered. Returns per-group coverage, and splits groups into
    ``undercovered`` (n >= min_group_n and below target) vs ``small`` (too few to
    trust — the Mondrian caveat that coverage on a tiny group is unstable).
    """
    covered = np.asarray(covered, dtype=bool).reshape(-1)
    target = 1.0 - float(alpha)
    out = {"alpha": float(alpha), "target": target,
           "min_group_n": int(min_group_n), "n": int(len(covered)),
           "groups": {}, "undercovered": [], "small": []}
    for g, m in _iter_groups(groups):
        n = int(m.sum())
        cov = float(covered[m].mean()) if n else 0.0
        out["groups"][g] = {"n": n, "coverage": cov}
        if n < min_group_n:
            out["small"].append(g)
        elif cov < target:
            out["undercovered"].append(g)
    return out


def deferral_quality(actions, correct,
                     answered_action: str = "answer") -> Dict:
    """Does deferral target the model's likely errors rather than its wins?

    ``correct`` is the top-1 correctness the model *would* have had if forced to
    answer every item. Splitting the log at ``answered_action``:

    - ``acc_asserted`` vs ``acc_deferred`` — well-aimed deferral defers the harder
      set, so ``acc_deferred <= acc_asserted``.
    - ``error_enrichment`` — P(wrong | deferred) / P(wrong | asserted); >1 means
      the deferred set is enriched for the model's errors.
    - ``breakeven_human_acc`` — the accuracy a human needs on deferred items for
      deferral to lower system error. It equals ``acc_deferred``: deferral pays
      iff a human beats what the model would have scored there.
    """
    actions = np.asarray([str(a) for a in actions])
    correct = np.asarray(correct, dtype=bool).reshape(-1)
    answered = actions == answered_action
    out = {"n": int(len(correct)), "n_asserted": int(answered.sum()),
           "n_deferred": int((~answered).sum()),
           "deferral_rate": float((~answered).mean()) if len(correct) else 0.0}
    if answered.any():
        out["acc_asserted"] = float(correct[answered].mean())
    if (~answered).any():
        out["acc_deferred"] = float(correct[~answered].mean())
    a, d = out.get("acc_asserted"), out.get("acc_deferred")
    if a is not None and d is not None:
        err_a, err_d = 1.0 - a, 1.0 - d
        out["breakeven_human_acc"] = d
        out["error_enrichment"] = (err_d / err_a) if err_a > 1e-9 else None
        out["well_aimed"] = bool(d <= a + 1e-9)
    return out


def rejection_by_group(actions, groups, min_group_n: int = 50,
                       concentration: float = 3.0) -> Dict:
    """Does abstention concentrate on a slice? (minority over-rejection)

    Reports each group's non-answer rate and the largest group-to-overall ratio
    among populated groups. This is a *signal*, not a verdict — a genuinely
    harder slice legitimately defers more, so concentration is surfaced for
    review rather than failed outright.
    """
    actions = np.asarray([str(a) for a in actions])
    reject = np.isin(actions, list(NON_ANSWER))
    overall = float(reject.mean()) if len(reject) else 0.0
    out = {"overall_non_answer": overall, "min_group_n": int(min_group_n),
           "concentration": float(concentration), "groups": {},
           "max_ratio": None, "concentrated": []}
    ratios = {}
    for g, m in _iter_groups(groups):
        n = int(m.sum())
        rate = float(reject[m].mean()) if n else 0.0
        out["groups"][g] = {"n": n, "non_answer_rate": rate}
        if n >= min_group_n and overall > 0:
            ratios[g] = rate / overall
    if ratios:
        out["max_ratio"] = float(max(ratios.values()))
        out["concentrated"] = sorted(g for g, r in ratios.items()
                                     if r > concentration)
    return out


def memorization_verdict(rotation_consistency: Optional[float],
                         withheld: Optional[Dict], n_options: int) -> Dict:
    """The reference eval's two probes, adapted to a fitted head.

    ``rotation_consistency`` — fraction of test items whose chosen *label* is
    unchanged under a candidate-order permutation (~1.0; a positional model
    tracks positions, not labels). ``withheld`` — ``{"top_prob": …,
    "answered": n}`` measured on uninformative states; a model reading real
    signal must not answer (``answered == 0``) and should sit near uniform.
    ``None`` inputs are reported, not passed.
    """
    chance = 1.0 / max(int(n_options), 1)
    rot_ok = (rotation_consistency is not None
              and rotation_consistency >= 0.999)
    w_answered = None if withheld is None else int(withheld.get("answered", 0))
    w_prob = None if withheld is None else float(withheld.get("top_prob", 0.0))
    w_ok = withheld is not None and w_answered == 0
    return {"n_options": int(n_options), "chance_top_prob": chance,
            "rotation_consistency": rotation_consistency,
            "withheld_top_prob": w_prob, "withheld_answered": w_answered,
            "rotation_invariant": bool(rot_ok),
            "withholding_collapses": bool(w_ok),
            "passed": bool(rot_ok and w_ok)}


def noul_placement(p_yes: np.ndarray, gold: np.ndarray) -> Dict:
    """The binary-judgment placement diagnostic (arXiv:2609.37647).

    The reference evaluation found binary ``P(yes)`` **ranks** well
    (AUROC high) while being **poorly placed** at a fixed 0.5 — the mean
    predicted probability sits off the observed positive rate, so
    threshold-based metrics suffer even though the ranking is sound.
    Their numbers: mean P(yes) 0.465 vs observed rate 0.518 across eight
    binary datasets (systematic under-confidence). We already report
    F1@0.5 *and* F1 at a tuned threshold; what was missing is the
    *placement gap* itself — the number that says whether a well-ranked
    but misplaced P(yes) is the failure, before any threshold is fitted.

    Args:
        p_yes: predicted P(yes) per example.
        gold: boolean positive labels.

    Returns:
        ``mean_p_yes`` vs ``observed_rate`` and the signed ``gap``
        (positive = over-confident: predicts yes more often than it
        occurs; negative = under-confident). ``base_rate`` mirrors the
        leg's existing field name. ``None`` on empty input.
    """
    p = np.asarray(p_yes, dtype=np.float64).reshape(-1)
    g = np.asarray(gold, dtype=float).reshape(-1)
    if len(p) == 0 or len(p) != len(g):
        return {"mean_p_yes": None, "observed_rate": None, "gap": None}
    mean_p = float(p.mean())
    rate = float(g.mean())
    return {"mean_p_yes": mean_p, "observed_rate": rate,
            "base_rate": rate, "gap": mean_p - rate}
