"""Late fusion for multimodal states — decision-level, not embedding-level.

The v27 multimodal leg measured ``full`` (image+text mean-pooled) at .428
*below* ``image_only`` at .433 — early fusion done badly. The fusion
taxonomy (Baltrusaitis et al., *Multimodal Deep Learning* §4.2.4: early /
joint / late) names why: pooling at the input layer assumes the
modalities are commensurable before any task signal exists. Mean-pooling
an image vector and a text vector averages two representations whose
scales and semantics were never aligned.

**Late fusion scores each modality separately and aggregates the
*predictions*, not the embeddings.** Each modality produces its own
probability vector over the same option set; the aggregation combines
them. Every piece already exists — ``DecisionModel.decide`` per modality
— so this module is the aggregation rule plus the routing, nothing new
on the scoring side.

Aggregation rules (the ensemble-classifier family, per the taxonomy's
late-fusion section):

- ``mean`` — arithmetic mean of probability vectors. The default; the
  simplest decision-level rule, robust when neither modality dominates.
- ``product`` — normalized geometric mean. Sharper than the mean when
  the modalities agree, collapses toward uniform when they disagree —
  the "experts must concur" rule.
- ``max`` — elementwise max, renormalized. Optimistic: an option any
  modality supports strongly survives.

The weight vector ``w`` (one per modality, summing to 1) is the late-
fusion dial the taxonomy leaves open. Uniform weights are the honest
zero-knowledge default; a fitted weight vector is the E4 aspiration.

No model weights run here — the aggregator consumes probability dicts,
so the whole path is stub-testable locally.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

#: Supported aggregation rules, named for what they do to disagreement.
RULES = ("mean", "product", "max")


def aggregate(prob_dicts: Sequence[Dict[str, float]],
              weights: Optional[Sequence[float]] = None,
              rule: str = "mean") -> Dict[str, float]:
    """Combine per-modality probability vectors into one distribution.

    Args:
        prob_dicts: One probability dict per modality, all over the same
            option set (keys must match across modalities).
        weights: Per-modality weights; normalized to sum to 1. ``None``
            is uniform. Length must match ``prob_dicts``.
        rule: ``"mean"`` | ``"product"`` | ``"max"`` (see module docstring).

    Returns:
        A probability dict over the same options, summing to 1.

    Raises:
        ValueError: on empty input, mismatched option sets, a bad rule,
            or negative weights.
    """
    if not prob_dicts:
        raise ValueError("aggregate needs at least one probability vector")
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")

    keys = set(prob_dicts[0])
    for d in prob_dicts[1:]:
        if set(d) != keys:
            raise ValueError("all modalities must score the same option set")

    n = len(prob_dicts)
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"{len(weights)} weights for {n} modalities")
        if any(x < 0 for x in weights):
            raise ValueError("weights must be non-negative")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = [x / total for x in weights]

    labels = sorted(keys)
    out: Dict[str, float] = {}
    for lab in labels:
        ps = [max(float(d[lab]), 1e-12) for d in prob_dicts]
        if rule == "mean":
            v = sum(wi * p for wi, p in zip(w, ps))
        elif rule == "product":
            # geometric mean under the weights: prod(p^w)
            v = 1.0
            for wi, p in zip(w, ps):
                v *= p ** wi
        else:  # max
            v = max(ps)
        out[lab] = v

    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else \
        {k: 1.0 / len(labels) for k in labels}


def late_fusion_decide(model, state_groups: Sequence, question,
                       weights: Optional[Sequence[float]] = None,
                       rule: str = "mean"):
    """Decide a multimodal state by scoring each modality, then aggregating.

    Args:
        model: A ``DecisionModel`` (or anything with ``decide(state,
            question)``). The gate is deliberately bypassed: fusion
            operates on probabilities, and the gate's out-of-schema check
            is a raw-score rule that has no meaning on an aggregated
            distribution. Callers that want gating run it after fusion.
        state_groups: One state per modality — e.g. ``[["image item"],
            ["question text"]]`` for an image+text pair. Each group is
            passed to ``decide`` as its own state.
        question: The shared ``Question`` over the option set.
        weights: Per-modality aggregation weights (see ``aggregate``).
        rule: Aggregation rule (see ``aggregate``).

    Returns:
        ``(probabilities, per_modality)`` — the aggregated distribution
        and the raw per-modality probability dicts, so the fusion is
        auditable (which modality drove the answer).
    """
    per_modality = []
    for group in state_groups:
        pred = model.decide(group, question)
        per_modality.append(dict(pred.probabilities))
    probs = aggregate(per_modality, weights=weights, rule=rule)
    return probs, per_modality
