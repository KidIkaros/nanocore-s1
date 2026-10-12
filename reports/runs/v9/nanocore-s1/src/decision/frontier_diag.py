"""Frontier bottleneck diagnostic (#17).

The post-v1 frontier question: does the typed-decision core have a DEPTH
problem (errors grow with how hard the decision is over accumulated evidence)
or a VOLUME problem (errors grow with how much competing history is stored), or
neither (decide-on-embeddings is the ceiling)?

The sleep paper's method, applied to a typed decision head. Operates on the
per-example records a leg already collects (top1 correct, top_prob, intent), so
it needs no new encoder pass — it re-ranks the SAME held-out examples along two
axes and reports accuracy per bucket. The verdict comes from which axis moves
accuracy:

  - accuracy falls with DIFFICULTY (flat volume)  -> depth bottleneck -> build
    the predictive/consequence organ (RWM-shaped).
  - accuracy falls with VOLUME (flat difficulty)   -> volume bottleneck -> build
    the typed-memory organ (memory-type routing + typed edges).
  - accuracy flat across both                      -> NEITHER: harden v1, build
    no organ. A valid, recorded negative result.

Axes are built from what a per-example record honestly exposes:
  - difficulty  = 1 - top_prob (decision confidence; low confidence = hard)
  - volume      = how many label classes are within the margin of the winner
                  (embedding-space crowding = competing stored evidence)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class DiagVerdict:
    depth_bottleneck: bool
    volume_bottleneck: bool
    neither: bool
    depth_slope: float        # accuracy drop across difficulty buckets
    volume_slope: float       # accuracy drop across volume buckets
    buckets: Dict[str, List[dict]]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "depth_bottleneck": self.depth_bottleneck,
            "volume_bottleneck": self.volume_bottleneck,
            "neither": self.neither,
            "depth_slope": round(self.depth_slope, 4),
            "volume_slope": round(self.volume_slope, 4),
            "recommendation": self.recommendation,
            "buckets": self.buckets,
        }


def _bucket_by(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Assign each example to a bucket (0=easy/empty .. n-1=hard/crowded) by
    quantile so buckets are balanced regardless of the raw distribution."""
    qs = np.quantile(values, np.linspace(0, 1, n_buckets + 1)[1:-1])
    return np.searchsorted(qs, values, side="right")


def diagnose_frontier(rec: dict, scores: np.ndarray, n_buckets: int = 4,
                      slope_thresh: float = 0.10) -> DiagVerdict:
    """Run the depth-vs-volume diagnostic over a leg's per-example records.

    ``rec`` is a leg's per-example record dict: keys ``top1`` (bool correct),
    ``top_prob`` (float), ``intent`` (int). ``scores`` is the per-example score
    matrix (n x K) that produced those records — used to measure volume (how
    many classes crowd the winner). Returns a DiagVerdict.
    """
    top1 = np.asarray(rec["top1"], dtype=bool)
    top_prob = np.asarray(rec["top_prob"], dtype=float)
    scores = np.asarray(scores, dtype=float)

    # depth axis: difficulty = 1 - confidence. Low confidence = hard decision.
    difficulty = 1.0 - top_prob
    # volume axis: crowd = number of classes whose score is within a margin of
    # the winner (competing stored evidence). More crowd = volume bottleneck.
    srt = np.sort(scores, axis=1)[:, ::-1]
    winner = srt[:, 0]
    # classes within 1.0 of the winner's score (the "near-miss" competition)
    crowd = (scores >= (winner[:, None] - 1.0)).sum(axis=1).astype(float)

    def slope_and_buckets(axis: np.ndarray) -> (float, List[dict]):
        b = _bucket_by(axis, n_buckets)
        buckets = []
        accs = []
        for bi in range(n_buckets):
            m = b == bi
            acc = float(top1[m].mean()) if m.sum() else float("nan")
            accs.append(acc)
            buckets.append({"bucket": bi, "n": int(m.sum()), "accuracy": round(acc, 4),
                            "axis_mean": round(float(axis[m].mean()), 4) if m.sum() else None})
        # slope = easy-bucket acc minus hard-bucket acc (positive => accuracy
        # falls as the axis rises => that axis is the bottleneck)
        finite = [a for a in accs if a == a]
        slope = (finite[0] - finite[-1]) if len(finite) >= 2 else 0.0
        return slope, buckets

    depth_slope, depth_buckets = slope_and_buckets(difficulty)
    volume_slope, volume_buckets = slope_and_buckets(crowd)

    depth_bottleneck = depth_slope >= slope_thresh and depth_slope >= volume_slope
    volume_bottleneck = volume_slope >= slope_thresh and volume_slope > depth_slope
    neither = not (depth_bottleneck or volume_bottleneck)

    if depth_bottleneck:
        rec_text = ("predictive organ first (RWM consequence prediction): "
                    "accuracy falls with decision difficulty")
    elif volume_bottleneck:
        rec_text = ("typed-memory organ first (memory-type routing + typed "
                    "edges): accuracy falls with competing stored evidence")
    else:
        rec_text = ("neither: decide-on-embeddings is the ceiling for this head "
                    "— harden v1, build no organ (recorded negative result)")

    return DiagVerdict(
        depth_bottleneck=depth_bottleneck, volume_bottleneck=volume_bottleneck,
        neither=neither, depth_slope=depth_slope, volume_slope=volume_slope,
        buckets={"difficulty": depth_buckets, "volume": volume_buckets},
        recommendation=rec_text)
