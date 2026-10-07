#!/usr/bin/env python3
"""Restore the abstention helper definitions in notebooks/s1_calibration.

Why this exists
---------------
The calibration kernel (v2) errored with `NameError: name 'selective_curve' is not
defined`. The notebook generator pruned a scaffold cell that held a placeholder
line *and* the two helper definitions, so the helpers were deleted along with the
placeholder. The abstention step is the product-relevant part of the experiment,
so the run has to be repeated with the helpers restored.

This patch is stdlib-only and performs no execution of notebook code: it rewrites
JSON. Nothing here imports torch, numpy, or touches an array — see the execution
boundary in docs/ARCHITECTURE-DECISION-MODEL.md section 0.

Usage: python fix_missing_defs.py [notebook.ipynb]
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HELPERS = '''# Restored by fix_missing_defs.py — these were pruned with a scaffold cell.
def selective_curve(probs, steps=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_test).astype(float)
    order = np.argsort(-conf)
    curve = []
    for c in np.linspace(0.1, 1.0, steps):
        n = max(1, int(c * len(y_test)))
        keep = order[:n]
        curve.append({"coverage": round(float(c), 2),
                      "selective_accuracy": float(correct[keep].mean())})
    return curve

def coverage_at(probs, target=0.90):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y_test).astype(float)
    order = np.argsort(-conf)
    best = 0.0
    for n in range(1, len(y_test) + 1):
        if correct[order[:n]].mean() >= target:
            best = n / len(y_test)
    return float(best)

'''


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "notebooks/s1_calibration/s1_calibration.ipynb")
    nb = json.loads(path.read_text())

    defines = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and "def selective_curve" in "".join(c["source"])]
    uses = [i for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and "selective_curve(probs)" in "".join(c["source"])]

    if defines:
        print(f"helpers already present in cell(s) {defines}; nothing to do")
        return 0
    if not uses:
        raise SystemExit("could not find the abstention step that uses selective_curve(probs)")

    target = uses[0]
    source = "".join(nb["cells"][target]["source"])
    nb["cells"][target]["source"] = (HELPERS + source).splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1))

    # static verification: every code cell parses, and the definitions precede use
    reparsed = json.loads(path.read_text())
    for i, c in enumerate(reparsed["cells"]):
        if c["cell_type"] == "code":
            ast.parse("".join(c["source"]))
    assert any("def selective_curve" in "".join(c["source"]) for c in reparsed["cells"]), "patch failed"
    print(f"restored helpers at the top of cell {target}; all code cells parse; "
          f"{len(reparsed['cells'])} cells total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
