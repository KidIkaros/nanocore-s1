#!/usr/bin/env python3
"""Generate notebooks/s1_tests/s1_tests.ipynb — the fast-fail test harness.

A CPU-only Kaggle kernel that extracts the source bundle and runs the whole pytest
suite. ADR-0006 requires that a notebook be validated by running it on Kaggle, never
locally, and that cheap fast-fail happen by pushing a *reduced* kernel rather than by
executing code on the host. This is that reduced kernel: no model weights, no data,
no GPU, no quota, ~2 minutes.

Use it after any change to src/decision/ or tests/ to get a verdict before spending a
real experiment run.

Run it with: python generate.py <path/to/notebook.ipynb>
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path


def code(s):
    return {"cell_type": "code", "metadata": {},
            "source": s.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def md(s):
    return {"cell_type": "markdown", "metadata": {},
            "source": s.strip().splitlines(keepends=True)}


cells = [
    md('''
# Fast-fail test harness (CPU, no weights, no quota)

Runs the full `pytest` suite for `src/decision/` on Kaggle. Exists because ADR-0006
forbids executing code on the host and requires cheap validation to happen by pushing
a reduced kernel.

Includes the two protocol regression tests:

- `test_recovers_a_small_temperature_that_a_narrow_grid_would_miss` — the grid-edge bug
  that made cosine scoring look catastrophically miscalibrated.
- `test_compares_against_best_baseline_not_weakest` — the verdict bug that reported a
  Brier win which does not exist against the strongest baseline.
'''),
    code('''
import json, subprocess, sys, time
from pathlib import Path

print("CPU-only harness — no accelerator, no model weights")
import numpy as np
print("numpy", np.__version__)

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/s1-work")
WORK.mkdir(parents=True, exist_ok=True)
STATUS = {"status": "running", "stage": "start", "started": time.time()}
(WORK / "run_status.json").write_text(json.dumps(STATUS, indent=2))
'''),
    code('''
import base64, io, sys, zipfile

BUNDLE = "__SOURCE_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    names = z.namelist()
    z.extractall(REPO)
print(f"extracted {len(names)} files to {REPO}")
print("tests:", sorted(n for n in names if n.startswith("tests/")))
sys.path.insert(0, REPO)
'''),
    code('''
# Run the decision-model suite. Scoped to tests/test_decision_*.py: the legacy
# decoder/tokenizer tests need models/tokenizer.json, which is not bundled here and
# is unrelated to this line of work. The repo's own quality gate covers those.
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_schema.py",
                    "tests/test_decision_protocol.py", "tests/test_decision_head.py",
                    "tests/test_decision_composer.py", "tests/test_decision_model.py",
                    "-q", "--no-header"],
                   capture_output=True, text=True, cwd=REPO)
out = r.stdout + r.stderr
print(out[-6000:])

import re
passed = int(m.group(1)) if (m := re.search(r"(\\d+) passed", out)) else 0
failed = int(m.group(1)) if (m := re.search(r"(\\d+) failed", out)) else 0
errors = int(m.group(1)) if (m := re.search(r"(\\d+) error", out)) else 0

RESULTS = {"returncode": r.returncode, "passed": passed, "failed": failed,
           "errors": errors, "ok": r.returncode == 0}
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
print("\\nSUMMARY:", json.dumps(RESULTS))
STATUS.update({"status": "complete" if r.returncode == 0 else "failed",
               "stage": "done", "ended": time.time()})
(WORK / "run_status.json").write_text(json.dumps(STATUS, indent=2))
assert r.returncode == 0, "test suite failed on Kaggle"
'''),
]

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = [
    ROOT / "src" / "__init__.py",
    *sorted((ROOT / "src" / "decision").glob("*.py")),
    ROOT / "tests" / "__init__.py",
    *sorted((ROOT / "tests").glob("test_decision_*.py")),
]
payload = io.BytesIO()
with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
    for source in BUNDLE_FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.write(source, source.relative_to(ROOT).as_posix())
bundle_b64 = base64.b64encode(payload.getvalue()).decode("ascii")

bootstrap = next(c for c in cells if "__SOURCE_BUNDLE__" in "".join(c["source"]))
bootstrap["source"] = "".join(bootstrap["source"]).replace(
    "__SOURCE_BUNDLE__", bundle_b64).splitlines(keepends=True)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(sys.argv[1])
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} — {len(json.loads(out.read_text())['cells'])} cells, "
      f"{len(BUNDLE_FILES)} files bundled")
