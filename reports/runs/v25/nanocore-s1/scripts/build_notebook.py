"""Build a kernel notebook: run the generator, bundle the repo, inject, check.

One place builds the bundle so the tree the kernel extracts cannot silently
diverge from the repo. That divergence is not hypothetical: v11 shipped a
bundle without ``scripts/``, so ``tests/test_evaluate.py`` failed to import at
collection time and the contract gate aborted the run before any experiment
started.

Usage:
    python scripts/build_notebook.py [--family s1_verify] [--root .]
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

BUNDLE_DIRS = ("src", "tests", "scripts")
BUNDLE_FILES = ("pytest.ini", "pyproject.toml", "setup.cfg", "requirements.txt")


def build_zip(root: Path) -> bytes:
    """Every .py the tests can import, plus the pytest config if present."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in BUNDLE_DIRS:
            for path in sorted((root / name).rglob("*.py")):
                archive.write(path, path.relative_to(root))
        for name in BUNDLE_FILES:
            if (root / name).exists():
                archive.write(root / name, name)
    return buf.getvalue()


def git_provenance(root: Path) -> str:
    """The revision the notebook will report, with a dirty marker.

    A qualification that cannot say which revision it qualified is not
    reproducible, and a *dirty* tree is worse: the bundle the kernel runs comes
    from the working tree, not from the commit, so the two can differ. Recording
    the marker is the honest option — it makes the difference visible.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"
    # The notebook is the build product of this very step, so its own
    # uncommitted state says nothing about the tree the kernel will run. What
    # matters is whether the *sources* differ from the commit.
    dirty = "\n".join(l for l in dirty.splitlines() if not l.endswith(".ipynb")).strip()
    return commit + ("+dirty" if dirty else "")


def inject(nb_path: Path, blob: str, provenance: str = "unknown") -> tuple:
    """Replace the bundle/provenance placeholders and syntax-check every cell."""
    nb = json.loads(nb_path.read_text())
    injected, checked, bad = 0, 0, 0
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "BUNDLE = " in source:
            source = re.sub(r'BUNDLE = "[^"]*"', f'BUNDLE = "{blob}"', source, count=1)
            cell["source"] = [source]
            injected += 1
        if 'GIT_COMMIT = "' in source:
            source = re.sub(r'GIT_COMMIT = "[^"]*"',
                            f'GIT_COMMIT = "{provenance}"', source, count=1)
            cell["source"] = [source]
        checked += 1
        try:
            ast.parse(source)
        except SyntaxError as exc:
            bad += 1
            print(f"  cell {index}: {exc}")
    if injected != 1:
        raise ValueError(f"expected exactly one bundle cell, found {injected}")
    nb_path.write_text(json.dumps(nb, indent=1))
    return checked, bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="s1_verify")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    nb_dir = root / "notebooks" / args.family
    subprocess.run([sys.executable, "generate.py"], cwd=nb_dir, check=True)
    nb_path = nb_dir / f"{args.family}.ipynb"

    blob = base64.b64encode(build_zip(root)).decode()
    provenance = git_provenance(root)
    checked, bad = inject(nb_path, blob, provenance)
    print(f"{nb_path.relative_to(root)}: bundle {len(blob) // 1024}KB from "
          f"{', '.join(BUNDLE_DIRS)}, {checked} code cells, {bad} syntax errors, "
          f"provenance {provenance}")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
