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


def inject(nb_path: Path, blob: str) -> tuple:
    """Replace the bundle placeholder and syntax-check every code cell."""
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
    checked, bad = inject(nb_path, blob)
    print(f"{nb_path.relative_to(root)}: bundle {len(blob) // 1024}KB from "
          f"{', '.join(BUNDLE_DIRS)}, {checked} code cells, {bad} syntax errors")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
