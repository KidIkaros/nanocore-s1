"""Dry-run the test suite without torch — the local-safe subset.

Why this exists: the full suite OOM'd this host once (pytest collection imports
the torch tier: ``head``/``legacy``/``composer``/``encoder`` are module-level
torch, and ``adapt()``/``*.fit()`` pull it lazily). A "no weights" suite is not
a no-torch suite — *importing* torch alone costs ~1-2 GB.

What a dry run proves (and what it doesn't):
- PROVES: syntax of every test file, and the pass/fail of the torch-free tier —
  gate, slow state, serve, monitor, schema, scoring math, trajectory logging.
- DOES NOT PROVE: anything torch-backed — head/composer/legacy fitting, token
  training, real encoders. That tier is Kaggle's job (the kernel runs the
  protocol test); locally it gets a syntax check, nothing more.

Usage:  python scripts/dry_run.py            # classify + run torch-free tier
        python scripts/dry_run.py --list     # show the split, run nothing
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# src modules that import torch at MODULE level (collection-time cost)
TORCH_TOP = {"src.decision.head", "src.decision.legacy", "src.decision.composer",
             "src.decision.encoder", "src.model", "src.tokenizer"}
# callables that pull torch lazily inside the function body — including the
# CLI's adapt subcommand, which reaches the same fit path with no literal
# `adapt(` in the test source
TORCH_CALLS = ("adapt(", ".fit(", "TaskHead(", "OrdinalScorer(",
               '"adapt"', "'adapt'")


def classify(path: Path) -> str:
    """'torch' if the test can reach torch (directly or transitively), else 'free'."""
    src = path.read_text()
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
    if "torch" in mods or mods & TORCH_TOP:
        return "torch"
    if any(call in src for call in TORCH_CALLS):
        return "torch"
    return "free"


def main() -> int:
    files = sorted(TESTS.glob("test_*.py"))
    # syntax-check EVERY file first — the torch tier still gets this
    for f in files:
        try:
            ast.parse(f.read_text())
        except SyntaxError as e:
            print(f"SYNTAX FAIL {f.name}: {e}")
            return 1

    tiers = {"free": [], "torch": []}
    for f in files:
        tiers[classify(f)].append(f)

    print(f"torch-free tier: {len(tiers['free'])} files")
    print(f"torch tier (syntax-checked only, Kaggle runs them): "
          f"{len(tiers['torch'])} files")
    for f in tiers["torch"]:
        print(f"  [torch] {f.name}")
    if "--list" in sys.argv:
        for f in tiers["free"]:
            print(f"  [free]  {f.name}")
        return 0
    if not tiers["free"]:
        print("nothing to run")
        return 0
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "--no-header",
         *[str(f) for f in tiers["free"]]], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
