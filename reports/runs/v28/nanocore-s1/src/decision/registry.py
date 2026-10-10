"""Model registry — versioned bundles with lineage, promotion, rollback.

A filesystem registry is honest infrastructure for one-model scale: every
registered bundle is immutable, self-describing, and traceable to the data +
config that produced it. ``current`` and ``stage`` are pointers, not copies —
rollback is repointing, which is instant and reversible.

Layout:

    registry/<name>/<version>/
        bundle/            the DecisionModel.save() output
        lineage.json       {data_hash, config, metrics, created, parent}
    registry/<name>/current      file containing the live version
    registry/<name>/previous     file containing the last live version

A model with no lineage is undebuggable; a registry with no rollback is a
commit log. Both are provided.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── registration ──────────────────────────────────────────────────────

    def register(self, name: str, bundle_dir, report: Dict,
                 data_hash: str = "", config: Optional[Dict] = None) -> str:
        """Copy ``bundle_dir`` in as the next version; write lineage.
        Returns the version string (``vN``)."""
        dest_parent = self.root / name
        dest_parent.mkdir(parents=True, exist_ok=True)
        existing = sorted(d.name for d in dest_parent.glob("v*")
                          if d.is_dir() and (d / "bundle").exists())
        version = f"v{len(existing) + 1}"
        dest = dest_parent / version
        shutil.copytree(bundle_dir, dest / "bundle")
        lineage = {"name": name, "version": version,
                   "created": time.time(), "data_hash": data_hash,
                   "config": config or {}, "metrics": report.get("test", {}),
                   "parent": self.current(name)}
        (dest / "lineage.json").write_text(json.dumps(lineage, indent=2))
        return version

    # ── pointers ──────────────────────────────────────────────────────────

    def current(self, name: str) -> Optional[str]:
        p = self.root / name / "current"
        return p.read_text().strip() if p.exists() else None

    def promote(self, name: str, version: str) -> str:
        """Point ``current`` at ``version``; demote the old pointer to
        ``previous`` so rollback is one call."""
        d = self.root / name
        if not (d / version / "bundle" / "manifest.json").exists():
            raise ValueError(f"no bundle at {d/version}")
        prev = self.current(name)
        if prev:
            (d / "previous").write_text(prev)
        (d / "current").write_text(version)
        return version

    def rollback(self, name: str) -> Optional[str]:
        """Repoint ``current`` to ``previous``. Returns the restored version."""
        d = self.root / name
        prev = d / "previous"
        if not prev.exists():
            return None
        restored = prev.read_text().strip()
        cur = self.current(name)
        prev.write_text(cur or "")
        (d / "current").write_text(restored)
        return restored

    # ── inspection ────────────────────────────────────────────────────────

    def lineage(self, name: str, version: Optional[str] = None) -> Dict:
        v = version or self.current(name)
        if v is None:
            raise ValueError(f"no registered version for {name!r}")
        return json.loads((self.root / name / v / "lineage.json").read_text())

    def versions(self, name: str) -> list:
        d = self.root / name
        return sorted(x.name for x in d.glob("v*")
                      if x.is_dir() and (x / "lineage.json").exists())

    def bundle_path(self, name: str, version: Optional[str] = None) -> Path:
        v = version or self.current(name)
        return self.root / name / v / "bundle"


def data_fingerprint(texts, labels) -> str:
    """Content hash of the training data — the "which data made this" key."""
    h = hashlib.sha256()
    for t, l in zip(texts, labels):
        h.update(str(t).encode()); h.update(b"\x00")
        h.update(str(l).encode()); h.update(b"\x00")
    return h.hexdigest()[:16]
