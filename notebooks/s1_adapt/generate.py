#!/usr/bin/env python3
"""Generate notebooks/s1_adapt/s1_adapt.ipynb — the adaptation harness proof.

Roadmap Phase 2: one call turns labeled data into a calibrated bundle.
Proof target: Banking77 (77 intents) — a task we have never calibrated.

  1. contract suite (tests/, no weights) — runs in-session
  2. real EmbeddingGemma 2 load (text, T4)
  3. Banking77 load; 20 classes held out as OOS for the τ_in_schema leg
  4. adapt(texts, labels) -> TaskHead + calibrated ConformalGate + bundle
  5. reload the bundle, reproduce held-out metrics exactly
  6. CLI leg — the shipped entry point answers through the adapted bundle
  7. artifacts: adapt_report.json, bundle/, results.json, run_status.json

Run it with: python generate.py <path/to/notebook.ipynb>
"""
import json
import sys
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {},
            "source": s.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}


cells = [
    md('''# s1_adapt — adaptation harness on a task we've never calibrated

`adapt(texts, labels, encoder)` must take raw labeled data and emit a bundle
(head + calibrated gate) whose held-out metrics reproduce exactly on reload.
Banking77: 13k customer-service queries over 77 intents.

Acceptance (roadmap Phase 2): one command → bundle; bundle reproduces
held-out metrics within tolerance; in-schema (OOS) threshold fitted on
classes the head never saw.
'''),

    md('## 0. Environment'),
    code('''
import json, subprocess, time
from pathlib import Path

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
print(gpu)
assert "T4" in gpu, f"Expected T4, got {gpu!r}"

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/s1-adapt")
WORK.mkdir(parents=True, exist_ok=True)
STATUS = {"status": "running", "stage": "env", "started": time.time()}
def write_status(stage=None, error=None, done=False):
    if stage: STATUS["stage"] = stage
    if error: STATUS["error"] = str(error); STATUS["status"] = "failed"
    if done: STATUS["status"] = "complete"
    STATUS["ended"] = time.time()
    (WORK / "run_status.json").write_text(json.dumps(STATUS, indent=2))
write_status()
'''),

    code('''
import base64, io, sys, zipfile

BUNDLE = "__DECISION_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)

r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                   capture_output=True, text=True, cwd=REPO)
print((r.stdout + r.stderr)[-3000:])
assert r.returncode == 0, "test suite failed on Kaggle"
write_status("tests-green")
'''),

    code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=6.1.0", "transformers>=5.19.0",
                "datasets>=2.20.0", "scikit-learn>=1.4"], check=True)

import numpy as np
import torch
from src.decision.encoder import StateEncoder, to_numpy

encoder = StateEncoder(modalities=("text",), device="cuda")
write_status("encoder")
'''),

    md('## 1. Banking77 — load and hold out OOS classes'),
    code('''
def load_banking77():
    """Namespaced HF first; GitHub CSV fallback (legacy no-namespace ids fail)."""
    try:
        import datasets
        bank = datasets.load_dataset("PolyAI/banking77")
        names = bank["train"].features["label"].names
        tr = (bank["train"]["text"], [names[i] for i in bank["train"]["label"]])
        te = (bank["test"]["text"], [names[i] for i in bank["test"]["label"]])
        return names, tr, te, "hf datasets"
    except Exception as e:
        print("hf path failed:", repr(e)[:140])
    import pandas as pd
    base = ("https://raw.githubusercontent.com/PolyAI-LDN/"
            "task-specific-datasets/master/banking_data/")
    trdf = pd.read_csv(base + "train.csv"); tedf = pd.read_csv(base + "test.csv")
    names = sorted(set(trdf["category"]) | set(tedf["category"]))
    return (names, (trdf["text"].tolist(), trdf["category"].tolist()),
            (tedf["text"].tolist(), tedf["category"].tolist()), "github csv")

names, (tr_x, tr_y), (te_x, te_y), src = load_banking77()
print(f"banking77 via {src}: {len(tr_x)} train / {len(te_x)} test, "
      f"{len(names)} intents")

rng = np.random.default_rng(0)
perm = rng.permutation(len(names))
oos_names = set(names[i] for i in perm[:20])          # head never sees these
in_names  = [n for n in names if n not in oos_names]  # 57 in-schema intents

def rows(xs, ys):
    out_x, out_y = [], []
    for t, l in zip(xs, ys):
        if l not in oos_names:
            out_x.append(t); out_y.append(l)
    return out_x, out_y

X_texts, y_labels = rows(tr_x, tr_y)
Xte_texts, yte_labels = rows(te_x, te_y)
oos_texts = [t for t, l in zip(te_x, te_y) if l in oos_names]
print(f"in-schema: {len(X_texts)} train / {len(Xte_texts)} test | "
      f"oos: {len(oos_texts)} | classes: {len(in_names)}")
RESULTS = {"n_train": len(X_texts), "n_test": len(Xte_texts),
           "n_oos": len(oos_texts), "n_classes": len(in_names)}
write_status("data")
'''),

    md('''## 2. `adapt()` — one call: split → encode → head → gate → evaluate

Stratified 50/25/25 fit/cal/test inside the harness; the gate sub-splits
cal into A (T_prob+τ) and B (q̂). OOS texts fit τ_in_schema.
'''),
    code('''
from src.decision.adapt import AdaptConfig, adapt

res = adapt(X_texts, y_labels, encoder,
            cfg=AdaptConfig(head_kind="linear", alpha=0.10,
                            head_kwargs={"epochs": 60}, seed=0),
            oos_texts=oos_texts,
            out_dir=WORK / "adapted")

rep = res.report
RESULTS["report"] = rep
print(json.dumps(rep["headroom"], indent=1))
print(f"head: {rep['head']['kind']} acc={rep['test']['accuracy']:.3f} "
      f"brier={rep['test']['brier']:.3f} "
      f"coverage={rep['test']['conformal_coverage']:.3f} "
      f"resolved={rep['test']['resolved_at_tau']:.3f} "
      f"set={rep['test']['mean_set_size']:.1f}")
print(f"tau_answer={rep['calibration']['tau_answer']:.4f} "
      f"t_prob={rep['calibration']['t_prob']:.3f} "
      f"qhat={rep['calibration']['qhat']:.3f} "
      f"tau_in_schema={rep['tau_in_schema']}")
write_status("adapted")
'''),

    md('''## 3. Bundle reload + external test

Two checks: (a) the reloaded bundle produces **identical** predictions to the
in-memory model on the same vectors; (b) it scores Banking77's official test
split — data the harness never touched during fit or calibration.
'''),
    code('''
from src.decision.model import DecisionModel

rebuilt = DecisionModel.load(WORK / "adapted" / "bundle", encoder=None)
labels_ = rebuilt.scorer.labels_

# (a) reload identity on fresh encodings of held-in-schema test texts
Xte = to_numpy(encoder.encode(Xte_texts[:400], prompt_name="Classification",
                              batch_size=64))
same, same_acc = 0, 0
yte_idx = np.array([labels_.index(l) for l in yte_labels[:400]])
for i in range(len(Xte)):
    a = res.model.gate.decide(res.model.scorer.scores(Xte[i]), labels_)
    b = rebuilt.gate.decide(rebuilt.scorer.scores(Xte[i]), labels_)
    same += int(a.action == b.action and a.prediction_set == b.prediction_set
                and abs(a.top_prob - b.top_prob) < 1e-6)
    same_acc += int(int(np.argmax(rebuilt.scorer.scores(Xte[i]))) == yte_idx[i])
RESULTS["reload_identity"] = {"n": len(Xte), "identical": same}
RESULTS["external_test_acc"] = same_acc / len(Xte)
print(f"reload identity: {same}/{len(Xte)} | external test acc: "
      f"{same_acc/len(Xte):.3f}")
assert same == len(Xte), "bundle round-trip mismatch"
write_status("reload")
'''),

    md('## 4. CLI leg — the entry point answers through the adapted bundle'),
    code('''
RESULTS["cli"] = {"status": "skipped"}
try:
    demos = ["how do i activate my new card",
             "my card was swallowed by the atm",
             "i want to change my pin",
             "when does my transfer arrive",
             "tell me about my savings account",
             "i got charged twice for the same thing",
             "what is this fee on my statement",
             "the atm ate my card this morning",
             "how do i top up my account",
             "i want to close my account",
             "blorple snizzle wumpus quack",
             "what is the meaning of life"]
    (WORK / "demos.txt").write_text("\\n".join(demos))
    opts = ",".join(rebuilt.scorer.labels_)
    r = subprocess.run(
        [sys.executable, "-m", "src.decision.cli", "decide",
         "--inputs", str(WORK / "demos.txt"),
         "--options", opts,
         "--bundle", str(WORK / "adapted" / "bundle"),
         "--cache", str(WORK / "cli-cache"),
         "--json"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"CLI exit {r.returncode}: {r.stderr[-2000:]}"
    cli_out = json.loads(r.stdout)
    assert len(cli_out) == len(demos)
    for o in cli_out:
        print(f"{o['input'][:50]!r:55} -> {o['action']:9} {o['top_prob']:.3f} "
              f"{o['prediction_set'][:3]}")
    RESULTS["cli"] = {"status": "ran", "n": len(cli_out),
                      "transcript": [{"input": o["input"], "action": o["action"],
                                      "top_prob": round(o["top_prob"], 4),
                                      "set": o["prediction_set"][:4]}
                                     for o in cli_out]}
except Exception as e:
    RESULTS["cli"] = {"status": "failed", "error": repr(e)[:300]}
    print("cli leg failed:", repr(e)[:200])
write_status("cli")
'''),

    md('## 5. Verdict'),
    code('''
rep = RESULTS["report"]
verdict = {
    "suite_green": True,
    "adapt_ran": "report" in RESULTS,
    "head_beats_zeroshot": (rep["test"]["accuracy"]
                           > rep["headroom"]["zeroshot_test_acc"]),
    "no_undercoverage": rep["test"]["conformal_coverage"] >= 1 - rep["config"]["alpha"] - 0.05,
    "bundle_written": (WORK / "adapted" / "bundle" / "manifest.json").exists(),
    "reload_identical": RESULTS["reload_identity"]["identical"] == 400,
    "tau_in_schema_fitted": rep["tau_in_schema"] is not None,
    "cli_ran": RESULTS["cli"]["status"] == "ran",
}
RESULTS["verdict"] = verdict
print(json.dumps(verdict, indent=2))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
write_status(done=True)
'''),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("s1_adapt.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}")
