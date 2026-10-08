#!/usr/bin/env python3
"""Generate notebooks/s1_goemotions/s1_goemotions.ipynb — the headroom + first Noul battery.

The question (docs/COMPETITIVE-LANDSCAPE.md §4): is there a task where EG2 zero-shot is NOT
at ceiling? Banking77 and BFCL are saturated (~93%), so no trained component can show a win
there (ADR-0011). GoEmotions is the candidate — multilabel, fine-grained, and the Jev paper's
own table shows every model degrading on it (Jev macro-F1 0.243 → 0.353 tuned).

Formulation: 28 emotions are a **Noul battery** — each label is a yes/no question answered
with a cosine score to the label's text. That matches Jev's per-label binary protocol, makes
the F1 comparison honest, and is the first time `Noul` is exercised as a scored primitive
(it has only existed as the BFCL relevance signal).

Protocol mirrors the Jev paper's Table 4: F1 at the fixed 0.5 threshold AND at per-label
thresholds tuned on the validation split. Per-label Platt sigmoids are fitted on validation;
everything is evaluated on test. That is the fit/eval separation the three-way-split rule
formalised for conformal work.

Artifact design: per-example score matrices are saved (scores_dev/test, gold_dev/test),
which this project has never done — every prior run kept only aggregates. These are the
calibration data Stage 3's conformal gate needs.

Run it with: python generate.py <path/to/notebook.ipynb>
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {},
            "source": s.strip().splitlines(keepends=True),
            "execution_count": None, "outputs": []}


cells = [
    md('''
# GoEmotions — the headroom check, run as a Noul battery

Banking77 and BFCL are saturated at ~93% zero-shot — no trained component can demonstrate
value there (ADR-0011). This run asks the decisive question on a task where *every* model
degrades: **is there headroom, and can cosine + calibration alone exploit it?**

| | |
|---|---|
| Task | 28 emotions → **28 `Noul` questions** per comment (multilabel-native) |
| Data | `google-research-datasets/go_emotions` — ungated, Reddit comments |
| Protocol | Jev paper Table 4: F1 at fixed 0.5 **and** per-label tuned thresholds |
| Calibration | per-label Platt sigmoid, fitted on validation only |
| Firsts | first real `Noul` evaluation · first saved per-example score arrays |

Jev paper reference points (arXiv:2609.37647, GoEmotions macro/micro-F1, fixed 0.5 → tuned):

| model | macro-F1 | micro-F1 |
|---|---:|---:|
| Jev | 0.243 → 0.353 | 0.239 → 0.387 |
| Qwen3.8-27B | 0.255 → 0.323 | 0.236 → 0.317 |
| Gemma-4-E4B | 0.211 → 0.267 | 0.207 → 0.268 |

For context, fine-tuned-BERT-era results on this dataset sit around macro-F1 ~0.46 — a
zero-shot score near 0.4 would be competitive with *trained* systems.
'''),

    md('## 0. Environment'),
    code('''
import json, subprocess, time
from pathlib import Path

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
print(gpu)
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
assert "T4" in gpu, f"Expected T4, got {gpu!r} — set machine_shape to NvidiaTeslaT4"

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/s1-work")
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

# The protocol is the instrument, so verify it before using it.
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_protocol.py",
                    "-q", "--no-header"], capture_output=True, text=True, cwd=REPO)
print((r.stdout + r.stderr)[-2500:])
assert r.returncode == 0, "protocol tests failed"
write_status("contract-tests")
'''),

    code('''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=6.1.0", "transformers>=5.19.0",
                "datasets>=2.20.0", "scikit-learn>=1.4"], check=True)

import numpy as np
from src.decision import protocol
from src.decision.encoder import StateEncoder, to_numpy

# Text-only: comments and label texts are text, so 270M suffices.
encoder = StateEncoder(modalities=("text",), device="cuda")
print("encoder params:", round(sum(p.numel() for p in encoder.model.parameters()) / 1e6), "M",
      "| dtype:", encoder.dtype)
write_status("encoder")
'''),

    md('''
## 1. Load GoEmotions

`simplified` config: `labels` is a list of emotion ids (multilabel). Fallback is the
upstream repo's suggested train/dev/test TSVs.
'''),
    code('''
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]
L = len(EMOTIONS)

def load_via_datasets():
    import datasets
    ds = datasets.load_dataset("google-research-datasets/go_emotions", "simplified")
    def rows(split):
        return [(r["text"], sorted(set(r["labels"]))) for r in ds[split]]
    return rows("validation"), rows("test")

def load_via_tsv():
    import requests
    base = ("https://raw.githubusercontent.com/google-research/google-research/"
            "master/goemotions/data/")
    def rows(name):
        out = []
        txt = requests.get(base + name, timeout=300).text
        for line in txt.splitlines()[1:]:
            parts = line.split("\\t")
            if len(parts) < 2:
                continue
            ids = [int(x) for x in parts[1].split(",") if x.strip()]
            out.append((parts[0], sorted(set(ids))))
        return out
    return rows("dev.tsv"), rows("test.tsv")

try:
    dev, test = load_via_datasets()
    src = "hf datasets"
except Exception as e:
    print("datasets path failed:", e)
    dev, test = load_via_tsv()
    src = "github tsv"

def gold_matrix(rows):
    Y = np.zeros((len(rows), L), dtype=np.float32)
    for i, (_, labs) in enumerate(rows):
        for lab in labs:
            if lab < L:
                Y[i, lab] = 1.0
    return Y

X_dev_text  = [t for t, _ in dev]
X_test_text = [t for t, _ in test]
Y_dev, Y_test = gold_matrix(dev), gold_matrix(test)

pos_dev  = Y_dev.sum(0); pos_test = Y_test.sum(0)
print(f"source: {src} | dev {len(dev)} | test {len(test)} | labels {L}")
print(f"multilabel fraction dev: {(Y_dev.sum(1) > 1).mean():.3f}")
for i, e in enumerate(EMOTIONS):
    if pos_test[i] < 20:
        print(f"  rare label: {e} has only {int(pos_test[i])} positives in test")
print(f"min positives test: {int(pos_test.min())} | max: {int(pos_test.max())}")
assert len(test) > 2000 and Y_test.sum(1).min() >= 0, "GoEmotions layout changed"
write_status("data")
'''),

    md('''
## 2. Encode

States get `SearchQuery`, label texts get `Document` — the pairing the prompt ablation
selected. Two label-text templates are encoded: the bare emotion name and an explicit
`"a comment expressing {emotion}"`, so the template choice is an ablation, not a guess.
'''),
    code('''
STATE_PROMPT, OPTION_PROMPT = "SearchQuery", "Document"

texts = X_dev_text + X_test_text
templates = {
    "name":     [e for e in EMOTIONS],
    "sentence": [f"a comment expressing {e}" for e in EMOTIONS],
}
S = to_numpy(encoder.encode(texts, prompt_name=STATE_PROMPT, batch_size=64))
torch.cuda.empty_cache()
LVECS = {k: to_numpy(encoder.encode(v, prompt_name=OPTION_PROMPT, batch_size=64))
         for k, v in templates.items()}
torch.cuda.empty_cache()
print("peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
n_dev = len(X_dev_text)
print("S:", S.shape, "| label vecs:", {k: v.shape for k, v in LVECS.items()})
write_status("encoded")
'''),

    md('''
## 3. Scores — cosine per (comment, emotion) pair

`scores[i, l] = s_i · e_l`. These matrices are the run's most durable artifact: they are
saved with the gold matrix, giving Stage 3's conformal gate real calibration data and any
future head a real target — fit on validation, evaluate on test (the three-way rule).
'''),
    code('''
SCORES = {}
for k, lv in LVECS.items():
    sc = S @ lv.T
    SCORES[k] = {"dev": sc[:n_dev], "test": sc[n_dev:]}

np.savez_compressed(WORK / "goemotions_scores.npz",
                    scores_dev_name=SCORES["name"]["dev"],
                    scores_test_name=SCORES["name"]["test"],
                    scores_dev_sentence=SCORES["sentence"]["dev"],
                    scores_test_sentence=SCORES["sentence"]["test"],
                    gold_dev=Y_dev, gold_test=Y_test,
                    emotions=np.array(EMOTIONS))
print("saved goemotions_scores.npz — the conformal gate's future calibration data")
write_status("scored")
'''),

    md('''
## 4. Calibration + evaluation

Per label: a Platt sigmoid fitted on validation scores, then two operating points —
the fixed 0.5 threshold (the number Jev's Table 4 calls "poorly placed") and a per-label
threshold tuned for F1 on validation. Both applied untouched to test.
'''),
    code('''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, log_loss

def platt_fit(sc_dev, y_dev):
    """One sigmoid per label; falls back to a constant where a class is degenerate."""
    models = []
    for l in range(L):
        x = sc_dev[:, l].reshape(-1, 1)
        y = y_dev[:, l]
        if y.min() == y.max():
            models.append(("const", float(y[0])))
            continue
        m = LogisticRegression(C=10.0, max_iter=500).fit(x, y)
        models.append(("platt", m))
    return models

def platt_apply(models, sc):
    P = np.zeros_like(sc)
    for l, (kind, m) in enumerate(models):
        if kind == "const":
            P[:, l] = m
        else:
            P[:, l] = m.predict_proba(sc[:, l].reshape(-1, 1))[:, 1]
    return P

def best_f1_threshold(probs, y, grid=None):
    grid = grid if grid is not None else np.linspace(0.02, 0.98, 49)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, probs >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

def evaluate(P, Y, thresholds):
    pred = P >= thresholds
    return {"macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0))}

RESULTS = {"label_templates": {}, "per_label_auroc": {}, "verdict": {}}
def save(stage):
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    write_status(stage)

for name, sc in SCORES.items():
    models = platt_fit(sc["dev"], Y_dev)
    P_dev, P_test = platt_apply(models, sc["dev"]), platt_apply(models, sc["test"])

    tuned = np.array([best_f1_threshold(P_dev[:, l], Y_dev[:, l]) for l in range(L)])
    fixed = np.full(L, 0.5)

    row = {"fixed_0.5":  evaluate(P_test, Y_test, fixed),
           "tuned_dev":  evaluate(P_test, Y_test, tuned),
           "log_loss":   float(np.mean([log_loss(Y_test[:, l], P_test[:, l],
                                                 labels=[0, 1]) for l in range(L)])),
           "tuned_thresholds": dict(zip(EMOTIONS, [float(t) for t in tuned]))}
    auroc = {}
    for l, e in enumerate(EMOTIONS):
        if Y_test[:, l].min() < Y_test[:, l].max():
            auroc[e] = float(roc_auc_score(Y_test[:, l], sc["test"][:, l]))
    row["macro_auroc"] = float(np.mean(list(auroc.values())))
    RESULTS["label_templates"][name] = row
    RESULTS["per_label_auroc"][name] = auroc

    print(f"\\n[{name}]  macro-F1 {row['fixed_0.5']['macro_f1']:.3f} -> "
          f"{row['tuned_dev']['macro_f1']:.3f} | micro-F1 {row['fixed_0.5']['micro_f1']:.3f} -> "
          f"{row['tuned_dev']['micro_f1']:.3f} | macro-AUROC {row['macro_auroc']:.3f}")
save("evaluated")
'''),

    md('''
## 5. Verdict — headroom, and against the published table

Two questions: (a) does this benchmark have room for trained components (ADR-0011's
headroom check), and (b) where does the typed-decision approach land against the three
systems the Jev paper already scored.
'''),
    code('''
best = max(RESULTS["label_templates"].items(),
           key=lambda kv: kv[1]["tuned_dev"]["macro_f1"])
bname, brow = best
print("best template:", bname)

hr = protocol.headroom_check(brow["tuned_dev"]["macro_f1"])
RESULTS["verdict"]["headroom"] = hr
print("headroom check:", json.dumps(hr))

JEV_TABLE = {
    "jev":          {"macro": (0.243, 0.353), "micro": (0.239, 0.387)},
    "qwen3.8-27b":  {"macro": (0.255, 0.323), "micro": (0.236, 0.317)},
    "gemma-4-e4b":  {"macro": (0.211, 0.267), "micro": (0.207, 0.268)},
}
ours = {"macro": (brow["fixed_0.5"]["macro_f1"], brow["tuned_dev"]["macro_f1"]),
        "micro": (brow["fixed_0.5"]["micro_f1"], brow["tuned_dev"]["micro_f1"])}
RESULTS["verdict"]["ours"] = {"template": bname, "macro": ours["macro"], "micro": ours["micro"]}
RESULTS["verdict"]["jev_table"] = JEV_TABLE

print(f"\\n{'model':<14} {'macro@0.5':>9} {'macro@tuned':>11} {'micro@0.5':>9} {'micro@tuned':>11}")
print(f"{'ours (EG2+Platt)':<14} {ours['macro'][0]:>9.3f} {ours['macro'][1]:>11.3f} "
      f"{ours['micro'][0]:>9.3f} {ours['micro'][1]:>11.3f}")
for name, t in JEV_TABLE.items():
    print(f"{name:<14} {t['macro'][0]:>9.3f} {t['macro'][1]:>11.3f} "
          f"{t['micro'][0]:>9.3f} {t['micro'][1]:>11.3f}")

# The honest read: AUROC high + F1 low means calibration fixes it; AUROC low means
# genuine headroom where a learned component could win.
RESULTS["verdict"]["read"] = {
    "auroc_vs_f1_gap": float(brow["macro_auroc"] - brow["tuned_dev"]["macro_f1"]),
    "interpretation": ("AUROC >> F1: ordering is good, thresholds/calibration are the "
                       "bottleneck — a scoring-rule fix, not a head opportunity"
                       if brow["macro_auroc"] - brow["tuned_dev"]["macro_f1"] > 0.25 else
                       "AUROC ~ F1: ordering itself is weak — real headroom for trained "
                       "components"),
}
print("\\nread:", json.dumps(RESULTS["verdict"]["read"]))

(WORK / "REPORT.md").write_text("# GoEmotions Noul battery\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
save("verdict")
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json", "goemotions_scores.npz"]:
        p = WORK / f
        if p.exists():
            zf.write(p, f)
STATUS["status"] = "complete"
write_status("done", done=True)
'''),
]

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = [
    ROOT / "src" / "__init__.py",
    *sorted((ROOT / "src" / "decision").glob("*.py")),
    ROOT / "tests" / "__init__.py",
    *(ROOT / "tests" / n for n in ("test_decision_schema.py", "test_decision_protocol.py",
                                   "test_decision_head.py", "test_decision_model.py")),
]
payload = io.BytesIO()
with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
    for source in BUNDLE_FILES:
        if not source.is_file():
            raise FileNotFoundError(source)
        archive.write(source, source.relative_to(ROOT).as_posix())
bundle_b64 = base64.b64encode(payload.getvalue()).decode("ascii")

bootstrap = next(c for c in cells if "__DECISION_BUNDLE__" in "".join(c["source"]))
bootstrap["source"] = "".join(bootstrap["source"]).replace(
    "__DECISION_BUNDLE__", bundle_b64).splitlines(keepends=True)

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
print(f"wrote {out} — {len(json.loads(out.read_text())['cells'])} cells, valid JSON")
