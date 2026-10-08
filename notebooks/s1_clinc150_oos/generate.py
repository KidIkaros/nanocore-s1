#!/usr/bin/env python3
"""Generate notebooks/s1_clinc150_oos/s1_clinc150_oos.ipynb — the knowing-when benchmark.

ADR-0013 added a `clarify` action and meta-routed heads on the premise that ambiguity is
*detectable*: when the state can't separate intents, scores go flat and a conformal set
goes wide. CLINC150 is the benchmark built for exactly this — 150 intents plus an
explicit out-of-scope (OOS) class. The question: do cheap ambiguity signals separate
OOS inputs from in-scope ones, and does the prediction set actually contain the true
intent when it's small (the `clarify` premise)?

Arms (all on the same cached cosine scores):
  max_sim   — raw top-1 cosine (low → OOS)
  max_prob  — temperature-scaled top-1 softmax (temp fit on dev in-scope)
  margin    — top1−top2 gap
  entropy   — predictive entropy of the softmax
  aps_set   — APS conformal set size (wide/empty → ambiguous)

Metrics: in-scope 150-way accuracy, OOS-detection AUROC per signal, selective
accuracy-vs-coverage, and the clarify-resolution rate P(true intent ∈ set | |set| ≤ 3).
Score matrices are saved so Stage-3 conformal work reuses them without re-encoding.

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
# CLINC150 out-of-scope — can it tell *when* it doesn't know?

ADR-0013 premise under test: ambiguity is measurable. 150 intents + an explicit OOS
class. Every signal below runs on the same cached cosine scores — the only thing that
varies is *how "I don't know" is computed*.

| signal | hypothesis |
|---|---|
| `max_sim` | OOS inputs match no intent well |
| `max_prob` | temperature-scaled confidence drops OOS |
| `margin` | ambiguous inputs have two close candidates |
| `entropy` | flat distributions mark confusion |
| `aps_set` | conformal sets go wide (or empty) out-of-scope |

Plus the `clarify` premise directly: when the set is small (≤3), how often does it
contain the true intent? If that number is high, a one-question follow-up resolves the
ambiguity — the curve-softening mechanism from ADR-0013 has teeth.
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
assert "T4" in gpu, f"Expected T4, got {gpu!r}"

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
from src.decision.encoder import StateEncoder, to_numpy

encoder = StateEncoder(modalities=("text",), device="cuda")
print("encoder params:", round(sum(p.numel() for p in encoder.model.parameters()) / 1e6), "M")
write_status("encoder")
'''),

    md('## 1. Load CLINC150 ("plus" config — includes OOS examples in every split)'),
    code('''
import datasets

ds = datasets.load_dataset("clinc_oos", "plus")
names = ds["test"].features["intent"].names
OOS = names.index("oos") if "oos" in names else None
print("classes:", len(names), "| oos index:", OOS)
assert OOS is not None, "oos label not found — dataset layout changed"

splits = {k: (ds[k]["text"], np.array(ds[k]["intent"])) for k in ("train", "validation", "test")}
for k, (x, y) in splits.items():
    print(f"{k}: {len(x)} total | oos {(y == OOS).sum()}")

X_val_text, y_val = splits["validation"]
X_te_text, y_te = splits["test"]
val_in = y_val != OOS
te_in = y_te != OOS
write_status("data")
'''),

    md('''
## 2. Encode utterances + intent labels

Intent names are snake_case (`restaurant_reviews`); humanised with spaces — the
GoEmotions ablation showed label-text verbosity hurts, so one rendering only.
'''),
    code('''
INTENT_TEXTS = [n.replace("_", " ") for n in names if n != "oos"]
INTENT_IDS = [i for i, n in enumerate(names) if n != "oos"]

texts = list(X_val_text) + list(X_te_text)
S = to_numpy(encoder.encode(texts, prompt_name="SearchQuery", batch_size=64))
LV = to_numpy(encoder.encode(INTENT_TEXTS, prompt_name="Document", batch_size=64))
torch.cuda.empty_cache()

S_val, S_te = S[:len(X_val_text)], S[len(X_val_text):]
SC_val, SC_te = S_val @ LV.T, S_te @ LV.T
print("scores:", SC_val.shape, SC_te.shape,
      "| peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))

np.savez_compressed(WORK / "clinc_scores.npz",
                    scores_val=SC_val, scores_test=SC_te,
                    y_val=y_val, y_test=y_te, intent_ids=np.array(INTENT_IDS),
                    names=np.array(names))
write_status("encoded")
'''),

    md('''
## 3. Ambiguity signals — same scores, five ways to say "I don't know"

`y` values for in-scope items are CLINC intent ids; `INTENT_IDS` maps them to score
columns (oos is excluded from the option set — matching deployment, where OOS isn't a
choice, it's a *detection*).
'''),
    code('''
from sklearn.metrics import roc_auc_score, accuracy_score

col_of = {intent_id: c for c, intent_id in enumerate(INTENT_IDS)}
y_val_c = np.array([col_of.get(int(v), -1) for v in y_val])
y_te_c = np.array([col_of.get(int(v), -1) for v in y_te])

def softmax(sc, T):
    z = sc / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)

# temperature on dev in-scope (minimize log-loss of true intent)
best_T, best_ll = 1.0, np.inf
for T in np.logspace(-2, 1, 61):
    P = softmax(SC_val[val_in], T)
    ll = -np.log(np.clip(P[np.arange(len(P)), y_val_c[val_in]], 1e-9, 1)).mean()
    if ll < best_ll:
        best_ll, best_T = ll, T
print(f"temperature: {best_T:.3f} (dev log-loss {best_ll:.4f})")

P_val, P_te = softmax(SC_val, best_T), softmax(SC_te, best_T)

def signals(sc, P):
    srt = np.sort(sc, axis=1)
    ent = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
    return {"max_sim": sc.max(1), "max_prob": P.max(1),
            "margin": srt[:, -1] - srt[:, -2], "neg_entropy": -ent}

def aps_sets(P, q):
    """APS: smallest top-prob set covering mass q. Returns set-size per row."""
    srt = -np.sort(-P, axis=1)
    cum = np.cumsum(srt, axis=1)
    return (cum < q).sum(1) + 1

# conformal quantile of needed mass on dev in-scope, alpha=0.10
def aps_mass_needed(P, y_c):
    srt = -np.sort(-P, axis=1)
    cum = np.cumsum(srt, axis=1)
    idx = np.arange(len(P))
    rank = (srt > P[idx, y_c][:, None]).sum(1)          # position of true label
    return cum[idx, np.minimum(rank, srt.shape[1] - 1)]

ALPHA = 0.10
masses = aps_mass_needed(P_val[val_in], y_val_c[val_in])
n = len(masses)
qhat = np.quantile(masses, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n))
print(f"APS q̂ (α={ALPHA}, dev n={n}): {qhat:.4f}")

SIG_val = signals(SC_val, P_val)
SIG_te = signals(SC_te, P_te)
SIG_val["neg_set_size"] = -aps_sets(P_val, qhat)
SIG_te["neg_set_size"] = -aps_sets(P_te, qhat)
RESULTS = {"temperature": float(best_T), "qhat": float(qhat), "alpha": ALPHA}
write_status("signals")
'''),

    md('''
## 4. Measure — in-scope accuracy, OOS separation, coverage, clarify-resolution

- **In-scope accuracy**: plain 150-way argmax (the base capability)
- **OOS AUROC**: does the signal rank in-scope above OOS?
- **Selective accuracy vs coverage**: abstain on the weakest signal → how fast does
  accuracy climb?
- **Clarify resolution**: of in-scope items with |APS set| ≤ 3, how often does the set
  contain the true intent — i.e., how often would one follow-up question fix it?
'''),
    code('''
# in-scope accuracy
acc_val = accuracy_score(y_val_c[val_in], SC_val[val_in].argmax(1))
acc_te = accuracy_score(y_te_c[te_in], SC_te[te_in].argmax(1))
RESULTS["in_scope_accuracy"] = {"val": float(acc_val), "test": float(acc_te)}
print(f"in-scope 150-way accuracy: val {acc_val:.3f} | test {acc_te:.3f}")

# OOS AUROC per signal
RESULTS["oos_auroc"] = {}
for name in SIG_te:
    a_val = roc_auc_score(val_in, SIG_val[name])
    a_te = roc_auc_score(te_in, SIG_te[name])
    RESULTS["oos_auroc"][name] = {"val": float(a_val), "test": float(a_te)}
    print(f"OOS AUROC {name:<14} val {a_val:.3f} | test {a_te:.3f}")
'''),

    code('''
# selective prediction: sort test items by signal strength, measure in-scope
# accuracy + OOS rejection as coverage shrinks
RESULTS["selective"] = {}
for name in SIG_te:
    sig = SIG_te[name]
    order = np.argsort(-sig)                      # strongest first
    curve = []
    for cov in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        k = int(len(order) * cov)
        idx = order[:k]
        in_idx = idx[te_in[idx]]
        acc = accuracy_score(y_te_c[in_idx], SC_te[in_idx].argmax(1)) if len(in_idx) else np.nan
        curve.append({"coverage": cov, "in_scope_acc": float(acc)})
    RESULTS["selective"][name] = curve
best = max(RESULTS["oos_auroc"], key=lambda k: RESULTS["oos_auroc"][k]["test"])
print(f"best OOS signal: {best} (AUROC {RESULTS['oos_auroc'][best]['test']:.3f})")
print("its coverage curve:", RESULTS["selective"][best])
'''),

    code('''
# clarify-resolution + set-size stats on test
set_val, set_te = aps_sets(P_val, qhat), aps_sets(P_te, qhat)
def set_contains(P, q, y_c):
    srt = np.argsort(-P, axis=1)
    cum = np.cumsum(-np.sort(-P, axis=1), axis=1)
    sizes = (cum < q).sum(1) + 1
    hit = np.zeros(len(P), dtype=bool)
    for i in range(len(P)):
        hit[i] = y_c[i] in srt[i, :sizes[i]]
    return hit

hit_val = set_contains(P_val, qhat, np.clip(y_val_c, 0, None))
mask_val = val_in & (set_val <= 3)
hit_te = set_contains(P_te, qhat, np.clip(y_te_c, 0, None))
mask_te = te_in & (set_te <= 3)

RESULTS["clarify"] = {
    "in_scope_with_small_set_frac": float(mask_te.mean()),
    "true_intent_in_small_set": float(hit_te[mask_te].mean()) if mask_te.any() else None,
    "mean_set_size_in_scope": float(set_te[te_in].mean()),
    "mean_set_size_oos": float(set_te[~te_in].mean()),
    "coverage_at_alpha": float(hit_te[te_in].mean()),
}
print(json.dumps(RESULTS["clarify"], indent=2))
RESULTS["headroom_check"] = "n/a — this is a detection benchmark, not a head comparison"
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
write_status("measured")
'''),

    md('''
## 5. Verdict — does ADR-0013 have teeth?

The `clarify` action is justified iff:

1. **OOS is separable** — best AUROC meaningfully above 0.5 (deployment-grade is ≥0.9)
2. **The set is honest** — coverage at α=0.10 ≈ 90% on in-scope (conformal guarantee)
3. **The set is useful** — `true_intent_in_small_set` high → a one-question follow-up
   resolves most ambiguous-but-in-scope inputs

Failure of (1) means OOS needs a dedicated detector arm, not emergent behaviour.
Failure of (3) means `clarify` would ask questions that don't resolve — abstain instead.
'''),
    code('''
verdict = {
    "oos_separable": RESULTS["oos_auroc"][best]["test"] >= 0.9,
    "coverage_holds": abs(RESULTS["clarify"]["coverage_at_alpha"] - (1 - ALPHA)) < 0.03,
    "clarify_resolves": (RESULTS["clarify"]["true_intent_in_small_set"] or 0) >= 0.8,
}
RESULTS["verdict"] = verdict
print(json.dumps(verdict, indent=2))

(WORK / "REPORT.md").write_text("# CLINC150 OOS — knowing-when benchmark\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json", "clinc_scores.npz"]:
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
