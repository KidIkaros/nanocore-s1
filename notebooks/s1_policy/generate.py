#!/usr/bin/env python3
"""Generate notebooks/s1_policy/s1_policy.ipynb — the corrected policy, verified.

s1_policy showed the first-draft trigger (answer iff |APS set| == 1) is unreachable on a
flat softmax — zero answers, 72% escalated, -0.26 vs threshold gating. The corrected
mapping keys actions on *calibrated confidence*, with the set as the clarify payload:

    not in_schema (max_sim < τ)            → escalate
    top_prob ≥ τ_answer (calibrated)       → answer
    APS set ≤ k_clarify (3)                → clarify
    else                                   → escalate

Protocol fixes applied (this is the Stage-3 gate prototype):
  - three-way splits: train → head fit; val_A → temperature + thresholds;
    val_B → conformal q̂; test → evaluated once
  - dual temperature: T_prob (log-loss, floored at 0.25) for reported probabilities;
    T_set = 1.0 for APS set construction
  - τ_answer chosen on val_A as the smallest threshold reaching ≥0.90 precision on
    answered items (selective-prediction target)

Policies compared: always_answer, threshold (incumbent, resolved 0.712), ambiguity_v2,
ambiguity_v2 with a fitted linear TaskHead as scorer (its own T_prob/q̂ per scorer —
the third sharpening-pathology fix).

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
# The corrected ambiguity-aware policy — gate prototype v2

v1's lesson: trigger on calibrated confidence, not absolute set size. This run is the
full gate machinery as designed for Stage 3:

- **dual temperature** — T_prob (floored, log-loss-fitted) reports probabilities;
  T_set=1.0 builds APS sets
- **three-way splits** — val_A fits temperature + thresholds, val_B calibrates q̂,
  test is touched once
- **τ_answer** — smallest threshold reaching ≥0.90 precision among answered items
- **per-scorer calibration** — the TaskHead gets its own T_prob and q̂ (it would
  otherwise produce degenerate sets: q̂≈1.0 measured in v1)

Verdict question: does `ambiguity_v2` beat `threshold` (0.712) on resolved fraction?
'''),

    md('## 0. Environment'),
    code('''
import json, subprocess, time
from pathlib import Path

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout
print(gpu)
import torch
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
assert r.returncode == 0
write_status("contract-tests")
'''),

    code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=6.1.0", "transformers>=5.19.0",
                "datasets>=2.20.0", "scikit-learn>=1.4"], check=True)

import numpy as np
from src.decision.encoder import StateEncoder, to_numpy
import torch.nn as nn

encoder = StateEncoder(modalities=("text",), device="cuda")
write_status("encoder")
'''),

    md('## 1. Load CLINC150 (dual-path loader)'),
    code('''
def load_via_datasets():
    import datasets
    ds = datasets.load_dataset("clinc_oos", "plus")
    names = list(ds["test"].features["intent"].names)
    return names, {k: (list(ds[k]["text"]), np.array(ds[k]["intent"]))
                   for k in ("train", "validation", "test")}

def load_via_github():
    import requests
    url = ("https://raw.githubusercontent.com/clinc/oos-eval/master/"
           "data/data_oos_plus.json")
    raw = requests.get(url, timeout=300).json()
    labels = sorted({lab for k, rows in raw.items() if rows for _, lab in rows})
    names = labels if "oos" in labels else labels + ["oos"]
    idx = {lab: i for i, lab in enumerate(names)}
    splits = {}
    for split, keys in (("validation", ("val", "oos_val")),
                        ("test", ("test", "oos_test")),
                        ("train", ("train", "oos_train"))):
        rows = [r for k in keys for r in raw.get(k, [])]
        splits[split] = ([t for t, _ in rows],
                         np.array([idx[l] for _, l in rows]))
    return names, splits

try:
    names, splits = load_via_datasets(); src = "hf datasets"
except Exception as e:
    print("datasets path failed:", repr(e)[:160])
    names, splits = load_via_github(); src = "github json"

OOS = names.index("oos")
X_tr, y_tr = splits["train"]
X_val, y_val = splits["validation"]
X_te, y_te = splits["test"]
tr_in, val_in, te_in = y_tr != OOS, y_val != OOS, y_te != OOS
X_tr_i = [t for t, in_ in zip(X_tr, tr_in) if in_]
y_tr_i = y_tr[tr_in]
print(f"source: {src} | train(in-scope) {len(X_tr_i)} | val {len(X_val)} | test {len(X_te)}")
assert len(X_te) == 5500 and (y_te == OOS).sum() == 1000
write_status("data")
'''),

    code('''
INTENT_TEXTS = [n.replace("_", " ") for n in names if n != "oos"]
INTENT_IDS = [i for i, n in enumerate(names) if n != "oos"]
col_of = {int(v): c for c, v in enumerate(INTENT_IDS)}
def to_cols(y):
    return np.array([col_of.get(int(v), -1) for v in y])

texts = X_tr_i + X_val + X_te
S = to_numpy(encoder.encode(texts, prompt_name="SearchQuery", batch_size=64))
LV = to_numpy(encoder.encode(INTENT_TEXTS, prompt_name="Document", batch_size=64))
torch.cuda.empty_cache()

n_tr, n_val = len(X_tr_i), len(X_val)
S_tr, S_val, S_te = S[:n_tr], S[n_tr:n_tr+n_val], S[n_tr+n_val:]
SC_val, SC_te = S_val @ LV.T, S_te @ LV.T
print("peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))

np.savez_compressed(WORK / "policy_v2_scores.npz",
                    emb_train=S_tr, emb_val=S_val, scores_val=SC_val, scores_test=SC_te,
                    y_train=y_tr_i, y_val=y_val, y_test=y_te)
write_status("encoded")
'''),

    md('''
## 2. Three-way splits — the Stage-3 protocol in code

`val` splits into two disjoint halves: **val_A** fits temperature + all thresholds
(τ, τ_answer), **val_B** calibrates the conformal q̂. Test is touched once. Sharing
calibration data with fit data is the optimistic-coverage bug the protocol forbids.
'''),
    code('''
rng = np.random.default_rng(13)
perm = rng.permutation(len(X_val))
A_idx, B_idx = perm[:len(perm)//2], perm[len(perm)//2:]

def softmax(sc, T):
    z = sc / T; z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def mass_needed(P, y_c):
    srt = -np.sort(-P, axis=1); cum = np.cumsum(srt, axis=1)
    i = np.arange(len(P)); rank = (srt > P[i, y_c][:, None]).sum(1)
    return cum[i, np.minimum(rank, srt.shape[1] - 1)]

def aps_members(P, q):
    srt = np.argsort(-P, axis=1)
    cum = np.cumsum(-np.sort(-P, axis=1), axis=1)
    sizes = (cum < q).sum(1) + 1
    return [srt[i, :sizes[i]] for i in range(len(P))]

y_val_c, y_te_c = to_cols(y_val), to_cols(y_te)
A_in, B_in = val_in[A_idx], val_in[B_idx]

def calibrate_scorer(SC_val, tag):
    """Returns the scorer's calibration bundle: T_prob (floored), tau_answer, qhat."""
    # T_prob on val_A in-scope, floored at 0.25 — no degenerate sharpening
    best_T, best_ll = 1.0, np.inf
    for T in np.linspace(0.25, 4.0, 32):
        P = softmax(SC_val[A_idx][A_in], T)
        ll = -np.log(np.clip(P[np.arange(len(P)), y_val_c[A_idx][A_in]], 1e-9, 1)).mean()
        if ll < best_ll:
            best_ll, best_T = ll, T
    P_A = softmax(SC_val[A_idx], best_T)
    # τ_answer: smallest top-prob reaching ≥0.90 precision among answered in-scope items
    tp = P_A.max(1)
    correct = SC_val[A_idx].argmax(1) == y_val_c[A_idx]
    cand = np.quantile(tp[A_in], np.linspace(0, 0.95, 40))
    tau_answer, cov = None, 0.0
    for t in sorted(cand):
        m = (tp >= t) & A_in
        if m.sum() < 30:
            continue
        prec = correct[m].mean()
        if prec >= 0.90:
            tau_answer, cov = float(t), float(m.mean())
            break
    if tau_answer is None:
        tau_answer, cov = float(np.quantile(tp[A_in], 0.9)), 0.1   # fail-closed: answer least
    # q̂ on val_B in-scope at T_set=1.0 — disjoint calibration data
    T_SET = 1.0
    m = mass_needed(softmax(SC_val[B_idx][B_in], T_SET), y_val_c[B_idx][B_in])
    qhat = float(np.quantile(m, min(1.0, np.ceil((len(m)+1)*0.9)/len(m))))
    print(f"{tag}: T_prob={best_T:.2f} τ_answer={tau_answer:.4f} (val_A answer-cov {cov:.2f}) "
          f"q̂={qhat:.4f}")
    return {"t_prob": best_T, "tau_answer": tau_answer, "qhat": qhat,
            "t_set": T_SET, "val_a_answer_coverage": cov}

COS = calibrate_scorer(SC_val, "cosine")
RESULTS = {"alpha": 0.10, "calibration": {"cosine": COS}}
write_status("calibrated")
'''),

    code('''
from sklearn.metrics import roc_curve

ms_val, ms_te = SC_val.max(1), SC_te.max(1)
fpr, tpr, ths = roc_curve(val_in[A_idx], ms_val[A_idx])
tau = float(ths[np.argmax(tpr - fpr)])
in_schema_te = ms_te >= tau
RESULTS["tau_in_schema"] = tau
RESULTS["in_schema"] = {"oos_caught": float((ms_te[~te_in] < tau).mean()),
                       "in_scope_false_reject": float((ms_te[te_in] < tau).mean())}
print(f"τ={tau:.4f} | OOS caught {RESULTS['in_schema']['oos_caught']:.3f} "
      f"| in-scope false-reject {RESULTS['in_schema']['in_scope_false_reject']:.3f}")
'''),

    md('''
## 3. Policies — v2 trigger geometry

| action | trigger |
|---|---|
| escalate | `max_sim < τ` (out-of-schema) |
| answer | `top_prob ≥ τ_answer` (calibrated, floored T_prob) |
| clarify | below τ_answer, APS set ≤ 3 |
| escalate | APS set > 3 (diffuse) |
'''),
    code('''
def evaluate_policy_v2(SC_test, cal, in_schema_mask):
    P = softmax(SC_test, cal["t_prob"])
    tp = P.max(1)
    P_set = softmax(SC_test, cal["t_set"])
    sets = aps_members(P_set, cal["qhat"])
    sizes = np.array([len(s) for s in sets])
    argmax = SC_test.argmax(1)
    out = {"actions": {}, "resolved": 0.0, "encodes": 0, "coverage_hits": 0, "n_in": 0}
    for i in range(len(y_te)):
        correct = te_in[i] and argmax[i] == y_te_c[i]
        if not in_schema_mask[i]:
            action, ok, cost = "escalate", not te_in[i], 1
        elif tp[i] >= cal["tau_answer"]:
            action, ok, cost = "answer", correct, 1
        elif sizes[i] <= 3:
            action, ok, cost = "clarify", te_in[i] and y_te_c[i] in sets[i], 2
        else:
            action, ok, cost = "escalate", not te_in[i], 1
        out["actions"][action] = out["actions"].get(action, 0) + 1
        out["resolved"] += float(bool(ok)); out["encodes"] += cost
        if te_in[i]:
            out["n_in"] += 1
            out["coverage_hits"] += int(y_te_c[i] in sets[i])
    out["resolved"] /= len(y_te)
    out["encodes_per_item"] = out["encodes"] / len(y_te)
    out["set_coverage_in_scope"] = out["coverage_hits"] / max(out["n_in"], 1)
    return out

RESULTS["policies"] = {}

# baselines
argmax_te = SC_te.argmax(1)
base = {"actions": {"answer": len(y_te)},
        "resolved": float((te_in & (argmax_te == y_te_c)).mean()),
        "encodes_per_item": 1.0}
RESULTS["policies"]["always_answer"] = base
thr = evaluate_policy_v2(SC_te, {**COS, "tau_answer": -np.inf, "t_set": 1.0, "qhat": 0.0},
                         in_schema_te)
thr["actions"] = {"abstain/escalate": int((~in_schema_te).sum()),
                  "answer": int(in_schema_te.sum())}
RESULTS["policies"]["threshold"] = thr

# v2 corrected
v2 = evaluate_policy_v2(SC_te, COS, in_schema_te)
RESULTS["policies"]["ambiguity_v2"] = v2
for name, r in RESULTS["policies"].items():
    print(f"{name:<16} resolved {r['resolved']:.3f} | enc/item {r['encodes_per_item']:.2f} "
          f"| {r['actions']}")
print(f"v2 set coverage in-scope: {v2['set_coverage_in_scope']:.3f} (target ~0.90)")
'''),

    md('''
## 4. TaskHead leg — fitted scorer, *per-scorer* calibration

Same policy, scorer replaced by a linear head on 768-dim states. It gets its own
calibrate_scorer call — v1 showed a cross-entropy head breaks set construction without
one (q̂≈1.0).
'''),
    code('''
def train_linear_head(S_tr, y_tr_i, seed=0, epochs=40, lr=1e-3, bs=256):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    L_head = len(INTENT_IDS)
    y_cols = np.array([col_of[int(v)] for v in y_tr_i])
    mu, sd = S_tr.mean(0, keepdims=True), S_tr.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((S_tr - mu) / sd, dtype=torch.float32, device=dev)
    Yt = torch.tensor(y_cols, dtype=torch.long, device=dev)
    head = nn.Linear(S_tr.shape[1], L_head).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            nn.functional.cross_entropy(head(Xt[idx]), Yt[idx]).backward()
            opt.step()
    def predict_probs(S):
        with torch.no_grad():
            Xs = torch.tensor((S - mu) / sd, dtype=torch.float32, device=dev)
            return torch.softmax(head(Xs), dim=1).cpu().numpy()
    return predict_probs, mu, sd

head_predict, h_mu, h_sd = train_linear_head(S_tr, y_tr_i)
PH_val, PH_te = head_predict(S_val), head_predict(S_te)
head_acc = float((PH_te[te_in].argmax(1) == y_te_c[te_in]).mean())
print(f"TaskHead in-scope accuracy: {head_acc:.3f}")

# head scores for the in-schema check use head max-prob (its own OOS signal, AUROC 0.968)
fpr_h, tpr_h, ths_h = roc_curve(val_in[A_idx], PH_val[A_idx].max(1))
tau_h = float(ths_h[np.argmax(tpr_h - fpr_h)])
in_schema_h = PH_te.max(1) >= tau_h

# per-scorer calibration: pass log-probs so softmax(x/T) is a proper temperature
LOGP_val = np.log(np.clip(PH_val, 1e-9, 1))
HCAL = calibrate_scorer(LOGP_val, "taskhead")
RESULTS["calibration"]["taskhead"] = HCAL

def evaluate_policy_head(PH_test, cal, in_schema_mask):
    P_cal = softmax(np.log(np.clip(PH_test, 1e-9, 1)), cal["t_prob"])
    tp = P_cal.max(1)
    # head sets: APS on probabilities at t_set (1.0 = the head's raw probs)
    P_set = PH_test if cal["t_set"] == 1.0 else softmax(
        np.log(np.clip(PH_test, 1e-9, 1)), cal["t_set"])
    sets = aps_members(P_set, cal["qhat"])
    sizes = np.array([len(s) for s in sets])
    argmax = PH_test.argmax(1)
    out = {"actions": {}, "resolved": 0.0, "encodes": 0, "coverage_hits": 0, "n_in": 0}
    for i in range(len(y_te)):
        correct = te_in[i] and argmax[i] == y_te_c[i]
        if not in_schema_mask[i]:
            action, ok, cost = "escalate", not te_in[i], 1
        elif tp[i] >= cal["tau_answer"]:
            action, ok, cost = "answer", correct, 1
        elif sizes[i] <= 3:
            action, ok, cost = "clarify", te_in[i] and y_te_c[i] in sets[i], 2
        else:
            action, ok, cost = "escalate", not te_in[i], 1
        out["actions"][action] = out["actions"].get(action, 0) + 1
        out["resolved"] += float(bool(ok)); out["encodes"] += cost
        if te_in[i]:
            out["n_in"] += 1
            out["coverage_hits"] += int(y_te_c[i] in sets[i])
    out["resolved"] /= len(y_te)
    out["encodes_per_item"] = out["encodes"] / len(y_te)
    out["set_coverage_in_scope"] = out["coverage_hits"] / max(out["n_in"], 1)
    return out

RESULTS["taskhead"] = {"in_scope_accuracy": head_acc, "tau": tau_h,
    "in_schema": {"oos_caught": float((PH_te.max(1)[~te_in] < tau_h).mean()),
                  "in_scope_false_reject": float((PH_te.max(1)[te_in] < tau_h).mean())}}
r = evaluate_policy_head(PH_te, HCAL, in_schema_h)
RESULTS["policies"]["ambiguity_v2_head"] = r
print(f"ambiguity_v2_head resolved {r['resolved']:.3f} | enc/item {r['encodes_per_item']:.2f} "
      f"| {r['actions']} | set-cov {r['set_coverage_in_scope']:.3f}")
'''),

    md('## 5. Verdict'),
    code('''
pol = RESULTS["policies"]
RESULTS["verdict"] = {
    "v2_beats_threshold": pol["ambiguity_v2"]["resolved"] > pol["threshold"]["resolved"],
    "v2_beats_base": pol["ambiguity_v2"]["resolved"] > pol["always_answer"]["resolved"],
    "coverage_holds": abs(pol["ambiguity_v2"]["set_coverage_in_scope"] - 0.90) < 0.05,
    "deltas": {k: round(pol[k]["resolved"] - pol["threshold"]["resolved"], 4)
               for k in pol if k != "threshold"},
}
print(json.dumps(RESULTS["verdict"], indent=2))

(WORK / "REPORT.md").write_text("# Ambiguity-aware policy v2 — corrected triggers\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json", "policy_v2_scores.npz"]:
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
