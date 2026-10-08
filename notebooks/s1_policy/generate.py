#!/usr/bin/env python3
"""Generate notebooks/s1_policy/s1_policy.ipynb — the amended design as a policy.

ADR-0013 added `clarify` and meta-routing; s1_clinc150_oos measured the pieces
(OOS AUROC 0.934, coverage at T=1.0, clarify-resolves 97.5%). What hasn't been measured
is the *policy* — the composed pipeline scored as a system:

    input → in-schema check → scorer → APS set → action ∈ {answer, clarify, escalate}

Three policies on the same cached scores, plus a fitted-TaskHead leg to test whether a
trained head changes the ambiguity structure:

  always_answer   — argmax on everything (baseline: OOS inputs all resolved WRONG)
  threshold       — max_sim < τ → abstain; else answer (the ADR-0005 incumbent)
  ambiguity_aware — OOS→escalate; |set|=1→answer; |set|≤3→clarify; else→escalate

Composite metric: resolved_fraction = P(in-scope correct answer) + P(in-scope clarify
that would resolve) + P(OOS correctly routed out). Also encode-cost per resolved item.

Everything runs on cached scores; the encoder runs once for all of train+val+test.

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
# The ambiguity-aware policy, end-to-end

ADR-0013's pieces measured separately: OOS detection (AUROC 0.934), conformal coverage
at T=1.0 (0.913), clarify resolution (97.5%). This run composes them into the actual
decision pipeline and scores it **as a system**:

```
input → in-schema (max_sim < τ → out)
      → scorer → APS prediction set @ T=1.0
      → |set| = 1 → answer | ≤ 3 → clarify | else → escalate
```

Three policies, one composite metric — **resolved fraction** = correct answers +
clarify-resolved + OOS correctly routed. Cost model: `answer` = 1 encode, `clarify` =
2 encodes (the follow-up + re-decision), `escalate`/`abstain` = 1 encode + an external
call we don't price.

A second leg re-runs the same policies with a **fitted TaskHead** (linear on embeddings,
CLINC train split) in place of cosine scoring — does a trained head shrink or just
reshuffle the ambiguity sets?
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

    md('## 1. Load CLINC150 (with the GitHub fallback the last run needed)'),
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
# keep the in-scope train split for the TaskHead leg
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

np.savez_compressed(WORK / "policy_scores.npz",
                    emb_train=S_tr, scores_val=SC_val, scores_test=SC_te,
                    y_train=y_tr_i, y_val=y_val, y_test=y_te)
write_status("encoded")
'''),

    md('''
## 2. Shared machinery — APS sets at T=1.0 (the dual-temperature fix)

`s1_clinc150_oos` showed log-loss fitting picks T=0.02 and destroys set structure.
Here: **T=1.0 raw softmax for sets** (coverage held: 0.913 @ α=0.10), and calibrated
probabilities reported separately. q̂ comes from val in-scope, α=0.10.
'''),
    code('''
def softmax(sc, T):
    z = sc / T; z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def aps_sets(P, q):
    cum = np.cumsum(-np.sort(-P, axis=1), axis=1)
    return (cum < q).sum(1) + 1

def aps_members(P, q):
    """Return list-of-arrays: option indices inside each row's APS set."""
    srt = np.argsort(-P, axis=1)
    cum = np.cumsum(-np.sort(-P, axis=1), axis=1)
    sizes = (cum < q).sum(1) + 1
    return [srt[i, :sizes[i]] for i in range(len(P))]

def mass_needed(P, y_c):
    srt = -np.sort(-P, axis=1); cum = np.cumsum(srt, axis=1)
    i = np.arange(len(P)); rank = (srt > P[i, y_c][:, None]).sum(1)
    return cum[i, np.minimum(rank, srt.shape[1] - 1)]

ALPHA, T_SET = 0.10, 1.0
y_val_c, y_te_c = to_cols(y_val), to_cols(y_te)
P_val, P_te = softmax(SC_val, T_SET), softmax(SC_te, T_SET)
m = mass_needed(P_val[val_in], y_val_c[val_in])
qhat = float(np.quantile(m, min(1.0, np.ceil((len(m) + 1) * (1 - ALPHA)) / len(m))))
print(f"q̂ = {qhat:.4f} (α={ALPHA})")

set_val, set_te = aps_members(P_val, qhat), aps_members(P_te, qhat)
sz_val = np.array([len(s) for s in set_val])
sz_te = np.array([len(s) for s in set_te])
hit_val = np.array([y_val_c[i] in set_val[i] for i in range(len(P_val))])
hit_te = np.array([y_te_c[i] in set_te[i] for i in range(len(P_te))])
print(f"coverage in-scope: val {hit_val[val_in].mean():.3f} | test {hit_te[te_in].mean():.3f}")
RESULTS = {"alpha": ALPHA, "t_set": T_SET, "qhat": qhat,
           "coverage_in_scope": {"val": float(hit_val[val_in].mean()),
                                  "test": float(hit_te[te_in].mean())}}
'''),

    md('''
## 3. In-schema threshold — tune τ on val so that OOS routing is calibrated there

τ is the val score where OOS-vs-in-scope separation is best (Youden J on the score
distribution). Simple, honest, matches deployment: the gate's first decision.
'''),
    code('''
from sklearn.metrics import roc_curve

ms_val, ms_te = SC_val.max(1), SC_te.max(1)
fpr, tpr, ths = roc_curve(val_in, ms_val)
tau = float(ths[np.argmax(tpr - fpr)])
print(f"τ (max_sim in-schema threshold) = {tau:.4f}")
in_schema_te = ms_te >= tau
print(f"test: routed-in {in_schema_te.mean():.3f} | OOS caught {(~in_schema_te & ~te_in).mean():.3f} "
      f"of OOS {(~te_in).mean():.3f} | in-scope wrongly rejected {(~in_schema_te & te_in).mean():.3f}")
RESULTS["tau"] = tau
RESULTS["in_schema"] = {"oos_caught_rate": float((ms_te[~te_in] < tau).mean()),
                       "in_scope_false_reject": float((ms_te[te_in] < tau).mean())}
'''),

    md('''
## 4. The three policies — scored as systems

`clarify` resolves iff the true intent is inside the set — an optimistic bound on the
follow-up question (the user might not pick right), but the honest measure of whether
the ambiguity set *contains* the answer, which is the property ADR-0013 needs.
'''),
    code('''
def evaluate_policy(name, SC, sets, sizes, in_schema):
    """Score a policy over the test split. resolved = correct final outcome."""
    argmax = SC.argmax(1)
    out = {"n": int(len(y_te)), "actions": {}, "resolved": 0.0, "encodes": 0}
    for i in range(len(y_te)):
        correct_argmax = te_in[i] and argmax[i] == y_te_c[i]
        if name == "always_answer":
            action, ok, cost = "answer", correct_argmax, 1
        elif name == "threshold":
            if not in_schema[i]:
                action, ok, cost = "abstain", not te_in[i], 1
            else:
                action, ok, cost = "answer", correct_argmax, 1
        else:  # ambiguity_aware
            if not in_schema[i]:
                action, ok, cost = "escalate", not te_in[i], 1
            elif sizes[i] == 1:
                action, ok, cost = "answer", correct_argmax, 1
            elif sizes[i] <= 3:
                action, ok, cost = "clarify", te_in[i] and y_te_c[i] in sets[i], 2
            else:
                action, ok, cost = "escalate", not te_in[i], 1
        out["actions"][action] = out["actions"].get(action, 0) + 1
        out["resolved"] += float(bool(ok))
        out["encodes"] += cost
    out["resolved"] /= len(y_te)
    out["encodes_per_item"] = out["encodes"] / len(y_te)
    return out

RESULTS["policies"] = {}
for name in ("always_answer", "threshold", "ambiguity_aware"):
    r = evaluate_policy(name, SC_te, set_te, sz_te, in_schema_te)
    RESULTS["policies"][name] = r
    print(f"{name:<16} resolved {r['resolved']:.3f} | encodes/item {r['encodes_per_item']:.2f} "
          f"| actions {r['actions']}")
'''),

    md('''
## 5. TaskHead leg — does a fitted head change the ambiguity structure?

Linear head on the 768-dim state (the scaling curve's sweet spot), trained on the 15k
in-scope train split. Same policy machinery, head probabilities in place of cosine.
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
    def predict(S):
        with torch.no_grad():
            Xs = torch.tensor((S - mu) / sd, dtype=torch.float32, device=dev)
            return torch.softmax(head(Xs), dim=1).cpu().numpy()
    return predict

head_predict = train_linear_head(S_tr, y_tr_i)
PH_val, PH_te = head_predict(S_val), head_predict(S_te)

# head in-schema signal: use head max prob threshold tuned on val the same way
fpr_h, tpr_h, ths_h = roc_curve(val_in, PH_val.max(1))
tau_h = float(ths_h[np.argmax(tpr_h - fpr_h)])
in_schema_h = PH_te.max(1) >= tau_h

# head APS sets: same procedure, its own qhat
m_h = mass_needed(PH_val[val_in], y_val_c[val_in])
qhat_h = float(np.quantile(m_h, min(1.0, np.ceil((len(m_h) + 1) * (1 - ALPHA)) / len(m_h))))
set_h = aps_members(PH_te, qhat_h)
sz_h = np.array([len(s) for s in set_h])
hit_h = np.array([y_te_c[i] in set_h[i] for i in range(len(PH_te))])

RESULTS["taskhead"] = {"qhat": qhat_h, "tau": tau_h,
    "in_scope_accuracy": float((PH_te[te_in].argmax(1) == y_te_c[te_in]).mean()),
    "coverage_in_scope": float(hit_h[te_in].mean()),
    "mean_set_in_scope": float(sz_h[te_in].mean()),
    "oos_auroc_maxprob": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"])
                               .roc_auc_score(te_in, PH_te.max(1)))}
print(json.dumps(RESULTS["taskhead"], indent=2))

r = evaluate_policy("ambiguity_aware", PH_te, set_h, sz_h, in_schema_h)
RESULTS["policies"]["ambiguity_aware_head"] = r
print(f"ambiguity_aware_head resolved {r['resolved']:.3f} | actions {r['actions']}")
'''),

    md('''
## 6. Verdict

The design claim under test: **the ambiguity-aware policy should beat always-answer and
threshold on resolved fraction**, because clarify converts the ambiguous-but-recoverable
slice into wins instead of losses. If `ambiguity_aware` ≤ `threshold`, clarify adds
machinery without payoff and ADR-0013 gets rescoped.
'''),
    code('''
pol = RESULTS["policies"]
base, thr, amb = pol["always_answer"], pol["threshold"], pol["ambiguity_aware"]
RESULTS["verdict"] = {
    "clarify_pays": amb["resolved"] > thr["resolved"],
    "gate_pays": thr["resolved"] > base["resolved"],
    "deltas": {"amb_vs_thr": amb["resolved"] - thr["resolved"],
               "amb_vs_base": amb["resolved"] - base["resolved"],
               "head_vs_cosine": pol["ambiguity_aware_head"]["resolved"] - amb["resolved"]},
}
print(json.dumps(RESULTS["verdict"], indent=2))

(WORK / "REPORT.md").write_text("# Ambiguity-aware policy, end-to-end\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json", "policy_scores.npz"]:
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
