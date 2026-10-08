#!/usr/bin/env python3
"""Generate notebooks/s1_goemotions_head/s1_goemotions_head.ipynb — the first fair fight.

s1_goemotions established that GoEmotions has real headroom (zero-shot macro-F1 0.287,
AUROC 0.824, gate usable under ADR-0011) and saved the score matrices. This run asks the
question the earlier run made meaningful: **can a learned component beat per-label Platt
thresholds — and if so, is it the scores' co-occurrence signal or the raw state that wins?**

The five-arm ladder isolates the answer:

| arm | input | functional class | what it tests |
|---|---|---|---|
| platt | 28-dim cosine scores | independent sigmoids (dev-fitted) | incumbent |
| linear_scores | 28-dim scores | joint linear map | shared fitting alone |
| mlp_scores | 28-dim scores | nonlinear + co-occurrence | label correlations, no state |
| linear_emb | 768-dim state | linear head | what the embedding adds |
| mlp_emb | 768-dim state | nonlinear head | the real candidate |

An `oracle_threshold` row (thresholds tuned on test) marks the ceiling of ANY thresholding
on the score features — a diagnostic, not a candidate.

Splits are the three-way rule in action: train (43k) fits heads, dev (5.4k) early-stops and
tunes thresholds, test (5.4k) is touched once.

Also saved this time: the embeddings themselves (train+dev+test), so no future head
experiment re-encodes — the cache lesson applied.

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
# GoEmotions head ladder — can a learned component win on a fair task?

Banking77 and BFCL retired the head on *saturated* benchmarks — null results were
uninformative there. GoEmotions has 0.713 headroom (`s1_goemotions`), so a null here
means something. Five arms, one question: **where, if anywhere, does learning beat
per-label Platt thresholds?**

| arm | input | what it isolates |
|---|---|---|
| `platt` | 28 scores | incumbent — per-label sigmoids fitted on dev |
| `linear_scores` | 28 scores | joint linear rescoring |
| `mlp_scores` | 28 scores | nonlinearity + label co-occurrence (state-blind) |
| `linear_emb` | 768-dim state | linear access to the full state |
| `mlp_emb` | 768-dim state | nonlinear head — the real candidate |

Plus `oracle_threshold` — thresholds tuned **on test**, the ceiling of any thresholding on
the score features. Marked as a bound, never a candidate.

train (43k) fits · dev (5.4k) early-stops + tunes thresholds · test (5.4k) evaluated once.
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

encoder = StateEncoder(modalities=("text",), device="cuda")
print("encoder params:", round(sum(p.numel() for p in encoder.model.parameters()) / 1e6), "M",
      "| dtype:", encoder.dtype)
write_status("encoder")
'''),

    md('## 1. Load GoEmotions (train + validation + test)'),
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
    return rows("train"), rows("validation"), rows("test")

def load_via_tsv():
    import requests
    base = ("https://raw.githubusercontent.com/google-research/google-research/"
            "master/goemotions/data/")
    def rows(name):
        out = []
        for line in requests.get(base + name, timeout=300).text.splitlines()[1:]:
            parts = line.split("\\t")
            if len(parts) < 2:
                continue
            ids = [int(x) for x in parts[1].split(",") if x.strip()]
            out.append((parts[0], sorted(set(ids))))
        return out
    return rows("train.tsv"), rows("dev.tsv"), rows("test.tsv")

try:
    train, dev, test = load_via_datasets()
    src = "hf datasets"
except Exception as e:
    print("datasets path failed:", e)
    train, dev, test = load_via_tsv()
    src = "github tsv"

def gold_matrix(rows):
    Y = np.zeros((len(rows), L), dtype=np.float32)
    for i, (_, labs) in enumerate(rows):
        for lab in labs:
            if lab < L:
                Y[i, lab] = 1.0
    return Y

X_tr_text  = [t for t, _ in train]
X_dev_text = [t for t, _ in dev]
X_te_text  = [t for t, _ in test]
Y_tr, Y_dev, Y_te = gold_matrix(train), gold_matrix(dev), gold_matrix(test)

print(f"source: {src} | train {len(train)} | dev {len(dev)} | test {len(test)}")
assert len(train) > 30000 and len(test) > 2000, "GoEmotions layout changed"
write_status("data")
'''),

    md('''
## 2. Encode all three splits — and save the embeddings this time

Bare emotion names won the template ablation (AUROC 0.824 vs 0.789), so labels encode as
names. The embedding matrices are persisted so no future head run re-encodes.
'''),
    code('''
STATE_PROMPT, OPTION_PROMPT = "SearchQuery", "Document"

texts = X_tr_text + X_dev_text + X_te_text
n_tr, n_dev = len(X_tr_text), len(X_dev_text)
print(f"texts to encode: {len(texts)}")

S = to_numpy(encoder.encode(texts, prompt_name=STATE_PROMPT, batch_size=64))
torch.cuda.empty_cache()
LV = to_numpy(encoder.encode(EMOTIONS, prompt_name=OPTION_PROMPT, batch_size=64))
torch.cuda.empty_cache()
print("peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2),
      "| S:", S.shape, "| LV:", LV.shape)

S_tr, S_dev, S_te = S[:n_tr], S[n_tr:n_tr+n_dev], S[n_tr+n_dev:]
SC_tr, SC_dev, SC_te = S_tr @ LV.T, S_dev @ LV.T, S_te @ LV.T   # 28-dim score features

np.savez_compressed(WORK / "goemotions_embeddings.npz",
                    emb_train=S_tr, emb_dev=S_dev, emb_test=S_te,
                    scores_train=SC_tr, scores_dev=SC_dev, scores_test=SC_te,
                    gold_train=Y_tr, gold_dev=Y_dev, gold_test=Y_te,
                    emotions=np.array(EMOTIONS))
print("saved goemotions_embeddings.npz — no future head run re-encodes")
write_status("encoded")
'''),

    md('''
## 3. The ladder

All torch heads share one trainer: standardised inputs, BCE with per-label `pos_weight`
(rare labels need it — grief has ~0.1% prevalence), early stop on dev log-loss, 3 seeds.
Thresholds are tuned on dev *after* fitting; test is touched once, at the end.
'''),
    code('''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, log_loss
import torch.nn as nn

def platt_fit(sc, y):
    models = []
    for l in range(L):
        x, yv = sc[:, l].reshape(-1, 1), y[:, l]
        if yv.min() == yv.max():
            models.append(("const", float(yv[0]))); continue
        models.append(("platt", LogisticRegression(C=10.0, max_iter=500).fit(x, yv)))
    return models

def platt_apply(models, sc):
    P = np.zeros_like(sc)
    for l, (kind, m) in enumerate(models):
        P[:, l] = m if kind == "const" else m.predict_proba(sc[:, l].reshape(-1, 1))[:, 1]
    return P

def make_head(kind, d_in, hidden=256):
    if kind == "linear":
        return nn.Linear(d_in, L)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Dropout(0.1), nn.Linear(hidden, L))

def train_head(X_tr, Y_tr, X_dev, Y_dev, kind, seed, epochs=60, lr=1e-3, bs=256):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=dev)
    Xv = torch.tensor((X_dev - mu) / sd, dtype=torch.float32, device=dev)
    Yv = torch.tensor(Y_dev, dtype=torch.float32, device=dev)
    pos = Y_tr.sum(0); neg = len(Y_tr) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 30.0),
                      dtype=torch.float32, device=dev)

    head = make_head(kind, X_tr.shape[1]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(head(Xt[idx]), Yt[idx])
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = float(nn.functional.binary_cross_entropy_with_logits(
                head(Xv), Yv, pos_weight=pw))
        if vl < best - 1e-4:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    head.load_state_dict(best_state)
    def predict_proba(X):
        Xs = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
        with torch.no_grad():
            return torch.sigmoid(head(Xs)).cpu().numpy()
    return predict_proba, best
'''),

    code('''
def best_f1_threshold(probs, y, grid=None):
    grid = grid if grid is not None else np.linspace(0.02, 0.98, 49)
    best_t, best_f = 0.5, -1.0
    for t in grid:
        f = f1_score(y, probs >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

def evaluate(P_dev, P_te, label=""):
    """Tune thresholds on dev, evaluate once on test. Returns the full row."""
    tuned = np.array([best_f1_threshold(P_dev[:, l], Y_dev[:, l]) for l in range(L)])
    out = {}
    for tag, P in (("dev", P_dev), ("test", P_te)):
        Y = Y_dev if tag == "dev" else Y_te
        for name, th in (("fixed_0.5", np.full(L, 0.5)), ("tuned", tuned)):
            pred = P >= th
            out[f"{tag}_{name}"] = {
                "macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
                "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0))}
        out[f"{tag}_log_loss"] = float(np.mean(
            [log_loss(Y[:, l], np.clip(P[:, l], 1e-7, 1 - 1e-7), labels=[0, 1])
             for l in range(L)]))
        auroc = [roc_auc_score(Y[:, l], P[:, l])
                 for l in range(L) if Y[:, l].min() < Y[:, l].max()]
        out[f"{tag}_auroc"] = float(np.mean(auroc))
    print(f"{label:<16} test macro-F1 {out['test_tuned']['macro_f1']:.3f} "
          f"| micro {out['test_tuned']['micro_f1']:.3f} | auroc {out['test_auroc']:.3f}")
    return out

RESULTS = {"arms": {}, "oracle": {}, "splits": {"train": int(n_tr), "dev": int(n_dev),
                                               "test": int(len(X_te_text))}}
def save(stage):
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    write_status(stage)

# --- arm 0: incumbent Platt (dev-fitted, reproduces s1_goemotions) ---
models = platt_fit(SC_dev, Y_dev)
RESULTS["arms"]["platt"] = evaluate(platt_apply(models, SC_dev),
                                  platt_apply(models, SC_te), "platt")
save("platt")
'''),

    code('''
# --- arms 1-4: learned heads, 3 seeds each ---
ARM_SPECS = [
    ("linear_scores", SC_tr, SC_dev, SC_te, "linear", 3e-3),
    ("mlp_scores",    SC_tr, SC_dev, SC_te, "mlp",    1e-3),
    ("linear_emb",    S_tr,  S_dev,  S_te,  "linear", 1e-3),
    ("mlp_emb",       S_tr,  S_dev,  S_te,  "mlp",    1e-3),
]
for name, Xa_tr, Xa_dev, Xa_te, kind, lr in ARM_SPECS:
    runs = []
    for seed in (0, 1, 2):
        predict, vl = train_head(Xa_tr, Y_tr, Xa_dev, Y_dev, kind, seed, lr=lr)
        row = evaluate(predict(Xa_dev), predict(Xa_te), f"{name}[s{seed}]")
        runs.append(row)
        save(f"{name}-seed{seed}")
    keys = [k for k in runs[0] if k.startswith("test_")]
    mean = {k: float(np.mean([r[k] for r in runs])) if isinstance(runs[0][k], float)
            else {m: float(np.mean([r[k][m] for r in runs])) for m in runs[0][k]}
            for k in keys}
    std_macro = float(np.std([r["test_tuned"]["macro_f1"] for r in runs]))
    std_micro = float(np.std([r["test_tuned"]["micro_f1"] for r in runs]))
    RESULTS["arms"][name] = {"mean": mean, "seeds": runs,
                             "macro_f1_std": std_macro, "micro_f1_std": std_micro}
    print(f"{name:<16} mean test macro {mean['test_tuned']['macro_f1']:.3f}±{std_macro:.3f} "
          f"| micro {mean['test_tuned']['micro_f1']:.3f}±{std_micro:.3f} "
          f"| auroc {mean['test_auroc']:.3f}")
save("heads")
'''),

    code('''
# --- diagnostic: oracle thresholds (tuned ON TEST — an upper bound, never a candidate) ---
P_platt_te = platt_apply(models, SC_te)
oracle = np.array([best_f1_threshold(P_platt_te[:, l], Y_te[:, l]) for l in range(L)])
pred = P_platt_te >= oracle
RESULTS["oracle"] = {"macro_f1": float(f1_score(Y_te, pred, average="macro",
                                               zero_division=0)),
                     "micro_f1": float(f1_score(Y_te, pred, average="micro",
                                               zero_division=0)),
                     "note": "thresholds tuned on test — the ceiling of ANY thresholding "
                             "on score features; diagnostic bound, not a candidate"}
print("oracle (ceiling of thresholding on scores):", RESULTS["oracle"]["macro_f1"],
      RESULTS["oracle"]["micro_f1"])
save("oracle")
'''),

    md('''
## 4. Verdict — where does learning pay?

Read the ladder, not just the winner:

- `linear_scores` ≈ `platt` → joint fitting alone adds nothing.
- `mlp_scores` > `linear_scores` → label co-occurrence carries signal (state-blind).
- `linear_emb` > `linear_scores` → the raw state carries signal the 28 scores discard.
- `mlp_emb` > `linear_emb` → nonlinearity on the state is worth parameters.

Whichever arm wins, the honest claim is the **smallest arm that beats Platt** — anything
bigger is showing off.
'''),
    code('''
inc = RESULTS["arms"]["platt"]["test_tuned"]
print(f"{'arm':<16} {'macro-F1':>9} {'micro-F1':>9} {'auroc':>7} {'vs platt':>9}")
print(f"{'platt':<16} {inc['macro_f1']:>9.3f} {inc['micro_f1']:>9.3f} "
      f"{RESULTS['arms']['platt']['test_auroc']:>7.3f} {'—':>9}")
table = {}
for name in ("linear_scores", "mlp_scores", "linear_emb", "mlp_emb"):
    m = RESULTS["arms"][name]["mean"]
    d = m["test_tuned"]["macro_f1"] - inc["macro_f1"]
    table[name] = {"macro": m["test_tuned"]["macro_f1"], "micro": m["test_tuned"]["micro_f1"],
                   "auroc": m["test_auroc"], "delta_vs_platt": d}
    print(f"{name:<16} {m['test_tuned']['macro_f1']:>9.3f} {m['test_tuned']['micro_f1']:>9.3f} "
          f"{m['test_auroc']:>7.3f} {d:>+9.3f}")
print(f"{'oracle(bound)':<16} {RESULTS['oracle']['macro_f1']:>9.3f} "
      f"{RESULTS['oracle']['micro_f1']:>9.3f}")

RESULTS["verdict"] = {
    "incumbent": inc, "table": table, "oracle": RESULTS["oracle"],
    "smallest_winner": min(
        (n for n, r in table.items() if r["delta_vs_platt"] > 0),
        key=lambda n: ("linear_scores", "mlp_scores", "linear_emb", "mlp_emb").index(n),
        default=None),
}
print("\\nverdict:", json.dumps(RESULTS["verdict"]["smallest_winner"]))

(WORK / "REPORT.md").write_text("# GoEmotions head ladder\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
save("verdict")
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json", "goemotions_embeddings.npz"]:
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
