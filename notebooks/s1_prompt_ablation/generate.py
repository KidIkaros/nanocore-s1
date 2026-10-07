#!/usr/bin/env python3
"""Generate notebooks/s1_prompt_ablation/s1_prompt_ablation.ipynb.

Open question 1 from docs/ARCHITECTURE-DECISION-MODEL.md section 8.

The incumbent pipeline encodes states with prompt_name="Classification" and options
with prompt_name="Document". EmbeddingGemma 2 is trained with task instruction
prefixes that steer the representation, so this asymmetry places the two sides of a
dot product in different prompt-conditioned subspaces. Nothing ever verified they are
comparable -- and that dot product is the scoring function of both the head and the
cosine baseline.

This kernel re-encodes Banking77 from the encoder in-kernel (no pre-baked artifact)
under four prompt configurations and measures each. It answers, cheaply, whether the
unablated assumption was suppressing head quality -- which would change ADR-0001's
risk assessment before we invest in the composer.

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
# Prompt ablation — was the state/option prompt asymmetry hurting the head?

States were encoded with `prompt_name="Classification"`, options with
`prompt_name="Document"`. EG2's prefixes steer representations, so a dot product
between the two sides compares vectors from different prompt-conditioned subspaces.
That assumption was never tested, and it is the scoring function of *both* the head
and the cosine baseline.

Four configurations, each re-encoded **from the encoder in this kernel** (no cached
artifact):

| # | state prompt | option prompt | rationale |
|---|---|---|---|
| 1 | `Classification` | `Document` | the incumbent, unprincipled asymmetry |
| 2 | `SearchQuery` | `Document` | the canonical query→document retrieval pairing |
| 3 | `Classification` | `Classification` | the canonical symmetric pairing |
| 4 | *(none)* | *(none)* | no task prefix at all |

Per configuration: zero-shot cosine, kNN-5 with a fitted temperature, and a
fingerprint head over 3 seeds, plus abstention curves.

Every configuration writes its results to disk as it finishes, so a late failure
still yields everything completed — the lesson from two runs that produced nothing.
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
print("work dir:", WORK)
'''),

    code('''
# Self-contained source bundle — no mounted worktree dataset required.
import base64, io, sys, zipfile

BUNDLE = "__DECISION_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)
print("source extracted to", REPO)
'''),

    code('''
# Dependency floors. No -U and no torchvision: force-upgrading already-satisfied
# deps on this image breaks its custom torch build.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers>=6.1.0", "transformers>=5.19.0", "datasets"],
               check=True)
import transformers, sentence_transformers
print("transformers", transformers.__version__,
      "| sentence-transformers", sentence_transformers.__version__)
'''),

    code('''
# Contract tests on this image before spending the run.
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_schema.py",
                    "tests/test_decision_head.py", "tests/test_decision_model.py", "-x", "-q"],
                   capture_output=True, text=True, cwd=REPO)
print(r.stdout[-1500:])
assert r.returncode == 0, "contract tests failed on T4"
write_status("contract-tests")
'''),

    md('''
## 1. Encoder

`text+vision` (439M) rather than text-only, so the absolute numbers stay comparable
to the v13 baseline and the calibration run, which both used this configuration.
bf16 — never fp16.
'''),
    code('''
import numpy as np
from src.decision.encoder import StateEncoder, to_numpy

encoder = StateEncoder(modalities=("text", "vision"), device="cuda")
print("params:", round(sum(p.numel() for p in encoder.model.parameters()) / 1e6), "M",
      "| dtype:", encoder.dtype, "| dim:", encoder.embedding_dim)
write_status("encoder-loaded")
'''),

    md('''
## 2. Data

Banking77 upstream is script-only and `datasets` 4.x refuses loading scripts, so the
canonical CSVs are read directly and the label order is taken from the upstream
script's `names=[...]` list.
'''),
    code('''
import ast, requests, pandas as pd

RAW = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
script = requests.get(
    "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.py").text
i, j = script.index("names=["), script.index("]", script.index("names=["))
label_names = ast.literal_eval(script[i + 6:j + 1])
train_df = pd.read_csv(RAW + "train.csv")
test_df = pd.read_csv(RAW + "test.csv")

option_texts = [n.replace("_", " ") for n in label_names]
K = len(option_texts)
text_col, label_col = train_df.columns[0], train_df.columns[1]
to_idx = lambda v: label_names.index(v) if isinstance(v, str) else int(v)

train_texts = train_df[text_col].tolist()
train_y = np.array([to_idx(v) for v in train_df[label_col]])
test_texts = test_df[text_col].tolist()
test_y = np.array([to_idx(v) for v in test_df[label_col]])
print(f"train {len(train_texts)} | test {len(test_texts)} | classes {K}")

g = np.random.default_rng(0)
perm = g.permutation(len(train_texts))
n_val = int(0.1 * len(perm))
val_idx, fit_idx = perm[:n_val], perm[n_val:]

def onehot(y):
    m = np.zeros((len(y), K))
    m[np.arange(len(y)), y] = 1.0
    return m

Y_TEST = onehot(test_y)
write_status("data")
'''),

    md('## 3. Helpers'),
    code('''
from src.decision import metrics
from src.decision.head import DecisionHead
from src.decision.schema import DecisionExample

RESULTS = {"configs": {}, "protocol": {
    "encoder": "google/embeddinggemma-2 text+vision, bf16",
    "head": "fingerprint, 120 epochs, seeds 0/1/2, hidden unused",
    "split": "90% fit / 10% temperature-fit, test untouched",
}}

def save_results(stage):
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    write_status(stage)

def softmax(s):
    s = np.asarray(s, dtype=np.float64)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def fit_tau(val_scores, val_y, lo=-0.3, hi=1.5, n=37):
    grid = [round(10 ** e, 4) for e in np.linspace(lo, hi, n)]
    best_t, best_v = 1.0, float("inf")
    for t in grid:
        v = metrics.log_score(softmax(val_scores / t), onehot(val_y))
        if v < best_v:
            best_t, best_v = float(t), v
    return best_t

def knn_class_scores(Xq, Xr, yr, k):
    sims = Xq @ Xr.T
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    out = np.zeros((len(Xq), K))
    for i in range(len(Xq)):
        np.add.at(out[i], yr[idx[i]], sims[i, idx[i]])
    return out

def selective_curve(probs, steps=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == test_y).astype(float)
    order = np.argsort(-conf)
    return [{"coverage": round(float(c), 2),
             "selective_accuracy": float(correct[order[:max(1, int(c * len(test_y)))]].mean())}
            for c in np.linspace(0.1, 1.0, steps)]

def coverage_at(probs, target):
    conf = probs.max(1)
    correct = (probs.argmax(1) == test_y).astype(float)
    order = np.argsort(-conf)
    best = 0.0
    for n in range(1, len(test_y) + 1):
        if correct[order[:n]].mean() >= target:
            best = n / len(test_y)
    return float(best)

def score_block(probs):
    return {
        "accuracy": float((probs.argmax(1) == test_y).mean()),
        "log_score": float(metrics.log_score(probs, Y_TEST)),
        "brier": float(metrics.brier_score(probs, Y_TEST)),
        "ece_report_only": float(metrics.expected_calibration_error(probs, Y_TEST)),
        "coverage_at_90pct": coverage_at(probs, 0.90),
        "coverage_at_95pct": coverage_at(probs, 0.95),
    }

print("helpers ready")
'''),

    md('''
## 4. The ablation

Each configuration encodes states and options under its own prompts, then runs the
same baselines and the same head protocol. Results are written to disk after every
configuration.
'''),
    code('''
CONFIGS = [
    ("Classification", "Document"),
    ("SearchQuery", "Document"),
    ("Classification", "Classification"),
    (None, None),
]

for state_prompt, option_prompt in CONFIGS:
    tag = f"state={state_prompt}|opt={option_prompt}"
    print(f"=== {tag} ===", flush=True)

    X_train = to_numpy(encoder.encode(train_texts, prompt_name=state_prompt, batch_size=64))
    X_test = to_numpy(encoder.encode(test_texts, prompt_name=state_prompt, batch_size=64))
    O = to_numpy(encoder.encode(option_texts, prompt_name=option_prompt, batch_size=64))

    X_fit, y_fit = X_train[fit_idx], train_y[fit_idx]
    X_val, y_val = X_train[val_idx], train_y[val_idx]

    entry = {"state_prompt": state_prompt, "option_prompt": option_prompt}

    # zero-shot cosine, no training at all
    entry["zero_shot_cosine"] = {
        "accuracy": float(((X_test @ O.T).argmax(1) == test_y).mean())
    }

    # kNN-5 with a fitted temperature
    tau_k = fit_tau(knn_class_scores(X_val, X_fit, y_fit, 5), y_val)
    probs_knn = softmax(knn_class_scores(X_test, X_fit, y_fit, 5) / tau_k)
    entry["knn5_tau"] = dict(score_block(probs_knn), temperature=tau_k)

    # fingerprint head, 3 seeds
    fit_ex = [DecisionExample(state_embedding=X_fit[i], qtype="choice", labels=option_texts,
                              target=[1.0 if j == y_fit[i] else 0.0 for j in range(K)],
                              option_embeddings=[O[j] for j in range(K)])
              for i in range(len(X_fit))]
    val_ex = [DecisionExample(state_embedding=X_val[i], qtype="choice", labels=option_texts,
                              target=[1.0 if j == y_val[i] else 0.0 for j in range(K)],
                              option_embeddings=[O[j] for j in range(K)])
              for i in range(len(X_val))]

    runs, seed0_probs = [], None
    for seed in (0, 1, 2):
        head = DecisionHead(dim=768, mode="fingerprint")
        head.fit(fit_ex, epochs=120, lr=0.05, batch_size=64, seed=seed)
        head.fit_temperature(val_ex)
        probs = np.zeros((len(X_test), K))
        for i in range(len(X_test)):
            p = head.predict(X_test[i], [O[j] for j in range(K)],
                             qtype="choice", labels=option_texts)
            probs[i] = [p.probabilities[l] for l in option_texts]
        runs.append(score_block(probs))
        if seed == 0:
            seed0_probs = probs
            head.save(WORK / f"head_{tag.replace('=', '').replace('|', '_')}_seed0.json")

    keys = ("accuracy", "log_score", "brier", "ece_report_only")
    entry["fingerprint"] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    entry["fingerprint_std"] = {k: float(np.std([r[k] for r in runs])) for k in keys}
    entry["fingerprint"]["coverage_at_90pct"] = runs[0]["coverage_at_90pct"]
    entry["fingerprint"]["coverage_at_95pct"] = runs[0]["coverage_at_95pct"]
    entry["abstention_curve_fingerprint"] = selective_curve(seed0_probs)

    RESULTS["configs"][tag] = entry
    save_results(f"config-{tag}")
    print(f"    zero-shot {entry['zero_shot_cosine']['accuracy']:.4f} | "
          f"knn5 {entry['knn5_tau']['accuracy']:.4f} | "
          f"head {entry['fingerprint']['accuracy']:.4f} "
          f"log {entry['fingerprint']['log_score']:.3f} "
          f"brier {entry['fingerprint']['brier']:.3f} "
          f"cov@90 {entry['fingerprint']['coverage_at_90pct']:.3f}", flush=True)
'''),

    md('## 5. Summary and verdict'),
    code('''
incumbent = "state=Classification|opt=Document"
rows = []
for tag, e in RESULTS["configs"].items():
    rows.append({
        "config": tag,
        "zero_shot": round(e["zero_shot_cosine"]["accuracy"], 4),
        "knn5": round(e["knn5_tau"]["accuracy"], 4),
        "head_acc": round(e["fingerprint"]["accuracy"], 4),
        "head_log": round(e["fingerprint"]["log_score"], 3),
        "head_brier": round(e["fingerprint"]["brier"], 3),
        "head_ece": round(e["fingerprint"]["ece_report_only"], 3),
        "cov@90": round(e["fingerprint"]["coverage_at_90pct"], 3),
    })
rows.sort(key=lambda r: -r["head_acc"])
print(f"{'config':<42}{'zero':>7}{'knn5':>7}{'head':>7}{'log':>8}{'brier':>7}{'ece':>7}{'cov90':>7}")
for r in rows:
    print(f"{r['config']:<42}{r['zero_shot']:>7.4f}{r['knn5']:>7.4f}{r['head_acc']:>7.4f}"
          f"{r['head_log']:>8.3f}{r['head_brier']:>7.3f}{r['head_ece']:>7.3f}{r['cov@90']:>7.3f}")

best = rows[0]["config"]
RESULTS["verdict"] = {
    "best_config_by_head_accuracy": best,
    "incumbent": incumbent,
    "incumbent_head_accuracy": RESULTS["configs"][incumbent]["fingerprint"]["accuracy"],
    "best_head_accuracy": rows[0]["head_acc"],
    "prompt_choice_matters": bool(abs(rows[0]["head_acc"]
                                      - RESULTS["configs"][incumbent]["fingerprint"]["accuracy"]) > 0.005),
    "head_beats_knn5_in_best_config": bool(rows[0]["head_acc"] > rows[0]["knn5"]),
    "note": "differences below ~0.005 are within seed noise (measured ~0.0002) plus split noise (unmeasured)",
}
for k, v in RESULTS["verdict"].items():
    print(f"{k}: {v}")

save_results("verdict")
(WORK / "REPORT.md").write_text(
    "# Prompt ablation\\n\\n" + json.dumps({"rows": rows, "verdict": RESULTS["verdict"]}, indent=2))
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json"]:
        p = WORK / f
        if p.exists():
            zf.write(p, f)
write_status("done", done=True)
'''),
]

# --- embed the source bundle so no dataset mount is required
ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = [
    ROOT / "src" / "__init__.py",
    *sorted((ROOT / "src" / "decision").glob("*.py")),
    ROOT / "tests" / "__init__.py",
    *(ROOT / "tests" / n for n in ("test_decision_schema.py", "test_decision_head.py",
                                   "test_decision_model.py")),
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
