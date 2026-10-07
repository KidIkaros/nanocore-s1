#!/usr/bin/env python3
"""Generate notebooks/s1_sharpening/s1_sharpening.ipynb — the cheap cluster.

Implements proposals S3, S5, S7 and S9 from docs/ARCHITECTURE-SHARPENING.md, all on
cached embeddings, so it runs on a CPU-only Kaggle kernel and spends no GPU quota.

  S3  initialize fingerprints from the option embeddings (ZS-LP) instead of random
  S5  normalize fingerprints (Zero-Bias Corollary 1, second remedy) and finally run
      the separability diagnostic that was built and never used
  S7  Matryoshka 256-d, to make the "lightweight" claim measurable
  S9  protocol fixes: wide temperature grid, verdicts against the BEST baseline, and
      abstention thresholds above the task's base error rate

Arms, all on one split with the same baselines:
  random-init fingerprint            (the incumbent, 93.11%)
  init-from-options + normalized, UNTRAINED   (must equal zero-shot cosine exactly)
  init-from-options + normalized, trained     (S3 + S5)
  init-from-options, NOT normalized, trained  (isolates S5)

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
# Sharpening cluster — S3, S5, S7, S9

The head does not beat kNN-5 on accuracy, Brier or ECE. The literature says the
problem is not head *capacity* but head *initialization*: zero-shot cosine already
scores 92.92%, and a randomly initialized head has to rediscover that from noise.

| Proposal | Change |
|---|---|
| **S3** | Initialize fingerprints from the option embeddings (ZS-LP), so the head is a learned *correction* to the zero-shot baseline |
| **S5** | Unit-normalize fingerprints (Zero-Bias Corollary 1's second remedy) and run the separability diagnostic that was built but never used |
| **S7** | Evaluate at Matryoshka 256-d to make the "lightweight" claim measurable |
| **S9** | Widen the temperature grid, compare verdicts against the **best** baseline, and set abstention thresholds above the base error rate |

An untrained initialized head is included as a correctness check: it must reproduce
the zero-shot cosine baseline exactly. If it does not, S3 is misimplemented.
'''),

    md('## 0. Environment'),
    code('''
import json, shutil, subprocess, time
from pathlib import Path

gpu = "none (expected: this analysis needs no accelerator)"
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name"],
                         capture_output=True, text=True).stdout.strip()
print("GPU:", gpu)

import numpy as np, torch
print("numpy", np.__version__, "| torch", torch.__version__)

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

    md('## 1. Source bundle and contract tests'),
    code('''
import base64, io, sys, zipfile

BUNDLE = "__DECISION_BUNDLE__"
REPO = str(WORK / "nanocore-s1")
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as z:
    z.extractall(REPO)
sys.path.insert(0, REPO)

r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_decision_schema.py",
                    "tests/test_decision_head.py", "tests/test_decision_composer.py",
                    "tests/test_decision_model.py", "-q"],
                   capture_output=True, text=True, cwd=REPO)
print(r.stdout[-2500:])
assert r.returncode == 0, "contract tests failed — the new head code is wrong"
write_status("contract-tests")
'''),

    md('## 2. Cached embeddings'),
    code('''
import glob

hits = glob.glob("/kaggle/input/**/banking77_embeddings.npz", recursive=True)
assert hits, "embeddings not found — attach mauricew/nanocore-s1-banking77-emb"
z = np.load(hits[0], allow_pickle=True)
X_train, y_train = z["X_train"], z["y_train"]
X_test, y_test = z["X_test"], z["y_test"]
O768 = z["O"]
option_texts = [str(s) for s in z["option_texts"]]
K = len(option_texts)

g = np.random.default_rng(0)
perm = g.permutation(len(X_train))
n_val = int(0.1 * len(perm))
val_idx, fit_idx = perm[:n_val], perm[n_val:]
print(f"train {X_train.shape} | test {X_test.shape} | options {O768.shape} | classes {K}")
write_status("embeddings")
'''),

    md('## 3. Helpers (S9: wide temperature grid)'),
    code('''
from src.decision import metrics
from src.decision.head import DecisionHead
from src.decision.schema import DecisionExample

RESULTS = {"baselines": {}, "heads": {}, "diagnostics": {}, "matryoshka": {}, "protocol": {}}

def save_results(stage):
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    write_status(stage)

def onehot(y, k):
    m = np.zeros((len(y), k))
    m[np.arange(len(y)), y] = 1.0
    return m

def softmax(s):
    s = np.asarray(s, dtype=np.float64)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

# S9: the previous grid bottomed out at 10**-0.3 and three baselines fitted that
# lower bound, so their log scores were not their best. Widen it.
TAU_GRID = [round(10 ** e, 4) for e in np.linspace(-3.0, 2.0, 51)]

def fit_tau(val_scores, val_y):
    best_t, best_v = 1.0, float("inf")
    for t in TAU_GRID:
        v = metrics.log_score(softmax(val_scores / t), onehot(val_y, K))
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

def selective_curve(probs, y, steps=10):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    order = np.argsort(-conf)
    return [{"coverage": round(float(c), 2),
             "selective_accuracy": float(correct[order[:max(1, int(c * len(y)))]].mean())}
            for c in np.linspace(0.1, 1.0, steps)]

def coverage_at(probs, y, target):
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    order = np.argsort(-conf)
    best = 0.0
    for n in range(1, len(y) + 1):
        if correct[order[:n]].mean() >= target:
            best = n / len(y)
    return float(best)

def score_block(probs, y, k):
    Y = onehot(y, k)
    return {
        "accuracy": float((probs.argmax(1) == y).mean()),
        "log_score": float(metrics.log_score(probs, Y)),
        "brier": float(metrics.brier_score(probs, Y)),
        "ece_report_only": float(metrics.expected_calibration_error(probs, Y)),
        # S9: thresholds must sit above the base error rate, so 90% is useless here.
        "coverage_at_95pct": coverage_at(probs, y, 0.95),
        "coverage_at_98pct": coverage_at(probs, y, 0.98),
    }

RESULTS["protocol"] = {"tau_grid": [TAU_GRID[0], TAU_GRID[-1], len(TAU_GRID)],
                       "abstention_thresholds": [0.95, 0.98]}
print("helpers ready | tau grid", TAU_GRID[0], "..", TAU_GRID[-1], f"({len(TAU_GRID)} values)")
'''),

    md('## 4. Baselines (S9: properly fitted temperatures)'),
    code('''
X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
X_val, y_val = X_train[val_idx], y_train[val_idx]
Y_TEST = onehot(y_test, K)

tau_cos = fit_tau(X_val @ O768.T, y_val)
probs_cos = softmax((X_test @ O768.T) / tau_cos)
RESULTS["baselines"]["cosine_tau"] = dict(score_block(probs_cos, y_test, K), temperature=tau_cos)

KNN_PROBS = {}
for k in (1, 5, 20):
    tau_k = fit_tau(knn_class_scores(X_val, X_fit, y_fit, k), y_val)
    p = softmax(knn_class_scores(X_test, X_fit, y_fit, k) / tau_k)
    KNN_PROBS[f"knn{k}_tau"] = p
    RESULTS["baselines"][f"knn{k}_tau"] = dict(score_block(p, y_test, K), temperature=tau_k, k=k)

for name, r in RESULTS["baselines"].items():
    print(f"{name:<14} acc={r['accuracy']:.4f} log={r['log_score']:.3f} brier={r['brier']:.3f} "
          f"ece={r['ece_report_only']:.3f} tau={r['temperature']}")
save_results("baselines")
'''),

    md('''
## 5. The head arms (S3 + S5)

`init+norm, untrained` is the correctness check: it must reproduce the zero-shot
cosine baseline exactly. `init, unnormalized` isolates S5 from S3.
'''),
    code('''
labels = list(option_texts)
fit_ex = [DecisionExample(state_embedding=X_fit[i], qtype="choice", labels=labels,
                          target=[1.0 if j == y_fit[i] else 0.0 for j in range(K)],
                          option_embeddings=[O768[j] for j in range(K)])
          for i in range(len(X_fit))]
val_ex = [DecisionExample(state_embedding=X_val[i], qtype="choice", labels=labels,
                          target=[1.0 if j == y_val[i] else 0.0 for j in range(K)],
                          option_embeddings=[O768[j] for j in range(K)])
          for i in range(len(X_val))]

def head_probs(head, X):
    out = np.zeros((len(X), K))
    for i in range(len(X)):
        p = head.predict(X[i], [O768[j] for j in range(K)], qtype="choice", labels=labels)
        out[i] = [p.probabilities[l] for l in labels]
    return out

def make_head(init_from_options, normalize, seed=0):
    head = DecisionHead(dim=768, mode="fingerprint", normalize_fingerprints=normalize)
    if init_from_options:
        head.initialize_fingerprints("choice", labels, [O768[j] for j in range(K)],
                                     normalize=True)
    return head

def run_arm(tag, init_from_options, normalize, epochs, seeds):
    runs, kept = [], None
    for seed in seeds:
        head = make_head(init_from_options, normalize, seed)
        if epochs:
            head.fit(fit_ex, epochs=epochs, lr=0.05, batch_size=64, seed=seed)
        head.fit_temperature(val_ex)
        probs = head_probs(head, X_test)
        runs.append(score_block(probs, y_test, K))
        if seed == seeds[0]:
            kept = (head, probs)
    keys = ("accuracy", "log_score", "brier", "ece_report_only")
    RESULTS["heads"][tag] = {
        **{k: float(np.mean([r[k] for r in runs])) for k in keys},
        **{f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys},
        "coverage_at_95pct": runs[0]["coverage_at_95pct"],
        "coverage_at_98pct": runs[0]["coverage_at_98pct"],
        "epochs": epochs, "seeds": list(seeds),
        "init_from_options": init_from_options, "normalize_fingerprints": normalize,
    }
    r = RESULTS["heads"][tag]
    print(f"{tag:<28} acc={r['accuracy']:.4f}+-{r['accuracy_std']:.4f} log={r['log_score']:.3f} "
          f"brier={r['brier']:.3f} ece={r['ece_report_only']:.3f} "
          f"cov@95={r['coverage_at_95pct']:.3f}", flush=True)
    save_results(f"head-{tag}")
    return kept

kept_random = run_arm("random_init", False, False, 120, (0, 1, 2))
kept_untrained = run_arm("init_norm_untrained", True, True, 0, (0,))
kept_init_norm = run_arm("init_norm_trained", True, True, 120, (0, 1, 2))
kept_init_raw = run_arm("init_unnormalized_trained", True, False, 120, (0, 1))

# S3 correctness check: untrained initialized head == zero-shot cosine baseline
zs_acc = RESULTS["baselines"]["cosine_tau"]["accuracy"]
got_acc = RESULTS["heads"]["init_norm_untrained"]["accuracy"]
RESULTS["checks"] = {
    "untrained_init_matches_zero_shot": bool(abs(got_acc - zs_acc) < 1e-9),
    "zero_shot_accuracy": zs_acc,
    "untrained_init_accuracy": got_acc,
}
print("S3 check — untrained initialized head equals zero-shot:", RESULTS["checks"])
'''),

    md('''
## 6. S5 — run the separability diagnostic we built and never used

Zero-Bias Corollary 2: mutually close fingerprints mean the head cannot separate
those options. This checks whether the diagnostic actually predicts errors.
'''),
    code('''
head_best, probs_best = kept_init_norm
fp_labels, matrix = head_best.fingerprint_distance_matrix("choice", labels)
norms = head_best.fingerprint_norms("choice")
off_diag = matrix[~np.eye(len(fp_labels), dtype=bool)]

pairs = []
for a in range(len(fp_labels)):
    for b in range(a + 1, len(fp_labels)):
        pairs.append((float(matrix[a][b]), fp_labels[a], fp_labels[b]))
pairs.sort(reverse=True)

# does fingerprint proximity predict errors?
pred = probs_best.argmax(1)
wrong = pred != y_test
idx_of = {l: i for i, l in enumerate(fp_labels)}
sim_wrong, sim_right = [], []
for i in range(len(y_test)):
    s = float(matrix[idx_of[labels[y_test[i]]]][idx_of[labels[pred[i]]]])
    (sim_wrong if wrong[i] else sim_right).append(s)

RESULTS["diagnostics"] = {
    "fingerprint_norm_min": float(min(norms.values())),
    "fingerprint_norm_max": float(max(norms.values())),
    "max_pairwise_similarity": float(off_diag.max()),
    "mean_pairwise_similarity": float(off_diag.mean()),
    "most_confusable_pairs": [{"similarity": round(s, 4), "a": a, "b": b} for s, a, b in pairs[:10]],
    "mean_similarity_true_vs_predicted_when_wrong": float(np.mean(sim_wrong)) if sim_wrong else None,
    "mean_similarity_true_vs_predicted_when_right": float(np.mean(sim_right)) if sim_right else None,
    "wrong_count": int(wrong.sum()),
}
d = RESULTS["diagnostics"]
print(f"fingerprint norms: {d['fingerprint_norm_min']:.4f} .. {d['fingerprint_norm_max']:.4f}")
print(f"pairwise similarity: mean {d['mean_pairwise_similarity']:.4f} max {d['max_pairwise_similarity']:.4f}")
print("most confusable pairs:")
for p in d["most_confusable_pairs"][:5]:
    print(f"   {p['similarity']:.4f}  {p['a']} / {p['b']}")
print(f"true-vs-predicted similarity | wrong {d['mean_similarity_true_vs_predicted_when_wrong']} "
      f"| right {d['mean_similarity_true_vs_predicted_when_right']}")
save_results("diagnostics")
'''),

    md('## 7. S7 — Matryoshka 256-d'),
    code('''
def truncate(X, dim):
    Xt = X[:, :dim].astype(np.float32)
    return Xt / np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)

D = 256
Xt_train, Xt_test, Ot = truncate(X_train, D), truncate(X_test, D), truncate(O768, D)
Xt_fit, Xt_val = Xt_train[fit_idx], Xt_train[val_idx]

tau_cos256 = fit_tau(Xt_val @ Ot.T, y_val)
p_cos256 = softmax((Xt_test @ Ot.T) / tau_cos256)
RESULTS["matryoshka"]["cosine_256d"] = dict(score_block(p_cos256, y_test, K), temperature=tau_cos256)

tau_k256 = fit_tau(knn_class_scores(Xt_val, Xt_fit, y_fit, 5), y_val)
p_knn256 = softmax(knn_class_scores(Xt_test, Xt_fit, y_fit, 5) / tau_k256)
RESULTS["matryoshka"]["knn5_256d"] = dict(score_block(p_knn256, y_test, K), temperature=tau_k256)

head256 = DecisionHead(dim=D, mode="fingerprint", normalize_fingerprints=True)
head256.initialize_fingerprints("choice", labels, [Ot[j] for j in range(K)], normalize=True)
fit_ex256 = [DecisionExample(state_embedding=Xt_fit[i], qtype="choice", labels=labels,
                             target=[1.0 if j == y_fit[i] else 0.0 for j in range(K)],
                             option_embeddings=[Ot[j] for j in range(K)])
             for i in range(len(Xt_fit))]
head256.fit(fit_ex256, epochs=120, lr=0.05, batch_size=64, seed=0)
p256 = np.zeros((len(Xt_test), K))
for i in range(len(Xt_test)):
    pr = head256.predict(Xt_test[i], [Ot[j] for j in range(K)], qtype="choice", labels=labels)
    p256[i] = [pr.probabilities[l] for l in labels]
RESULTS["matryoshka"]["init_norm_head_256d"] = score_block(p256, y_test, K)

for name, r in RESULTS["matryoshka"].items():
    print(f"{name:<26} acc={r['accuracy']:.4f} log={r['log_score']:.3f} brier={r['brier']:.3f}")
save_results("matryoshka")
'''),

    md('## 8. Verdict — against the BEST baseline (S9)'),
    code('''
best_base_name = max(RESULTS["baselines"], key=lambda n: RESULTS["baselines"][n]["accuracy"])
best_base = RESULTS["baselines"][best_base_name]
incumbent = RESULTS["heads"]["random_init"]
best_head_name = max(
    (n for n in RESULTS["heads"] if n != "init_norm_untrained"),
    key=lambda n: RESULTS["heads"][n]["accuracy"])
best_head = RESULTS["heads"][best_head_name]

rows = [{"arm": f"baseline:{n}", "acc": round(r["accuracy"], 4), "log": round(r["log_score"], 3),
         "brier": round(r["brier"], 3), "cov95": round(r["coverage_at_95pct"], 3)}
        for n, r in RESULTS["baselines"].items()]
rows += [{"arm": f"head:{n}", "acc": round(r["accuracy"], 4), "log": round(r["log_score"], 3),
          "brier": round(r["brier"], 3), "cov95": round(r["coverage_at_95pct"], 3)}
         for n, r in RESULTS["heads"].items()]
rows.sort(key=lambda r: -r["acc"])
print(f"{'arm':<34}{'acc':>8}{'log':>8}{'brier':>8}{'cov95':>8}")
for r in rows:
    print(f"{r['arm']:<34}{r['acc']:>8.4f}{r['log']:>8.3f}{r['brier']:>8.3f}{r['cov95']:>8.3f}")

RESULTS["verdict"] = {
    "best_baseline": best_base_name,
    "best_baseline_accuracy": best_base["accuracy"],
    "best_head": best_head_name,
    "best_head_accuracy": best_head["accuracy"],
    "incumbent_head_accuracy": incumbent["accuracy"],
    "s3_s5_improved_accuracy_over_incumbent": bool(best_head["accuracy"] > incumbent["accuracy"] + 0.002),
    "head_beats_best_baseline_on_accuracy": bool(best_head["accuracy"] > best_base["accuracy"]),
    "head_beats_best_baseline_on_log_score": bool(best_head["log_score"] < best_base["log_score"]),
    "head_beats_best_baseline_on_brier": bool(best_head["brier"] < best_base["brier"]),
    "head_beats_best_baseline_on_abstention": bool(best_head["coverage_at_95pct"] > best_base["coverage_at_95pct"]),
    "normalization_helped": bool(RESULTS["heads"]["init_norm_trained"]["accuracy"]
                                 > RESULTS["heads"]["init_unnormalized_trained"]["accuracy"]),
    "matryoshka_256d_accuracy": RESULTS["matryoshka"]["init_norm_head_256d"]["accuracy"],
    "matryoshka_retention_vs_768d": float(RESULTS["matryoshka"]["init_norm_head_256d"]["accuracy"]
                                          / max(best_head["accuracy"], 1e-9)),
}
print()
for k, v in RESULTS["verdict"].items():
    print(f"{k}: {v}")

(WORK / "REPORT.md").write_text(
    "# Sharpening cluster S3/S5/S7/S9\\n\\n" + json.dumps(RESULTS, indent=2, default=str))
save_results("done")
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json"]:
        p = WORK / f
        if p.exists():
            zf.write(p, f)
STATUS["status"] = "complete"
write_status("done", done=True)
'''),
]

# --- embed source + tests so no worktree dataset mount is needed
ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = [
    ROOT / "src" / "__init__.py",
    *sorted((ROOT / "src" / "decision").glob("*.py")),
    ROOT / "tests" / "__init__.py",
    *(ROOT / "tests" / n for n in ("test_decision_schema.py", "test_decision_head.py",
                                   "test_decision_composer.py", "test_decision_model.py")),
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
