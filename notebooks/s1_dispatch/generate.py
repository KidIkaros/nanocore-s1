#!/usr/bin/env python3
"""Generate notebooks/s1_dispatch/s1_dispatch.ipynb — the dispatch benchmark.

The target task (docs/RESEARCH-NOTES.md R9): a general assistant's small model should
decide **dispatch** — which tool handles a request, and whether any does. That is a
typed decision over a supplied option set, which is what this architecture is.

Data: BFCL v3 (Berkeley Function Calling Leaderboard), ungated.

  routing   (Choice)  BFCL_v3_multiple.json + BFCL_v3_live_multiple.json
                      1,253 queries, 2-37 candidate tools each. Label = the function
                      named in possible_answer/<file>.json.
  abstention (Noul)   BFCL_v3_irrelevance.json + BFCL_v3_live_irrelevance.json
                      882 queries where NO candidate tool is relevant.

Everything is measured with src/decision/protocol.py, including ragged_metric_block,
which exists because dispatch option sets vary from 2 to 37 wide.

Architectural note: the **interaction** head is used for training, not the fingerprint
head. The fingerprint head stores one learned vector per *label*, which is wrong here —
positional labels would collapse "option 0" across queries into a single vector and
teach it that position 0 is correct. The interaction head scores (state, option) pairs
from option *embeddings*, which is the design's answer to a variable option set
(ADR-0003: a new option needs no retraining).

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
# Dispatch benchmark — which tool, and is there one at all?

The model's job is **dispatch**, not content (R9). This measures it on real data:

| Task | Primitive | Data | Size |
|---|---|---|---|
| **Routing** — which tool handles this? | `Choice` | BFCL `multiple` + `live_multiple` | 1,253 queries, **2–37** candidate tools |
| **Abstention** — is any tool relevant? | `Noul` | BFCL `irrelevance` + `live_irrelevance` | 882 queries with no relevant tool |

Why not Banking77: it is saturated at 92.9% zero-shot, so no head can show a difference
there (R1). Tool routing has a **variable option set**, which is also the first real test
of order invariance.

All measurement goes through `src/decision/protocol.py` — the instrument this run also
exercises in anger.

**Abstention uses the raw score, not the softmax probability.** Most irrelevance entries
have a single candidate tool, where softmax is trivially 1.0. The open-set signal is the
pre-softmax match, as the zero-bias design intends.
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
                "sentence-transformers>=6.1.0", "transformers>=5.19.0"], check=True)

import numpy as np
from src.decision import protocol
from src.decision.encoder import StateEncoder, to_numpy
from src.decision.head import DecisionHead
from src.decision.schema import DecisionExample

# Text-only: routing states and tool descriptions are all text, so 270M suffices.
encoder = StateEncoder(modalities=("text",), device="cuda")
print("encoder params:", round(sum(p.numel() for p in encoder.model.parameters()) / 1e6), "M",
      "| dtype:", encoder.dtype)
write_status("encoder")
'''),

    md('''
## 1. Load BFCL

`question` is `[[{role, content}, ...]]`; the label is the single key of
`ground_truth[0]` in the matching `possible_answer` file.
'''),
    code('''
import collections, requests

BASE = ("https://huggingface.co/datasets/gorilla-llm/"
        "Berkeley-Function-Calling-Leaderboard/resolve/main/")

def jsonl(name):
    txt = requests.get(BASE + name, timeout=180).text
    return [json.loads(l) for l in txt.splitlines() if l.strip()]

def state_text(question):
    parts = []
    for turn in question:
        for msg in turn:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
    return " ".join(parts).strip()

def option_text(fn, limit=280):
    desc = str(fn.get("description", "")).replace("\\n", " ").strip()
    return (fn["name"] + ": " + desc)[:limit]

routing = []
for name in ["BFCL_v3_multiple.json", "BFCL_v3_live_multiple.json"]:
    qs = jsonl(name)
    ans = {r["id"]: r for r in jsonl("possible_answer/" + name)}
    kept = skipped = 0
    for q in qs:
        cands = q.get("function") or []
        gt = ans.get(q["id"], {}).get("ground_truth") or []
        if len(cands) < 2 or not gt or not isinstance(gt[0], dict):
            skipped += 1; continue
        target = next(iter(gt[0].keys()))
        names = [f["name"] for f in cands]
        if target not in names:
            skipped += 1; continue
        routing.append({"id": q["id"], "state": state_text(q["question"]),
                        "options": [option_text(f) for f in cands],
                        "label": names.index(target)})
        kept += 1
    print(f"{name}: kept {kept} | skipped {skipped}")

irrelevant = []
for name in ["BFCL_v3_irrelevance.json", "BFCL_v3_live_irrelevance.json"]:
    for q in jsonl(name):
        cands = q.get("function") or []
        if cands:
            irrelevant.append({"id": q["id"], "state": state_text(q["question"]),
                               "options": [option_text(f) for f in cands]})

dist = collections.Counter(len(r["options"]) for r in routing)
print(f"\\nrouting: {len(routing)} | candidates/query: {dict(sorted(dist.items()))}")
print(f"irrelevant: {len(irrelevant)}")
assert len(routing) > 500 and len(irrelevant) > 300, "BFCL layout changed — check the loader"
write_status("data")
'''),

    md('## 2. Encode'),
    code('''
STATE_PROMPT, OPTION_PROMPT = "SearchQuery", "Document"   # pairing adopted after the prompt ablation

uniq = {}
for r in routing + irrelevant:
    for o in r["options"]:
        uniq.setdefault(o, None)
uniq_options = list(uniq)
oidx = {o: i for i, o in enumerate(uniq_options)}

# BFCL conversations are long, and SentenceTransformer pads every batch to its longest
# member, so a large batch over long texts blows up attention memory. Clip and use small
# batches; the first attempt OOM'd at 6.96 GiB with batch_size=64.
def clip(s, n=400):
    s = str(s)
    return s if len(s) <= n else s[:n]

states = [clip(r["state"]) for r in routing]
states_irr = [clip(r["state"]) for r in irrelevant]
print("max state chars:", max(len(s) for s in states + states_irr),
      "| max option chars:", max(len(o) for o in uniq_options),
      "| texts to encode:", len(states) + len(states_irr) + len(uniq_options))

import torch
S = to_numpy(encoder.encode(states, prompt_name=STATE_PROMPT, batch_size=16))
torch.cuda.empty_cache()
S_irr = to_numpy(encoder.encode(states_irr, prompt_name=STATE_PROMPT, batch_size=16))
torch.cuda.empty_cache()
O = to_numpy(encoder.encode(uniq_options, prompt_name=OPTION_PROMPT, batch_size=16))
torch.cuda.empty_cache()

print("unique tool descriptions:", len(uniq_options), "| shapes:", S.shape, S_irr.shape, O.shape)
print("peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
np.savez_compressed(WORK / "dispatch_embeddings.npz", S=S, S_irr=S_irr, O=O)
write_status("encoded")
'''),

    md('''
## 3. Split, headroom, baselines

Headroom first (R1): if zero-shot is near ceiling, this benchmark cannot discriminate
either and we should say so before drawing conclusions. Every candidate is scored with
`ragged_metric_block`, which fits a temperature per option-count group.
'''),
    code('''
g = np.random.default_rng(0)
perm = g.permutation(len(routing))
n_test = int(0.3 * len(perm))
test_idx, train_idx = np.sort(perm[:n_test]), np.sort(perm[n_test:])
test = [routing[i] for i in test_idx]
train = [routing[i] for i in train_idx]
X_te, X_tr = S[test_idx], S[train_idx]
print(f"train {len(train)} | test {len(test)}")

def targets_for(items):
    out = []
    for r in items:
        y = np.zeros(len(r["options"]))
        y[r["label"]] = 1.0
        out.append(y)
    return out

T_te, T_tr = targets_for(test), targets_for(train)

RESULTS = {"routing": {}, "abstention": {}, "order_invariance": {}, "protocol": {}}
def save(stage):
    (WORK / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str))
    write_status(stage)

def report(name, scores_list, items, targets):
    block = protocol.ragged_metric_block(scores_list, targets)
    RESULTS["routing"][name] = block
    print(f"{name:<22} acc={block['accuracy']:.4f} log={block['log_score']:.3f} "
          f"brier={block['brier']:.3f} ece={block['ece_report_only']:.3f}", flush=True)
    return block

# --- floor
rng = np.random.default_rng(1)
report("random", [rng.normal(size=len(r["options"])) for r in test], test, T_te)

# --- zero-shot cosine: state vs tool description
cos_te = [X_te[i] @ np.stack([O[oidx[o]] for o in r["options"]]).T
          for i, r in enumerate(test)]
report("zero_shot_cosine", cos_te, test, T_te)

# --- kNN: which tool did similar past queries choose?
sims = X_te @ X_tr.T
train_tools = [train[i]["options"][train[i]["label"]] for i in range(len(train))]
def knn_scores(k):
    out = []
    for i in range(len(test)):
        top = np.argsort(-sims[i])[:k]
        votes = {}
        for t in top:
            votes[train_tools[t]] = votes.get(train_tools[t], 0.0) + 1.0
        out.append(np.array([votes.get(o, 0.0) for o in test[i]["options"]]))
    return out
for k in (5, 20):
    report(f"knn{k}_votes", knn_scores(k), test, T_te)

hr = protocol.headroom_check(RESULTS["routing"]["zero_shot_cosine"]["accuracy"])
RESULTS["protocol"]["headroom"] = hr
print("\\nheadroom check:", json.dumps(hr))
save("baselines")
'''),

    md('''
## 4. Trained interaction head

Positional labels (`opt_0`…) are used deliberately: they make the head group decisions by
**option count**, which is exactly the batching the ragged scoring needs. The interaction
head never reads the label as a class identity — it scores `(state, option_embedding)`
pairs — so this is correct here, and would be wrong for the fingerprint head.
'''),
    code('''
def make_examples(X, items):
    ex = []
    for i, r in enumerate(items):
        vecs = np.stack([O[oidx[o]] for o in r["options"]])
        n = len(vecs)
        ex.append(DecisionExample(
            state_embedding=X[i], qtype="choice",
            labels=[f"opt_{j}" for j in range(n)],
            target=[1.0 if j == r["label"] else 0.0 for j in range(n)],
            option_embeddings=[vecs[j] for j in range(n)]))
    return ex

tr_ex = make_examples(X_tr, train)

def head_raw_scores(head, X, items):
    out = []
    for i, r in enumerate(items):
        vecs = np.stack([O[oidx[o]] for o in r["options"]])
        labels = [f"opt_{j}" for j in range(len(vecs))]
        out.append(np.asarray(head.raw_scores(X[i], [vecs[j] for j in range(len(vecs))],
                                             qtype="choice", labels=labels)))
    return out

trained = None
for seed in (0, 1, 2):
    head = DecisionHead(dim=768, mode="interaction", hidden=64)
    head.fit(tr_ex, epochs=60, lr=0.02, batch_size=32, seed=seed)
    block = report(f"interaction_seed{seed}",
                   head_raw_scores(head, X_te, test), test, T_te)
    if seed == 0:
        trained = head
        head.save(WORK / "head_interaction_seed0.json")
    save(f"head-seed{seed}")

keys = ("accuracy", "log_score", "brier", "ece_report_only")
runs = [RESULTS["routing"][f"interaction_seed{s}"] for s in (0, 1, 2)]
RESULTS["routing"]["interaction_mean"] = {k: float(np.mean([r[k] for r in runs])) for k in keys}
RESULTS["routing"]["interaction_mean"]["accuracy_std"] = float(np.std([r["accuracy"] for r in runs]))
print("interaction mean:", json.dumps(RESULTS["routing"]["interaction_mean"]))
save("head-mean")
'''),

    md('''
## 5. Order invariance on real tool sets

Previously validated on Banking77's fixed 77 labels. Here the option set is **variable**
and up to 37 wide, which is the real test: permuting the tools must permute the scores
and nothing else.
'''),
    code('''
rng2 = np.random.default_rng(7)
sample = rng2.choice(len(test), min(80, len(test)), replace=False)
max_diff, flips = 0.0, 0
for i in sample:
    r = test[i]
    n = len(r["options"])
    vecs = [O[oidx[o]] for o in r["options"]]
    labels = [f"opt_{j}" for j in range(n)]
    p0 = trained.predict(X_te[i], vecs, qtype="choice", labels=labels)
    order = rng2.permutation(n)
    p1 = trained.predict(X_te[i], [vecs[j] for j in order], qtype="choice",
                         labels=[labels[j] for j in order])
    if p0.choice != p1.choice:
        flips += 1
    for j in range(n):
        max_diff = max(max_diff, abs(p0.probabilities[labels[j]] - p1.probabilities[labels[j]]))

RESULTS["order_invariance"] = {"states_checked": int(len(sample)),
                               "max_abs_probability_difference": float(max_diff),
                               "argmax_flips": int(flips),
                               "option_cardinality": "2-37, variable per request"}
print("order invariance:", json.dumps(RESULTS["order_invariance"]))
save("order-invariance")
'''),

    md('''
## 6. Abstention (Noul) — is any tool relevant?

Open-set signal is the **raw** match, not softmax: most irrelevance queries carry a single
candidate, where softmax is trivially 1.0.

Positives: each query paired with its ground-truth tool. Negatives: irrelevance queries
with every candidate tool. Then the product question — how much can be auto-routed at a
given precision.
'''),
    code('''
pos = np.array([float(S[i] @ O[oidx[r["options"][r["label"]]]]) for i, r in enumerate(routing)])
neg = np.array([float(S_irr[j] @ O[oidx[o]]) for j, r in enumerate(irrelevant) for o in r["options"]])
print(f"relevant pairs {len(pos)} | irrelevant pairs {len(neg)}")
print(f"cosine | relevant mean {pos.mean():.4f} | irrelevant mean {neg.mean():.4f}")

scores = np.concatenate([pos, neg])
correct = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
esc90 = protocol.escalation_summary(scores, correct, quality_target=0.90, strong_cost=100.0)
esc95 = protocol.escalation_summary(scores, correct, quality_target=0.95, strong_cost=100.0)
RESULTS["abstention"]["zero_shot_raw_score"] = {
    "relevant_mean": float(pos.mean()), "irrelevant_mean": float(neg.mean()),
    "separation": float(pos.mean() - neg.mean()),
    "escalation_at_90pct": esc90, "escalation_at_95pct": esc95,
    "n_relevant": int(len(pos)), "n_irrelevant": int(len(neg)),
}
print("escalation @90%:", json.dumps(esc90, indent=2))
print("escalation @95%:", json.dumps(esc95, indent=2))
save("abstention")
'''),

    md('## 7. Verdict — against the best baseline (S9)'),
    code('''
cand = {k: v for k, v in RESULTS["routing"].items()
        if isinstance(v, dict) and v.get("accuracy") is not None and "std" not in k}
baselines = [k for k in ("random", "zero_shot_cosine", "knn5_votes", "knn20_votes") if k in cand]
verdict = protocol.compare_to_best(cand, baselines=baselines, min_effect=0.0)
RESULTS["verdict"] = verdict

acc = verdict["per_metric"]["accuracy"]
print(f"headroom usable: {RESULTS['protocol']['headroom']['usable']} "
      f"(zero-shot {RESULTS['protocol']['headroom']['zero_shot_accuracy']:.4f})")
print(f"best baseline: {acc['best_baseline']} = {acc['best_baseline_value']:.4f}")
for name, row in sorted(acc["rows"].items(), key=lambda kv: -kv[1]["value"]):
    print(f"  {name:<22} acc={row['value']:.4f} delta={row['delta_vs_best_baseline']:+.4f} "
          f"beats_best={row['beats_best_baseline']}")
for metric in ("log_score", "brier"):
    m = verdict["per_metric"][metric]
    print(f"{metric}: best baseline {m['best_baseline']} = {m['best_baseline_value']:.4f}")
print("winners:", json.dumps(verdict["winners"]))

(WORK / "REPORT.md").write_text("# Dispatch benchmark\\n\\n" +
                                json.dumps(RESULTS, indent=2, default=str))
save("verdict")
with zipfile.ZipFile(WORK / "artifacts.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in ["results.json", "REPORT.md", "run_status.json"]:
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
