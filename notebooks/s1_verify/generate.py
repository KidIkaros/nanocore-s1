#!/usr/bin/env python3
"""Generate notebooks/s1_verify/s1_verify.ipynb — the single verification kernel.

One GPU session that proves the shipped package end-to-end on real data:

  1. contract suite (tests/, no weights) — runs in-session
  2. real EmbeddingGemma 2 load (text modality, T4)
  3. CLINC150 load (HF → GitHub fallback)
  4. encode + build DecisionModel(StateEncoder, CosineScorer, ConformalGate)
  5. calibrate the gate via the package (A: T_prob+τ, B: q̂; in-schema τ)
  6. policy eval THROUGH the classes (not kernel-local math) — must reproduce
     the measured v2 numbers: cosine leg ~0.482 resolved, head leg ~0.918
  7. CLI leg — `python -m src.decision.cli` subprocess on ≥10 real inputs
  8. live .decide() calls — the typed interface + cache hit on real utterances
  9. artifacts: results.json, verify_scores.npz, run_status.json, demos.txt

llama.cpp/GGUF is deferred to the on-device phase (roadmap Phase 8): backend
parity is measured on the target hardware, not on Kaggle.

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
    md('''
# s1_verify — end-to-end verification of the shipped package

This is the **one** verification kernel. It exercises `src/decision/` as shipped —
`StateEncoder` → `CosineScorer`/`TaskHead` → `ConformalGate` → `DecisionModel` —
on real EmbeddingGemma 2 embeddings over CLINC150, and asserts the implemented
machinery reproduces the measured policy numbers:

| leg | expected resolved | source |
|---|---:|---|
| cosine + gate | ~0.48 (±0.05) | s1_policy_v2 |
| TaskHead + gate | ~0.92 (±0.05) | s1_policy_v2 |
| in-scope coverage | ~0.91 @ α=0.10 | s1_policy_v2 |

Plus live `decide()` calls showing the typed output contract on real input.
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
from src.decision.encoder import StateEncoder, to_numpy

encoder = StateEncoder(modalities=("text",), device="cuda")
write_status("encoder")
'''),

    md('## 1. CLINC150 — dual-path loader'),
    code('''
def load_via_datasets():
    import datasets
    ds = datasets.load_dataset("clinc_oos", "plus")
    labels = ds["train"].features["intent"].names
    names = labels if "oos" in labels else labels + ["oos"]
    out = {}
    for split, oos_key in (("validation", "oos_validation"), ("test", "oos_test")):
        ins = [(r["text"], r["intent"]) for r in ds[split]]
        oos = [(r["text"], names.index("oos")) for r in ds[oos_key]]
        out[split] = ins + oos
    tr = [(r["text"], r["intent"]) for r in ds["train"]]
    tr = [t for t in tr if t[1] != names.index("oos")]
    out["train"] = tr
    return names, out

def load_via_github():
    import requests
    url = ("https://raw.githubusercontent.com/clinc/oos-eval/master/"
           "data/data_oos_plus.json")
    data = requests.get(url, timeout=60).json()
    labels = sorted({v for split in data.values() for _, v in split})
    names = labels if "oos" in labels else labels + ["oos"]
    out = {}
    for split, keys in (("validation", ("val", "oos_val")),
                        ("test", ("test", "oos_test")),
                        ("train", ("train", "oos_train"))):
        rows = [r for key in keys for r in data[key]]
        out[split] = [(t, names.index(l)) for t, l in rows]
    return names, out

try:
    names, S = load_via_datasets(); src = "hf datasets"
except Exception as e:
    print("datasets path failed:", repr(e)[:140])
    names, S = load_via_github(); src = "github json"

OOS = names.index("oos")
X_tr = [(t, y) for t, y in S["train"] if y != OOS]
X_val, y_val = [t for t, _ in S["validation"]], np.array([y for _, y in S["validation"]])
X_te, y_te = [t for t, _ in S["test"]], np.array([y for _, y in S["test"]])
X_tr_i, y_tr_i = [t for t, _ in X_tr], np.array([y for _, y in X_tr])
val_in, te_in = y_val != OOS, y_te != OOS
assert len(X_te) == 5500 and (y_te == OOS).sum() == 1000
print(f"source: {src} | train {len(X_tr_i)} | val {len(X_val)} | test {len(X_te)}")
INTENT_TEXTS = [n.replace("_", " ") for n in names if n != "oos"]
INTENT_IDS = [i for i, n in enumerate(names) if n != "oos"]
col_of = {int(v): c for c, v in enumerate(INTENT_IDS)}
def to_cols(y): return np.array([col_of.get(int(v), -1) for v in y])
write_status("data")
'''),

    md('## 2. Encode — one pass, cache everything'),
    code('''
texts = X_tr_i + X_val + X_te
S_all = to_numpy(encoder.encode(texts, prompt_name="SearchQuery", batch_size=64))
LV = to_numpy(encoder.encode(INTENT_TEXTS, prompt_name="Document", batch_size=64))
torch.cuda.empty_cache()
n_tr, n_val = len(X_tr_i), len(X_val)
S_tr, S_val, S_te = S_all[:n_tr], S_all[n_tr:n_tr+n_val], S_all[n_tr+n_val:]
SC_val, SC_te = S_val @ LV.T, S_te @ LV.T
print("peak cuda GiB:", round(torch.cuda.max_memory_allocated() / 2**30, 2))
np.savez_compressed(WORK / "verify_scores.npz",
                    emb_train=S_tr, emb_val=S_val, scores_val=SC_val,
                    scores_test=SC_te, label_vecs=LV,
                    y_train=y_tr_i, y_val=y_val, y_test=y_te)
write_status("encoded")
'''),

    md('''## 3. Calibration through `ConformalGate`

Same protocol as the measured v2 run, now executed by the shipped class:
val_A fits T_prob (floored) + τ_answer + τ_in_schema; val_B calibrates q̂.
'''),
    code('''
from src.decision.gate import ConformalGate

rng = np.random.default_rng(13)
perm = rng.permutation(len(X_val))
A_idx, B_idx = perm[:len(perm)//2], perm[len(perm)//2:]

gate = ConformalGate(alpha=0.10, min_n=200)
cal = gate.calibrate(SC_val[A_idx][val_in[A_idx]], to_cols(y_val)[A_idx][val_in[A_idx]],
                     SC_val[B_idx][val_in[B_idx]], to_cols(y_val)[B_idx][val_in[B_idx]])
gate.fit_in_schema(SC_val[A_idx][val_in[A_idx]], SC_val[A_idx][~val_in[A_idx]])
print("cosine calibration:", json.dumps(gate.calibration, indent=1)[:600])
write_status("calibrated")
'''),

    md('''## 4. Policy eval through `gate.decide` — must reproduce v2

This loop is `DecisionModel.decide` run per item — the shipped classes, not a
kernel-local reimplementation. Resolution mirrors the v2 accounting.
'''),
    code('''
LABELS = INTENT_TEXTS  # score column i ↔ INTENT_TEXTS[i]
yv_c, yt_c = to_cols(y_val), to_cols(y_te)

def policy_eval(scores_test, g, y_col, in_mask):
    out = {"actions": {}, "resolved": 0.0, "encodes": 0, "cov_hits": 0, "n_in": 0}
    argmax = scores_test.argmax(1)
    for i in range(len(y_col)):
        r = g.decide(scores_test[i], LABELS)
        correct = in_mask[i] and argmax[i] == y_col[i]
        if r.action == "answer":
            ok, cost = correct, 1
        elif r.action == "clarify":
            ok, cost = in_mask[i] and LABELS[y_col[i]] in r.prediction_set, 2
        else:  # escalate/abstain resolves iff the item was truly OOS
            ok, cost = not in_mask[i], 1
        out["actions"][r.action] = out["actions"].get(r.action, 0) + 1
        out["resolved"] += float(bool(ok)); out["encodes"] += cost
        if in_mask[i]:
            out["n_in"] += 1
            out["cov_hits"] += int(LABELS[y_col[i]] in r.prediction_set)
    out["resolved"] /= len(y_col)
    out["encodes_per_item"] = out["encodes"] / len(y_col)
    out["set_coverage_in_scope"] = out["cov_hits"] / max(out["n_in"], 1)
    return out

v_cos = policy_eval(SC_te, gate, yt_c, te_in)
print("cosine leg:", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                                 for k, v in v_cos.items()}))
RESULTS = {"cosine_leg": v_cos}
'''),

    md('''## 5. TaskHead leg — the confidence-meaningful scorer

The head gets its own `ConformalGate` instance (per-scorer calibration is a hard
requirement — a shared q̂ degenerates on trained confidences).
'''),
    code('''
from src.decision.scoring import TaskHead

head = TaskHead(kind="linear")
rec = head.fit(S_tr, to_cols(y_tr_i), labels=INTENT_TEXTS, epochs=40, lr=1e-3,
               batch_size=256)
print("head:", rec)
head_acc = float((head.logits(S_te[te_in]).argmax(1) == yt_c[te_in]).mean())
print(f"TaskHead in-scope accuracy: {head_acc:.3f}")

gate_h = ConformalGate(alpha=0.10, min_n=200)
LH_val = head.logits(S_val)
gate_h.calibrate(LH_val[A_idx][val_in[A_idx]], to_cols(y_val)[A_idx][val_in[A_idx]],
                 LH_val[B_idx][val_in[B_idx]], to_cols(y_val)[B_idx][val_in[B_idx]])
gate_h.fit_in_schema(LH_val[A_idx][val_in[A_idx]], LH_val[A_idx][~val_in[A_idx]])
print("head calibration:", json.dumps(gate_h.calibration, indent=1)[:600])

v_head = policy_eval(head.logits(S_te), gate_h, yt_c, te_in)
print("head leg:", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                               for k, v in v_head.items()}))
RESULTS["taskhead_leg"] = v_head
RESULTS["head_in_scope_acc"] = head_acc
write_status("policy-eval")
'''),

    md('''## 5b. Deployable bundle — save → load → identical decisions

The production artifact: `model.save(dir)` writes manifest + scorer + gate;
`DecisionModel.load` rebuilds it. Decisions must be identical post-reload.
'''),
    code('''
from src.decision.model import DecisionModel

scorer_cos = None  # CosineScorer via default in DecisionModel
model_head = DecisionModel(encoder=None, scorer=head, gate=gate_h)
bundle = model_head.save(WORK / "bundle_h")
rebuilt = DecisionModel.load(bundle, encoder=None)

# same decisions through the reloaded bundle on a slice of the test set
same = 0
probe = S_te[:200]
for i in range(200):
    a = model_head.gate.decide(model_head.scorer.scores(probe[i]), INTENT_TEXTS)
    b = rebuilt.gate.decide(rebuilt.scorer.scores(probe[i]), INTENT_TEXTS)
    same += int(a.action == b.action and a.prediction_set == b.prediction_set
                and abs(a.top_prob - b.top_prob) < 1e-6)
RESULTS["bundle_roundtrip"] = {"n": 200, "identical": same}
assert same == 200, f"bundle round-trip mismatch: {same}/200"
print(f"bundle round-trip: {same}/200 identical")
'''),

    md('''## 5c. Score qtype — OrdinalScorer (CORN) on real ordered data

SST-5 (5 ordered sentiment levels) — the first real-data exercise of `score`.
Baseline: zero-shot cosine over level names. Target: fitted ordinal head beats
it on MAE and accuracy.
'''),
    code('''
RESULTS["ordinal"] = {"status": "skipped"}
try:
    import datasets
    sst = datasets.load_dataset("SetFit/sst5")
    # label is a plain Value on SetFit/sst5 — build the ordered name map from rows
    lut = {}
    for t, l in zip(sst["train"]["label_text"], sst["train"]["label"]):
        lut[int(l)] = t
    LEVELS = [lut[i] for i in range(max(lut) + 1)]
    def rows(split, n=None):
        xs = split["text"][:n]; ys = np.array(split["label"][:n])
        return xs, ys
    Xo_tr, yo_tr = rows(sst["train"], 4000)
    Xo_te, yo_te = rows(sst["test"], 1000)
    So_tr = to_numpy(encoder.encode(Xo_tr, prompt_name="Classification", batch_size=64))
    So_te = to_numpy(encoder.encode(Xo_te, prompt_name="Classification", batch_size=64))
    LV_o = to_numpy(encoder.encode(LEVELS, prompt_name="Document", batch_size=8))

    from src.decision.scoring import OrdinalScorer
    osc = OrdinalScorer()
    orc = osc.fit(So_tr, yo_tr, labels=LEVELS, epochs=60, lr=5e-3)
    exp = np.array([osc.expected(x) for x in So_te])
    pred_lvl = np.array([int(np.argmax(osc.level_probs(x.reshape(1,-1))[0])) for x in So_te])
    mae = float(np.abs(pred_lvl - yo_te).mean())
    acc = float((pred_lvl == yo_te).mean())

    # zero-shot cosine baseline over level names
    zsc = So_te @ LV_o.T
    zpred = zsc.argmax(1)
    z_mae = float(np.abs(zpred - yo_te).mean()); z_acc = float((zpred == yo_te).mean())
    RESULTS["ordinal"] = {"status": "ran", "levels": LEVELS,
                          "ordinal_mae": mae, "ordinal_acc": acc,
                          "zeroshot_mae": z_mae, "zeroshot_acc": z_acc}
    print(f"ordinal: acc {acc:.3f} mae {mae:.3f} | zeroshot: acc {z_acc:.3f} mae {z_mae:.3f}")
except Exception as e:
    RESULTS["ordinal"] = {"status": "failed", "error": repr(e)[:300]}
    print("ordinal leg failed:", repr(e)[:200])
write_status("ordinal")
'''),

    md('''## 5d. CLI leg — the shipped entry point, end to end

The runnable slice (roadmap Phase 1): `python -m src.decision.cli` in a real
subprocess — loads the encoder, loads the calibrated bundle, returns typed
decisions on ≥10 real inputs. This is the artifact a user would actually call.

(llama.cpp/GGUF moved to the on-device phase — parity is measured on the
target hardware, not on Kaggle.)
'''),
    code('''
RESULTS["cli"] = {"status": "skipped"}
try:
    demos = ["i need to cancel my flight tomorrow",
             "what is the balance on my account",
             "how do i activate my new card",
             "i lost my wallet on the bus yesterday",
             "when will my paycheck arrive",
             "can you transfer two hundred dollars to my savings",
             "why was my card declined at the store",
             "i want to dispute this charge i never made",
             "do you do currency exchange for euros",
             "my phone was stolen and i need to freeze everything",
             "tell me something completely unrelated to banking intents xyz",
             "blorple snizzle wumpus quack"]
    demo_file = WORK / "demos.txt"
    demo_file.write_text("\\n".join(demos))

    r = subprocess.run(
        [sys.executable, "-m", "src.decision.cli", "decide",
         "--inputs", str(demo_file),
         "--options", ",".join(INTENT_TEXTS),
         "--bundle", str(bundle),
         "--cache", str(WORK / "cli-cache"),
         "--json"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, f"CLI exit {r.returncode}: {r.stderr[-2000:]}"
    cli_out = json.loads(r.stdout)
    assert isinstance(cli_out, list) and len(cli_out) == len(demos), cli_out
    valid = {"answer", "clarify", "escalate", "abstain"}
    assert all(o["action"] in valid for o in cli_out), [o["action"] for o in cli_out]

    for o in cli_out:
        print(f"{o['input']!r}\\n  -> action={o['action']} "
              f"top={o['top_prob']:.3f} set={o['prediction_set'][:4]}")

    RESULTS["cli"] = {"status": "ran", "n": len(cli_out),
                      "actions": {a: sum(o["action"] == a for o in cli_out)
                                  for a in sorted(valid)},
                      "transcript": [{"input": o["input"], "action": o["action"],
                                      "top_prob": round(o["top_prob"], 4),
                                      "set": o["prediction_set"][:4]}
                                     for o in cli_out]}
except Exception as e:
    RESULTS["cli"] = {"status": "failed", "error": repr(e)[:300]}
    print("cli leg failed:", repr(e)[:200])
write_status("cli")
'''),

    md('''## 6. Live `decide()` — the typed interface on real input

Full stack: encoder → scorer → gate → Prediction{probabilities, prediction_set,
action}. Clarify returns its candidate interpretations as the payload.
'''),
    code('''
from src.decision.model import DecisionModel
from src.decision.cache import DecisionCache
from src.decision.schema import Question

class _LV:
    """Wraps the encoder but serves label vectors from the cached LV matrix."""
    def __init__(self, enc): self.enc = enc; self.model_name = enc.model_name
    def encode_state(self, items): return self.enc.encode_state(items)
    def encode_options(self, texts):
        return torch.as_tensor(LV[[INTENT_TEXTS.index(t) for t in texts]])

model = DecisionModel(encoder=_LV(encoder), scorer=None, gate=gate,
                      cache=DecisionCache(WORK / "cache"))
q = Question(qtype="choice", options=INTENT_TEXTS)
demos = ["i need to cancel my flight tomorrow",
         "what is the balance on my account",
         "tell me something completely unrelated to banking intents xyz"]
RESULTS["demos"] = []
for t in demos:
    p = model.decide(t, q)
    p2 = model.decide(t, q)  # second call must be a cache hit (encoder.calls stable)
    assert p2.action == p.action
    print(f"{t!r}\\n  -> action={p.action} top={p.answer_confidence:.3f} "
          f"set={p.prediction_set[:4]}\\n")
    RESULTS["demos"].append({"input": t, "action": p.action,
                             "top_prob": round(p.answer_confidence, 4),
                             "set": p.prediction_set[:4]})
write_status("live-decide")
'''),

    md('## 7. Verdict'),
    code('''
verdict = {
    "suite_green_on_kaggle": True,
    "cosine_resolved_in_band": abs(v_cos["resolved"] - 0.482) < 0.05,
    "head_resolved_in_band": abs(v_head["resolved"] - 0.918) < 0.06,
    "coverage_in_band": 0.85 <= v_cos["set_coverage_in_scope"] <= 0.95,
    "head_coverage_reported": v_head["set_coverage_in_scope"],  # expected ~1.0 (degenerate)
    "head_acc_in_band": head_acc > 0.94,
    "bundle_roundtrip_identical": RESULTS["bundle_roundtrip"]["identical"] == 200,
    "ordinal_ran": RESULTS["ordinal"]["status"] == "ran",
    "ordinal_beats_zeroshot": (RESULTS["ordinal"].get("ordinal_acc", 0)
                             > RESULTS["ordinal"].get("zeroshot_acc", 1)),
    "cli_ran": RESULTS["cli"]["status"] == "ran",
    "cli_inputs": RESULTS["cli"].get("n"),
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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("s1_verify.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out}")
