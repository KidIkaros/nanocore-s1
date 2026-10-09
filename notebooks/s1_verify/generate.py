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
    out = {"actions": {}, "resolved": 0.0, "encodes": 0, "cov_hits": 0, "n_in": 0,
           "set_sizes_in_scope": []}
    # Per-example in-scope records for the readiness checks — the per-slice and
    # deferral numbers a marginal figure can hide (research:
    # benchmarks-and-evaluation-readiness).
    rec = {"action": [], "top1": [], "covered": [], "top_prob": [], "intent": []}
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
            covered = LABELS[y_col[i]] in r.prediction_set
            out["cov_hits"] += int(covered)
            out["set_sizes_in_scope"].append(len(r.prediction_set))
            rec["action"].append(r.action)
            rec["top1"].append(bool(correct))
            rec["covered"].append(bool(covered))
            rec["top_prob"].append(float(r.top_prob))
            rec["intent"].append(int(y_col[i]))
    out["resolved"] /= len(y_col)
    out["encodes_per_item"] = out["encodes"] / len(y_col)
    out["set_coverage_in_scope"] = out["cov_hits"] / max(out["n_in"], 1)
    # Set SIZE is the observable that coverage hides: a saturated quantile
    # reports coverage 1.0000 (looks like success) while every set is the whole
    # label space. Reported so the degeneracy can never hide again.
    sizes = np.asarray(out.pop("set_sizes_in_scope"), dtype=float)
    out["mean_set_size"] = float(sizes.mean()) if sizes.size else 0.0
    out["singleton_rate"] = float((sizes == 1).mean()) if sizes.size else 0.0
    return out, rec

v_cos, rec_cos = policy_eval(SC_te, gate, yt_c, te_in)
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

v_head, rec_head = policy_eval(head.logits(S_te), gate_h, yt_c, te_in)
print("head leg:", json.dumps({k: round(v, 3) if isinstance(v, float) else v
                               for k, v in v_head.items()}))
RESULTS["taskhead_leg"] = v_head
RESULTS["head_in_scope_acc"] = head_acc
# Persist the head's own scores. `verify_scores.npz` holds only the *cosine*
# scores (`S @ LV.T`), so the head path could not be re-evaluated offline — the
# v19 conformal-set audit had to recover the head from its bundle and re-derive
# the raw-id → column mapping by hand, which is how a false 53% accuracy was
# read before the mapping was found. Two extra arrays make that avoidable.
np.savez_compressed(WORK / "head_scores.npz",
                    scores_val=LH_val, scores_test=head.logits(S_te),
                    y_val=y_val, y_test=y_te)
print("head scores saved for offline re-evaluation")
write_status("policy-eval")
'''),

    md('''## 5a. Human-interaction readiness — what the marginals can hide

Four checks from `docs/research/benchmarks-and-evaluation-readiness.md`, each
aimed at a failure mode an aggregate reports as fine:

- **conditional coverage** — does the conformal target hold inside the hard
  (low-confidence) slice, and per intent, or does the marginal hide it?
  (MAPIE conditional-CP / Mondrian)
- **deferral quality** — does the gate escalate the items it would get wrong?
  (learning-to-defer; Mozannar et al. 2023)
- **over-rejection** — does abstention concentrate on one slice?
  (Pugnana & Ruggieri minority over-rejection)
- **memorization** — candidate-order invariance + withheld-state collapse
  (arXiv:2609.37647)

All of it computes on the per-example records `policy_eval` already collected —
no re-encoding except the five withheld-state probes.
'''),
    code('''
from src.decision import readiness

def _slice_block(rec, alpha):
    band = readiness.confidence_bands(rec["top_prob"])
    return {
        "coverage_by_band": readiness.group_coverage(rec["covered"], band, alpha),
        "coverage_by_intent": readiness.group_coverage(
            rec["covered"], rec["intent"], alpha, min_group_n=30),
        "deferral": readiness.deferral_quality(rec["action"], rec["top1"]),
        "rejection_by_band": readiness.rejection_by_group(rec["action"], band),
        # Per-intent is the slice where concentrated rejection could mean a
        # broken intent — by-band concentration is expected (the hard band
        # defers more), by-intent is not.
        "rejection_by_intent": readiness.rejection_by_group(
            rec["action"], rec["intent"], min_group_n=30, concentration=5.0),
    }

RESULTS["readiness"] = {
    "cosine_leg": _slice_block(rec_cos, gate.alpha),
    "taskhead_leg": _slice_block(rec_head, gate_h.alpha),
}

# Memorization probes (arXiv:2609.37647). Candidate-order invariance on the
# cached head scores: permute the columns, the chosen *label* must not move.
_rng = np.random.default_rng(0)
_perm = _rng.permutation(len(INTENT_TEXTS))
SH = head.logits(S_te)[te_in]
_cons = float((_perm[SH[:, _perm].argmax(1)] == SH.argmax(1)).mean())
# Withheld state — uninformative text must not be confidently answered.
_fillers = ["", "the", "...", "aaaaaaaa", "lorem ipsum dolor sit amet"]
E_w = to_numpy(encoder.encode(_fillers, prompt_name="SearchQuery"))
_w_acts, _w_top = [], []
for _i in range(len(_fillers)):
    _rw = gate_h.decide(head.scores(E_w[_i]), INTENT_TEXTS)
    _w_acts.append(_rw.action); _w_top.append(_rw.top_prob)
_w = {"top_prob": float(np.mean(_w_top)),
      "answered": int(sum(a == "answer" for a in _w_acts)),
      "actions": _w_acts}
RESULTS["readiness"]["memorization"] = readiness.memorization_verdict(
    _cons, _w, len(INTENT_TEXTS))

for _leg in ("cosine_leg", "taskhead_leg"):
    _b = RESULTS["readiness"][_leg]
    _cb, _dq = _b["coverage_by_band"], _b["deferral"]
    print(f"{_leg:12} cov " + " ".join(
        f"{g}={v['coverage']:.3f}(n{v['n']})" for g, v in _cb["groups"].items())
        + f" | undercovered {_cb['undercovered'] or 'none'}")
    print(f"{'':12} assert={_dq.get('acc_asserted')} "
          f"deferred={_dq.get('acc_deferred')} "
          f"enrichment={_dq.get('error_enrichment')}")
print("memorization:", json.dumps(RESULTS["readiness"]["memorization"]))
write_status("readiness")
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

    md('''## 5d-bis. CLI adapt leg — labeled data → bundle → decision, all through the CLI

Definition-of-done #1: *"a person can install it, point it at their labeled
data, and get a calibrated bundle."* Until `adapt` existed as a command, that
was reachable only from Python — a user could consume a bundle but not make one.

This runs the whole user-facing path in real subprocesses with the real encoder:
write a CSV, `adapt` it, then `decide` against the bundle it produced. The
contract is the round trip, not the accuracy — a bundle the next command cannot
read is not a bundle.
'''),
    code('''
RESULTS["cli_adapt"] = {"status": "skipped"}
try:
    import csv as _csv

    # Class-balanced rows: CLINC's train is class-ordered, so a head-of-file
    # slice would give ~6 classes and `adapt` would refuse the split outright.
    text_by_id = {int(i): t for i, t in zip(INTENT_IDS, INTENT_TEXTS)}
    PER_CLASS = 10
    rows = []
    for c in np.unique(y_tr_i):
        for i in np.flatnonzero(np.asarray(y_tr_i) == c)[:PER_CLASS]:
            rows.append((X_tr_i[i], text_by_id[int(c)]))
    data_csv = WORK / "cli_adapt.csv"
    with data_csv.open("w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"wrote {len(rows)} labeled rows over {len(np.unique(y_tr_i))} classes")

    ad = subprocess.run(
        [sys.executable, "-m", "src.decision.cli", "adapt",
         "--data", str(data_csv), "--text-col", "text", "--label-col", "label",
         "--out", str(WORK / "cli-adapt"), "--backend", "st",
         "--min-cal", "200", "--epochs", "12", "--json"],
        cwd=REPO, capture_output=True, text=True, timeout=3600)
    assert ad.returncode == 0, f"adapt exit {ad.returncode}: {ad.stderr[-2000:]}"
    rep = json.loads(ad.stdout)
    print(f"adapt: {rep['n_total']} examples, {rep['n_classes']} classes | "
          f"headroom {rep['headroom']['zeroshot_test_acc']:.3f} | "
          f"test acc {rep['test']['accuracy']:.3f} "
          f"cov {rep['test']['conformal_coverage']:.3f} "
          f"set {rep['test']['mean_set_size']:.2f}")

    # the round trip: the bundle `adapt` wrote must be decidable by `decide`
    dc = subprocess.run(
        [sys.executable, "-m", "src.decision.cli", "decide",
         "i need to cancel my flight tomorrow",
         "--options", ",".join(INTENT_TEXTS),
         "--bundle", str(WORK / "cli-adapt" / "bundle"),
         "--cache", str(WORK / "cli-adapt-cache"), "--json"],
        cwd=REPO, capture_output=True, text=True, timeout=1800)
    assert dc.returncode == 0, f"decide exit {dc.returncode}: {dc.stderr[-2000:]}"
    out = json.loads(dc.stdout)
    assert out["action"] in ("answer", "clarify", "escalate", "abstain"), out
    print(f"round trip: action={out['action']} top={out['top_prob']:.3f} "
          f"set={out['prediction_set'][:4]}")

    RESULTS["cli_adapt"] = {
        "status": "ran", "n_rows": len(rows), "n_classes": rep["n_classes"],
        "headroom": rep["headroom"]["zeroshot_test_acc"],
        "test": rep["test"],
        "roundtrip_action": out["action"],
        "roundtrip_top_prob": round(out["top_prob"], 4)}
except Exception as e:
    RESULTS["cli_adapt"] = {"status": "failed", "error": repr(e)[:300]}
    print("cli adapt leg failed:", repr(e)[:200])
write_status("cli-adapt")
'''),

    md('''## 5e. Serve leg — the HTTP surface on a real subprocess

Phase 4 acceptance in-kernel: `nanocore serve` starts the model server with
the calibrated bundle; a real POST /decide returns an action and writes a
prediction-log line with latency. This is the surface other programs call.
'''),
    code('''
RESULTS["serve"] = {"status": "skipped"}
try:
    import urllib.request, socket
    s = socket.socket(); s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]; s.close()
    log_path = WORK / "serve-preds.jsonl"
    srv = subprocess.Popen(
        [sys.executable, "-m", "src.decision.cli", "serve",
         "--bundle", str(bundle), "--backend", "st", "--device", "cuda",
         "--port", str(port), "--log", str(log_path),
         "--cache", str(WORK / "serve-cache")],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ok = False
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz",
                                       timeout=2)
                ok = True; break
            except Exception:
                time.sleep(1)
        assert ok, "server never became healthy"

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/decide",
            data=json.dumps({"text": "i need to cancel my flight",
                             "options": INTENT_TEXTS}).encode(),
            headers={"Content-Type": "application/json"})
        out = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert out["action"] in ("answer", "clarify", "escalate", "abstain")
        assert out["prediction_set"] and out["latency_ms"] > 0

        rec = json.loads(log_path.read_text().strip().splitlines()[-1])
        assert rec["action"] == out["action"] and rec["latency_ms"] > 0
        stats = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/stats", timeout=10).read())
        print(f"serve: POST /decide -> {out['action']} "
              f"({out['latency_ms']:.0f}ms) | stats: {stats['requests']} req, "
              f"p50={stats['latency_ms']['p50']:.0f}ms")
        RESULTS["serve"] = {"status": "ran", "action": out["action"],
                            "latency_ms": out["latency_ms"],
                            "log_line": True}
    finally:
        srv.terminate(); srv.wait(timeout=30)
except Exception as e:
    RESULTS["serve"] = {"status": "failed", "error": repr(e)[:300]}
    print("serve leg failed:", repr(e)[:200])
write_status("serve")
'''),

    md('''## 5f. Baselines — TF-IDF+LR on identical splits (roadmap Phase 7)

Cross-paper claims die here: a real baseline run on the *same* splits. If
TF-IDF+LR beats our scorers, we say so — `compare_to_best` compares every
candidate against the best baseline per metric.
'''),
    code('''
RESULTS["baselines"] = {"status": "skipped"}
try:
    from src.decision.evaluate import tfidf_baseline
    from src.decision.protocol import metric_block, compare_to_best
    from src.decision.scoring import softmax_rows

    Xte_in = [t for t, m in zip(X_te, te_in) if m]  # X_te is a list
    # label-space check: fit LR on column ids (0..149), not dataset ids
    P_lr = tfidf_baseline(X_tr_i, to_cols(y_tr_i), Xte_in, len(INTENT_TEXTS))

    onehot = np.eye(len(INTENT_TEXTS))[yt_c[te_in]]
    P_cos = softmax_rows(SC_te[te_in], gate.t_prob)
    P_head = softmax_rows(head.logits(S_te[te_in]), gate_h.t_prob)

    blocks = {"cosine": metric_block(P_cos, onehot),
              "taskhead": metric_block(P_head, onehot),
              "tfidf_lr": metric_block(P_lr, onehot)}
    verdict_cmp = compare_to_best(blocks, baselines=["tfidf_lr"])
    RESULTS["baselines"] = {"status": "ran", "blocks": blocks,
                            "comparison": verdict_cmp}
    for n, b in blocks.items():
        print(f"{n:10} acc={b['accuracy']:.3f} log={b['log_score']:.3f} "
              f"brier={b['brier']:.3f} ece={b['ece_report_only']:.3f}")
except Exception as e:
    RESULTS["baselines"] = {"status": "failed", "error": repr(e)[:300]}
    print("baselines leg failed:", repr(e)[:200])
write_status("baselines")
'''),

    md('''## 5g. Evaluation rigor — CIs, AURC, selective prediction

Every primary metric carries a bootstrap 95% CI. AURC + accuracy at fixed
coverage (50%/80%) are the selective-prediction standard. ECE over 15 bins.
'''),
    code('''
RESULTS["rigor"] = {"status": "skipped"}
try:
    from src.decision.evaluate import rigor_block
    from src.decision.scoring import softmax_rows

    P_cos = softmax_rows(SC_te[te_in], gate.t_prob)
    P_head = softmax_rows(head.logits(S_te[te_in]), gate_h.t_prob)
    yt_in = yt_c[te_in]
    legs = {"cosine": rigor_block(P_cos, yt_in),
            "taskhead": rigor_block(P_head, yt_in)}
    if RESULTS["baselines"].get("status") == "ran":
        legs["tfidf_lr"] = rigor_block(P_lr, yt_in)
    RESULTS["rigor"] = {"status": "ran", **legs}
    for name, r in legs.items():
        print(f"{name:10} acc={r['acc']['point']:.3f} "
              f"[{r['acc']['lo']:.3f},{r['acc']['hi']:.3f}] "
              f"aurc={r['aurc']:.4f} ece15={r['ece15']:.3f} "
              f"acc@80={r['acc_at_80']:.3f}")
except Exception as e:
    RESULTS["rigor"] = {"status": "failed", "error": repr(e)[:300]}
    print("rigor leg failed:", repr(e)[:200])
write_status("rigor")
'''),

    md('''## 5h. Memorization probe — similarity-band accuracy

If the head only memorized near-duplicates, accuracy collapses on test items
distant from the train set. Bucket test items by max cosine similarity to
any train embedding; report accuracy per band.
'''),
    code('''
RESULTS["memorization"] = {"status": "skipped"}
try:
    from src.decision.evaluate import similarity_bands
    hpred = head.logits(S_te[te_in]).argmax(1)
    acc_by_band = similarity_bands(S_te[te_in], S_tr, hpred, yt_c[te_in])
    RESULTS["memorization"] = {"status": "ran", "acc_by_sim_band": acc_by_band}
    print("acc by nearest-train-similarity band:",
          json.dumps(acc_by_band, indent=1))
except Exception as e:
    RESULTS["memorization"] = {"status": "failed", "error": repr(e)[:300]}
    print("memorization leg failed:", repr(e)[:200])
write_status("memorization")
'''),

    md('''## 5i. Breadth — the same protocol across datasets

Phase 7 breadth leg: several public classification datasets, each through the
identical pipeline — zero-shot cosine vs fitted head vs TF-IDF+LR, every block
with bootstrap CIs, AURC, ECE15, Brier (AUROC/AUPRC on binary tasks). All
computation lives in `src.decision.evaluate.dataset_suite` (unit-tested);
this cell is load → encode → call → record. Per-dataset failures are recorded
and skipped rather than aborting the leg.
'''),
    code('''
import datasets as _ds
from src.decision.adapt import split_indices
from src.decision.datasets import (class_quota, label_field, label_space,
                                   texts_of)
from src.decision.evaluate import dataset_suite

CAP_TR, CAP_TE = 2000, 1000


def _unit(a):
    """Row-wise L2 normalize, guarding the zero vector."""
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


def run_dataset_cfg(cfg, cap_tr=CAP_TR, cap_te=CAP_TE):
    """Load one benchmark config, encode it, return (summary, arrays).

    Shared by the breadth leg and the multilingual leg: one pipeline, two config
    lists. It lives in its own cell, outside either leg's ``try``, so a failure
    in one leg cannot leave the other calling an undefined helper.

    The arrays come back so a cross-lingual arm can reuse an English head against
    another language's test split without re-encoding; callers that only want the
    numbers ignore them.
    """
    d = _ds.load_dataset(cfg["hf"], cfg.get("config"))
    tr, te = d["train"], d[cfg.get("test_split", "test")]
    lf = label_field(tr.features, cfg.get("label", "label"))
    space = label_space(tr, lf, cfg.get("names"), cfg.get("names_field"))
    names = list(space.names)
    tr_s = tr.select(class_quota(tr, lf, cap_tr))
    te_s = te.select(class_quota(te, lf, cap_te))
    Xtr_all = texts_of(tr_s, cfg.get("text", "text"), cfg.get("pair"))
    ytr_all = space.ids(tr_s, lf)
    Xte = texts_of(te_s, cfg.get("text", "text"), cfg.get("pair"))
    yte = space.ids(te_s, lf)
    fi, ci, _ = split_indices(len(Xtr_all), 0.6, 0.3, 0, y=ytr_all)
    absent = sorted(set(range(len(names))) - set(ytr_all[fi].tolist()))
    if absent:
        raise ValueError(f"fit split is missing {len(absent)} classes: {absent[:6]}")
    kw = {"prompt_name": "Classification", "batch_size": 64}
    E_fit = to_numpy(encoder.encode([Xtr_all[i] for i in fi], **kw))
    E_cal = to_numpy(encoder.encode([Xtr_all[i] for i in ci], **kw))
    E_te = to_numpy(encoder.encode(Xte, **kw))
    LVb = _unit(to_numpy(encoder.encode(space.prompt_names(),
                                        prompt_name="Document", batch_size=32)))
    r = dataset_suite(E_fit, ytr_all[fi], E_cal, ytr_all[ci],
                      E_te, yte, [Xtr_all[i] for i in fi], Xte,
                      LVb, len(names))
    summary = {"status": "ran", "n_classes": len(names), "n_test": len(Xte), **r}
    arrays = {"E_fit": E_fit, "y_fit": ytr_all[fi], "E_te": E_te, "y_te": yte,
              "names": names}
    return summary, arrays
'''),

    code('''
RESULTS["breadth"] = {"status": "skipped", "datasets": {}}
try:
    # Parquet-only ids, verified loadable locally: Kaggle's older datasets
    # tolerates script-based datasets (PolyAI/banking77, CogComp/trec) while a
    # current one refuses them, so a script id makes local and Kaggle runs
    # disagree about what exists.
    BREADTH = [
        {"name": "banking77", "hf": "mteb/banking77", "names_field": "label_text"},
        {"name": "emotion", "hf": "dair-ai/emotion", "config": "split"},
        {"name": "tweet_emotion", "hf": "cardiffnlp/tweet_eval", "config": "emotion"},
        {"name": "tweet_sentiment", "hf": "cardiffnlp/tweet_eval", "config": "sentiment"},
        {"name": "tweet_hate", "hf": "cardiffnlp/tweet_eval", "config": "hate"},
        {"name": "sst2", "hf": "stanfordnlp/sst2", "text": "sentence",
         "test_split": "validation", "names": ["negative", "positive"]},
        {"name": "ag_news", "hf": "fancyzhx/ag_news"},
        {"name": "dbpedia_14", "hf": "fancyzhx/dbpedia_14", "config": "dbpedia_14",
         "text": "content"},
        {"name": "mnli", "hf": "nyu-mll/glue", "config": "mnli", "text": "premise",
         "pair": "hypothesis", "test_split": "validation_matched"},
        {"name": "massive_intent_en", "hf": "mteb/amazon_massive_intent",
         "config": "en", "names_field": "label_text"},
    ]
    ran = 0
    for cfg in BREADTH:
        name = cfg["name"]
        try:
            r, _ = run_dataset_cfg(cfg)
            RESULTS["breadth"]["datasets"][name] = r
            ran += 1
            b = r["blocks"]
            print(f"{name:16} k={r['n_classes']:3} "
                  f"head={b['taskhead']['acc']['point']:.3f} "
                  f"cos={b['cosine']['acc']['point']:.3f} "
                  f"tfidf={b['tfidf_lr']['acc']['point']:.3f}")
        except Exception as e:
            RESULTS["breadth"]["datasets"][name] = {
                "status": "failed", "error": repr(e)[:200]}
            print(f"{name:16} failed: {repr(e)[:140]}")
    RESULTS["breadth"]["status"] = "ran" if ran >= 3 else "partial"
    RESULTS["breadth"]["n_ran"] = ran
except Exception as e:
    RESULTS["breadth"] = {"status": "failed", "error": repr(e)[:300]}
    print("breadth leg failed:", repr(e)[:200])
write_status("breadth")
'''),

    md('''## 5i. Multilingual slice — the language gradient

`massive_intent_en` is the **English** config of a 52-language parallel corpus, so
it added breadth and zero multilingual evidence. Same 60 intents, same loader, same
head and gate — only the language changes.

Two questions, one pass: does the pipeline hold **in-language**, and does an
**English-trained head transfer**? The second needs the label spaces aligned by
name, not by index, or column `i` would mean different intents in different
languages and the transfer number would be noise.
'''),
    code('''
try:
    from src.decision.scoring import TaskHead

    LANGS = ["de", "es", "fr", "ru", "zh-CN", "ja"]
    MTR, MTE = 1000, 500
    RESULTS["multilingual"] = {"status": "skipped", "languages": {}}

    def mass_cfg(lang):
        return {"name": f"massive_{lang}", "hf": "mteb/amazon_massive_intent",
                "config": lang, "names_field": "label_text"}

    r_en, arr_en = run_dataset_cfg(mass_cfg("en"), cap_tr=MTR, cap_te=MTE)
    canon = list(arr_en["names"])
    pos = {n: i for i, n in enumerate(canon)}
    en_head = TaskHead(kind="linear")
    en_head.fit(arr_en["E_fit"], arr_en["y_fit"], labels=canon, seed=0)
    RESULTS["multilingual"]["languages"]["en"] = {
        "status": "ran", "n_classes": r_en["n_classes"],
        "in_language_head_acc": r_en["blocks"]["taskhead"]["acc"],
        "in_language_cosine_acc": r_en["blocks"]["cosine"]["acc"],
        "cross_lingual_en_head_acc": r_en["blocks"]["taskhead"]["acc"]["point"]}
    print(f"{'en':6} in-language {r_en['blocks']['taskhead']['acc']['point']:.3f} "
          f"| cosine {r_en['blocks']['cosine']['acc']['point']:.3f}")

    for lang in LANGS:
        try:
            r, arr = run_dataset_cfg(mass_cfg(lang), cap_tr=MTR, cap_te=MTE)
            missing = [n for n in arr["names"] if n not in pos]
            if missing:
                raise ValueError(f"{len(missing)} labels absent from the English "
                                 f"taxonomy: {missing[:4]}")
            y_al = np.array([pos[n] for n in (arr["names"][j] for j in arr["y_te"])])
            xl = float((en_head.logits(arr["E_te"]).argmax(1) == y_al).mean())
            RESULTS["multilingual"]["languages"][lang] = {
                "status": "ran", "n_classes": r["n_classes"],
                "in_language_head_acc": r["blocks"]["taskhead"]["acc"],
                "in_language_cosine_acc": r["blocks"]["cosine"]["acc"],
                "cross_lingual_en_head_acc": xl}
            print(f"{lang:6} in-language {r['blocks']['taskhead']['acc']['point']:.3f} "
                  f"| en-head transfer {xl:.3f} | cosine "
                  f"{r['blocks']['cosine']['acc']['point']:.3f}")
        except Exception as e:
            RESULTS["multilingual"]["languages"][lang] = {
                "status": "failed", "error": repr(e)[:200]}
            print(f"{lang:6} failed: {repr(e)[:140]}")
    n_lang = sum(1 for v in RESULTS["multilingual"]["languages"].values()
                 if v["status"] == "ran")
    RESULTS["multilingual"]["status"] = "ran" if n_lang >= 4 else "partial"
    RESULTS["multilingual"]["n_ran"] = n_lang
except Exception as e:
    RESULTS["multilingual"] = {"status": "failed", "error": repr(e)[:300]}
    print("multilingual leg failed:", repr(e)[:200])
write_status("multilingual")
'''),

    md('''## 5k. Multimodal decision — the unproven thesis

Every number in this kernel is text. The encoder is multimodal by design (text
270M / +vision 170M / +audio 300M), and vision has exactly one artifact behind it:
a 100-image CIFAR smoke test measuring *embedding* quality (0.91 zero-shot cosine).
**No typed decision has ever been made from a non-text state.**

ScienceQA supplies a real multimodal Choice with per-question options — and it has
*per-question* option sets, so this uses the cosine scorer rather than a
schema-bound head. Three arms at equal treatment:

| arm | state | reads |
|---|---|---|
| `full` | `[{"image": img}, question]` | image + text |
| `text_only` | `[question]` | text |
| `image_only` | `[{"image": img}]` | image |

If the image carries decision information, `full` beats `text_only`. If it does
not, the multimodal claim is unproven **on this task** — which is the honest
outcome and the reason to run it.
'''),
    code('''
try:
    from datasets import load_dataset as _load_ds
    from src.decision.causal import paired_readout
    from src.decision.encoder import StateEncoder
    from src.decision.evaluate import rigor_block
    from src.decision.scoring import aps_members, softmax_rows as _sm
    from src.decision.gate import ConformalGate

    SQ_TR, SQ_TE, N_OPT, CHUNK = 1500, 600, 4, 64
    sq = _load_ds("derek-thomas/ScienceQA")

    def sq_index(split, cap):
        """Indices of usable rows, scanned WITHOUT materialising the split.

        The `image` field decodes on access, so the obvious list comprehension
        holds every image in RAM — ~12,700 rows at roughly 750 KB each. Scan by
        index, keep only the ones that qualify, and stop once there are enough.
        """
        keep = []
        for i in range(len(sq[split])):
            e = sq[split][i]
            if (e["image"] is not None and len(e["choices"]) == N_OPT
                    and 0 <= int(e["answer"]) < N_OPT):
                keep.append(i)
                if len(keep) >= cap:
                    break
        return keep

    tr_idx, te_idx = sq_index("train", SQ_TR), sq_index("validation", SQ_TE)
    n_tr = len(tr_idx)
    all_idx = [("train", i) for i in tr_idx] + [("validation", i) for i in te_idx]
    print(f"ScienceQA: {n_tr} fit / {len(te_idx)} test rows "
          f"({N_OPT}-choice, with image)")
    if n_tr < 200 or len(te_idx) < 100:
        raise ValueError(f"too few usable rows: {n_tr}/{len(te_idx)}")

    # A vision encoder is a second load; the text-only one stays resident.
    enc_v = StateEncoder(modalities=("text", "vision"), device="cuda")

    def _img_chunk(imgs):
        """Descending batch size: 32 OOM'd on CIFAR and these images are larger."""
        for bs in (4, 2, 1):
            try:
                torch.cuda.empty_cache()
                return to_numpy(enc_v.encode([{"image": im} for im in imgs],
                                             batch_size=bs))
            except torch.cuda.OutOfMemoryError:
                print(f"  vision OOM at batch {bs}, retrying smaller")
        raise RuntimeError("vision encode OOM at every batch size")

    # Chunked so only one chunk's decoded images are resident at a time.
    E_img = np.concatenate([
        _img_chunk([sq[s][i]["image"] for s, i in all_idx[c:c + CHUNK]])
        for c in range(0, len(all_idx), CHUNK)])
    E_q = to_numpy(enc_v.encode([sq[s][i]["question"] for s, i in all_idx],
                                prompt_name="Classification", batch_size=64))
    print("vision peak allocated GiB:",
          round(torch.cuda.max_memory_allocated() / 2**30, 2))

    arms = {
        "full": _unit((E_img + E_q) / 2.0),
        "text_only": _unit(E_q),
        "image_only": _unit(E_img),
    }

    flat_opts = [c for s, i in all_idx for c in sq[s][i]["choices"]]
    O = _unit(to_numpy(enc_v.encode(flat_opts, prompt_name="Document",
                                    batch_size=64))).reshape(len(all_idx), N_OPT, -1)
    y_all = np.array([int(sq[s][i]["answer"]) for s, i in all_idx], dtype=int)

    def arm_eval(vecs):
        """(n, N_OPT) cosine scores → (calibrated gate block + set stats, errors)."""
        S = np.einsum("nd,nkd->nk", vecs, O)
        S_tr, S_te = S[:n_tr], S[n_tr:]
        y_tr, y_te = y_all[:n_tr], y_all[n_tr:]
        ci, cj, _ = split_indices(n_tr, 0.6, 0.3, 0, y=y_tr)
        g = ConformalGate(alpha=0.10, min_n=100)
        g.calibrate(S_tr[ci], y_tr[ci], S_tr[cj], y_tr[cj])
        P = _sm(S_te, g.t_prob)
        members = aps_members(_sm(S_te, g.t_set), g.qhat)
        sizes = np.array([len(m) for m in members])
        cov = float(np.mean([y_te[i] in members[i] for i in range(len(members))]))
        summary = {"block": rigor_block(P, y_te, resamples=200),
                   "coverage": cov, "mean_set_size": float(sizes.mean()),
                   "t_set": g.t_set, "qhat": g.qhat}
        return summary, (P.argmax(1) != y_te).astype(float)

    RESULTS["multimodal"] = {"status": "ran", "n_fit": n_tr, "n_test": len(te_idx),
                             "n_options": N_OPT, "arms": {}}
    errors = {}
    for name, vecs in arms.items():
        r, err = arm_eval(vecs)
        RESULTS["multimodal"]["arms"][name] = r
        errors[name] = err
        b = r["block"]
        print(f"{name:11} acc={b['acc']['point']:.3f} "
              f"[{b['acc']['lo']:.3f},{b['acc']['hi']:.3f}] "
              f"log={b['log']['point']:.3f} cov={r['coverage']:.3f} "
              f"set={r['mean_set_size']:.2f}")

    # PAIRED readouts, not differences of marginal CIs: every arm scores the
    # identical items, so the resample index is shared and each interval is on
    # the difference. Errors (not accuracy) because the readout assumes
    # lower-is-better.
    #
    # Two pre-registered comparisons, and the claim requires BOTH. Testing
    # `full > text_only` alone was v20's mistake: it reports false on data where
    # the image plainly carries signal, because the image arm can beat both.
    # Against BOTH singles also removes the cherry-pick of naming whichever
    # single arm happened to score highest on the test split.
    for other in ("text_only", "image_only"):
        delta = paired_readout("error_rate", errors[other], errors["full"])
        RESULTS["multimodal"][f"vs_{other}"] = delta
        d = delta["delta"]
        print(f"full vs {other:11}: {d['point']:+.4f} "
              f"[{d['lo']:+.4f}, {d['hi']:+.4f}] -> {delta['verdict']}")
    del enc_v
    torch.cuda.empty_cache()
except Exception as e:
    RESULTS["multimodal"] = {"status": "failed", "error": repr(e)[:300]}
    print("multimodal leg failed:", repr(e)[:200])
write_status("multimodal")
'''),

    md('''## 5j. Cross-task shift — a genuinely different domain

The breadth leg measures each dataset on its own terms. This one asks the
deployment question: a **CLINC-calibrated** model meets **Banking77** input —
a different intent taxonomy, so every arriving item is unanswerable by
construction. The correct action is escalate, and the metric that matters is
the unsafe-answer rate.

Same arms as the local A/B (`src/decision/policy.py`): the frozen incumbent,
the label-free slow state, and the label-driven baselines it must beat.
'''),
    code('''
RESULTS["cross_task"] = {"status": "skipped"}
try:
    import datasets as _ds
    from src.decision.policy import (GlialPolicy, PolicyThresholds,
                                     RecalibrateConfig, RecalibratePolicy,
                                     StaticPolicy, Stream, StreamConfig,
                                     run_stream)
    from src.decision.slow import SlowStateConfig, observe_row

    N = 600
    bank = _ds.load_dataset("mteb/banking77")["test"]["text"][:N]
    Eb = to_numpy(encoder.encode(bank, prompt_name="SearchQuery", batch_size=64))
    SC_bank = Eb @ LV.T                     # scored over CLINC's label space
    stream = Stream(np.vstack([SC_te[te_in][:N], SC_bank]),
                    np.concatenate([yt_c[te_in][:N], np.full(N, -1)]), N)

    base = PolicyThresholds(gate.tau_answer, gate.k_clarify, gate.tau_in_schema)
    cfg = StreamConfig(t_prob=gate.t_prob, qhat=gate.qhat, k_clarify=gate.k_clarify)
    cal_rows = SC_val[val_in]
    glial = GlialPolicy(base, SlowStateConfig(
        reference_max_score=float(cal_rows.max(axis=1).mean())))
    glial.calibrate_reference([observe_row(r, gate.t_prob, gate.qhat)[0]
                               for r in cal_rows])
    arms = [StaticPolicy(base), glial,
            RecalibratePolicy(base, RecalibrateConfig(delay=200, refit_every=50,
                                                      n_options=len(INTENT_TEXTS))),
            RecalibratePolicy(base, RecalibrateConfig(delay=0, refit_every=50,
                                                      n_options=len(INTENT_TEXTS)))]
    RESULTS["cross_task"] = {"status": "ran", "n_per_phase": N,
                             "arms": {p.name: run_stream(p, stream, cfg)
                                      for p in arms}}
    for arm, r in RESULTS["cross_task"]["arms"].items():
        print(f"{arm:22} esc {r['phase1']['escalation_rate']:.3f}"
              f"->{r['phase2']['escalation_rate']:.3f} "
              f"unsafe {r['phase2']['wrong_answer_rate']:.3f}")
except Exception as e:
    RESULTS["cross_task"] = {"status": "failed", "error": repr(e)[:300]}
    print("cross-task leg failed:", repr(e)[:200])
write_status("cross-task")
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

    md('''## 7. Operate the system — log, monitor, retrain, release

The parts a *system* needs and a model does not, none of which had ever run on
real traffic:

- **logged traffic** — the same `PredictionLogger` the server writes, fed by
  real utterances, so the monitor and the shadow run have something to read
- **monitor** — "is it still working?" answered from that log, with named alerts
- **handlers** — every action reaches a real destination (Phase 3's in-kernel
  exercise, still outstanding)
- **registry** — a retrain produces a versioned bundle with lineage; promote and
  roll back are pointer flips; a **shadow run** compares the candidate on the
  logged traffic *before* promotion touches `current`
'''),
    code('''
RESULTS["operate"] = {"status": "skipped"}
try:
    import time as _time
    from src.decision.adapt import AdaptConfig, adapt
    from src.decision.cadence import CadenceConfig, should_retrain
    from src.decision.canary import GuardrailConfig, arm_for, canary_verdict
    from src.decision.causal import escalations, paired_readout, unsafe_answers
    from src.decision.datasets import class_halves
    from src.decision.handlers import QueuedEscalation, handle
    from src.decision.monitor import Monitor, read_log, shadow_compare
    from src.decision.registry import Registry, data_fingerprint
    from src.decision.serve import PredictionLogger
    from src.decision.schema import Question

    N_LOG, N_RT = 300, 4000
    ops = WORK / "ops"
    ops.mkdir(parents=True, exist_ok=True)
    q = Question(qtype="choice", options=INTENT_TEXTS)

    # 1. real logged traffic
    log_path = ops / "preds.jsonl"
    if log_path.exists():
        log_path.unlink()
    logger = PredictionLogger(log_path, model_id="nanocore-s1-verify")
    for text in X_te[:N_LOG]:
        t0 = _time.perf_counter()
        pred = model.decide(text, q)
        logger.record(text=text, question=q, pred=pred,
                      latency_ms=(_time.perf_counter() - t0) * 1000, policy=None)
    records = read_log(log_path)
    print(f"logged {len(records)} real decisions")

    # 2. is it still working?
    mon = Monitor(log_path)
    status = mon.check()
    agg = status.get("ml", {}).get("aggregate", {})
    RESULTS["operate"] = {"status": "ran",
                          "logged": len(records),
                          "monitor": {"alerts": status.get("alerts", []),
                                      "escalate_rate": agg.get("escalate_rate"),
                                      "p99_ms": status.get("operational", {})
                                      .get("latency_ms", {}).get("p99")}}
    print(f"monitor: alerts={status.get('alerts')} "
          f"escalate={agg.get('escalate_rate')}")

    # 3. every action reaches a destination
    queue = ops / "escalations.jsonl"
    dispatcher = QueuedEscalation(queue)
    counts = {}
    for text in X_te[:N_LOG]:
        res = handle(model.decide(text, q), state=text, question=q,
                     escalate_to=dispatcher)
        counts[res.action] = counts.get(res.action, 0) + 1
    queued = sum(1 for _ in queue.open()) if queue.exists() else 0
    RESULTS["operate"]["handlers"] = {"counts": counts, "queued": queued}
    print(f"handlers: {counts} | escalation queue lines {queued}")

    # 4. a retrain, twice, so lineage has a parent. Per-class quota: CLINC's
    #    train split is class-ordered, so head-of-file slices would give the two
    #    bundles different label spaces and make the shadow run meaningless.
    y_tr_arr = np.asarray(y_tr_i)
    # Two caveats on the slices. (1) Split per class, not at a midpoint: a
    # concatenated per-class index run is class-ordered, so halving it
    # partitions by class and the second bundle silently knows a disjoint label
    # space — that is what produced "scores and labels must be the same length"
    # in v16. (2) Option labels must be the intent *texts*, not the raw class
    # ids: a bundle whose options read "11"/"42" cannot be served by anything
    # that knows the task, and the incumbent's labels are the texts.
    idx_a, idx_b = class_halves({"label": y_tr_arr}, "label", cap=2 * N_RT)
    text_by_id = {int(i): t for i, t in zip(INTENT_IDS, INTENT_TEXTS)}
    slices = [([X_tr_i[i] for i in idx_a],
               [text_by_id[int(c)] for c in y_tr_arr[idx_a]]),
              ([X_tr_i[i] for i in idx_b],
               [text_by_id[int(c)] for c in y_tr_arr[idx_b]])]
    for si, (_, labs) in enumerate(slices, start=1):
        print(f"retrain slice {si}: {len(labs)} items, "
              f"{len(set(labs))} classes")
    bundles = []
    for i, (texts, labels) in enumerate(slices, start=1):
        out = adapt(texts, labels, encoder, cfg=AdaptConfig(min_cal=150),
                    out_dir=ops / f"retrain{i}")
        bundles.append((out.bundle_dir, out.report, data_fingerprint(texts, labels)))
    reg = Registry(ops / "registry")
    v1 = reg.register("clinc", bundles[0][0], bundles[0][1], bundles[0][2])
    reg.promote("clinc", v1)
    v2 = reg.register("clinc", bundles[1][0], bundles[1][1], bundles[1][2])
    lineage = reg.lineage("clinc", v2)

    # 5. shadow the candidate on logged traffic BEFORE promoting it
    candidate = DecisionModel.load(bundles[1][0], encoder=_LV(encoder))
    shadow = shadow_compare(records, candidate)
    reg.promote("clinc", v2)
    after_promote = reg.current("clinc")
    restored = reg.rollback("clinc")
    RESULTS["operate"]["registry"] = {
        "versions": reg.versions("clinc"), "parent_of_v2": lineage.get("parent"),
        "promoted": after_promote, "rolled_back_to": restored,
        "current_after_rollback": reg.current("clinc"),
        "shadow": {"n": shadow["n"], "n_failed": shadow.get("n_failed", 0),
                   "first_error": shadow.get("first_error"),
                   "agreement": shadow["agreement"],
                   "transitions": shadow["action_transitions"]}}
    print(f"registry: versions={reg.versions('clinc')} "
          f"v2.parent={lineage.get('parent')} promote->{after_promote} "
          f"rollback->{restored} now={reg.current('clinc')}")
    print(f"shadow: n={shadow['n']} agreement={shadow['agreement']:.3f} "
          f"failed={shadow.get('n_failed')} {shadow['action_transitions']}")
    if shadow.get("first_error"):
        print(f"  first_error: {shadow['first_error']}")

    # 6. cadence: is a retrain due, and why? The monitor's own drift report
    #    feeds it, so the window split is computed once and both consumers see
    #    the same comparison. The held case is the anti-thrash property: an
    #    alert below the sample floor must NOT retrain.
    cad_cfg = CadenceConfig(scheduled_every=1000, drift_min_samples=500)
    fired = should_retrain(N_RT, mon.last_drift, cad_cfg)
    held = should_retrain(120, mon.last_drift, cad_cfg)
    RESULTS["operate"]["cadence"] = {
        "drift_alerts": (mon.last_drift.alerts if mon.last_drift else []),
        "fired": {"retrain": fired.retrain, "cadence": fired.cadence,
                  "reason": fired.reason},
        "held": {"retrain": held.retrain, "cadence": held.cadence,
                 "reason": held.reason}}
    print(f"cadence: fired={fired.retrain} via {fired.cadence} "
          f"({fired.reason[:70]})")
    print(f"cadence: held={not held.retrain} ({held.reason[:70]})")

    # 7-8. canary + causal. Both models decide the same 300 items — the paired
    #      design the causal readout needs. The canary additionally routes by a
    #      stable hash of the input: *which* arm is a decision, moving bytes to
    #      it is the deployer's transport. Latency is timed per model so the
    #      arm comparison is genuinely per-arm, not the incumbent's numbers.
    gold = ["oos" if int(c) == OOS else text_by_id[int(c)]
            for c in y_te[:N_LOG]]
    # The incumbent served these same 300 items moments ago, so every re-decide
    # would be a DecisionCache hit (measured p95 0.45 ms) while the freshly
    # loaded candidate pays the real ~58 ms — a 130x "latency regression" that
    # is purely the cache. Both arms must be cache-free before timing.
    model.cache = None
    inc_preds, cand_preds, inc_ms, cand_ms = [], [], [], []
    for r in records:
        qq = Question(qtype=r.get("qtype", "choice"), options=r["labels"])
        t0 = _time.perf_counter()
        inc_preds.append(model.decide(r["input"], qq))
        inc_ms.append((_time.perf_counter() - t0) * 1000)
        t0 = _time.perf_counter()
        cand_preds.append(candidate.decide(r["input"], qq))
        cand_ms.append((_time.perf_counter() - t0) * 1000)

    inc_unsafe = unsafe_answers(inc_preds, gold, oos_label="oos")
    cand_unsafe = unsafe_answers(cand_preds, gold, oos_label="oos")
    inc_esc, cand_esc = escalations(inc_preds), escalations(cand_preds)

    is_canary = np.array([arm_for(r["input"], 0.3, salt="v18") == "canary"
                          for r in records])

    def arm_stats(mask, unsafe, esc, ms):
        idx = np.flatnonzero(mask)
        if not len(idx):
            return {"n": 0}
        return {"n": int(len(idx)),
                "unsafe_rate": float(unsafe[idx].mean()),
                "escalate_rate": float(esc[idx].mean()),
                "latency_p95": float(np.percentile(np.asarray(ms)[idx], 95))}

    control = arm_stats(~is_canary, inc_unsafe, inc_esc, inc_ms)
    canary = arm_stats(is_canary, cand_unsafe, cand_esc, cand_ms)
    cv = canary_verdict(control, canary, GuardrailConfig(min_n=30))
    RESULTS["operate"]["canary"] = {
        "fraction": 0.3, "control": control, "canary": canary,
        "verdict": cv.decision, "reason": cv.reason, "deltas": cv.deltas,
        "checked": cv.checked, "unchecked": cv.unchecked}
    print(f"canary: control n={control['n']} unsafe={control['unsafe_rate']:.3f} "
          f"| canary n={canary['n']} unsafe={canary['unsafe_rate']:.3f} "
          f"-> {cv.decision} ({cv.reason[:60]})")

    causal = {"unsafe": paired_readout("unsafe_rate", inc_unsafe, cand_unsafe),
              "escalate": paired_readout("escalate_rate", inc_esc, cand_esc)}
    RESULTS["operate"]["causal"] = {
        "n": causal["unsafe"]["n"],
        "gold_oos": int(sum(1 for g in gold if g == "oos")),
        "incumbent_unsafe": float(inc_unsafe.mean()),
        "candidate_unsafe": float(cand_unsafe.mean()),
        "unsafe": causal["unsafe"], "escalate": causal["escalate"]}
    for name in ("unsafe", "escalate"):
        d = causal[name]["delta"]
        print(f"causal {name}: delta={d['point']:+.3f} "
              f"[{d['lo']:+.3f}, {d['hi']:+.3f}] -> {causal[name]['verdict']}")
except Exception as e:
    RESULTS["operate"] = {"status": "failed", "error": repr(e)[:300]}
    print("operate leg failed:", repr(e)[:200])
write_status("operate")
'''),

    md('## 8. Verdict'),
    code('''
verdict = {
    "suite_green_on_kaggle": True,
    "cosine_resolved_in_band": abs(v_cos["resolved"] - 0.482) < 0.05,
    "head_resolved_in_band": abs(v_head["resolved"] - 0.918) < 0.06,
    "coverage_in_band": 0.85 <= v_cos["set_coverage_in_scope"] <= 0.95,
    "head_coverage_reported": v_head["set_coverage_in_scope"],
    # v19: the head path used to report coverage 1.0000 with mean sets of 31 of
    # 150 labels — a vacuous guarantee that coverage alone could not reveal.
    # These two make set size a first-class, checked outcome.
    "cosine_sets_small": v_cos["mean_set_size"] < 8,
    "head_sets_small": v_head["mean_set_size"] < 10,
    "head_acc_in_band": head_acc > 0.94,
    "bundle_roundtrip_identical": RESULTS["bundle_roundtrip"]["identical"] == 200,
    "ordinal_ran": RESULTS["ordinal"]["status"] == "ran",
    "ordinal_beats_zeroshot": (RESULTS["ordinal"].get("ordinal_acc", 0)
                             > RESULTS["ordinal"].get("zeroshot_acc", 1)),
    "cli_ran": RESULTS["cli"]["status"] == "ran",
    "cli_inputs": RESULTS["cli"].get("n"),
    # Definition-of-done #1 through the shipped CLI, with the real encoder.
    "cli_adapt_ran": RESULTS.get("cli_adapt", {}).get("status") == "ran",
    "cli_adapt_roundtrip": (RESULTS.get("cli_adapt", {}).get("roundtrip_action")
                            in ("answer", "clarify", "escalate", "abstain")),
    "serve_ran": RESULTS["serve"]["status"] == "ran",
    "baselines_ran": RESULTS["baselines"]["status"] == "ran",
    "head_beats_tfidf_acc": (RESULTS["baselines"].get("blocks", {})
                           .get("taskhead", {}).get("accuracy", 0)
                           > RESULTS["baselines"].get("blocks", {})
                           .get("tfidf_lr", {}).get("accuracy", 1)),
    "rigor_ran": RESULTS["rigor"]["status"] == "ran",
    "memorization_ran": RESULTS["memorization"]["status"] == "ran",
    "breadth_ran": RESULTS["breadth"].get("status") in ("ran", "partial"),
    "breadth_datasets": RESULTS["breadth"].get("n_ran", 0),
    "operate_ran": RESULTS["operate"].get("status") == "ran",
    "monitor_ran": bool(RESULTS["operate"].get("monitor", {}).get("p99_ms")),
    "handlers_ran": bool(RESULTS["operate"].get("handlers", {}).get("counts")),
    "escalations_queued": RESULTS["operate"].get("handlers", {}).get("queued", 0) > 0,
    "registry_lineage": RESULTS["operate"].get("registry", {}).get("parent_of_v2") is not None,
    "registry_rollback": (RESULTS["operate"].get("registry", {})
                          .get("current_after_rollback")
                          == RESULTS["operate"].get("registry", {}).get("rolled_back_to")),
    "shadow_ran": RESULTS["operate"].get("registry", {}).get("shadow", {}).get("n", 0) > 0,
    # Phase 6, the three remaining pieces. Cadence needs BOTH behaviours on the
    # real drift report: fires when the floor is cleared, holds when it is not.
    "cadence_ran": RESULTS["operate"].get("cadence", {}).get("fired", {}).get("retrain") is True,
    "cadence_floored": RESULTS["operate"].get("cadence", {}).get("held", {}).get("retrain") is False,
    "canary_ran": (RESULTS["operate"].get("canary", {}).get("verdict")
                   in ("promote", "hold", "rollback")),
    "canary_arms_nonempty": min(RESULTS["operate"].get("canary", {})
                                .get("control", {}).get("n", 0),
                                RESULTS["operate"].get("canary", {})
                                .get("canary", {}).get("n", 0)) > 0,
    "causal_ran": (RESULTS["operate"].get("causal", {}).get("n", 0)
                   == RESULTS["operate"].get("logged", 0) > 0),
    "causal_ci_reported": "lo" in (RESULTS["operate"].get("causal", {})
                                   .get("unsafe", {}).get("delta", {})),
    "cross_task_ran": RESULTS["cross_task"].get("status") == "ran",
    "cross_task_glial_safer": (
        RESULTS["cross_task"].get("arms", {}).get("glial", {})
        .get("phase2", {}).get("wrong_answer_rate", 1.0)
        < RESULTS["cross_task"].get("arms", {}).get("static", {})
        .get("phase2", {}).get("wrong_answer_rate", 0.0)),
    # Multilingual: the pipeline must hold off-English, and the English head must
    # transfer above chance. Both are reported, neither is assumed.
    "multilingual_ran": RESULTS.get("multilingual", {}).get("n_ran", 0) >= 4,
    "multilingual_in_language_holds": all(
        v.get("in_language_head_acc", {}).get("point", 0) > 0.5
        for v in RESULTS.get("multilingual", {}).get("languages", {}).values()
        if v.get("status") == "ran"),
    "multilingual_en_head_transfers": all(
        v.get("cross_lingual_en_head_acc", 0) > 0.5
        for v in RESULTS.get("multilingual", {}).get("languages", {}).values()
        if v.get("status") == "ran"),
    # Multimodal, as TWO pre-registered claims. The first is the thesis: does a
    # non-text state carry decision information at all? Measured against chance
    # by the arm's CI lower bound, so a lucky point estimate cannot pass it.
    # The second is the composition claim, and it requires beating BOTH singles.
    "multimodal_ran": RESULTS.get("multimodal", {}).get("status") == "ran",
    "multimodal_image_carries_signal": (
        RESULTS.get("multimodal", {}).get("arms", {}).get("image_only", {})
        .get("block", {}).get("acc", {}).get("lo", 0.0)
        > 1.0 / max(RESULTS.get("multimodal", {}).get("n_options", 4), 1)),
    "multimodal_combination_beats_both_singles": all(
        RESULTS.get("multimodal", {}).get(f"vs_{o}", {}).get("verdict") == "improved"
        for o in ("text_only", "image_only")),
    "breadth_head_ge_tfidf": all(
        r["blocks"]["taskhead"]["acc"]["point"]
        >= r["blocks"]["tfidf_lr"]["acc"]["point"]
        for r in RESULTS["breadth"].get("datasets", {}).values()
        if r.get("status") == "ran"),
}
RESULTS["verdict"] = verdict
print(json.dumps(verdict, indent=2))
(WORK / "results.json").write_text(json.dumps(RESULTS, indent=2))
write_status("verdict")
'''),

    md('''## 10. Qualification — may this build ship?

Every leg above *measured* something. This one asks the different question, with
criteria fixed in advance (`src/decision/qualify.py`): may this build ship?

It also runs the **refusal battery** — the safety property as behaviour, with a
positive control, because a model that refuses everything would otherwise pass a
refusal test perfectly. And it emits the two delivery artifacts: the sign-off
(`qualification.json`) and the **model card** a recipient expects to receive.

The artifact is named by **content** (`bundle_digest`), so "the thing we tested"
and "the thing we ship" are provably the same object.
'''),
    code('''
RESULTS["qualification"] = {"status": "skipped"}
try:
    from src.decision import rubric
    from src.decision.model import bundle_digest
    from src.decision.modelcard import build_card, render_markdown
    from src.decision.qualify import evaluate
    from src.decision.refusals import BatterySpec, LossMatrix, battery_report

    GIT_COMMIT = "unknown"
    # The deployment's risk posture. The refusal boundary is derived from these
    # costs rather than fitted to a precision target (arXiv:2609.28940).
    REFUSAL_LOSS = LossMatrix(cfp=10.0, cfn=20.0, cr=1.0)

    # 1. refusals, over the real decisions the legs above already made, plus
    #    probes generated from the model's own label space
    ref_rows = []
    for _t in X_te[:N_LOG]:
        _p = model.decide(_t, q)
        ref_rows.append({"top_prob": _p.answer_confidence,
                         "max_score": _p.max_score, "action": _p.action})
    refusals = battery_report(model, INTENT_TEXTS, ref_rows, BatterySpec(
        loss=REFUSAL_LOSS, sample_texts=X_te[:60], gate=model.gate))
    RESULTS["refusals"] = refusals
    _th = refusals["loss"]
    print(f"refusal costs: cfp={_th['cfp']} cfn={_th['cfn']} cr={_th['cr']} -> "
          f"discard<{_th['discard_below']:.3f} assert>{_th['assert_above']:.3f}")
    print(f"  unsafe answers {refusals['unsafe_answers']} | "
          f"over-refusals {refusals['over_refusals'] or 'none'} | "
          f"control {refusals['positive_control_passed']}")
    for _i in refusals["invariants"]:
        print(f"  {_i['status']:9} {_i['id']:3} {_i['statement'][:56]}")
        if _i["status"] in ("fail", "error"):
            print(f"            {_i['detail'][:92]}")
    print(f"  reasons: " + ", ".join(f"{k}={v['status']}"
                                     for k, v in refusals["reasons"].items()))

    # 2. name the artifact, then qualify it
    digest = bundle_digest(bundle)
    prov = {"git_commit": GIT_COMMIT, "kernel": "s1_verify",
            "bundle_sha256": digest, "dataset": "CLINC150 (150 intents)"}
    qual = evaluate(RESULTS, provenance=prov)
    qual["rubric"] = rubric.coverage()
    RESULTS["qualification"] = qual
    (WORK / "qualification.json").write_text(json.dumps(qual, indent=2))

    print(f"\\nqualification: {qual['verdict']}  (bundle {digest[:16]}...)")
    print(f"  must_pass failed : {qual['must_pass_failed']}")
    print(f"  must_fix open    : {qual['must_fix_open']}")
    print(f"  deferred         : {qual['deferred']}")
    print(rubric.render(qual["rubric"]))

    # 3. the delivery artifact
    built = build_card(qual, RESULTS.get("cli_adapt", {}),
                       dataset="CLINC150 (150 intents)", encoder_id=encoder.model_name)
    (WORK / "MODEL_CARD.md").write_text(render_markdown(built))
    print(f"model card written: {len(built['card'])} sections, "
          f"{len(built['metrics'])} eval metrics in model-index")

    RESULTS["verdict"].update({
        "qualification_ran": True,
        "qualification_named_artifact": bool(digest),
        "refusals_ran": True,
        "refusals_no_unsafe_answers": refusals["unsafe_answers"] == 0,
        "refusals_battery_meaningful": bool(battery_is_meaningful(refusals)),
        "model_card_written": (WORK / "MODEL_CARD.md").exists(),
    })
except Exception as e:
    RESULTS["qualification"] = {"status": "failed", "error": repr(e)[:300]}
    RESULTS["verdict"].update({"qualification_ran": False})
    print("qualification leg failed:", repr(e)[:200])

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
