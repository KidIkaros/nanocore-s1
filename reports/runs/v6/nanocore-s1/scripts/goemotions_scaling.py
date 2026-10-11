#!/usr/bin/env python3
"""GoEmotions data-scaling curve — where does a task-fitted head start earning its keep?

s1_goemotions_head showed a head trained on the full 43k train split beats per-label
Platt by +0.168 macro-F1. It said nothing about the curve: does the head still win with
1k labeled examples? 400? This script subsamples the *cached real* embeddings
(reports/runs/s1_goemotions_head/goemotions_embeddings.npz — no re-encoding, no model
loading) and refits every arm at each size, 3 seeds.

Local-safe by the compute boundary: head training on cached embeddings is ~100MB work —
no weights are loaded. GPU is used if present for speed.

Protocol (matches the kernel): fit on train subsample → early-stop + threshold-tune on
dev → evaluate once on test. Platt is fitted on the same subsample for a fair curve;
the dev-fitted incumbent (0.287 macro) is printed as the reference line.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "reports/runs/s1_goemotions_head/goemotions_embeddings.npz"
OUT = ROOT / "reports/runs/s1_goemotions_scaling"
SIZES = [100, 400, 1000, 4000, 16000, None]   # None = full train
SEEDS = (0, 1, 2)
EPOCHS, BS = 60, 256


def make_head(kind, d_in, L, hidden=256):
    if kind == "linear":
        return nn.Linear(d_in, L)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Dropout(0.1), nn.Linear(hidden, L))


def train_head(X_tr, Y_tr, X_dev, Y_dev, kind, seed, dev, lr):
    torch.manual_seed(seed)
    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True) + 1e-6
    Xt = torch.tensor((X_tr - mu) / sd, dtype=torch.float32, device=dev)
    Yt = torch.tensor(Y_tr, dtype=torch.float32, device=dev)
    Xv = torch.tensor((X_dev - mu) / sd, dtype=torch.float32, device=dev)
    Yv = torch.tensor(Y_dev, dtype=torch.float32, device=dev)
    pos = Y_tr.sum(0)
    neg = len(Y_tr) - pos
    pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, 30.0),
                      dtype=torch.float32, device=dev)
    head = make_head(kind, X_tr.shape[1], Y_tr.shape[1]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state, patience = np.inf, None, 0
    for _ in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            loss = lossf(head(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
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

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(head(
                torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev))).cpu().numpy()
    return predict


def platt_fit(sc, y, L):
    models = []
    for l in range(L):
        x, yv = sc[:, l].reshape(-1, 1), y[:, l]
        if yv.min() == yv.max():
            models.append(("const", float(yv[0])))
            continue
        models.append(("platt", LogisticRegression(C=10.0, max_iter=500).fit(x, yv)))
    return models


def platt_apply(models, sc):
    P = np.zeros_like(sc)
    for l, (kind, m) in enumerate(models):
        P[:, l] = m if kind == "const" else m.predict_proba(sc[:, l].reshape(-1, 1))[:, 1]
    return P


def best_f1_threshold(probs, y):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 49):
        f = f1_score(y, probs >= t, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t


def evaluate(P_dev, P_te, Y_dev, Y_te):
    L = P_te.shape[1]
    tuned = np.array([best_f1_threshold(P_dev[:, l], Y_dev[:, l]) for l in range(L)])
    pred = P_te >= tuned
    auroc = [roc_auc_score(Y_te[:, l], P_te[:, l])
             for l in range(L) if Y_te[:, l].min() < Y_te[:, l].max()]
    return {"macro_f1": float(f1_score(Y_te, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(Y_te, pred, average="micro", zero_division=0)),
            "auroc": float(np.mean(auroc))}


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}", flush=True)
    d = np.load(NPZ, allow_pickle=True)
    S_tr, S_dev, S_te = d["emb_train"], d["emb_dev"], d["emb_test"]
    SC_tr, SC_dev, SC_te = d["scores_train"], d["scores_dev"], d["scores_test"]
    Y_tr, Y_dev, Y_te = d["gold_train"], d["gold_dev"], d["gold_test"]
    L = Y_tr.shape[1]
    rng = np.random.default_rng(7)

    arms = [("platt", None), ("linear_scores", (SC_tr, SC_dev, SC_te, "linear", 3e-3)),
            ("mlp_scores", (SC_tr, SC_dev, SC_te, "mlp", 1e-3)),
            ("linear_emb", (S_tr, S_dev, S_te, "linear", 1e-3)),
            ("mlp_emb", (S_tr, S_dev, S_te, "mlp", 1e-3))]
    results = {"sizes": [s if s else len(S_tr) for s in SIZES], "arms": {}}
    t0 = time.time()

    for size in SIZES:
        n = size if size else len(S_tr)
        key = str(n)
        print(f"\n=== n={n} ===", flush=True)
        for name, spec in arms:
            runs = []
            for seed in SEEDS:
                sub = rng.choice(len(S_tr), size=n, replace=False)
                if spec is None:
                    models = platt_fit(SC_tr[sub], Y_tr[sub], L)
                    row = evaluate(platt_apply(models, SC_dev),
                                   platt_apply(models, SC_te), Y_dev, Y_te)
                else:
                    Xa_tr, Xa_dev, Xa_te, kind, lr = spec
                    predict = train_head(Xa_tr[sub], Y_tr[sub], Xa_dev, Y_dev,
                                         kind, seed, dev, lr)
                    row = evaluate(predict(Xa_dev), predict(Xa_te), Y_dev, Y_te)
                runs.append(row)
            mean = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            sd = {k: float(np.std([r[k] for r in runs])) for k in runs[0]}
            results["arms"].setdefault(name, {})[key] = {"mean": mean, "std": sd,
                                                        "runs": runs}
            print(f"  {name:<15} macro {mean['macro_f1']:.3f}±{sd['macro_f1']:.3f} "
                  f"| micro {mean['micro_f1']:.3f} | auroc {mean['auroc']:.3f}",
                  flush=True)

    results["meta"] = {"source": str(NPZ), "seeds": list(SEEDS), "sizes": SIZES,
                       "protocol": "fit on train subsample; dev early-stop + thresholds; "
                                   "test once", "wall_s": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'results.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
