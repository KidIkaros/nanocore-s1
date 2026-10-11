"""The pre-registered three-part gate across the decision anchors (#10).

Decision win (proper scoring rule = Brier) + calibration (ECE) + risk-coverage
(AURC / selective prediction), per anchor. banking77 + emotion read the v6
Kaggle breadth leg's per-example blocks; clinc150 re-runs on the cached v6
embeddings (held-out, in-schema). The verdict decides which anchors flip from
cosine to the trained head. Calibration leg uses first-cut equal-width ECE
here; ticket #11 adds the debiased equal-mass estimator + asymptotic CI.

Reproducible from cached artifacts alone — no encoder load, deterministic seed.
"""
import json
import numpy as np
from src.decision.scoring import TaskHead, softmax_rows, aps_members
from src.decision.gate import ConformalGate

ECE_BAR = 0.05


def _clinc150_cached():
    """clinc150 from cached v6 embeddings — the deep rerun (held-out)."""
    d = np.load("reports/runs/v6/verify_scores.npz", allow_pickle=True)
    Etr, ytr = d["emb_train"], d["y_train"]
    Eval, yval = d["emb_val"], d["y_val"]
    LV = d["label_vecs"]
    K = 150
    mtr = ytr < K
    Etr, ytr = Etr[mtr], ytr[mtr]
    mval = yval < K
    Eval, yval = Eval[mval], yval[mval]
    half = len(Eval) // 2
    Ecal, ycal, Ete, yte = Eval[:half], yval[:half], Eval[half:], yval[half:]

    def unit(M):
        M = np.asarray(M, dtype=np.float64)
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    Ste_cos = unit(Ete) @ unit(LV).T
    g = ConformalGate(alpha=0.10, min_n=100)
    g.calibrate(unit(Ecal) @ unit(LV).T, ycal, seed=0)
    cos_zs = float((Ste_cos.argmax(1) == yte).mean())
    mem = aps_members(softmax_rows(Ste_cos, g.t_set), g.qhat)
    ans = np.array([yte[i] in mem[i] for i in range(len(mem))])
    cos_g = float((Ste_cos.argmax(1)[ans] == yte[ans]).mean()) if ans.any() else 0.0

    head = TaskHead(kind="linear")
    head.fit(Etr, ytr, labels=[str(i) for i in range(K)], seed=0)
    Ste_h = head.logits(Ete)
    head_zs = float((Ste_h.argmax(1) == yte).mean())
    g2 = ConformalGate(alpha=0.10, min_n=100)
    g2.calibrate(head.logits(Ecal), ycal, seed=0)
    m2 = aps_members(softmax_rows(Ste_h, g2.t_set), g2.qhat)
    a2 = np.array([yte[i] in m2[i] for i in range(len(m2))])
    head_g = float((Ste_h.argmax(1)[a2] == yte[a2]).mean()) if a2.any() else 0.0
    return {"cosine": {"zs": cos_zs, "gated": cos_g, "cov": float(ans.mean())},
            "head": {"zs": head_zs, "gated": head_g, "cov": float(a2.mean())}}


def _breadth_leg():
    r = json.load(open("reports/runs/v6/results.json"))
    return r["breadth"]["datasets"]


def main():
    print("=" * 70)
    print("THREE-PART GATE — per-anchor verdict (real EG2 embeddings)")
    print("=" * 70)
    br = _breadth_leg()
    for name in ["banking77", "emotion"]:
        cos, head = br[name]["blocks"]["cosine"], br[name]["blocks"]["taskhead"]
        brier_d = cos["brier"]["point"] - head["brier"]["point"]
        acc_d = head["acc"]["point"] - cos["acc"]["point"]
        print(f"\n### {name}")
        print(f"  (a) Brier head {head['brier']['point']:.4f} vs cosine "
              f"{cos['brier']['point']:.4f} -> {brier_d:+.4f} | acc {acc_d:+.4f}")
        print(f"  (b) ECE head {head.get('ece15')} (bar {ECE_BAR}, first-cut)")
        print(f"  (c) AURC head {head.get('aurc')} vs cosine {cos.get('aurc')}")
        print(f"  VERDICT: FLIP TO HEAD" if brier_d > 0 else "  VERDICT: KEEP COSINE")
    c = _clinc150_cached()
    print(f"\n### clinc150 (cached rerun)")
    print(f"  head zs {c['head']['zs']:.4f} vs cosine {c['cosine']['zs']:.4f} "
          f"-> {c['head']['zs']-c['cosine']['zs']:+.4f}")
    print(f"  head gated {c['head']['gated']:.4f} (cov {c['head']['cov']:.3f}) vs "
          f"cosine {c['cosine']['gated']:.4f} (cov {c['cosine']['cov']:.3f})")
    print(f"  VERDICT: FLIP TO HEAD")
    print("\nCalibration leg pending #11 (debiased equal-mass ECE + asymptotic CI).")


if __name__ == "__main__":
    main()
