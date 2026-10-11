"""Regional-error clustering probe — ADR-0014's first cheap evidence.

Premise under test: the glial regulator's "regional competence field" only has
something to learn if decision errors cluster in *regions* of embedding space
rather than spreading uniformly. If errors are spatially diffuse, a regional
regulator degenerates to the global threshold nudge we already ship.

Method (pure array logic, no model loading — runs locally on cached matrices):
  1. errors = argmax(scores) != y on the val split
  2. for each error point, local error rate among its k nearest cosine neighbors
  3. permutation test: shuffle the error mask, compare the observed local error
     rate against the null distribution
  4. spatial cohesion: mean cosine similarity between error pairs vs random pairs

Usage: python scripts/regional_error_probe.py reports/runs/v26/verify_scores.npz
"""
import sys
import numpy as np


def main(path: str, k: int = 10, n_perm: int = 1000, seed: int = 0) -> int:
    d = np.load(path)
    emb = d["emb_val"].astype(np.float64)
    scores = d["scores_val"]
    y = d["y_val"]

    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    err = scores.argmax(axis=1) != y
    n_err = int(err.sum())
    base_rate = n_err / len(err)
    print(f"n={len(err)}  errors={n_err}  base_error_rate={base_rate:.4f}")

    # nearest neighbors by cosine similarity (exclude self)
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]

    def local_err_rate(mask):
        """mean fraction of errors among kNN of error points"""
        return mask[nn[mask]].mean()

    obs = local_err_rate(err)
    print(f"local error rate around errors (k={k}): {obs:.4f}  vs baseline {base_rate:.4f}  lift {obs/base_rate:.2f}x")

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = np.zeros(len(err), dtype=bool)
        perm[rng.choice(len(err), n_err, replace=False)] = True
        null[i] = local_err_rate(perm)
    p = (null >= obs).mean()
    print(f"permutation test ({n_perm} shuffles): null mean {null.mean():.4f} ± {null.std():.4f}  p={p:.4f}")

    # spatial cohesion: do error points sit nearer each other than chance?
    err_idx = np.where(err)[0]
    if n_err > 2:
        pe = sim[np.ix_(err_idx, err_idx)]
        iu = np.triu_indices_from(pe, k=1)
        err_pair = pe[iu].mean()
        i, j = np.triu_indices(len(err), k=1)
        rng2 = np.random.default_rng(seed)
        sub = rng2.choice(len(i), min(20000, len(i)), replace=False)
        all_pair = sim[i[sub], j[sub]].mean()
        print(f"mean cosine sim — error pairs {err_pair:.4f} vs all pairs {all_pair:.4f}")

    # does clustering survive *within* the predicted class?
    # (errors obviously concentrate in hard classes; the field claim is finer)
    pred = scores.argmax(axis=1)
    lifts = []
    for c in np.unique(pred[err]):
        m = pred == c
        e = err & m
        if e.sum() < 5 or (~e & m).sum() < 5:
            continue
        sub_nn = np.argsort(-np.where(m[None, :], sim, -np.inf).max(axis=0))[:0]  # unused
        # neighbors restricted to same predicted class
        sim_c = np.where(m[None, :], sim, -np.inf)
        nn_c = np.argpartition(-sim_c, min(k, m.sum() - 1), axis=1)[:, :k]
        lifts.append((c, err[nn_c[e]].mean(), err[m].mean(), int(e.sum())))
    lifts.sort(key=lambda t: -t[3])
    within = [t for t in lifts if t[1] > t[2] * 1.5]
    print(f"within-class lift>1.5x in {len(within)}/{len(lifts)} classes with >=5 errors")
    for c, l, b, n in within[:10]:
        print(f"   class {c}: local {l:.3f} vs class base {b:.3f} ({n} errors)")

    verdict = "SUPPORTS regional field" if p < 0.05 and obs > base_rate * 1.2 else \
              "DOES NOT support regional structure" if p >= 0.05 else "WEAK clustering"
    print(f"\nverdict: {verdict} (p={p:.4f}, lift={obs/base_rate:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "reports/runs/v26/verify_scores.npz"))
