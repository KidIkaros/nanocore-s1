"""PCA error-submanifold probe — glial-regulator cost question (probe #2).

ADR-0014's field is a *regional competence field* over EG2 space. Before
sizing a learned field, ask whether error embeddings collapse onto a
low-rank submanifold: if errors live in few effective dimensions, the
field's learned form is cheap (a parametric GMM over error locations, not
a corpus-sized kNN). If errors are as diffuse as correct points, the
field must stay corpus-sized.

Pure array logic on cached embeddings — no model load.

Usage: python scripts/pca_error_submanifold.py reports/runs/v27/artifacts/verify_scores.npz
"""
import sys
import numpy as np


def effective_rank(var: np.ndarray) -> float:
    """Participation ratio of a normalized variance spectrum."""
    return float(1.0 / (var ** 2).sum())


def spectrum(E: np.ndarray) -> np.ndarray:
    Ec = E - E.mean(0)
    s = np.linalg.svd(Ec, compute_uv=False)
    v = s ** 2
    return v / v.sum()


def main(path: str) -> int:
    d = np.load(path)
    emb = d["emb_val"].astype(np.float64)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    y = d["y_val"]
    # column-space gold if present, else raw (oos last => identity)
    y_col = d["y_col"] if "y_col" in d else y
    scores = d["scores_val"]
    in_scope = y < 150 if "intent_ids" not in d else np.ones(len(y), bool)
    err = (scores.argmax(1) != y_col) & in_scope

    E_err, E_ok = emb[err], emb[in_scope & ~err]
    print(f"n={len(emb)}  errors={err.sum()}  correct={(in_scope & ~err).sum()}")
    ve, vo = spectrum(E_err), spectrum(E_ok)
    for k in (1, 5, 10, 25, 50):
        print(f"top-{k:2d} variance: errors {ve[:k].sum():.3f}  "
              f"correct {vo[:k].sum():.3f}")
    re_, ro_ = effective_rank(ve), effective_rank(vo)
    print(f"effective rank (of {emb.shape[1]}): errors {re_:.1f}  "
          f"correct {ro_:.1f}")

    cheap = re_ < ro_ * 0.7
    print()
    if cheap:
        print("verdict: errors are LOWER-rank — a parametric field is viable")
    else:
        print("verdict: errors are NOT lower-rank (diffuse >= correct) — "
              "the field must stay corpus-sized; no cheap parametric form")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else "reports/runs/v27/artifacts/verify_scores.npz"))
