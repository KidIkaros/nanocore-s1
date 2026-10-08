"""Evaluation legs — the computations kernels call, tested locally.

Why this module exists: three kernel pushes failed on logic that lived only
inside notebook cells (a wrong import module, a list masked by a bool array,
label-column misalignment between sklearn's ``classes_`` and the intent
columns). Notebook cells are orchestration; anything that computes belongs
here, where the test suite can run it on synthetic arrays before a GPU
session is spent.

Each function takes plain arrays/texts and returns JSON-able dicts.
"""
from __future__ import annotations

from numbers import Real
from typing import Dict, List, Sequence

import numpy as np

from src.decision import metrics as M
from src.decision.scoring import softmax_rows


def tfidf_baseline(train_texts: Sequence[str], train_cols: np.ndarray,
                   test_texts: Sequence[str], n_classes: int) -> np.ndarray:
    """TF-IDF + logistic-regression probabilities aligned to column ids.

    ``train_cols`` are *column* ids (0..n_classes-1), not dataset label ids —
    the kernel's earlier bug was fitting on dataset ids and misaligning the
    probability columns. Classes absent from the train split get uniform mass
    in the output row (renormalized), so the returned matrix always has shape
    ``(n_test, n_classes)``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train_cols = np.asarray(train_cols, dtype=int)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit(list(train_texts))
    lr = LogisticRegression(max_iter=1000, C=4.0).fit(
        tf.transform(list(train_texts)), train_cols)
    raw = lr.predict_proba(tf.transform(list(test_texts)))
    P = np.full((len(test_texts), n_classes), 1e-9)
    for col, cid in enumerate(lr.classes_):
        P[:, int(cid)] = raw[:, col]
    return P / P.sum(axis=1, keepdims=True)


def rigor_block(P: np.ndarray, y_idx: np.ndarray,
                resamples: int = 500, seed: int = 0) -> Dict:
    """The reference-eval metric block with uncertainty on it.

    Bootstrap CIs on accuracy and log score, AURC, selective accuracy at
    50%/80% coverage, ECE over 15 bins, Brier. Binary tasks additionally
    report AUROC and AUPRC (threshold-free, per the eval standard). Every
    number the comparison tables print comes from here so columns stay
    comparable across scorers.
    """
    P = np.asarray(P, dtype=np.float64)
    y_idx = np.asarray(y_idx, dtype=int)
    onehot = np.eye(P.shape[1])[y_idx]
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y_idx).astype(float)
    out = {
        "acc": M.bootstrap_ci(lambda a: a.mean(), correct,
                              resamples=resamples, seed=seed),
        "log": M.bootstrap_ci(lambda p, y: M.log_score(p, np.eye(p.shape[1])[y]),
                              P, y_idx, resamples=resamples, seed=seed),
        "aurc": M.aurc(conf, correct),
        "acc_at_50": M.selective_at_coverage(conf, correct, 0.5),
        "acc_at_80": M.selective_at_coverage(conf, correct, 0.8),
        "ece15": M.expected_calibration_error(P, onehot, n_bins=15),
        "brier": M.brier_score(P, onehot),
    }
    if P.shape[1] == 2:
        from sklearn.metrics import roc_auc_score, average_precision_score
        out["auroc"] = float(roc_auc_score(y_idx, P[:, 1]))
        out["auprc"] = float(average_precision_score(y_idx, P[:, 1]))
    return out


def seed_aggregate(runs: Sequence[Dict]) -> Dict:
    """Mean and spread of every numeric leaf across seeds.

    The verdicts are read off these means — one seed is luck. Leaves are
    discovered from the data rather than listed, so a metric added to a run
    cannot be silently left out of the aggregate.

    Non-numeric leaves (a policy name, an unreached ``recovery_latency``) are
    dropped: there is nothing to average.
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    out: Dict = {}
    for key, value in runs[0].items():
        if isinstance(value, dict):
            out[key] = seed_aggregate([run[key] for run in runs])
        elif isinstance(value, Real) and not isinstance(value, bool):
            values = [float(run[key]) for run in runs]
            out[key] = {"mean": float(np.mean(values)), "sd": float(np.std(values))}
    return out


def similarity_bands(test_vecs: np.ndarray, train_vecs: np.ndarray,
                     preds: np.ndarray, targets: np.ndarray,
                     bands=((-1, .8), (.8, .9), (.9, .95), (.95, 1.01))) -> Dict:
    """Accuracy by nearest-train-neighbor similarity — the memorization probe.

    If the model memorized, accuracy collapses on examples far from train.
    A flat or gently declining curve means the head generalized.
    """
    S_te = np.asarray(test_vecs); S_tr = np.asarray(train_vecs)
    sims = (S_te @ S_tr.T).max(axis=1)
    preds = np.asarray(preds); targets = np.asarray(targets)
    out = {}
    for lo, hi in bands:
        m = (sims >= lo) & (sims < hi)
        if int(m.sum()) >= 10:
            out[f"[{lo},{hi})"] = {"n": int(m.sum()),
                                   "acc": float((preds[m] == targets[m]).mean())}
    return out


def dataset_suite(train_vecs: np.ndarray, y_tr: np.ndarray,
                  cal_vecs: np.ndarray, y_cal: np.ndarray,
                  test_vecs: np.ndarray, y_te: np.ndarray,
                  train_texts: Sequence[str], test_texts: Sequence[str],
                  label_vecs: np.ndarray, n_classes: int,
                  alpha: float = 0.10, seed: int = 0,
                  resamples: int = 200) -> Dict:
    """One breadth dataset through the full comparison.

    Three probability matrices on the identical test split —
    zero-shot cosine (``label_vecs``), a fitted linear ``TaskHead``, and
    TF-IDF+LR — each wrapped in a ``rigor_block``, plus the head's
    memorization bands. Gate temperatures are calibrated on ``cal_*``,
    disjoint from fit and test, matching the three-way-split rule.

    Embeddings are injected precomputed: encoding happens in the kernel
    (encoder lives there); this function is pure array math and is what
    the local test suite exercises.
    """
    from src.decision.gate import ConformalGate
    from src.decision.scoring import TaskHead

    X_tr, X_cal, X_te = map(lambda a: np.asarray(a, dtype=np.float64),
                            (train_vecs, cal_vecs, test_vecs))
    y_tr, y_cal, y_te = (np.asarray(y, dtype=int) for y in (y_tr, y_cal, y_te))
    LV = np.asarray(label_vecs, dtype=np.float64)

    g_cos = ConformalGate(alpha=alpha, min_n=100)
    g_cos.calibrate(X_cal @ LV.T, y_cal, seed=seed)
    P_cos = softmax_rows(X_te @ LV.T, g_cos.t_prob)

    head = TaskHead(kind="linear")
    head.fit(X_tr, y_tr, labels=[str(i) for i in range(n_classes)], seed=seed)
    g_head = ConformalGate(alpha=alpha, min_n=100)
    g_head.calibrate(head.logits(X_cal), y_cal, seed=seed)
    head.temperature = g_head.t_prob
    P_head = softmax_rows(head.logits(X_te), g_head.t_prob)

    P_tfidf = tfidf_baseline(train_texts, y_tr, test_texts, n_classes)

    blocks = {"cosine": rigor_block(P_cos, y_te, resamples, seed),
              "taskhead": rigor_block(P_head, y_te, resamples, seed),
              "tfidf_lr": rigor_block(P_tfidf, y_te, resamples, seed)}
    return {
        "blocks": blocks,
        "memorization": similarity_bands(X_te, X_tr,
                                         P_head.argmax(axis=1), y_te),
        "calibration": {"cosine_t": g_cos.t_prob, "head_t": g_head.t_prob},
    }
