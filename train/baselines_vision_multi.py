#!/usr/bin/env python3
"""Per-question linear probes for the FracAtlas multi-question experiment.

    python train/baselines_vision_multi.py --encoder medsiglip --out-dir runs/fracatlas/multiq/lr-medsiglip

One standardised, balanced-class logistic regression per question (multinomial for questions with
more than two options), ``C`` chosen on val AUROC, fit on train only. This is the strongest simple
baseline for the questions it was trained on; unlike a CLM head it cannot answer a question it has
no labels for. Writes ``<question>.npz`` (``mean``, ``scale``, ``coef`` [K or 1, d], ``intercept``,
``C``) and ``summary.json``. The test split is never loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from finetune_vision_multi import auroc  # noqa: E402


def lr_probs(m: dict, x: np.ndarray, k: int) -> np.ndarray:
    """[n, K] class probabilities from a saved probe."""
    s = ((x - m["mean"]) / m["scale"]) @ m["coef"].T + m["intercept"]
    if k == 2:
        p1 = 1 / (1 + np.exp(-s[:, 0]))
        return np.stack([1 - p1, p1], 1)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--grid", type=float, nargs="+", default=list(np.logspace(-5, 1, 13)))
    a = ap.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")
    d = fracatlas_data.load_splits(a.encoder, ("train", "val"), a.data)
    xtr, xva = d["train"].x.numpy().astype(np.float64), d["val"].x.numpy().astype(np.float64)
    sc = StandardScaler().fit(xtr)
    xtr_s, xva_s = sc.transform(xtr), sc.transform(xva)
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {}
    for q in fq.QUESTIONS:
        keys, _ = fq.options(q)
        ytr, yva = fq.labels(q, d["train"].rows), fq.labels(q, d["val"].rows)
        best = None
        for c in a.grid:
            m = LogisticRegression(C=c, class_weight="balanced", max_iter=10000).fit(xtr_s, ytr)
            model = {"mean": sc.mean_, "scale": sc.scale_, "coef": m.coef_, "intercept": m.intercept_}
            if len(keys) > 2 and len(m.classes_) != len(keys):
                raise SystemExit(f"{q}: train labels miss a class")
            auc = auroc(yva, lr_probs(model, xva, len(keys)), len(keys), fq.positive_index(q))
            if best is None or auc > best[0] + 1e-9:
                best = (auc, c, model)
        auc, c, model = best
        np.savez(os.path.join(a.out_dir, f"{q}.npz"), C=c, **model)
        summary[q] = {"C": c, "val_auroc": auc}
        print(f"[lr-multiq] {d['train'].meta['encoder']} {q:<15} C={c:.1e} val auroc {auc:.4f}", flush=True)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
