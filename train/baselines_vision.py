#!/usr/bin/env python3
"""Linear-probe baseline on cached image embeddings (FracAtlas): no CLM head, no text tower.

    python train/baselines_vision.py --encoder qwen3-vl-4b --out-dir runs/fracatlas/lr-qwen3vl4b
    python train/baselines_vision.py --encoder medsiglip   --out-dir runs/fracatlas/lr-medsiglip

Standardise (train mean / std) -> L2-regularised logistic regression with balanced class
weights. ``C`` is picked on val AUROC from a log grid; the model is fit on train only, as
the CLM head is. The test split is never loaded here.

Writes ``model.npz`` (``mean``, ``scale``, ``coef``, ``intercept``, ``C``) and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracatlas_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--grid", type=float, nargs="+", default=list(np.logspace(-5, 1, 13)))
    a = ap.parse_args()

    d = fracatlas_data.load_splits(a.encoder, ("train", "val"), a.data)
    xtr, ytr = d["train"].x.numpy().astype(np.float64), d["train"].y.numpy()
    xva, yva = d["val"].x.numpy().astype(np.float64), d["val"].y.numpy()
    sc = StandardScaler().fit(xtr)
    xtr_s, xva_s = sc.transform(xtr), sc.transform(xva)

    # numpy 2.2 + Apple Accelerate raise spurious divide/overflow warnings inside matmul;
    # the fits are finite and match a float64 torch recomputation.
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")
    results = []
    for c in a.grid:
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=10000).fit(xtr_s, ytr)
        auc = roc_auc_score(yva, m.predict_proba(xva_s)[:, 1])
        results.append((auc, c, m))
        print(f"[lr] {d['train'].meta['encoder']} C={c:.2e} val auroc {auc:.4f}", flush=True)
    auc, c, m = max(results, key=lambda r: (r[0], -r[1]))   # ties -> stronger regularisation

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez(os.path.join(a.out_dir, "model.npz"), mean=sc.mean_, scale=sc.scale_,
             coef=m.coef_[0], intercept=m.intercept_[0], C=c)
    json.dump({"encoder": d["train"].meta["encoder"], "C": c, "val_auroc": auc,
               "grid": [{"C": cc, "val_auroc": aa} for aa, cc, _ in results]},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"[lr] best C={c:.2e} val auroc {auc:.4f} -> {a.out_dir}/model.npz", flush=True)


if __name__ == "__main__":
    main()
