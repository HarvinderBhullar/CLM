#!/usr/bin/env python3
"""Calibrate a vision head on val and store it in the checkpoint (``cfg["calibration"]``).

    python train/calibrate_vision.py --ckpt runs/.../best_head.pt --out checkpoints/fracatlas-medsiglip.pt

With ``logit_scale`` frozen at 100 the head ranks images well but its raw p(fracture) is
overconfident and its best cut is far below 0.5, so the plain ``choice`` misses many
fractures. For each option pair the head is served with (the ``choice`` and ``noul`` forms in
``options.pt``) this fits, on the **val** split only:

* Platt scaling of the logit difference: p = sigmoid(a * (l_pos - l_neg) + b);
* a decision threshold on the calibrated p: max Youden J (default) or the highest threshold
  that keeps val sensitivity >= ``--min-sensitivity``.

The scores are computed with ``clm.heads.HeadPair``, the code ``clm-serve`` runs, and the
engine applies the entry (``clm.calibration``) whenever a question's options are that pair.
Platt is monotone, so AUROC is unchanged. The test split is never loaded.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import fracatlas_data  # noqa: E402
from clm.heads import HeadPair  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")
POSITIVE_KEYS = ("fracture", "true")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def sens_spec(y, p, t):
    return float((p[y == 1] >= t).mean()), float((p[y == 0] < t).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="vision head (train/finetune_vision.py best_head.pt)")
    ap.add_argument("--out", required=True, help="calibrated checkpoint to write (may equal --ckpt)")
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--options", default=None, help="default DATA/options.pt")
    ap.add_argument("--min-sensitivity", type=float, default=None,
                    help="threshold = highest cut with val sensitivity >= this (default: max Youden J)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if cfg.get("state_modality") != "image":
        raise SystemExit(f"{a.ckpt} is not a vision head (cfg.state_modality = {cfg.get('state_modality')!r})")
    head = HeadPair("calibrate", a.ckpt, device="cpu").ensure()
    val = fracatlas_data.load_split(cfg["state_encoder"], "val", a.data)
    y = val.y.numpy()
    zs = head.project_states(val.x.numpy())
    opts = torch.load(a.options or os.path.join(a.data, "options.pt"), map_location="cpu")

    entries = []
    for form, oset in opts["sets"].items():
        keys, texts = oset["keys"], oset["texts"]
        pos = next(i for i, k in enumerate(keys) if k in POSITIVE_KEYS)
        za = head.project_actions(oset["raw"].float().numpy())
        logits = (head.scale * zs @ za.T).double().numpy()                # what the engine computes
        d = logits[:, pos] - logits[:, 1 - pos]
        p_raw = 1 / (1 + np.exp(-np.clip(d, -500, 500)))
        lr = LogisticRegression(C=1e6, max_iter=10000).fit(d[:, None], y)
        aa, bb = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        p_cal = 1 / (1 + np.exp(-(aa * d + bb)))
        fpr, tpr, thr = roc_curve(y, p_cal)
        if a.min_sensitivity is None:
            t, rule = float(thr[np.argmax(tpr - fpr)]), "max Youden J on val"
        else:
            ok = tpr >= a.min_sensitivity
            t, rule = float(thr[ok][0]), f"highest cut with val sensitivity >= {a.min_sensitivity}"
        t = min(t, 1.0)
        se0, sp0 = sens_spec(y, p_raw, 0.5)
        se1, sp1 = sens_spec(y, p_cal, t)
        entry = {"form": form, "texts": {"positive": texts[pos], "negative": texts[1 - pos]},
                 "a": aa, "b": bb, "threshold": t, "rule": rule, "fit_on": "val", "n": int(len(y)),
                 "n_positive": int(y.sum()),
                 "val": {"auroc": float(roc_auc_score(y, d)), "ece_raw": ece(y, p_raw), "ece_calibrated": ece(y, p_cal),
                         "sens_spec_raw_at_0.5": [se0, sp0], "sens_spec_at_threshold": [se1, sp1]}}
        entries.append(entry)
        print(f"[calibrate] {form:<6} a {aa:.4f} b {bb:+.3f} | threshold {t:.3f} ({rule})\n"
              f"            val AUROC {entry['val']['auroc']:.3f} | ECE {entry['val']['ece_raw']:.3f} -> "
              f"{entry['val']['ece_calibrated']:.3f} | sens/spec: raw p>0.5 {se0:.2f}/{sp0:.2f} -> "
              f"calibrated p>={t:.3f} {se1:.2f}/{sp1:.2f}", flush=True)

    ck["cfg"] = {**cfg, "calibration": entries}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, a.out)          # atomic: a running clm-serve hot-reloads a complete file
    print(f"[calibrate] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
