#!/usr/bin/env python3
"""Small val-only hyperparameter sweep for ``train/finetune_vision.py`` (FracAtlas).

    python train/sweep_vision.py --encoder qwen3-vl-4b
    python train/sweep_vision.py --encoder qwen3-vl-4b --ablations   # after the grid, on its best config

Every config runs over ``--seeds``; configs are ranked by mean val AUROC. ``--ablations``
reruns the best config with ``--train-logit-scale``, ``--no-standardize``,
``--class-weight none`` and ``--option-source random`` (random option vectors: a control for
whether the text action tower matters). Results go to ``runs/fracatlas/sweep/<encoder>/sweep.json``.
The test split is never touched.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = {"lr": [1e-5, 3e-5, 1e-4, 3e-4], "weight_decay": [0.01, 0.1], "input_dropout": [0.0, 0.3]}
ABLATIONS = {"train_logit_scale": ["--train-logit-scale"], "no_standardize": ["--no-standardize"],
             "no_class_weight": ["--class-weight", "none"], "random_options": ["--option-source", "random"]}


def name_of(cfg: dict) -> str:
    return "_".join(f"{k}{v:g}" for k, v in cfg.items())


def run(encoder: str, out: str, cfg: dict, extra: list[str], seed: int) -> float:
    path = os.path.join(out, f"s{seed}")
    summ = os.path.join(path, "summary.json")
    if not os.path.exists(summ):          # resumable: finished runs are reused
        cmd = [sys.executable, os.path.join(HERE, "finetune_vision.py"), "--encoder", encoder, "--out-dir", path,
               "--seed", str(seed), *[x for k, v in cfg.items() for x in (f"--{k.replace('_', '-')}", str(v))], *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.load(open(summ))["val_auroc"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", default="runs/fracatlas/sweep")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ablations", action="store_true")
    a = ap.parse_args()

    root = os.path.join(a.out, a.encoder)
    rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, vals))
        aucs = [run(a.encoder, os.path.join(root, "grid", name_of(cfg)), cfg, [], s) for s in a.seeds]
        rows.append({"config": cfg, "name": name_of(cfg), "val_aurocs": aucs,
                     "mean": statistics.mean(aucs), "sd": statistics.stdev(aucs) if len(aucs) > 1 else 0.0})
        print(f"[sweep] {a.encoder} {name_of(cfg):<40} val auroc {rows[-1]['mean']:.4f} ± {rows[-1]['sd']:.4f}",
              flush=True)
    rows.sort(key=lambda r: -r["mean"])
    best = rows[0]
    print(f"[sweep] best {best['name']} {best['mean']:.4f} ± {best['sd']:.4f}", flush=True)

    ablations = {}
    if a.ablations:
        for k, extra in ABLATIONS.items():
            d = os.path.join(root, "ablation", best["name"], k)
            aucs = [run(a.encoder, d, best["config"], extra, s) for s in a.seeds]
            ablations[k] = {"val_aurocs": aucs, "mean": statistics.mean(aucs),
                            "sd": statistics.stdev(aucs) if len(aucs) > 1 else 0.0, "dir": d}
            print(f"[sweep] ablation {k:<18} val auroc {ablations[k]['mean']:.4f} ± {ablations[k]['sd']:.4f}",
                  flush=True)
    json.dump({"encoder": a.encoder, "seeds": a.seeds, "grid": rows,
               "best": {**best, "dir": os.path.join(root, "grid", best["name"])}, "ablations": ablations},
              open(os.path.join(root, "sweep.json"), "w"), indent=1)
    print(f"[sweep] -> {root}/sweep.json", flush=True)


if __name__ == "__main__":
    main()
