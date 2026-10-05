#!/usr/bin/env python3
"""Val-only sweep and controls for the FracAtlas multi-question experiment.

    python train/sweep_vision_multi.py --encoder medsiglip

1. grid: one multi-question head (all questions but the held-out ones), lr x weight decay, over
   ``--seeds``; ranked by mean val AUROC over the training questions (held-out questions never
   enter selection);
2. controls on the best config: the same head with random option vectors, and one single-question
   head per question (including the held-out ones, as their supervised reference).

Results go to ``runs/fracatlas/multiq/<encoder>/sweep.json``. Finished runs are reused.
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
sys.path.insert(0, HERE)
import fracatlas_questions as fq  # noqa: E402

GRID = {"lr": [1e-5, 3e-5, 1e-4], "weight_decay": [0.01, 0.1]}


def run(encoder: str, out: str, args: list[str], seed: int) -> dict:
    path = os.path.join(out, f"s{seed}")
    summ = os.path.join(path, "summary.json")
    if not os.path.exists(summ):
        cmd = [sys.executable, os.path.join(HERE, "finetune_vision_multi.py"), "--encoder", encoder,
               "--out-dir", path, "--seed", str(seed), *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.load(open(summ))


def agg(summaries: list[dict]) -> dict:
    v = [s["val_mean_auroc"] for s in summaries]
    return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "val": v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out", default="runs/fracatlas/multiq")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    root = os.path.join(a.out, a.encoder)

    grid = []
    for lr, wd in itertools.product(*GRID.values()):
        name = f"lr{lr:g}_wd{wd:g}"
        args = ["--lr", str(lr), "--weight-decay", str(wd)]
        res = agg([run(a.encoder, os.path.join(root, "multi", name), args, s) for s in a.seeds])
        grid.append({"name": name, "args": args, "dir": os.path.join(root, "multi", name), **res})
        print(f"[sweep-multiq] {a.encoder} multi {name:<16} val mean auroc {res['mean']:.4f} ± {res['sd']:.4f}",
              flush=True)
    grid.sort(key=lambda r: -r["mean"])
    best = grid[0]
    print(f"[sweep-multiq] best {best['name']} {best['mean']:.4f}", flush=True)

    controls = {}
    d = os.path.join(root, "random_options", best["name"])
    controls["random_options"] = {"dir": d, **agg([run(a.encoder, d, best["args"] + ["--option-source", "random"], s)
                                                   for s in a.seeds])}
    print(f"[sweep-multiq] random options     val mean auroc {controls['random_options']['mean']:.4f}", flush=True)
    for q in fq.QUESTIONS:
        d = os.path.join(root, "single", best["name"], q)
        controls[f"single:{q}"] = {"dir": d, **agg([run(a.encoder, d, best["args"] + ["--questions", q], s)
                                                    for s in a.seeds])}
        print(f"[sweep-multiq] single {q:<15} val auroc {controls[f'single:{q}']['mean']:.4f}", flush=True)
    json.dump({"encoder": a.encoder, "seeds": a.seeds, "holdout": list(fq.HOLDOUT), "grid": grid, "best": best,
               "controls": controls}, open(os.path.join(root, "sweep.json"), "w"), indent=1)
    print(f"[sweep-multiq] -> {root}/sweep.json", flush=True)


if __name__ == "__main__":
    main()
