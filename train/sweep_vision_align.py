#!/usr/bin/env python3
"""Runs and selections of the pre-registered alignment experiment (research/protocol_alignment.md).

    python train/sweep_vision_align.py

* A1 (text-only alignment), per corpus (strict, general): lr in {3e-5, 1e-4} x gap shift in {off, on},
  seeds 0-2; the config is chosen by mean val zero-shot AUROC over the trained questions (dev set).
* A2: A1-strict init, fine-tuned on the trained questions without the alignment loss.
* A3: A1-strict init, fine-tuned with lambda in {0.1, 0.3, 1, 3} (all reported; lambda = 1 is primary).
* A3g: A1-general init, lambda = 1.
Seed s of a fine-tuned variant starts from seed s of its A1 run. Held-out questions and the test split
are never touched. Writes runs/fracatlas/align/medsiglip/sweep.json; finished runs are reused.
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
LAMBDAS = [0.1, 0.3, 1.0, 3.0]


def run(out: str, args: list[str], seed: int) -> dict:
    path = os.path.join(out, f"s{seed}")
    summ = os.path.join(path, "summary.json")
    if not os.path.exists(summ):
        cmd = [sys.executable, os.path.join(HERE, "finetune_vision_align.py"), "--out-dir", path,
               "--seed", str(seed), *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        print("   " + r.stdout.strip().splitlines()[-1], flush=True)
    return json.load(open(summ))


def agg(runs: list[dict]) -> dict:
    v = [r["val_mean_auroc"] for r in runs]
    return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "val": v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/fracatlas/align/medsiglip")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    a1 = {}
    for corpus in ("strict", "general"):
        grid = []
        for lr, gap in itertools.product([3e-5, 1e-4], [False, True]):
            name = f"lr{lr:g}_gap{int(gap)}"
            d = os.path.join(a.out, "A1", corpus, name)
            args = ["--stage", "align", "--corpus", corpus, "--lr", str(lr)] + (["--gap-shift"] if gap else [])
            res = agg([run(d, args, s) for s in a.seeds])
            grid.append({"name": name, "dir": d, **res})
            print(f"[sweep-align] A1 {corpus:<7} {name:<14} val zero-shot (trained questions) "
                  f"{res['mean']:.4f} ± {res['sd']:.4f}", flush=True)
        grid.sort(key=lambda r: -r["mean"])
        a1[corpus] = {"grid": grid, "best": grid[0]}
        print(f"[sweep-align] A1 {corpus} best {grid[0]['name']}", flush=True)

    variants = {"A2": ("strict", 0.0)}
    variants.update({f"A3_l{l:g}": ("strict", l) for l in LAMBDAS})
    variants["A3g_l1"] = ("general", 1.0)
    ft = {}
    for vid, (corpus, lam) in variants.items():
        d = os.path.join(a.out, vid)
        init = a1[corpus]["best"]["dir"]
        runs = [run(d, ["--stage", "finetune", "--init", os.path.join(init, f"s{s}"), "--align-weight", str(lam)], s)
                for s in a.seeds]
        ft[vid] = {"dir": d, "corpus": corpus, "align_weight": lam, "init": init, **agg(runs)}
        print(f"[sweep-align] {vid:<8} corpus {corpus:<7} lambda {lam:<4g} val mean auroc (trained) "
              f"{ft[vid]['mean']:.4f} ± {ft[vid]['sd']:.4f}", flush=True)
    json.dump({"seeds": a.seeds, "A1": a1, "finetune": ft, "primary": "A3_l1"},
              open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
    print(f"[sweep-align] -> {a.out}/sweep.json", flush=True)


if __name__ == "__main__":
    main()
