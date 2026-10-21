#!/usr/bin/env python3
"""Runs and selections of the pre-registered distillation experiment (research/protocol_distill.md).

    python train/sweep_vision_distill.py

* D1 (distillation only, no image labels), per corpus (strict, general): lr in {3e-5, 1e-4} x teacher scale
  in {10, 30, 100}, seeds 0-2; the config is chosen by mean val zero-shot AUROC over the trained questions.
* D2: D1-general init, fine-tuned on the trained questions without the distillation loss.
* D3: D1-general init, fine-tuned with lambda in {0.1, 0.3, 1, 3} (all reported; lambda = 1 is primary).
* W: per seed, (1 - alpha) * D1-general + alpha * D2 for alpha in {0.25, 0.5, 0.75} (all reported; 0.5 primary).
Seed s of a fine-tuned variant starts from seed s of its D1 run. Held-out questions and the test split are
never touched. Writes runs/fracatlas/distill/medsiglip/sweep.json; finished runs are reused.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from clm.heads import make_head  # noqa: E402
from finetune_vision import fold_standardisation  # noqa: E402

LAMBDAS = [0.1, 0.3, 1.0, 3.0]
ALPHAS = [0.25, 0.5, 0.75]


def run(out: str, args: list[str], seed: int) -> dict:
    path = os.path.join(out, f"s{seed}")
    summ = os.path.join(path, "summary.json")
    if not os.path.exists(summ):
        cmd = [sys.executable, os.path.join(HERE, "finetune_vision_distill.py"), "--out-dir", path,
               "--seed", str(seed), *args]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        print("   " + r.stdout.strip().splitlines()[-1], flush=True)
    return json.load(open(summ))


def agg(runs: list[dict]) -> dict:
    v = [r["val_mean_auroc"] for r in runs]
    return {"mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "val": v}


def interpolate(zs_dir: str, ft_dir: str, alpha: float, out_dir: str) -> None:
    """Write best_head.pt for (1 - alpha) * zero-shot + alpha * fine-tuned weights (same standardisation)."""
    a = torch.load(os.path.join(zs_dir, "train_state.pt"), map_location="cpu", weights_only=False)
    b = torch.load(os.path.join(ft_dir, "train_state.pt"), map_location="cpu", weights_only=False)
    if not (torch.equal(a["mu"], b["mu"]) and torch.equal(a["sd"], b["sd"])):
        raise SystemExit(f"{zs_dir} and {ft_dir} use different input standardisation")
    ck = torch.load(os.path.join(ft_dir, "best_head.pt"), map_location="cpu", weights_only=False)
    c = ck["cfg"]
    head = make_head(proj=c["projection_dim"], hidden=c["hidden_size"],
                     **{k: c[k] for k in ("width", "depth", "activation", "layernorm", "residual")})
    head.load_state_dict({k: (1 - alpha) * a["state_head_unfolded"][k] + alpha * v
                          for k, v in b["state_head_unfolded"].items()})
    fold_standardisation(head, b["mu"], b["sd"])
    os.makedirs(out_dir, exist_ok=True)
    torch.save({**ck, "state_head": head.state_dict(),
                "cfg": {**c, "task": "vision_distill_interp", "alpha": alpha, "zero_shot": zs_dir, "finetuned": ft_dir},
                "epoch": None, "metrics": None}, os.path.join(out_dir, "best_head.pt"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/fracatlas/distill/medsiglip")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    d1 = {}
    for corpus in ("strict", "general"):
        grid = []
        for lr, t in itertools.product([3e-5, 1e-4], [10, 30, 100]):
            name = f"lr{lr:g}_T{t}"
            d = os.path.join(a.out, "D1", corpus, name)
            args = ["--stage", "distill", "--corpus", corpus, "--lr", str(lr), "--teacher-scale", str(t)]
            res = agg([run(d, args, s) for s in a.seeds])
            grid.append({"name": name, "dir": d, **res})
            print(f"[sweep-distill] D1 {corpus:<7} {name:<14} val zero-shot (trained questions) "
                  f"{res['mean']:.4f} ± {res['sd']:.4f}", flush=True)
        grid.sort(key=lambda r: -r["mean"])
        d1[corpus] = {"grid": grid, "best": grid[0]}
        print(f"[sweep-distill] D1 {corpus} best {grid[0]['name']}", flush=True)

    init = d1["general"]["best"]["dir"]
    variants = {"D2": 0.0, **{f"D3_l{l:g}": l for l in LAMBDAS}}
    ft = {}
    for vid, lam in variants.items():
        d = os.path.join(a.out, vid)
        runs = [run(d, ["--stage", "finetune", "--init", os.path.join(init, f"s{s}"), "--distill-weight", str(lam)], s)
                for s in a.seeds]
        ft[vid] = {"dir": d, "distill_weight": lam, "init": init, **agg(runs)}
        print(f"[sweep-distill] {vid:<8} lambda {lam:<4g} val mean auroc (trained) "
              f"{ft[vid]['mean']:.4f} ± {ft[vid]['sd']:.4f}", flush=True)
    interp = {}
    for alpha in ALPHAS:
        d = os.path.join(a.out, f"W_a{alpha:g}")
        for s in a.seeds:
            interpolate(os.path.join(init, f"s{s}"), os.path.join(ft["D2"]["dir"], f"s{s}"), alpha,
                        os.path.join(d, f"s{s}"))
        interp[f"W_a{alpha:g}"] = {"dir": d, "alpha": alpha}
    json.dump({"seeds": a.seeds, "D1": d1, "finetune": ft, "interp": interp,
               "primary": {"H4": "D3_l1", "H5": "W_a0.5"}},
              open(os.path.join(a.out, "sweep.json"), "w"), indent=1)
    print(f"[sweep-distill] -> {a.out}/sweep.json", flush=True)


if __name__ == "__main__":
    main()
