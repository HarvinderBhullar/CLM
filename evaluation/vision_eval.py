#!/usr/bin/env python3
"""Evaluate FracAtlas fracture classifiers on the held-out test split; write results/fracatlas.md.

    python evaluation/vision_eval.py --dry-run     # val stands in for test: debug without looking at test
    python evaluation/vision_eval.py               # the one test evaluation

Methods (all chosen on val by the training scripts, nothing here is tuned on test):
* CLM head (``train/finetune_vision.py``, best config of ``train/sweep_vision.py``), every seed,
  scored exactly as served: image embedding -> state head; option text embeddings -> the
  checkpoint's action head; softmax over ``scale * cos``.
* CLM ablations from the sweep (trainable scale, no standardisation, no class weights,
  random option vectors).
* Logistic-regression linear probe (``train/baselines_vision.py``).
* Resolution only: logistic regression on log(pixel count), fit on train (a shortcut reference).

Metrics on test: AUROC (95% bootstrap CI), sensitivity / specificity at the threshold that
maximises Youden's J on val, sensitivity at 90% specificity (read off the test ROC), ECE
(15 bins) raw and after Platt scaling fit on val, per-body-region AUROC, and a paired
bootstrap of the AUROC difference CLM - linear probe.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import warnings
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))
import fracatlas_data  # noqa: E402
from clm import calibration  # noqa: E402
from clm.heads import HeadPair, make_head  # noqa: E402

# numpy 2.2 + Apple Accelerate raise spurious divide/overflow warnings inside matmul
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")

ENCODERS = {"qwen3-vl-4b": "Qwen3-VL-4B", "medsiglip": "MedSigLIP-448"}
ABLATION_NAMES = {"train_logit_scale": "trainable logit_scale", "no_standardize": "no input standardisation",
                  "no_class_weight": "no class weights", "random_options": "random option vectors"}
MIN_REGION_POS = 5
N_BOOT = 2000


# --------------------------------------------------------------------------- scorers
def clm_scores(ckpt_path: str, options: dict, x: torch.Tensor) -> np.ndarray:
    """p(fracture) from a vision CLM checkpoint, the way the serving engine would compute it."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    kw = {k: cfg[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
    sh = make_head(proj=cfg["projection_dim"], hidden=cfg["hidden_size"], **kw).eval()
    ah = make_head(proj=cfg["projection_dim"], hidden=cfg["action_hidden_size"], **kw).eval()
    sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    oset = options["sets"][cfg["option_form"]]
    with torch.no_grad():
        if cfg.get("option_source", "text") == "random":
            a = F.normalize(torch.randn(oset["proj"].shape, generator=torch.Generator().manual_seed(12345)), dim=-1)
        else:
            a = F.normalize(ah(oset["raw"].float()), dim=-1)
        z = F.normalize(sh(x), dim=-1)
        scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
        p = torch.softmax(scale * z @ a.t(), -1)[:, oset["keys"].index(cfg["positive"])]
    return p.double().numpy()


def served_logit_diff(ckpt_path: str, options: dict, x: torch.Tensor) -> tuple[np.ndarray, dict | None]:
    """Raw logit difference fracture - no_fracture of a deployed checkpoint (``HeadPair``, the code
    ``clm-serve`` runs) for the ``choice`` options, and its calibration entry if it has one."""
    head = HeadPair("served", ckpt_path, device="cpu").ensure()
    oset = options["sets"]["choice"]
    pos = oset["keys"].index("fracture")
    lg = head.scale * head.project_states(x.numpy()) @ head.project_actions(oset["raw"].float().numpy()).T
    d = (lg[:, pos] - lg[:, 1 - pos]).double().numpy()
    return d, calibration.find(head.cfg, oset["texts"])


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def lr_scores(model_dir: str, x: torch.Tensor) -> np.ndarray:
    m = np.load(os.path.join(model_dir, "model.npz"))
    s = ((x.double().numpy() - m["mean"]) / m["scale"]) @ m["coef"] + float(m["intercept"])
    return 1 / (1 + np.exp(-s))


def log_pixels(data: str, rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        with Image.open(os.path.join(data, "raw", r["path"])) as im:
            out.append(math.log(im.size[0] * im.size[1]))
    return np.array(out)[:, None]


# --------------------------------------------------------------------------- metrics
def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sens_spec(y: np.ndarray, p: np.ndarray, t: float) -> tuple[float, float]:
    pred = p >= t
    return float(pred[y == 1].mean()), float((~pred[y == 0]).mean())


def sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float = 0.9) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec + 1e-12
    return float(tpr[ok].max())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def platt(yv: np.ndarray, pv: np.ndarray, pt: np.ndarray) -> np.ndarray:
    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))  # noqa: E731
    m = LogisticRegression(C=1e6, max_iter=10000).fit(lg(pv)[:, None], yv)
    return m.predict_proba(lg(pt)[:, None])[:, 1]


def boot_idx(y: np.ndarray, n: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified bootstrap resamples (keeps the positive count, so AUROC is always defined)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def metrics(yv, pv, yt, pt, boots) -> dict:
    t = youden_threshold(yv, pv)
    se, sp = sens_spec(yt, pt, t)
    aucs = [roc_auc_score(yt[i], pt[i]) for i in boots]
    return {"auroc": float(roc_auc_score(yt, pt)), "auroc_ci": [float(np.percentile(aucs, 2.5)),
                                                                 float(np.percentile(aucs, 97.5))],
            "threshold": t, "sens": se, "spec": sp, "sens_at_90spec": sens_at_spec(yt, pt),
            "ece": ece(yt, pt), "ece_platt": ece(yt, platt(yv, pv, pt)), "mean_p": float(pt.mean())}


def region_auroc(rows, y, p) -> dict:
    out = {}
    for reg in sorted({r["region"] for r in rows}):
        m = np.array([r["region"] == reg for r in rows])
        npos = int(y[m].sum())
        out[reg] = {"n": int(m.sum()), "pos": npos,
                    "auroc": float(roc_auc_score(y[m], p[m])) if 0 < npos < m.sum() else None}
    return out


# --------------------------------------------------------------------------- report
def fmt_ci(m):
    return f"{m['auroc']:.3f} [{m['auroc_ci'][0]:.3f}, {m['auroc_ci'][1]:.3f}]"


def mean_sd(ms: list[dict], k: str) -> str:
    v = [m[k] for m in ms]
    return f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"


def row(name, ms: list[dict]) -> str:
    if len(ms) == 1:
        m = ms[0]
        return (f"| {name} | {fmt_ci(m)} | {m['sens']:.3f} | {m['spec']:.3f} | {m['sens_at_90spec']:.3f} | "
                f"{m['ece']:.3f} | {m['ece_platt']:.3f} |")
    return (f"| {name} | {mean_sd(ms, 'auroc')} | {mean_sd(ms, 'sens')} | {mean_sd(ms, 'spec')} | "
            f"{mean_sd(ms, 'sens_at_90spec')} | {mean_sd(ms, 'ece')} | {mean_sd(ms, 'ece_platt')} |")


HEADER = ("| method | AUROC | sens @ val thr | spec @ val thr | sens @ 90% spec | ECE | ECE (val Platt) |\n"
          "|---|---|---|---|---|---|---|")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--sweep", default="runs/fracatlas/sweep")
    ap.add_argument("--lr-dir", default="runs/fracatlas/lr-{encoder}")
    ap.add_argument("--out", default="results/fracatlas.md")
    ap.add_argument("--dry-run", action="store_true", help="evaluate on val instead of test (no test access)")
    ap.add_argument("--served", nargs="*", default=None,
                    help="deployed (calibrated) vision checkpoints to report as served; default checkpoints/fracatlas-*.pt")
    ap.add_argument("--verdict", default="results/fracatlas_verdict.md",
                    help="hand-written verdict inserted into the report, if the file exists")
    a = ap.parse_args()

    eval_split = "val" if a.dry_run else "test"
    options = torch.load(os.path.join(a.data, "options.pt"), map_location="cpu")
    res: dict = {"eval_split": eval_split, "methods": {}, "regions": {}, "paired": {}, "sweeps": {}}

    # resolution-only reference (encoder independent)
    spl = {s: fracatlas_data.load_split("qwen3-vl-4b", s, a.data) for s in ("train", "val", eval_split)}
    yv, yt = spl["val"].y.numpy(), spl[eval_split].y.numpy()
    boots = boot_idx(yt, N_BOOT)
    feats = {s: log_pixels(a.data, spl[s].rows) for s in spl}
    res_m = LogisticRegression(class_weight="balanced").fit(feats["train"], spl["train"].y.numpy())
    pr_v, pr_t = res_m.predict_proba(feats["val"])[:, 1], res_m.predict_proba(feats[eval_split])[:, 1]
    res["methods"]["resolution only"] = {"runs": [metrics(yv, pr_v, yt, pr_t, boots)]}
    test_rows = spl[eval_split].rows

    for enc, label in ENCODERS.items():
        v, t = fracatlas_data.load_split(enc, "val", a.data), fracatlas_data.load_split(enc, eval_split, a.data)
        assert v.ids == spl["val"].ids and t.ids == spl[eval_split].ids
        sweep = json.load(open(os.path.join(a.sweep, enc, "sweep.json")))
        res["sweeps"][enc] = {"best": sweep["best"]["name"], "val_auroc": sweep["best"]["mean"],
                              "seeds": sweep["seeds"]}

        lr_dir = a.lr_dir.format(encoder=enc)
        lv, lt = lr_scores(lr_dir, v.x), lr_scores(lr_dir, t.x)
        res["methods"][f"Linear probe (LR) · {label}"] = {"runs": [metrics(yv, lv, yt, lt, boots)], "encoder": enc}
        res["regions"][f"Linear probe (LR) · {label}"] = region_auroc(test_rows, yt, lt)

        clm_runs, clm_t = [], []
        for s in sweep["seeds"]:
            ck = os.path.join(sweep["best"]["dir"], f"s{s}", "best_head.pt")
            cv, ct = clm_scores(ck, options, v.x), clm_scores(ck, options, t.x)
            clm_runs.append(metrics(yv, cv, yt, ct, boots)); clm_t.append(ct)
        name = f"CLM head · {label}"
        res["methods"][name] = {"runs": clm_runs, "encoder": enc}
        res["regions"][name] = {reg: {**d, "auroc": (statistics.mean(x) if None not in (x := [
            region_auroc(test_rows, yt, p)[reg]["auroc"] for p in clm_t]) else None)}
            for reg, d in region_auroc(test_rows, yt, clm_t[0]).items()}

        # paired bootstrap: mean-over-seeds CLM AUROC minus linear-probe AUROC on the same resample
        diffs = [statistics.mean(roc_auc_score(yt[i], p[i]) for p in clm_t) - roc_auc_score(yt[i], lt[i])
                 for i in boots]
        point = statistics.mean(roc_auc_score(yt, p) for p in clm_t) - roc_auc_score(yt, lt)
        res["paired"][label] = {"diff": point, "ci": [float(np.percentile(diffs, 2.5)),
                                                      float(np.percentile(diffs, 97.5))],
                                "p_le_0": float(np.mean(np.array(diffs) <= 0))}

        for k, abl in sweep["ablations"].items():
            runs = []
            for s in sweep["seeds"]:
                ck = os.path.join(abl["dir"], f"s{s}", "best_head.pt")
                runs.append(metrics(yv, clm_scores(ck, options, v.x), yt, clm_scores(ck, options, t.x), boots))
            res["methods"][f"CLM ablation: {ABLATION_NAMES[k]} · {label}"] = {"runs": runs, "encoder": enc,
                                                                              "ablation": k}

    # deployed checkpoints: the plain answer (raw p > 0.5) vs the calibrated, thresholded one
    res["served"] = {}
    served = a.served if a.served is not None else sorted(glob.glob("checkpoints/fracatlas-*.pt"))
    for ck in served:
        cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
        if cfg.get("state_modality") != "image":
            continue
        enc = cfg["state_encoder"]
        d, entry = served_logit_diff(ck, options, fracatlas_data.load_split(enc, eval_split, a.data).x)
        raw = sigmoid(d)
        se0, sp0 = sens_spec(yt, raw, 0.5)
        srow = {"ckpt": ck, "encoder": enc, "seed": cfg.get("seed"), "auroc": float(roc_auc_score(yt, d)),
               "plain": {"sens": se0, "spec": sp0, "acc": float(np.mean((raw >= 0.5) == yt)), "ece": ece(yt, raw)}}
        if entry:
            cal = sigmoid(entry["a"] * d + entry["b"])
            se1, sp1 = sens_spec(yt, cal, entry["threshold"])
            srow["calibrated"] = {"threshold": entry["threshold"], "sens": se1, "spec": sp1,
                                 "acc": float(np.mean((cal >= entry["threshold"]) == yt)), "ece": ece(yt, cal)}
        res["served"][os.path.basename(ck)] = srow

    # ---------------------------------------------------------------- write
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_md = a.out if not a.dry_run else a.out.replace(".md", ".dryrun-val.md")
    json.dump(res, open(out_md.replace(".md", ".json"), "w"), indent=1)
    meta = json.load(open(os.path.join(a.data, "splits", "split_meta.json")))
    n_t, pos_t = len(yt), int(yt.sum())
    main_rows = [k for k in res["methods"] if not k.startswith("CLM ablation")]
    half_ci = statistics.mean((m["runs"][0]["auroc_ci"][1] - m["runs"][0]["auroc_ci"][0]) / 2
                              for m in res["methods"].values() if len(m["runs"]) == 1)
    L = [f"# FracAtlas fracture detection: CLM vision head vs linear probes",
         "",
         f"Research prototype, not a clinical tool. Evaluated {date.today().isoformat()} on the "
         f"**{eval_split}** split ({n_t} images, {pos_t} fractured, {100 * pos_t / n_t:.1f}%)"
         + (" — DRY RUN: val stands in for test, numbers are not test results." if a.dry_run else
            ", once; every choice (configs, C, thresholds, Platt scaling) was made on val."),
         "",
         "## Test results" if not a.dry_run else "## Results (val, dry run)",
         "",
         "CLM rows: mean ± sd over seeds " + ", ".join(map(str, res["sweeps"]["qwen3-vl-4b"]["seeds"]))
         + " of the best val config. Single-model rows: AUROC with a 95% stratified bootstrap CI "
         f"({N_BOOT} resamples). Threshold = max Youden J on val. Sens @ 90% spec is read off the "
         f"{eval_split} ROC. ECE: 15 equal-width bins on p(fracture); both trained methods use balanced "
         "class weights, which inflates raw p(fracture) by design, so ECE is also shown after Platt "
         "scaling fit on val.",
         "",
         HEADER]
    L += [row(k, res["methods"][k]["runs"]) for k in main_rows]
    L += ["",
          "### Does the CLM head add value over the linear probe?",
          "",
          "Paired stratified bootstrap of AUROC(CLM, mean over seeds) − AUROC(linear probe), same encoder:",
          "",
          "| encoder | ΔAUROC | 95% CI | P(Δ ≤ 0) |", "|---|---|---|---|"]
    L += [f"| {k} | {v['diff']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['p_le_0']:.3f} |"
          for k, v in res["paired"].items()]
    verdict = open(a.verdict).read().strip() if os.path.exists(a.verdict) else "<!-- verdict -->"
    L += ["", verdict, ""]
    if res["served"]:
        L += ["## Served heads: plain answer vs calibrated decision", "",
              f"The deployed checkpoints (`checkpoints/`, seed 0 of the best config) scored with the serving code "
              f"on {eval_split}. *Plain*: the option with the higher raw probability (p > 0.5), which is what "
              "`choice` returned before calibration. *Calibrated*: Platt scaling and a threshold, both fit on val "
              "by `train/calibrate_vision.py` and applied by the engine. AUROC is the same for both.", "",
              "| checkpoint | AUROC | plain sens / spec | plain acc. | plain ECE | threshold | calibrated sens / spec "
              "| calibrated acc. | calibrated ECE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, r in res["served"].items():
            c = r.get("calibrated")
            L.append(f"| `{name}` | {r['auroc']:.3f} | {r['plain']['sens']:.3f} / {r['plain']['spec']:.3f} | "
                     f"{r['plain']['acc']:.3f} | {r['plain']['ece']:.3f} | "
                     + (f"{c['threshold']:.3f} | {c['sens']:.3f} / {c['spec']:.3f} | {c['acc']:.3f} | {c['ece']:.3f} |"
                        if c else "– | not calibrated | – | – |"))
        L += [""]
    L += ["## Ablations (CLM head, best config)", "",
          HEADER]
    L += [row(k.replace("CLM ablation: ", ""), v["runs"]) for k, v in res["methods"].items()
          if k.startswith("CLM ablation")]
    L += ["",
          f"## Per body region ({eval_split} AUROC)",
          "",
          f"Regions with fewer than {MIN_REGION_POS} fractured images are shown for completeness only.",
          ""]
    regs = list(next(iter(res["regions"].values())))
    L += ["| method | " + " | ".join(f"{r} (n={res['regions'][main_rows[1]][r]['n']}, "
                                    f"{res['regions'][main_rows[1]][r]['pos']}+)" for r in regs) + " |",
          "|---|" + "---|" * len(regs)]
    for k, regd in res["regions"].items():
        cells = []
        for r in regs:
            d = regd[r]
            c = "–" if d["auroc"] is None else f"{d['auroc']:.3f}"
            cells.append(c + ("*" if d["pos"] < MIN_REGION_POS else ""))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "\\* fewer than %d positives: not interpretable." % MIN_REGION_POS,
          "",
          "## Setup",
          "",
          f"- **Data**: FracAtlas (figshare 22363012, CC BY 4.0). {sum(c['n'] for c in meta['counts'].values())} "
          f"of 4083 images used: {len(meta['dropped_ambiguous'])} dropped (present in both class folders, "
          f"ambiguous label) and {len(meta['dropped_truncated'])} dropped (truncated JPEGs, all non-fractured; "
          "decoding leaves a grey band, a label shortcut). Stratified (label × body region) image-level "
          f"70/15/15 split, seed {meta['seed']}: "
          + ", ".join(f"{k} {v['n']} ({v['fractured']}+)" for k, v in meta["counts"].items()) + ".",
          "- **CLM head**: frozen image encoder → L2-normalised embedding → new `state_head` "
          "(make_head 1536×3, LayerNorm, GELU → 512, random init); options \"Radiograph showing an acute "
          "bone fracture.\" / \"Radiograph of intact bones with no fracture.\" embedded once by Qwen3-8B "
          "(last token, bf16) through the released `action_head`, both frozen; `logit_scale` frozen (100). "
          "Class-weighted CE over the two options, AdamW, early stopping on val AUROC. Config from a val-only "
          "grid (lr × weight decay × input dropout, 3 seeds): "
          + "; ".join(f"{ENCODERS[e]} `{s['best']}` (val {s['val_auroc']:.3f})" for e, s in res["sweeps"].items())
          + ".",
          "- **Encoders**: Qwen3-VL-4B-Instruct (fp16, MPS; chat turn = image ≤1024² px + \"Radiograph for "
          "fracture assessment.\", last-token final hidden state, 2560-d) and MedSigLIP-448 (pooled image "
          "embedding, 1152-d). The 24 GB Mac rules out Qwen3-VL-8B, so the released `state_head` cannot be "
          "used as a warm start (2560-d input vs 4096-d).",
          "- **Linear probe**: standardise → logistic regression, balanced class weights, C picked on val.",
          "- **Resolution only**: logistic regression on log(pixel count) — images > 1 MP are 29% fractured "
          "vs 16.5% for the rest, so image size alone leaks some label signal.",
          "",
          "## Caveats",
          "",
          "- **Leakage**: no patient ids; consecutive ids look like several views of one study, so the same "
          "patient can appear in train and test. Test numbers are optimistic for unseen patients.",
          f"- **Small {eval_split} set**: {pos_t} fractured images; single-model AUROC 95% CIs are "
          f"±{half_ci:.3f} wide on average, larger than most gaps between methods.",
          "- **Val-selected configs**: CLM configs and C were chosen on val, so val numbers are optimistic; "
          "the test split was not used for any choice.",
          "- **Small regions**: " + ", ".join(f"{r} has {d['pos']} fractured / {d['n']}" for r, d in
                                              res["regions"][main_rows[1]].items() if d["pos"] < MIN_REGION_POS)
          + f" {eval_split} images, too few for a per-region AUROC.",
          ""]
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[eval] -> {out_md} (+ .json)")


if __name__ == "__main__":
    main()
