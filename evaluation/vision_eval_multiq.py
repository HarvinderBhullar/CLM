#!/usr/bin/env python3
"""Evaluate the FracAtlas multi-question experiment; write results/fracatlas_multiq.md.

    python evaluation/vision_eval_multiq.py --encoder medsiglip --dry-run   # val stands in for test
    python evaluation/vision_eval_multiq.py --encoder medsiglip             # the one test evaluation

Methods (all selected on val by ``train/sweep_vision_multi.py``; nothing here is tuned on test):
* CLM multi-question head, best config, every seed: trained questions, and the held-out questions
  answered zero-shot;
* the same head with random option vectors (control: does the text tower matter?);
* one CLM head per question (control: does sharing one head across questions cost anything?);
* one logistic-regression probe per question (cannot answer questions it has no labels for).

Per question: AUROC (macro one-vs-rest above two options) and balanced accuracy (binary: threshold
at max Youden J on val; otherwise argmax), for option wording 0 (training) and paraphrases 1 and 2.
Paired stratified bootstrap of the macro AUROC over trained questions, CLM multi minus probes, and of
the zero-shot macro AUROC against chance.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import warnings
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from baselines_vision_multi import lr_probs  # noqa: E402
from clm.heads import make_head  # noqa: E402
from finetune_vision_multi import auroc, random_vector  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")
N_BOOT = 2000
NAMES = {"fracture": "fracture", "region": "body region", "view_frontal": "frontal view",
         "view_lateral": "lateral view", "view_oblique": "oblique view", "hardware": "hardware",
         "multiscan": "several scans", "fracture_count": "fracture count"}


class Head:
    """A multi-question checkpoint scored as the engine would: state head on the image embedding,
    the checkpoint's action head on each option text, softmax over 100 * cosine."""

    def __init__(self, path: str, bank: dict):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = self.cfg = ck["cfg"]
        kw = {k: cfg[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
        self.sh = make_head(proj=cfg["projection_dim"], hidden=cfg["hidden_size"], **kw).eval()
        self.ah = make_head(proj=cfg["projection_dim"], hidden=cfg["action_hidden_size"], **kw).eval()
        self.sh.load_state_dict(ck["state_head"]); self.ah.load_state_dict(ck["action_head"])
        self.scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
        self.bank, self.random = bank, cfg.get("option_source") == "random"
        self._z: dict[int, torch.Tensor] = {}

    def probs(self, x: torch.Tensor, qid: str, wording: int) -> np.ndarray:
        with torch.no_grad():
            z = self._z.setdefault(id(x), F.normalize(self.sh(x), dim=-1))
            texts = fq.options(qid, wording)[1]
            if self.random:
                a = torch.stack([random_vector(t, z.shape[1]) for t in texts])
            else:
                a = F.normalize(self.ah(torch.stack([self.bank[t] for t in texts]).float()), dim=-1)
            return torch.softmax(self.scale * z @ a.t(), -1).double().numpy()


def balanced_acc(qid: str, yv, pv, yt, pt) -> float:
    pos = fq.positive_index(qid)
    if pos is None:
        return float(balanced_accuracy_score(yt, pt.argmax(1)))
    fpr, tpr, thr = roc_curve(yv == pos, pv[:, pos])
    t = thr[np.argmax(tpr - fpr)]
    return float(balanced_accuracy_score(yt == pos, pt[:, pos] >= t))


def boot_idx(y: np.ndarray, n: int, seed: int = 0) -> list[np.ndarray]:
    """Bootstrap resamples stratified by the fracture label (every question stays defined)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def safe_auroc(y, p, k, pos):
    try:
        return auroc(y, p, k, pos)
    except ValueError:
        return float("nan")


def mean_sd(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)


def fmt(v):
    m, s = mean_sd(v) if isinstance(v, list) else (v, None)
    return "–" if m != m else (f"{m:.3f}" if not s else f"{m:.3f} ± {s:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="medsiglip")
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--root", default="runs/fracatlas/multiq")
    ap.add_argument("--out", default="results/fracatlas_multiq.md")
    ap.add_argument("--verdict", default="results/fracatlas_multiq_verdict.md")
    ap.add_argument("--dry-run", action="store_true", help="evaluate on val instead of test (no test access)")
    a = ap.parse_args()

    split = "val" if a.dry_run else "test"
    sweep = json.load(open(os.path.join(a.root, a.encoder, "sweep.json")))
    holdout, seeds = sweep["holdout"], sweep["seeds"]
    trained = [q for q in fq.QUESTIONS if q not in holdout]
    ob = torch.load(os.path.join(a.data, "options_multiq.pt"), map_location="cpu")
    bank = dict(zip(ob["texts"], ob["raw"]))
    v, t = fracatlas_data.load_split(a.encoder, "val", a.data), fracatlas_data.load_split(a.encoder, split, a.data)
    Y = {q: (fq.labels(q, v.rows), fq.labels(q, t.rows)) for q in fq.QUESTIONS}
    boots = boot_idx(t.y.numpy(), N_BOOT)
    K = {q: len(fq.options(q)[0]) for q in fq.QUESTIONS}
    P = {q: fq.positive_index(q) for q in fq.QUESTIONS}

    def score_head(path: str, questions) -> dict:
        h = Head(path, bank)
        out = {}
        for q in questions:
            yv, yt = Y[q]
            r = {}
            for w in range(fq.N_WORDINGS):
                pv, pt = h.probs(v.x, q, w), h.probs(t.x, q, w)
                r[w] = {"auroc": safe_auroc(yt, pt, K[q], P[q]), "bacc": balanced_acc(q, yv, pv, yt, pt), "pt": pt}
            out[q] = r
        return out

    methods = {}
    best = sweep["best"]["dir"]
    methods["clm_multi"] = [score_head(os.path.join(best, f"s{s}", "best_head.pt"), fq.QUESTIONS) for s in seeds]
    rnd = sweep["controls"]["random_options"]["dir"]
    methods["clm_multi_random"] = [score_head(os.path.join(rnd, f"s{s}", "best_head.pt"), fq.QUESTIONS) for s in seeds]
    single = {q: [score_head(os.path.join(sweep["controls"][f"single:{q}"]["dir"], f"s{s}", "best_head.pt"), [q])[q]
                  for s in seeds] for q in fq.QUESTIONS}
    lr_dir = os.path.join(a.root, f"lr-{a.encoder}")
    lr = {}
    for q in fq.QUESTIONS:
        m = dict(np.load(os.path.join(lr_dir, f"{q}.npz")))
        yv, yt = Y[q]
        pv, pt = lr_probs(m, v.x.numpy().astype(np.float64), K[q]), lr_probs(m, t.x.numpy().astype(np.float64), K[q])
        lr[q] = {"auroc": safe_auroc(yt, pt, K[q], P[q]), "bacc": balanced_acc(q, yv, pv, yt, pt), "pt": pt}

    def per_q(runs, q, w, key):
        return [r[q][w][key] for r in runs]

    # paired bootstrap: macro AUROC over trained questions, CLM multi (mean over seeds) - probes
    def macro(pts_by_q, idx):
        return np.mean([safe_auroc(Y[q][1][idx], pts_by_q[q][idx], K[q], P[q]) for q in pts_by_q])
    diffs, zs = [], []
    clm_mean_pt = {q: np.mean([r[q][0]["pt"] for r in methods["clm_multi"]], 0) for q in fq.QUESTIONS}
    for idx in boots:
        diffs.append(macro({q: clm_mean_pt[q] for q in trained}, idx) - macro({q: lr[q]["pt"] for q in trained}, idx))
        zs.append(macro({q: clm_mean_pt[q] for q in holdout}, idx))
    full = np.arange(len(t.ids))
    paired = {"diff": float(macro({q: clm_mean_pt[q] for q in trained}, full) - macro({q: lr[q]["pt"] for q in trained}, full)),
              "ci": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))]}
    zero = {"auroc": float(macro({q: clm_mean_pt[q] for q in holdout}, full)),
            "ci": [float(np.nanpercentile(zs, 2.5)), float(np.nanpercentile(zs, 97.5))]}

    # ---------------------------------------------------------------- report
    res = {"encoder": a.encoder, "split": split, "seeds": seeds, "holdout": holdout, "trained": trained,
           "best": sweep["best"]["name"], "paired_trained_macro": paired, "zero_shot_macro_ensemble": zero,
           "questions": {}}
    for q in fq.QUESTIONS:
        res["questions"][q] = {
            "clm_multi": {w: {k: per_q(methods["clm_multi"], q, w, k) for k in ("auroc", "bacc")} for w in range(3)},
            "clm_multi_random": {w: {k: per_q(methods["clm_multi_random"], q, w, k) for k in ("auroc", "bacc")}
                                 for w in range(3)},
            "clm_single": {w: {k: [r[w][k] for r in single[q]] for k in ("auroc", "bacc")} for w in range(3)},
            "lr": {"auroc": lr[q]["auroc"], "bacc": lr[q]["bacc"]}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_md = a.out if not a.dry_run else a.out.replace(".md", ".dryrun-val.md")
    json.dump(res, open(out_md.replace(".md", ".json"), "w"), indent=1)

    Q = res["questions"]
    def macro_of(method, qs, w=0, key="auroc"):
        if method == "lr":
            return statistics.mean(Q[q]["lr"][key] for q in qs)
        return [statistics.mean(Q[q][method][w][key][i] for q in qs) for i in range(len(seeds))]

    n, npos = len(t.ids), int(t.y.sum())
    L = ["# FracAtlas multi-question experiment: one CLM vision head, many typed questions", "",
         f"Research prototype, not a clinical tool. Encoder **{a.encoder}**, evaluated {date.today().isoformat()} on "
         f"the **{split}** split ({n} images, {npos} fractured)"
         + (" — DRY RUN: val stands in for test." if a.dry_run else
            ", once; every choice (config, C, thresholds) was made on val."), "",
         f"One state head is trained on {len(trained)} questions at once ({', '.join(NAMES[q] for q in trained)}); "
         f"the {len(holdout)} view questions ({', '.join(NAMES[q] for q in holdout)}) are **held out**: never "
         "trained on and never used for selection, so the head answers them zero-shot from the option text alone. "
         f"Best val config `{res['best']}`; CLM rows are mean ± sd over seeds {', '.join(map(str, seeds))}.", "",
         "## Summary", "",
         "| macro AUROC | trained questions | held-out questions (zero-shot) |", "|---|---:|---:|",
         f"| CLM multi-question head | {fmt(macro_of('clm_multi', trained))} | {fmt(macro_of('clm_multi', holdout))} |",
         f"| same head, random option vectors | {fmt(macro_of('clm_multi_random', trained))} | "
         f"{fmt(macro_of('clm_multi_random', holdout))} |",
         f"| one CLM head per question | {fmt(macro_of('clm_single', trained))} | "
         f"{fmt(macro_of('clm_single', holdout))} (supervised) |",
         f"| one linear probe per question | {fmt(macro_of('lr', trained))} | "
         f"{fmt(macro_of('lr', holdout))} (supervised) |", "",
         f"- Paired bootstrap ({N_BOOT} resamples), macro AUROC over trained questions, CLM multi (seed-mean "
         f"probabilities) − linear probes: **{paired['diff']:+.3f} [{paired['ci'][0]:+.3f}, {paired['ci'][1]:+.3f}]**.",
         f"- Zero-shot macro AUROC on the held-out questions (seed-mean probabilities): **{zero['auroc']:.3f} "
         f"[{zero['ci'][0]:.3f}, {zero['ci'][1]:.3f}]**; chance is 0.5. Rows marked *supervised* had labels for "
         "those questions and are an upper reference, not a comparison.", ""]
    verdict = open(a.verdict).read().strip() if os.path.exists(a.verdict) and not a.dry_run else "<!-- verdict -->"
    L += [verdict, "", "## Per question", "",
          "AUROC (macro one-vs-rest for region and fracture count); BA = balanced accuracy (binary: val Youden "
          "threshold; otherwise argmax).", "",
          "| question | positives / n | CLM multi AUROC | CLM multi BA | random options AUROC | single head AUROC "
          "| linear probe AUROC | linear probe BA |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for q in fq.QUESTIONS:
        r = Q[q]
        yt = Y[q][1]
        cnt = f"{int((yt == P[q]).sum())} / {n}" if P[q] is not None else f"{K[q]} classes"
        tag = " (held out)" if q in holdout else ""
        L.append(f"| {NAMES[q]}{tag} | {cnt} | {fmt(r['clm_multi'][0]['auroc'])} | {fmt(r['clm_multi'][0]['bacc'])} "
                 f"| {fmt(r['clm_multi_random'][0]['auroc'])} | {fmt(r['clm_single'][0]['auroc'])} "
                 f"| {fmt(r['lr']['auroc'])} | {fmt(r['lr']['bacc'])} |")
    L += ["", "## Paraphrased options", "",
          "The same heads, asked with option wordings 1 and 2, which no head saw during training. Random option "
          "vectors cannot follow a paraphrase (each wording is a new random vector), so they show what the text "
          "tower contributes.", "",
          "| question | CLM multi: wording 0 / 1 / 2 | random options: 0 / 1 / 2 | single head: 0 / 1 / 2 |",
          "|---|---|---|---|"]
    for q in fq.QUESTIONS:
        r = Q[q]
        cell = lambda m: " / ".join(f"{statistics.mean(r[m][w]['auroc']):.3f}" for w in range(3))  # noqa: E731
        L.append(f"| {NAMES[q]}{' (held out)' if q in holdout else ''} | {cell('clm_multi')} | "
                 f"{cell('clm_multi_random')} | {cell('clm_single')} |")
    L.append(f"| **macro, trained** | " + " | ".join(
        " / ".join(f"{statistics.mean(macro_of(m, trained, w)):.3f}" for w in range(3))
        for m in ("clm_multi", "clm_multi_random", "clm_single")) + " |")
    L += ["", "## Setup", "",
          "- Same data, splits, cached image embeddings and released action head as the binary experiment "
          "(`results/fracatlas.md`). Questions and wordings: `train/fracatlas_questions.py`; option texts embedded "
          "once with Qwen3-8B (`train/embed_options.py --multiq`).",
          "- One image embedding answers every question (the question text does not reach the image encoder).",
          "- Loss: class-weighted cross-entropy per question over its own options, averaged over trained questions; "
          "`logit_scale` frozen at 100; early stopping on mean val AUROC over trained questions.",
          "- Caveats: per-image splits (no patient ids); hardware has few positives "
          f"({int((Y['hardware'][1] == P['hardware']).sum())} in {split}) and 97 of the 99 hardware images are "
          "fractured, so it overlaps with the fracture question; region and view labels are easy for every method.",
          ""]
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[eval-multiq] -> {out_md} (+ .json)")


if __name__ == "__main__":
    main()
