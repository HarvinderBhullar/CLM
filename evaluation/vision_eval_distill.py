#!/usr/bin/env python3
"""Evaluate the pre-registered distillation experiment; write results/fracatlas_distill_medsiglip.md.

    python evaluation/vision_eval_distill.py --dry-run     # val, trained questions only
    python evaluation/vision_eval_distill.py               # the one test evaluation

Protocol: research/protocol_distill.md. Variants come from train/sweep_vision_distill.py (D1 strict / general,
D2, D3 with every lambda, W with every alpha) and train/sweep_vision_multi.py (A0); the reference is MedSigLIP's
own text tower with no training. The point estimate and its 95% stratified bootstrap CI are both computed on the
seed-averaged probabilities (a three-seed ensemble); seed mean ± sd is reported alongside. The dry run never
scores the held-out questions.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from vision_eval_multiq import N_BOOT, NAMES, Head, boot_idx, safe_auroc  # noqa: E402

A0_TRAINED = 0.952          # pre-registered reference (multi-question experiment, test)
TRAINED_MIN = A0_TRAINED - 0.01


def ci(zs):
    return [float(np.nanpercentile(zs, 2.5)), float(np.nanpercentile(zs, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--distill", default="runs/fracatlas/distill/medsiglip/sweep.json")
    ap.add_argument("--multiq", default="runs/fracatlas/multiq/medsiglip/sweep.json")
    ap.add_argument("--out", default="results/fracatlas_distill_medsiglip.md")
    ap.add_argument("--verdict", default="results/fracatlas_distill_medsiglip_verdict.md")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    split = "val" if a.dry_run else "test"
    sw, mq = json.load(open(a.distill)), json.load(open(a.multiq))
    seeds, holdout = sw["seeds"], mq["holdout"]
    trained = [q for q in fq.QUESTIONS if q not in holdout]
    scored = trained if a.dry_run else list(fq.QUESTIONS)
    ob = torch.load(os.path.join(a.data, "options_multiq.pt"), map_location="cpu")
    bank = dict(zip(ob["texts"], ob["raw"]))
    t = fracatlas_data.load_split("medsiglip", split, a.data)
    Y = {q: fq.labels(q, t.rows) for q in scored}
    K = {q: len(fq.options(q)[0]) for q in scored}
    P = {q: fq.positive_index(q) for q in scored}
    boots = boot_idx(t.y.numpy(), N_BOOT)

    variants = {"A0 from scratch (multi-question head)": mq["best"]["dir"],
                "D1 distillation only, strict corpus": sw["D1"]["strict"]["best"]["dir"],
                "D1 distillation only, general corpus": sw["D1"]["general"]["best"]["dir"]}
    for vid, v in sw["finetune"].items():
        variants["D2 D1 init, fine-tuned, λ = 0" if vid == "D2"
                 else f"D3 D1 init, fine-tuned + distillation, λ = {v['distill_weight']:g}"] = v["dir"]
    for vid, v in sw["interp"].items():
        variants[f"W interpolation D1 → D2, α = {v['alpha']:g}"] = v["dir"]

    def summarise(pts_by_q, qs):
        """Ensemble macro AUROC + CI over qs, from {q: [n, K] seed-mean probabilities}."""
        if not qs:
            return None, None
        est = float(np.mean([safe_auroc(Y[q], pts_by_q[q], K[q], P[q]) for q in qs]))
        zs = [np.mean([safe_auroc(Y[q][i], pts_by_q[q][i], K[q], P[q]) for q in qs]) for i in boots]
        return est, ci(zs)

    held = [q for q in holdout if q in scored]
    res = {"split": split, "seeds": seeds, "holdout": holdout, "variants": {}}
    for label, d in variants.items():
        heads = [Head(os.path.join(d, f"s{s}", "best_head.pt"), bank) for s in seeds]
        per, ens = {}, {}
        for q in scored:
            per[q] = {}
            for w in range(fq.N_WORDINGS):
                pts = [h.probs(t.x, q, w) for h in heads]
                per[q][w] = [safe_auroc(Y[q], p, K[q], P[q]) for p in pts]
                ens[(q, w)] = np.mean(pts, 0)
        seed_macro = lambda qs, w=0: [statistics.mean(per[q][w][i] for q in qs) for i in range(len(seeds))]  # noqa: E731
        v = {"dir": d, "per_question_seeds": {q: per[q] for q in scored},
             "per_question_ensemble": {q: safe_auroc(Y[q], ens[(q, 0)], K[q], P[q]) for q in scored}}
        for name, qs in (("trained", trained), ("held_out", held)):
            if not qs:
                continue
            v[f"{name}_seeds"] = seed_macro(qs)
            v[name], v[f"{name}_ci"] = summarise({q: ens[(q, 0)] for q in qs}, qs)
            v[f"{name}_paraphrase"] = [float(np.mean([safe_auroc(Y[q], ens[(q, w)], K[q], P[q]) for q in qs]))
                                       for w in (1, 2)]
        res["variants"][label] = v

    sb = torch.load(os.path.join(a.data, "options_multiq_medsiglip.pt"), map_location="cpu")
    stext, sscale = dict(zip(sb["texts"], sb["emb"])), sb["meta"]["logit_scale"]
    sp = {q: torch.softmax(sscale * t.x @ torch.stack([stext[x] for x in fq.descriptions(q, 0)[1]]).float().t(),
                           -1).double().numpy() for q in scored}
    s = {"per_question": {q: safe_auroc(Y[q], sp[q], K[q], P[q]) for q in scored}}
    s["trained"], s["trained_ci"] = summarise(sp, trained)
    if held:
        s["held_out"], s["held_out_ci"] = summarise(sp, held)
    res["siglip_zero_shot"] = s

    V = res["variants"]
    hyp = {}
    if not a.dry_run:
        d1 = V["D1 distillation only, general corpus"]
        d3 = V["D3 D1 init, fine-tuned + distillation, λ = 1"]
        w5 = V["W interpolation D1 → D2, α = 0.5"]
        hyp = {"H3": {"held_out_ci_low": d1["held_out_ci"][0], "supported": d1["held_out_ci"][0] > 0.5},
               "H4": {"held_out_ci_low": d3["held_out_ci"][0], "trained": d3["trained"],
                      "supported": d3["held_out_ci"][0] > 0.5 and d3["trained"] >= TRAINED_MIN},
               "H5": {"held_out_ci_low": w5["held_out_ci"][0], "trained": w5["trained"],
                      "supported": w5["held_out_ci"][0] > 0.5 and w5["trained"] >= TRAINED_MIN}}
    res["hypotheses"] = hyp
    out_md = a.out if not a.dry_run else a.out.replace(".md", ".dryrun-val.md")
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    json.dump(res, open(out_md.replace(".md", ".json"), "w"), indent=1)

    ms = lambda v: f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}"  # noqa: E731
    cis = lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731
    n, npos = len(t.ids), int(t.y.sum())
    L = ["# FracAtlas: distilling MedSigLIP's zero-shot ability into a CLM vision head", "",
         f"Research prototype, not a clinical tool. Pre-registered protocol: `research/protocol_distill.md`. "
         f"Evaluated {date.today().isoformat()} on the **{split}** split ({n} images, {npos} fractured)"
         + (" — DRY RUN: val, trained questions only." if a.dry_run else ", once, after all runs and selections."), "",
         f"Trained questions: {', '.join(NAMES[q] for q in trained)}. Held out (never trained on, never used for "
         f"selection): {', '.join(NAMES[q] for q in holdout)}. Estimates and 95% stratified bootstrap CIs are both "
         f"computed on the seed-averaged probabilities (seeds {', '.join(map(str, seeds))}); seed mean ± sd alongside.", "",
         "## Summary (macro AUROC)", ""]
    if a.dry_run:
        L += ["| variant | trained (ensemble) | trained CI | seed mean ± sd |", "|---|---:|---|---:|"]
        for label, v in V.items():
            L.append(f"| {label} | {v['trained']:.3f} | {cis(v['trained_ci'])} | {ms(v['trained_seeds'])} |")
        L.append(f"| MedSigLIP text tower, no training | {s['trained']:.3f} | {cis(s['trained_ci'])} | – |")
    else:
        L += ["| variant | image labels | trained | held out | held-out 95% CI | held out, seed mean ± sd |",
              "|---|---|---:|---:|---|---:|"]
        for label, v in V.items():
            lab = "none" if label.startswith("D1") else "5 trained questions"
            L.append(f"| {label} | {lab} | {v['trained']:.3f} | {v['held_out']:.3f} | {cis(v['held_out_ci'])} | "
                     f"{ms(v['held_out_seeds'])} |")
        L.append(f"| MedSigLIP text tower, no training (reference) | none | {s['trained']:.3f} | {s['held_out']:.3f} | "
                 f"{cis(s['held_out_ci'])} | – |")
        L += ["", "## Pre-registered hypotheses", ""]
        for h, txt in (("H3", "D1 general, held-out CI lower bound > 0.5"),
                       ("H4", f"D3 λ = 1: held-out CI lower bound > 0.5 and trained ≥ {TRAINED_MIN:.3f}"),
                       ("H5", f"W α = 0.5: held-out CI lower bound > 0.5 and trained ≥ {TRAINED_MIN:.3f}")):
            r = hyp[h]
            extra = f", trained {r['trained']:.3f}" if "trained" in r else ""
            L.append(f"- **{h}** ({txt}): lower bound {r['held_out_ci_low']:.3f}{extra} → "
                     f"**{'supported' if r['supported'] else 'not supported'}**.")
        verdict = open(a.verdict).read().strip() if os.path.exists(a.verdict) else "<!-- verdict -->"
        L += ["", verdict]
    L += ["", "## Per question (wording 0, ensemble)", "",
          "| variant | " + " | ".join(NAMES[q] + (" *" if q in holdout else "") for q in scored) + " |",
          "|---|" + "---:|" * len(scored)]
    for label, v in V.items():
        L.append(f"| {label} | " + " | ".join(f"{v['per_question_ensemble'][q]:.3f}" for q in scored) + " |")
    L.append("| MedSigLIP text tower | " + " | ".join(f"{s['per_question'][q]:.3f}" for q in scored) + " |")
    L += ["", "\\* held out.", "", "## Paraphrased options (macro AUROC, ensemble, wording 1 / 2)", "",
          "| variant | trained questions" + (" | held-out questions |" if held else " |"),
          "|---|---|" + ("---|" if held else "")]
    for label, v in V.items():
        row = f"| {label} | " + " / ".join(f"{x:.3f}" for x in v["trained_paraphrase"])
        if held:
            row += " | " + " / ".join(f"{x:.3f}" for x in v["held_out_paraphrase"])
        L.append(row + " |")
    L += ["", "## Setup", "",
          "- D1: KL(teacher ‖ student) on unlabelled train images; teacher softmax over 256 corpus sentences of "
          "T · cos(MedSigLIP image, MedSigLIP text), student the CLM scoring 100 · cos(state_head(image), "
          "action_head(Qwen3-8B text)). lr and T selected by mean val zero-shot AUROC over the trained questions.",
          "- D2 / D3: fine-tuning on the trained questions + λ · distillation loss; λ not selected.",
          "- W: per seed, (1 − α) · D1 general + α · D2 in weight space; α not selected.", ""]
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[eval-distill] -> {out_md} (+ .json)")


if __name__ == "__main__":
    main()
