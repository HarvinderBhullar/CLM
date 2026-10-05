#!/usr/bin/env python3
"""Evaluate the pre-registered alignment experiment; write results/fracatlas_align_medsiglip.md.

    python evaluation/vision_eval_align.py --dry-run     # val stands in for test
    python evaluation/vision_eval_align.py               # the one test evaluation

Protocol: research/protocol_alignment.md. Variants come from train/sweep_vision_align.py (A1, A2, A3 with
every lambda, A3g) and train/sweep_vision_multi.py (A0, the head trained from scratch); the reference is
MedSigLIP's own text tower with no training. Per variant and question: test AUROC (seed mean ± sd), macro
over trained and held-out questions, paraphrased wordings, and the 95% stratified bootstrap CI of the
held-out macro AUROC (seed-mean probabilities). H1 / H2 are checked with the pre-registered criteria.
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

A0_TRAINED = 0.952          # pre-registered reference values (multi-question experiment, test)
SIGLIP_HELD_OUT = 0.660


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--align", default="runs/fracatlas/align/medsiglip/sweep.json")
    ap.add_argument("--multiq", default="runs/fracatlas/multiq/medsiglip/sweep.json")
    ap.add_argument("--out", default="results/fracatlas_align_medsiglip.md")
    ap.add_argument("--verdict", default="results/fracatlas_align_medsiglip_verdict.md")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    split = "val" if a.dry_run else "test"
    sw, mq = json.load(open(a.align)), json.load(open(a.multiq))
    seeds, holdout = sw["seeds"], mq["holdout"]
    trained = [q for q in fq.QUESTIONS if q not in holdout]
    ob = torch.load(os.path.join(a.data, "options_multiq.pt"), map_location="cpu")
    bank = dict(zip(ob["texts"], ob["raw"]))
    t = fracatlas_data.load_split("medsiglip", split, a.data)
    Y = {q: fq.labels(q, t.rows) for q in fq.QUESTIONS}
    K = {q: len(fq.options(q)[0]) for q in fq.QUESTIONS}
    P = {q: fq.positive_index(q) for q in fq.QUESTIONS}
    boots = boot_idx(t.y.numpy(), N_BOOT)

    variants = {"A0 from scratch (multi-question head)": mq["best"]["dir"],
                "A1 text-only, strict corpus": sw["A1"]["strict"]["best"]["dir"],
                "A1 text-only, general corpus": sw["A1"]["general"]["best"]["dir"]}
    for vid, v in sw["finetune"].items():
        label = {"A2": "A2 A1-strict init, fine-tuned, λ = 0"}.get(vid) or (
            f"A3 A1-strict init, fine-tuned, λ = {v['align_weight']:g}" if vid.startswith("A3_")
            else f"A3g A1-general init, fine-tuned, λ = {v['align_weight']:g}")
        variants[label] = v["dir"]

    res = {"split": split, "seeds": seeds, "holdout": holdout, "variants": {}}
    for label, d in variants.items():
        heads = [Head(os.path.join(d, f"s{s}", "best_head.pt"), bank) for s in seeds]
        per = {q: {w: [] for w in range(fq.N_WORDINGS)} for q in fq.QUESTIONS}
        mean_pt = {}
        for q in fq.QUESTIONS:
            for w in range(fq.N_WORDINGS):
                pts = [h.probs(t.x, q, w) for h in heads]
                per[q][w] = [safe_auroc(Y[q], p, K[q], P[q]) for p in pts]
                if w == 0:
                    mean_pt[q] = np.mean(pts, 0)
        macro = lambda qs, w=0: [statistics.mean(per[q][w][i] for q in qs) for i in range(len(seeds))]  # noqa: E731
        zs = [np.mean([safe_auroc(Y[q][i], mean_pt[q][i], K[q], P[q]) for q in holdout]) for i in boots]
        res["variants"][label] = {
            "dir": d, "per_question": {q: per[q] for q in fq.QUESTIONS},
            "trained": macro(trained), "held_out": macro(holdout),
            "trained_paraphrase": [macro(trained, w) for w in (1, 2)],
            "held_out_paraphrase": [macro(holdout, w) for w in (1, 2)],
            "held_out_ensemble": float(np.mean([safe_auroc(Y[q], mean_pt[q], K[q], P[q]) for q in holdout])),
            "held_out_ci": [float(np.nanpercentile(zs, 2.5)), float(np.nanpercentile(zs, 97.5))]}

    # MedSigLIP's own text tower (no training)
    sb = torch.load(os.path.join(a.data, "options_multiq_medsiglip.pt"), map_location="cpu")
    stext, sscale = dict(zip(sb["texts"], sb["emb"])), sb["meta"]["logit_scale"]
    sp = {q: torch.softmax(sscale * t.x @ torch.stack([stext[x] for x in fq.descriptions(q, 0)[1]]).float().t(),
                           -1).double().numpy() for q in fq.QUESTIONS}
    s_per = {q: safe_auroc(Y[q], sp[q], K[q], P[q]) for q in fq.QUESTIONS}
    zs = [np.mean([safe_auroc(Y[q][i], sp[q][i], K[q], P[q]) for q in holdout]) for i in boots]
    res["siglip_zero_shot"] = {"per_question": s_per, "trained": statistics.mean(s_per[q] for q in trained),
                               "held_out": statistics.mean(s_per[q] for q in holdout),
                               "held_out_ci": [float(np.nanpercentile(zs, 2.5)), float(np.nanpercentile(zs, 97.5))]}

    V = res["variants"]
    a1 = V["A1 text-only, strict corpus"]
    a3 = next(v for k, v in V.items() if k.startswith("A3 ") and k.endswith("λ = 1"))
    res["hypotheses"] = {
        "H1": {"held_out_ci_low": a1["held_out_ci"][0], "supported": a1["held_out_ci"][0] > 0.5},
        "H2": {"held_out_ci_low": a3["held_out_ci"][0], "trained": statistics.mean(a3["trained"]),
               "supported": a3["held_out_ci"][0] > 0.5 and statistics.mean(a3["trained"]) >= A0_TRAINED - 0.01}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_md = a.out if not a.dry_run else a.out.replace(".md", ".dryrun-val.md")
    json.dump(res, open(out_md.replace(".md", ".json"), "w"), indent=1)

    ms = lambda v: f"{statistics.mean(v):.3f} ± {statistics.stdev(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}"  # noqa: E731
    n, npos = len(t.ids), int(t.y.sum())
    L = ["# FracAtlas: keeping zero-shot ability in a CLM vision head (MedSigLIP)", "",
         f"Research prototype, not a clinical tool. Pre-registered protocol: `research/protocol_alignment.md`. "
         f"Evaluated {date.today().isoformat()} on the **{split}** split ({n} images, {npos} fractured)"
         + (" — DRY RUN: val stands in for test." if a.dry_run else ", once, after all runs and selections."), "",
         f"Trained questions: {', '.join(NAMES[q] for q in trained)}. Held out (never trained on, never used for "
         f"selection): {', '.join(NAMES[q] for q in holdout)}. Rows are mean ± sd over seeds "
         f"{', '.join(map(str, seeds))}; the CI is the 95% stratified bootstrap of the held-out macro AUROC of the "
         "seed-mean probabilities.", "",
         "## Summary (macro AUROC)", "",
         "| variant | image labels | trained questions | held-out questions | held-out 95% CI |",
         "|---|---|---:|---:|---|"]
    for label, v in V.items():
        lab = "none" if label.startswith("A1") else "5 trained questions"
        L.append(f"| {label} | {lab} | {ms(v['trained'])} | {ms(v['held_out'])} | "
                 f"[{v['held_out_ci'][0]:.3f}, {v['held_out_ci'][1]:.3f}] |")
    s = res["siglip_zero_shot"]
    L += [f"| MedSigLIP text tower, no training (reference) | none | {s['trained']:.3f} | {s['held_out']:.3f} | "
          f"[{s['held_out_ci'][0]:.3f}, {s['held_out_ci'][1]:.3f}] |", "",
          "## Pre-registered hypotheses", "",
          f"- **H1** (A1 strict, held-out CI lower bound > 0.5): lower bound "
          f"{res['hypotheses']['H1']['held_out_ci_low']:.3f} → **{'supported' if res['hypotheses']['H1']['supported'] else 'not supported'}**.",
          f"- **H2** (A3 strict λ = 1: held-out CI lower bound > 0.5 and trained macro ≥ {A0_TRAINED - 0.01:.3f}): "
          f"lower bound {res['hypotheses']['H2']['held_out_ci_low']:.3f}, trained {res['hypotheses']['H2']['trained']:.3f} "
          f"→ **{'supported' if res['hypotheses']['H2']['supported'] else 'not supported'}**.", ""]
    verdict = open(a.verdict).read().strip() if os.path.exists(a.verdict) and not a.dry_run else "<!-- verdict -->"
    L += [verdict, "", "## Per question (wording 0)", "",
          "| variant | " + " | ".join(NAMES[q] + (" *" if q in holdout else "") for q in fq.QUESTIONS) + " |",
          "|---|" + "---:|" * len(fq.QUESTIONS)]
    for label, v in V.items():
        L.append(f"| {label} | " + " | ".join(f"{statistics.mean(v['per_question'][q][0]):.3f}" for q in fq.QUESTIONS) + " |")
    L.append("| MedSigLIP text tower | " + " | ".join(f"{s['per_question'][q]:.3f}" for q in fq.QUESTIONS) + " |")
    L += ["", "\\* held out.", "", "## Paraphrased options (macro AUROC, wording 1 / 2)", "",
          "| variant | trained questions | held-out questions |", "|---|---|---|"]
    for label, v in V.items():
        L.append(f"| {label} | " + " / ".join(f"{statistics.mean(x):.3f}" for x in v["trained_paraphrase"]) + " | "
                 + " / ".join(f"{statistics.mean(x):.3f}" for x in v["held_out_paraphrase"]) + " |")
    L += ["", "## Setup", "",
          "- Corpus (`train/fracatlas_text_corpus.py`): template radiograph descriptions. *Strict*: 6,156 sentences, "
          "none containing view vocabulary (checked by regex); *general*: strict + 648 sentences mentioning views. "
          "No sentence equals an option text. Embedded by MedSigLIP's text tower and by Qwen3-8B.",
          "- A1: symmetric InfoNCE between state_head(MedSigLIP text) and the frozen CLM action embedding of the same "
          "sentence; lr and modality-gap shift chosen by mean val zero-shot AUROC over the trained questions.",
          "- A2 / A3: fine-tuning on the trained questions (class-weighted CE per question) + λ · alignment loss on a "
          "random batch of 256 corpus sentences per step; early stopping on mean val AUROC over trained questions; "
          "λ not selected.",
          f"- References from the protocol: A0 trained macro {A0_TRAINED} and MedSigLIP zero-shot held-out "
          f"{SIGLIP_HELD_OUT} (both test, multi-question experiment).", ""]
    open(out_md, "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\n[eval-align] -> {out_md} (+ .json)")


if __name__ == "__main__":
    main()
