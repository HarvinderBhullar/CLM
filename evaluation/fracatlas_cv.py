#!/usr/bin/env python3
"""Repeated cross-validation of the FracAtlas findings; writes results/fracatlas_cv_<encoder>.md.

    python evaluation/fracatlas_cv.py --encoder medsiglip --scheme image
    python evaluation/fracatlas_cv.py --encoder medsiglip --scheme blocked

Post-hoc robustness analysis (not pre-registered). The single 70/15/15 split has 109 fractured test
images, so test AUROC intervals (about ±0.04) are wider than most differences between methods. Here
all 4,022 images are pooled and the main comparisons are repeated over ``--repeats`` × ``--folds``
outer folds. In each fold the remaining images are split again into train and val (``--val-frac``);
hyperparameters are the ones selected on the original val split (fixed, not re-tuned), linear-probe C
and early stopping use the inner val, and the outer fold is evaluated once.

Schemes:

* ``image``: folds stratified by the fracture label, per image (as the original split).
* ``blocked``: folds stratified by label over contiguous blocks of ``--block`` image ids. FracAtlas
  has no patient ids, but consecutive ids are often views of one study (lag-1 label correlation 0.66
  vs about 0.40 from lag 2 on), so keeping id blocks together removes most within-study leakage.

Methods per fold (one seed per fold; the fold-to-fold spread includes seed variance):

* fracture only: CLM head (text options), CLM head (random option vectors), linear probe;
  fracture AUROC also on the hardware-free subset (97 of 99 hardware images are fractured).
* multi-question: one CLM head on the five trained questions (views held out), one linear probe per
  trained question; macro AUROC on trained questions, held-out views (zero-shot) and paraphrases.

Differences CLM − probe are summarised by mean, sd, wins, and a 95% interval from the corrected
resampled t-test (Nadeau & Bengio, 2003), which accounts for overlapping training sets across folds.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import date  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "train"))
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from clm.heads import default_checkpoint, make_head  # noqa: E402
from finetune_vision_multi import auroc, option_vectors  # noqa: E402

# hyperparameters selected on the original val split (results/fracatlas*.md, research/log.md)
SELECTED = {"fracture": {"medsiglip": (3e-5, 0.1), "qwen3-vl-4b": (3e-5, 0.1)},
            "multiq": {"medsiglip": (3e-5, 0.1), "qwen3-vl-4b": (1e-5, 0.01)}}
C_GRID = list(np.logspace(-5, 1, 7))


def safe_auroc(y, p, k, pos):
    try:
        return auroc(y, p, k, pos)
    except ValueError:
        return float("nan")


def pooled(encoder: str, data: str):
    """All splits concatenated, ordered by image id."""
    d = fracatlas_data.load_splits(encoder, fracatlas_data.SPLITS, data)
    rows = sum((d[s].rows for s in fracatlas_data.SPLITS), [])
    x = torch.cat([d[s].x for s in fracatlas_data.SPLITS])
    o = np.argsort([int(r["image_id"][3:10]) for r in rows])
    return x[o], [rows[i] for i in o]


def folds(rows, scheme: str, k: int, repeats: int, block: int, val_frac: float):
    """Yield (repeat, fold, train idx, val idx, test idx)."""
    y = np.array([int(r["fractured"]) for r in rows])
    groups = np.array([int(r["image_id"][3:10]) // block for r in rows]) if scheme == "blocked" else np.arange(len(y))
    n_inner = max(2, round(1 / val_frac))
    for rep in range(repeats):
        outer = StratifiedGroupKFold(k, shuffle=True, random_state=rep) if scheme == "blocked" \
            else StratifiedKFold(k, shuffle=True, random_state=rep)
        for f, (rest, te) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
            inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=100 + rep * k + f) if scheme == "blocked" \
                else StratifiedKFold(n_inner, shuffle=True, random_state=100 + rep * k + f)
            tr_i, va_i = next(inner.split(np.zeros(len(rest)), y[rest], groups[rest]))
            yield rep, f, rest[tr_i], rest[va_i], te


def train_head(xtr, xva, qs, hidden, head_cfg, proj, scale, lr, wd, seed, dev, epochs=200, patience=20, batch=64):
    """Class-weighted CE per question, averaged; early stopping on mean val AUROC (as finetune_vision_multi)."""
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xt, xv = ((xtr - mu) / sd).to(dev), ((xva - mu) / sd).to(dev)
    sh = make_head(proj=proj, hidden=hidden, **head_cfg).to(dev)
    opt = torch.optim.AdamW(sh.parameters(), lr=lr, weight_decay=wd)

    @torch.no_grad()
    def val_score():
        sh.eval()
        z = F.normalize(sh(xv), dim=-1)
        sh.train()
        return float(np.mean([auroc(s["yva"], torch.softmax(scale * z @ s["A"].t(), -1).cpu().numpy(),
                                    len(s["A"]), s["pos"]) for s in qs]))

    best, bad, state = val_score(), 0, {k: v.clone() for k, v in sh.state_dict().items()}
    g = torch.Generator().manual_seed(seed)
    n = len(xt)
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=g).split(batch):
            idx = idx.to(dev)
            z = F.normalize(sh(xt[idx]), dim=-1)
            loss = torch.stack([F.cross_entropy(scale * z @ s["A"].t(), s["ytr"][idx], weight=s["w"])
                                for s in qs]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sh.parameters(), 1.0); opt.step()
        v = val_score()
        if v > best + 1e-6:
            best, bad, state = v, 0, {k: t.clone() for k, t in sh.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    sh.load_state_dict(state)
    sh.eval()

    @torch.no_grad()
    def embed(x):
        return F.normalize(sh(((x - mu) / sd).to(dev)), dim=-1)
    return embed


def probe(xtr, ytr, xva, yva, xte, k, pos):
    """Standardised balanced logistic regression, C on val AUROC; returns test probabilities."""
    sc = StandardScaler().fit(xtr)
    a, b, c = sc.transform(xtr), sc.transform(xva), sc.transform(xte)
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(a, ytr)
        s = safe_auroc(yva, m.predict_proba(b), k, pos)
        if best is None or s > best[0]:
            best = (s, m)
    return best[1].predict_proba(c)


def corrected_ci(d: list[float], n_train: int, n_test: int):
    """Nadeau-Bengio corrected resampled t interval for the mean of per-fold differences."""
    d = np.asarray([v for v in d if v == v])
    j = len(d)
    var = d.var(ddof=1) * (1 / j + n_test / n_train)
    h = stats.t.ppf(0.975, j - 1) * np.sqrt(var)
    return float(d.mean() - h), float(d.mean() + h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=list(fracatlas_data.ENCODERS))
    ap.add_argument("--scheme", choices=["image", "blocked"], default="image")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--out", default=None, help="default results/fracatlas_cv_<encoder>_<scheme>.md")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in (matmul|dot)")
    out = a.out or f"results/fracatlas_cv_{a.encoder}_{a.scheme}.md"
    dev = torch.device(a.device)

    x, rows = pooled(a.encoder, a.data)
    hidden = x.shape[1]
    ob = torch.load(os.path.join(a.data, "options_multiq.pt"), map_location="cpu")
    bank = dict(zip(ob["texts"], ob["raw"]))
    ck = torch.load(default_checkpoint(), map_location="cpu", weights_only=False)
    c0 = ck["cfg"]
    head_cfg = {k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
    proj = c0.get("projection_dim", 512)
    ah = make_head(proj=proj, hidden=c0.get("hidden_size", 4096), **head_cfg).eval()
    ah.load_state_dict(ck["action_head"])
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

    trained = [q for q in fq.QUESTIONS if q not in fq.HOLDOUT]
    Y = {q: fq.labels(q, rows) for q in fq.QUESTIONS}
    K = {q: len(fq.options(q)[0]) for q in fq.QUESTIONS}
    P = {q: fq.positive_index(q) for q in fq.QUESTIONS}
    hw = np.array([r["hardware"] == "1" for r in rows])
    A = {(q, w, src): option_vectors(fq.options(q, w)[1], src, bank, ah, proj).to(dev)
         for q in fq.QUESTIONS for w in range(fq.N_WORDINGS) for src in ("text", "random")}

    def qspec(q, src, tr, va):
        ytr = Y[q][tr]
        counts = np.bincount(ytr, minlength=K[q]).astype(np.float64)
        w = np.where(counts > 0, len(ytr) / (K[q] * np.maximum(counts, 1)), 0.0)
        return {"A": A[(q, 0, src)], "pos": P[q], "yva": Y[q][va], "ytr": torch.as_tensor(ytr, device=dev),
                "w": torch.as_tensor(w, dtype=torch.float32, device=dev)}

    def answer(embed, xte, q, w, src="text"):
        return torch.softmax(scale * embed(xte) @ A[(q, w, src)].t(), -1).cpu().double().numpy()

    res = {"encoder": a.encoder, "scheme": a.scheme, "folds": a.folds, "repeats": a.repeats, "block": a.block,
           "selected": {k: v[a.encoder] for k, v in SELECTED.items()}, "per_fold": []}
    t0 = time.time()
    for rep, f, tr, va, te in folds(rows, a.scheme, a.folds, a.repeats, a.block, a.val_frac):
        seed = rep * a.folds + f
        xtr, xva, xte = x[tr], x[va], x[te]
        nhw = ~hw[te]
        r = {"repeat": rep, "fold": f, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
             "test_pos": int((Y["fracture"][te] == P["fracture"]).sum())}
        yf, kf, pf = Y["fracture"][te], K["fracture"], P["fracture"]
        lr, wd = SELECTED["fracture"][a.encoder]
        for src in ("text", "random"):
            emb = train_head(xtr, xva, [qspec("fracture", src, tr, va)], hidden, head_cfg, proj, scale, lr, wd, seed, dev)
            p = answer(emb, xte, "fracture", 0, src)
            r[f"fracture_clm_{src}"] = safe_auroc(yf, p, kf, pf)
            r[f"fracture_clm_{src}_nohw"] = safe_auroc(yf[nhw], p[nhw], kf, pf)
        xn = {s: v.numpy().astype(np.float64) for s, v in (("tr", xtr), ("va", xva), ("te", xte))}
        pp = {}
        for q in trained:
            pp[q] = probe(xn["tr"], Y[q][tr], xn["va"], Y[q][va], xn["te"], K[q], P[q])
        r["fracture_probe"] = safe_auroc(yf, pp["fracture"], kf, pf)
        r["fracture_probe_nohw"] = safe_auroc(yf[nhw], pp["fracture"][nhw], kf, pf)

        lr, wd = SELECTED["multiq"][a.encoder]
        emb = train_head(xtr, xva, [qspec(q, "text", tr, va) for q in trained], hidden, head_cfg, proj, scale,
                         lr, wd, seed, dev)
        per = {q: [safe_auroc(Y[q][te], answer(emb, xte, q, w), K[q], P[q]) for w in range(fq.N_WORDINGS)]
               for q in fq.QUESTIONS}
        r["multiq_per_question"] = per
        r["multiq_trained"] = float(np.nanmean([per[q][0] for q in trained]))
        r["multiq_held_out"] = float(np.nanmean([per[q][0] for q in fq.HOLDOUT]))
        r["multiq_trained_paraphrase"] = [float(np.nanmean([per[q][w] for q in trained])) for w in (1, 2)]
        r["probe_trained"] = float(np.nanmean([safe_auroc(Y[q][te], pp[q], K[q], P[q]) for q in trained]))
        res["per_fold"].append(r)
        print(f"[cv] {a.encoder} {a.scheme} rep {rep} fold {f} ({(time.time() - t0) / 60:.1f} min): "
              f"fracture clm {r['fracture_clm_text']:.3f} rand {r['fracture_clm_random']:.3f} "
              f"probe {r['fracture_probe']:.3f} | multiq clm {r['multiq_trained']:.3f} probe "
              f"{r['probe_trained']:.3f} held-out {r['multiq_held_out']:.3f}", flush=True)

    F_ = res["per_fold"]
    col = lambda k: [r[k] for r in F_]  # noqa: E731
    ntr, nte = statistics.mean(r["n_train"] + r["n_val"] for r in F_), statistics.mean(r["n_test"] for r in F_)
    summ = {}
    for k in ("fracture_clm_text", "fracture_clm_random", "fracture_probe", "fracture_clm_text_nohw",
              "fracture_clm_random_nohw", "fracture_probe_nohw", "multiq_trained", "probe_trained", "multiq_held_out"):
        summ[k] = {"mean": float(np.nanmean(col(k))), "sd": float(np.nanstd(col(k), ddof=1))}
    for k in (0, 1):
        v = [r["multiq_trained_paraphrase"][k] for r in F_]
        summ[f"multiq_trained_paraphrase{k + 1}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    diffs = {"fracture: CLM − probe": ("fracture_clm_text", "fracture_probe"),
             "fracture: CLM − random options": ("fracture_clm_text", "fracture_clm_random"),
             "fracture, no hardware: CLM − probe": ("fracture_clm_text_nohw", "fracture_probe_nohw"),
             "five questions: shared CLM − probes": ("multiq_trained", "probe_trained"),
             "held-out views: CLM − 0.5": ("multiq_held_out", None)}
    dsum = {}
    for name, (k1, k2) in diffs.items():
        d = [r[k1] - (r[k2] if k2 else 0.5) for r in F_]
        dsum[name] = {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)), "wins": int(sum(v > 0 for v in d)),
                      "ci": corrected_ci(d, ntr, nte)}
    res["summary"], res["differences"] = summ, dsum
    res["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out.replace(".md", ".json"), "w"), indent=1)

    s = lambda k: f"{summ[k]['mean']:.3f} ± {summ[k]['sd']:.3f}"  # noqa: E731
    nf = len(F_)
    L = [f"# FracAtlas cross-validation: {a.encoder}, {a.scheme} folds", "",
         f"Post-hoc robustness analysis (not pre-registered), {date.today().isoformat()}. Research prototype, not a "
         "clinical tool. Script: `evaluation/fracatlas_cv.py`.", "",
         f"All 4,022 images pooled; {a.repeats} × {a.folds}-fold outer CV ({nf} folds, about {nte:.0f} test images "
         f"and {statistics.mean(r['test_pos'] for r in F_):.0f} fractured per fold)"
         + (f", folds built from blocks of {a.block} consecutive image ids (approximate study grouping)"
            if a.scheme == "blocked" else ", folds per image") + ". Inner val split for early stopping and probe C; "
         f"hyperparameters fixed at the values selected on the original val split. Values are mean ± sd over folds.", "",
         "## AUROC", "", "| | mean ± sd over folds |", "|---|---:|",
         f"| fracture, CLM head | {s('fracture_clm_text')} |",
         f"| fracture, CLM head with random option vectors | {s('fracture_clm_random')} |",
         f"| fracture, linear probe | {s('fracture_probe')} |",
         f"| fracture without hardware images, CLM head | {s('fracture_clm_text_nohw')} |",
         f"| fracture without hardware images, linear probe | {s('fracture_probe_nohw')} |",
         f"| five trained questions (macro), one shared CLM head | {s('multiq_trained')} |",
         f"| five trained questions (macro), one probe per question | {s('probe_trained')} |",
         f"| five trained questions, paraphrase 1 / 2 | {s('multiq_trained_paraphrase1')} / {s('multiq_trained_paraphrase2')} |",
         f"| three held-out view questions (macro, zero-shot) | {s('multiq_held_out')} |", "",
         "## Differences", "",
         "| comparison | mean | sd | folds > 0 | 95% CI (corrected resampled t) |", "|---|---:|---:|---:|---|"]
    for name, v in dsum.items():
        L.append(f"| {name} | {v['mean']:+.3f} | {v['sd']:.3f} | {v['wins']}/{nf} | "
                 f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |")
    L += ["", f"Runtime {res['minutes']} min on {a.device}.", ""]
    open(out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
