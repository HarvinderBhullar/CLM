#!/usr/bin/env python3
"""Alignment-preserving vision state head (pre-registered in research/protocol_alignment.md).

    # stage 1, text only (no image labels): MedSigLIP text embedding -> CLM action embedding
    python train/finetune_vision_align.py --stage align --corpus strict --out-dir runs/.../A1/s0
    # stage 2: fine-tune on the trained questions, keeping lambda * the alignment loss
    python train/finetune_vision_align.py --stage finetune --init runs/.../A1/s0 --align-weight 1 --out-dir ...

MedSigLIP's image and text towers share one space. Stage 1 trains the state head to map MedSigLIP *text*
embeddings of corpus sentences onto the frozen CLM action embedding of the same sentence (Qwen3-8B + the
released action head), with symmetric InfoNCE; applied to an *image* embedding, the head should then land
near the action embeddings of matching descriptions, so held-out questions can be answered zero-shot.

* Inputs are standardised with the corpus text mean / std. ``--gap-shift`` standardises images with the
  image mean instead (removes the offset between the modalities). Both are affine and fold into the head's
  first layer, so the checkpoint keeps the CLM format and takes raw L2-normalised image embeddings.
* Stage 1 selection: mean val **zero-shot** AUROC over the trained questions (a development set; stage 1
  never trains on their labels). Stage 2 selection: mean val AUROC over the trained questions.
* The held-out questions are never evaluated here, and the test split is never loaded.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from clm.heads import default_checkpoint, make_head  # noqa: E402
from finetune_vision import fold_standardisation  # noqa: E402
from finetune_vision_multi import auroc, option_vectors  # noqa: E402

ENCODER = "google/medsiglip-448"


def load_corpus(data: str, name: str):
    texts = json.load(open(os.path.join(data, f"corpus_{name}.json")))
    sig = torch.load(os.path.join(data, "corpus_general_medsiglip.pt"), map_location="cpu")
    qw = torch.load(os.path.join(data, "corpus_general_qwen3-8b.pt"), map_location="cpu")
    si, qi = {t: i for i, t in enumerate(sig["texts"])}, {t: i for i, t in enumerate(qw["texts"])}
    return texts, sig["emb"][[si[t] for t in texts]].float(), qw["raw"][[qi[t] for t in texts]].float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["align", "finetune"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--corpus", choices=["strict", "general"], default="strict")
    ap.add_argument("--gap-shift", action="store_true", help="standardise images with the image mean (align stage)")
    ap.add_argument("--init", default=None, help="finetune: an align-stage run directory")
    ap.add_argument("--align-weight", type=float, default=1.0, help="finetune: lambda on the alignment loss")
    ap.add_argument("--holdout", nargs="*", default=list(fq.HOLDOUT))
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 align, 3e-5 finetune")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=None, help="default 10 align, 20 finetune")
    ap.add_argument("--batch", type=int, default=64, help="images per step (finetune)")
    ap.add_argument("--text-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    lr = a.lr or (1e-4 if a.stage == "align" else 3e-5)
    patience = a.patience or (10 if a.stage == "align" else 20)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)

    init = None
    if a.stage == "finetune":
        if not a.init:
            raise SystemExit("--stage finetune needs --init <align run dir>")
        init = torch.load(os.path.join(a.init, "train_state.pt"), map_location="cpu", weights_only=False)
        a.corpus, a.gap_shift = init["corpus"], init["gap_shift"]
    d = fracatlas_data.load_splits("medsiglip", ("train", "val"), a.data)
    tr, va = d["train"], d["val"]
    if tr.meta["encoder"] != ENCODER:
        raise SystemExit("the alignment experiment needs MedSigLIP embeddings")
    hidden = tr.x.shape[1]
    texts, sig_t, qw_t = load_corpus(a.data, a.corpus)

    ck_path = default_checkpoint()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    c0 = ck["cfg"]
    head_cfg = {k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
    proj, action_hidden = c0.get("projection_dim", 512), c0.get("hidden_size", 4096)
    ah = make_head(proj=proj, hidden=action_hidden, **head_cfg).eval()
    ah.load_state_dict(ck["action_head"])
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    with torch.no_grad():
        target = F.normalize(ah(qw_t), dim=-1).to(dev)                      # [N, P] frozen action embeddings

    mu_t, sd_t = sig_t.mean(0), sig_t.std(0).clamp_min(1e-6)
    mu_img = tr.x.mean(0) if a.gap_shift else mu_t
    if init:
        mu_t, sd_t, mu_img = init["mu_t"], init["sd_t"], init["mu_img"]
    txt = ((sig_t - mu_t) / sd_t).to(dev)
    xtr, xva = ((tr.x - mu_img) / sd_t).to(dev), ((va.x - mu_img) / sd_t).to(dev)

    trained = [q for q in fq.QUESTIONS if q not in a.holdout]
    bank = dict(zip(*(lambda o: (o["texts"], o["raw"]))(torch.load(os.path.join(a.data, "options_multiq.pt")))))
    qs = {}
    for q in trained:
        keys, otexts = fq.options(q, 0)
        ytr = fq.labels(q, tr.rows)
        counts = np.bincount(ytr, minlength=len(keys)).astype(np.float64)
        w = np.where(counts > 0, len(ytr) / (len(keys) * np.maximum(counts, 1)), 0.0)
        qs[q] = {"keys": keys, "pos": fq.positive_index(q), "A": option_vectors(otexts, "text", bank, ah, proj).to(dev),
                 "ytr": torch.as_tensor(ytr, device=dev), "yva": fq.labels(q, va.rows),
                 "w": torch.as_tensor(w, dtype=torch.float32, device=dev)}

    sh = make_head(proj=proj, hidden=hidden, **head_cfg).to(dev)
    if init:
        sh.load_state_dict(init["state_head_unfolded"])
    opt = torch.optim.AdamW(sh.parameters(), lr=lr, weight_decay=a.weight_decay)
    g = torch.Generator().manual_seed(a.seed)
    n_txt = len(texts)

    def align_loss(idx):
        s = F.normalize(sh(txt[idx]), dim=-1)
        lg = scale * s @ target[idx].t()
        lab = torch.arange(len(idx), device=dev)
        return (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab)) / 2

    @torch.no_grad()
    def evaluate():
        sh.eval()
        z = F.normalize(sh(xva), dim=-1)
        sh.train()
        per = {q: auroc(s["yva"], torch.softmax(scale * z @ s["A"].t(), -1).cpu().numpy(), len(s["keys"]), s["pos"])
               for q, s in qs.items()}
        return {"per_question": per, "mean_auroc": float(np.mean(list(per.values())))}

    def save(epoch, metrics):
        head = make_head(proj=proj, hidden=hidden, **head_cfg)
        unfolded = {k: v.detach().cpu().clone() for k, v in sh.state_dict().items()}
        head.load_state_dict(unfolded)
        fold_standardisation(head, mu_img, sd_t)
        torch.save({"state_head": head.state_dict(), "action_head": ck["action_head"],
                    "logit_scale": torch.as_tensor(ck["logit_scale"]).clone(),
                    "cfg": {**head_cfg, "projection_dim": proj, "hidden_size": hidden,
                            "action_hidden_size": action_hidden, "task": f"vision_align_{a.stage}",
                            "state_encoder": ENCODER, "state_modality": "image", "state_embedding": tr.meta,
                            "action_encoder": "Qwen/Qwen3-8B", "questions": {q: fq.wire(q, 0) for q in trained},
                            "holdout": a.holdout, "option_source": "text", "corpus": a.corpus,
                            "corpus_size": n_txt, "gap_shift": a.gap_shift, "align_weight": a.align_weight,
                            "lr": lr, "weight_decay": a.weight_decay, "seed": a.seed,
                            "init": a.init, "init_ckpt": os.path.basename(ck_path)},
                    "epoch": epoch, "metrics": metrics}, os.path.join(a.out_dir, "best_head.pt"))
        torch.save({"state_head_unfolded": unfolded, "mu_t": mu_t, "sd_t": sd_t, "mu_img": mu_img,
                    "corpus": a.corpus, "gap_shift": a.gap_shift}, os.path.join(a.out_dir, "train_state.pt"))

    m0 = evaluate()
    best, best_ep, bad, history = m0["mean_auroc"], 0, 0, [{"epoch": 0, "val": m0}]
    save(0, m0)
    print(f"[align] stage {a.stage} corpus {a.corpus} ({n_txt} texts) gap_shift {a.gap_shift} lr {lr} "
          f"lambda {a.align_weight if a.stage == 'finetune' else '-'} | epoch 0 val mean auroc {best:.4f}", flush=True)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tot = nb = 0
        if a.stage == "align":
            batches = [(None, idx) for idx in torch.randperm(n_txt, generator=g).split(a.text_batch)]
        else:
            batches = [(idx, torch.randint(0, n_txt, (a.text_batch,), generator=g))
                       for idx in torch.randperm(len(tr.ids), generator=g).split(a.batch)]
        for img_idx, txt_idx in batches:
            loss = torch.zeros((), device=dev)
            if a.stage == "align" or a.align_weight > 0:
                loss = align_loss(txt_idx.to(dev)) * (1.0 if a.stage == "align" else a.align_weight)
            if img_idx is not None:
                img_idx = img_idx.to(dev)
                z = F.normalize(sh(xtr[img_idx]), dim=-1)
                loss = loss + torch.stack([F.cross_entropy(scale * z @ s["A"].t(), s["ytr"][img_idx], weight=s["w"])
                                           for s in qs.values()]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sh.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        mv = evaluate()
        history.append({"epoch": ep, "train_loss": tot / nb, "val": mv})
        if mv["mean_auroc"] > best + 1e-6:
            best, best_ep, bad = mv["mean_auroc"], ep, 0
            save(ep, mv)
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[align] epoch {ep:>3} loss {tot / nb:.4f} val mean auroc {mv['mean_auroc']:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            break
    json.dump(history, open(os.path.join(a.out_dir, "history.json"), "w"), indent=1)
    json.dump({"best_epoch": best_ep, "val_mean_auroc": best, "val_per_question": history[best_ep]["val"]["per_question"],
               "stage": a.stage, "corpus": a.corpus, "gap_shift": a.gap_shift, "lr": lr,
               "align_weight": a.align_weight if a.stage == "finetune" else None,
               "minutes": round((time.time() - t0) / 60, 2), "args": vars(a)},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"[align] best epoch {best_ep}: val mean auroc {best:.4f} "
          + " ".join(f"{q} {v:.3f}" for q, v in history[best_ep]["val"]["per_question"].items()), flush=True)


if __name__ == "__main__":
    main()
