#!/usr/bin/env python3
"""Distil MedSigLIP's zero-shot ability into a CLM vision head (pre-registered in research/protocol_distill.md).

    # stage 1, no image labels: match MedSigLIP's image->corpus-text distribution with the CLM scoring
    python train/finetune_vision_distill.py --stage distill --corpus general --out-dir runs/.../D1/s0
    # stage 2: fine-tune on the trained questions, keeping lambda * the distillation loss
    python train/finetune_vision_distill.py --stage finetune --init runs/.../D1/s0 --distill-weight 1 --out-dir ...

Teacher: ``softmax_j(T * cos(MedSigLIP_image(i), MedSigLIP_text(j)))`` over a batch of corpus sentences j.
Student: ``softmax_j(100 * cos(state_head(image i), action_head(Qwen3-8B(j))))``, the CLM scoring with the frozen
released action tower. Loss ``KL(teacher || student)``. Images come from the train split; their labels are unused
by this loss. Unlike the alignment experiment the head is trained on image embeddings, where it is used.

* Inputs are standardised with the train image mean / std (folded into the head's first layer, so the checkpoint
  keeps the CLM format); stage 2 keeps the stage-1 statistics so weights can be interpolated.
* Stage 1 selection: mean val **zero-shot** AUROC over the trained questions (a development set). Stage 2
  selection: mean val AUROC over the trained questions.
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
from finetune_vision_align import ENCODER, load_corpus  # noqa: E402
from finetune_vision_multi import auroc, option_vectors  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["distill", "finetune"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--corpus", choices=["strict", "general"], default="general")
    ap.add_argument("--teacher-scale", type=float, default=30.0, help="T in the teacher softmax (distill stage)")
    ap.add_argument("--init", default=None, help="finetune: a distill-stage run directory")
    ap.add_argument("--distill-weight", type=float, default=1.0, help="finetune: lambda on the distillation loss")
    ap.add_argument("--holdout", nargs="*", default=list(fq.HOLDOUT))
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 distill, 3e-5 finetune")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=None, help="default 100 distill, 200 finetune")
    ap.add_argument("--patience", type=int, default=None, help="default 10 distill, 20 finetune")
    ap.add_argument("--batch", type=int, default=64, help="images per step")
    ap.add_argument("--text-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    lr = a.lr or (1e-4 if a.stage == "distill" else 3e-5)
    epochs = a.epochs or (100 if a.stage == "distill" else 200)
    patience = a.patience or (10 if a.stage == "distill" else 20)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)

    init = None
    if a.stage == "finetune":
        if not a.init:
            raise SystemExit("--stage finetune needs --init <distill run dir>")
        init = torch.load(os.path.join(a.init, "train_state.pt"), map_location="cpu", weights_only=False)
        a.corpus, a.teacher_scale = init["corpus"], init["teacher_scale"]
    d = fracatlas_data.load_splits("medsiglip", ("train", "val"), a.data)
    tr, va = d["train"], d["val"]
    if tr.meta["encoder"] != ENCODER:
        raise SystemExit("the distillation experiment needs MedSigLIP embeddings")
    hidden = tr.x.shape[1]
    texts, sig_t, qw_t = load_corpus(a.data, a.corpus)
    sig_t = F.normalize(sig_t, dim=-1).to(dev)

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

    mu, sd = (init["mu"], init["sd"]) if init else (tr.x.mean(0), tr.x.std(0).clamp_min(1e-6))
    raw_tr = F.normalize(tr.x, dim=-1).to(dev)                              # teacher input
    xtr, xva = ((tr.x - mu) / sd).to(dev), ((va.x - mu) / sd).to(dev)

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
    n_txt, n_img = len(texts), len(tr.ids)

    def distill_loss(img_idx, txt_idx):
        with torch.no_grad():
            pt = torch.log_softmax(a.teacher_scale * raw_tr[img_idx] @ sig_t[txt_idx].t(), -1)
        z = F.normalize(sh(xtr[img_idx]), dim=-1)
        ps = torch.log_softmax(scale * z @ target[txt_idx].t(), -1)
        return F.kl_div(ps, pt, log_target=True, reduction="batchmean")

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
        fold_standardisation(head, mu, sd)
        torch.save({"state_head": head.state_dict(), "action_head": ck["action_head"],
                    "logit_scale": torch.as_tensor(ck["logit_scale"]).clone(),
                    "cfg": {**head_cfg, "projection_dim": proj, "hidden_size": hidden,
                            "action_hidden_size": action_hidden, "task": f"vision_distill_{a.stage}",
                            "state_encoder": ENCODER, "state_modality": "image", "state_embedding": tr.meta,
                            "action_encoder": "Qwen/Qwen3-8B", "questions": {q: fq.wire(q, 0) for q in trained},
                            "holdout": a.holdout, "option_source": "text", "corpus": a.corpus,
                            "corpus_size": n_txt, "teacher_scale": a.teacher_scale,
                            "distill_weight": a.distill_weight, "lr": lr, "weight_decay": a.weight_decay,
                            "seed": a.seed, "init": a.init, "init_ckpt": os.path.basename(ck_path)},
                    "epoch": epoch, "metrics": metrics}, os.path.join(a.out_dir, "best_head.pt"))
        torch.save({"state_head_unfolded": unfolded, "mu": mu, "sd": sd, "corpus": a.corpus,
                    "teacher_scale": a.teacher_scale}, os.path.join(a.out_dir, "train_state.pt"))

    m0 = evaluate()
    best, best_ep, bad, history = m0["mean_auroc"], 0, 0, [{"epoch": 0, "val": m0}]
    save(0, m0)
    print(f"[distill] stage {a.stage} corpus {a.corpus} ({n_txt} texts) T {a.teacher_scale:g} lr {lr} "
          f"lambda {a.distill_weight if a.stage == 'finetune' else '-'} | epoch 0 val mean auroc {best:.4f}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tot = nb = 0
        for img_idx in torch.randperm(n_img, generator=g).split(a.batch):
            txt_idx = torch.randint(0, n_txt, (a.text_batch,), generator=g).to(dev)
            img_idx = img_idx.to(dev)
            loss = torch.zeros((), device=dev)
            if a.stage == "distill" or a.distill_weight > 0:
                loss = distill_loss(img_idx, txt_idx) * (1.0 if a.stage == "distill" else a.distill_weight)
            if a.stage == "finetune":
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
            print(f"[distill] epoch {ep:>3} loss {tot / nb:.4f} val mean auroc {mv['mean_auroc']:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= patience:
            break
    json.dump(history, open(os.path.join(a.out_dir, "history.json"), "w"), indent=1)
    json.dump({"best_epoch": best_ep, "val_mean_auroc": best, "val_per_question": history[best_ep]["val"]["per_question"],
               "stage": a.stage, "corpus": a.corpus, "teacher_scale": a.teacher_scale, "lr": lr,
               "distill_weight": a.distill_weight if a.stage == "finetune" else None,
               "minutes": round((time.time() - t0) / 60, 2), "args": vars(a)},
              open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"[distill] best epoch {best_ep}: val mean auroc {best:.4f} "
          + " ".join(f"{q} {v:.3f}" for q, v in history[best_ep]["val"]["per_question"].items()), flush=True)


if __name__ == "__main__":
    main()
