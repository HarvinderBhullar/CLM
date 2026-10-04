#!/usr/bin/env python3
"""Train a CLM state head on image embeddings against the frozen text action tower (FracAtlas).

    python train/finetune_vision.py --encoder qwen3-vl-4b --out-dir runs/fracatlas/clm-qwen3vl4b/s0

Swap the state tower, keep the action tower: images are embedded once by
``train/embed_images.py``; the question's options are embedded once by Qwen3-8B and the
released action head (``train/embed_options.py``). Only ``state_head`` trains.

* logits = ``min(exp(logit_scale), 100) * cos(state_head(x), option)`` over the fixed options
  (``fracture``, ``no_fracture``); ``logit_scale`` is frozen at the released value unless
  ``--train-logit-scale`` (then it starts just under the cap, where ``clamp`` passes gradient).
* loss: class-weighted cross-entropy over those options (not in-batch InfoNCE: with two
  classes, same-label images would be pushed apart as negatives).
* inputs are standardised with train mean / std while training; the affine map is folded
  into the head's first Linear before saving, so the checkpoint keeps the ``make_head``
  architecture and takes the encoder's L2-normalised embedding directly.
* early stopping on val AUROC. The test split is never loaded here.

The checkpoint is the usual ``state_head`` / ``action_head`` / ``logit_scale`` / ``cfg`` dict;
``cfg`` adds ``state_encoder``, ``state_modality: "image"``, ``action_hidden_size`` and the
embedding / option recipe needed to reproduce the inputs at serving time.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import fracatlas_data  # noqa: E402
from clm.heads import default_checkpoint, make_head  # noqa: E402

FRACTURE = "fracture"


def fold_standardisation(head: nn.Module, mu: torch.Tensor, sd: torch.Tensor) -> None:
    """head(x) on (x - mu) / sd  ==  head'(x) on x, with W' = W / sd, b' = b - W' mu."""
    with torch.no_grad():
        w = head.inp.weight / sd
        head.inp.bias -= w @ mu
        head.inp.weight.copy_(w)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, help="qwen3-vl-4b, medsiglip or a HF id with cached embeddings")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--options", default=None, help="default DATA/options.pt")
    ap.add_argument("--option-form", choices=["choice", "noul"], default="choice")
    ap.add_argument("--option-source", choices=["text", "random"], default="text",
                    help="random: fixed random unit vectors instead of the text options (control: "
                         "does the action tower matter, or only the head?)")
    ap.add_argument("--init-ckpt", default=None, help="released checkpoint (action head, logit_scale, head cfg)")
    ap.add_argument("--init-state-head", action="store_true",
                    help="warm-start state_head from --init-ckpt (only when the encoder width matches)")
    ap.add_argument("--train-logit-scale", action="store_true")
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--input-dropout", type=float, default=0.0)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)

    d = fracatlas_data.load_splits(a.encoder, ("train", "val"), a.data)
    tr, va = d["train"], d["val"]
    hidden = tr.x.shape[1]
    if hidden != tr.meta["hidden_size"]:
        raise SystemExit(f"embedding width {hidden} != meta hidden_size {tr.meta['hidden_size']}")

    opts = torch.load(a.options or os.path.join(a.data, "options.pt"), map_location="cpu")
    oset = opts["sets"][a.option_form]
    pos_key = FRACTURE if a.option_form == "choice" else "true"
    pos = oset["keys"].index(pos_key)
    A = oset["proj"].float()                                              # [K, P], frozen, L2-normalised
    if a.option_source == "random":
        A = F.normalize(torch.randn(A.shape, generator=torch.Generator().manual_seed(12345)), dim=-1)
    A = A.to(dev)

    ck_path = a.init_ckpt or default_checkpoint()
    if not ck_path:
        raise SystemExit("no released checkpoint: run clm-download or pass --init-ckpt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    c0 = ck["cfg"]
    head_cfg = {k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
    proj = c0.get("projection_dim", ck.get("projection_dim", 512))
    action_hidden = c0.get("hidden_size", 4096)
    if A.shape[1] != proj:
        raise SystemExit(f"option projections are {A.shape[1]}-d, head projects to {proj}")

    sh = make_head(proj=proj, hidden=hidden, **head_cfg).to(dev)
    if a.init_state_head:
        if hidden != action_hidden:
            raise SystemExit(f"--init-state-head needs a {action_hidden}-d encoder, got {hidden}")
        sh.load_state_dict(ck["state_head"])
    released = float(torch.as_tensor(ck["logit_scale"]).float())
    init_scale = min(released, math.log(100.0) - 1e-3) if a.train_logit_scale else released
    logit_scale = nn.Parameter(torch.tensor(init_scale, device=dev), requires_grad=a.train_logit_scale)

    if a.no_standardize:
        mu, sd = torch.zeros(hidden), torch.ones(hidden)
    else:
        mu, sd = tr.x.mean(0), tr.x.std(0).clamp_min(1e-6)
    xtr, xva = ((tr.x - mu) / sd).to(dev), ((va.x - mu) / sd).to(dev)
    # option index of each image's gold answer
    ytr = torch.where(tr.y == 1, pos, 1 - pos).to(dev)
    n_pos = int(tr.y.sum()); n = len(tr.y)
    w = torch.ones(len(oset["keys"]))
    if a.class_weight == "balanced":
        w[pos], w[1 - pos] = n / (2 * n_pos), n / (2 * (n - n_pos))
    w = w.to(dev)

    def logits(x):
        z = F.normalize(sh(x), dim=-1)
        return logit_scale.exp().clamp(max=100.0) * z @ A.t()

    @torch.no_grad()
    def evaluate(x, y):
        sh.eval()
        lg = logits(x)
        sh.train()
        p = torch.softmax(lg, -1)[:, pos].cpu().numpy()
        yo = torch.where(y == 1, pos, 1 - pos).to(dev)
        return {"auroc": float(roc_auc_score(y.numpy(), p)),
                "loss": float(F.cross_entropy(lg, yo, weight=w)),
                "mean_p_pos": float(p.mean())}

    params = list(sh.parameters()) + ([logit_scale] if a.train_logit_scale else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    print(f"[vision] {tr.meta['encoder']} hidden {hidden} | train {n} ({n_pos} pos) val {len(va.y)} "
          f"({int(va.y.sum())} pos) | head {head_cfg} proj {proj} "
          f"{sum(p.numel() for p in sh.parameters()) / 1e6:.1f}M params | scale {math.exp(init_scale):.1f} "
          f"{'trainable' if a.train_logit_scale else 'frozen'} | init {'released' if a.init_state_head else 'random'}",
          flush=True)

    def blob(epoch, metrics):
        head = make_head(proj=proj, hidden=hidden, **head_cfg)
        head.load_state_dict({k: v.detach().cpu() for k, v in sh.state_dict().items()})
        fold_standardisation(head, mu, sd)
        return {"state_head": head.state_dict(), "action_head": ck["action_head"],
                "logit_scale": logit_scale.detach().cpu().clone(),
                "cfg": {**head_cfg, "projection_dim": proj, "hidden_size": hidden,
                        "action_hidden_size": action_hidden, "task": "vision_choice",
                        "state_encoder": tr.meta["encoder"], "state_modality": "image",
                        "state_embedding": tr.meta, "action_encoder": opts["meta"]["model"],
                        "instructions": opts["meta"]["instructions"], "option_form": a.option_form,
                        "options": dict(zip(oset["keys"], oset["texts"])), "positive": pos_key,
                        "option_source": a.option_source,
                        "init_ckpt": os.path.basename(ck_path), "init_state_head": a.init_state_head,
                        "train_logit_scale": a.train_logit_scale, "standardize": not a.no_standardize,
                        "lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch,
                        "input_dropout": a.input_dropout, "class_weight": a.class_weight, "seed": a.seed},
                "epoch": epoch, "metrics": metrics}

    m0 = evaluate(xva, va.y)
    best, best_ep, bad, history = m0["auroc"], 0, 0, [{"epoch": 0, "val": m0}]
    torch.save(blob(0, m0), os.path.join(a.out_dir, "best_head.pt"))
    print(f"[vision] epoch 0 val auroc {m0['auroc']:.4f}", flush=True)
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tot = nb = 0
        for idx in torch.randperm(n, generator=g).split(a.batch):
            idx = idx.to(dev)
            x = F.dropout(xtr[idx], a.input_dropout) if a.input_dropout else xtr[idx]
            loss = F.cross_entropy(logits(x), ytr[idx], weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item(); nb += 1
        mv = evaluate(xva, va.y)
        history.append({"epoch": ep, "train_loss": tot / nb, "val": mv, "scale": float(logit_scale.exp())})
        if mv["auroc"] > best + 1e-6:
            best, best_ep, bad = mv["auroc"], ep, 0
            torch.save(blob(ep, mv), os.path.join(a.out_dir, "best_head.pt"))
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[vision] epoch {ep:>3} loss {tot / nb:.4f} val auroc {mv['auroc']:.4f} loss {mv['loss']:.4f} "
                  f"scale {float(logit_scale.exp()):.1f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= a.patience:
            print(f"[vision] early stop at epoch {ep}", flush=True)
            break

    # the saved (folded) head must reproduce the trained one on raw embeddings
    best_ck = torch.load(os.path.join(a.out_dir, "best_head.pt"), map_location="cpu", weights_only=False)
    head = make_head(proj=proj, hidden=hidden, **head_cfg).eval()
    head.load_state_dict(best_ck["state_head"])
    with torch.no_grad():
        z = F.normalize(head(va.x), dim=-1) @ A.cpu().t() * float(best_ck["logit_scale"].exp().clamp(max=100.0))
        auroc_saved = float(roc_auc_score(va.y.numpy(), torch.softmax(z, -1)[:, pos].numpy()))
    if abs(auroc_saved - best) > 2e-3:
        raise SystemExit(f"saved head gives val auroc {auroc_saved:.4f}, trained {best:.4f}: folding is wrong")
    summary = {"best_epoch": best_ep, "val_auroc": best, "val_auroc_saved_head": auroc_saved,
               "init_val_auroc": m0["auroc"], "minutes": round((time.time() - t0) / 60, 2), "args": vars(a)}
    json.dump(history, open(os.path.join(a.out_dir, "history.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"[vision] best epoch {best_ep}: val auroc {best:.4f} (saved head {auroc_saved:.4f}) "
          f"-> {a.out_dir}/best_head.pt", flush=True)


if __name__ == "__main__":
    main()
