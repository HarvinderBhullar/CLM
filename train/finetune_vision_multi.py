#!/usr/bin/env python3
"""Train one vision state head on many typed questions at once (FracAtlas multi-question experiment).

    python train/finetune_vision_multi.py --encoder medsiglip --out-dir runs/fracatlas/multiq/clm/s0

One image embedding answers every question: the state head maps it to ``z_image``, each question's
option texts go through Qwen3-8B and the frozen released action head, and a question's answer is
the softmax of ``100 * cos(z_image, option)`` over its own options. The loss is class-weighted
cross-entropy per question, averaged over the training questions. Questions in ``--holdout`` are
never trained on and never used for model selection, so their answers are zero-shot.

* ``--questions fracture`` trains a single-question head (the per-question control).
* ``--option-source random`` replaces every option text by a fixed random unit vector (control:
  does the text tower matter, or only the head?).

Early stopping on the mean val AUROC over the training questions (macro one-vs-rest for questions
with more than two options). The test split is never loaded. The checkpoint keeps the CLM format
(released action head included); ``cfg`` lists the trained and held-out questions.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import fracatlas_data  # noqa: E402
import fracatlas_questions as fq  # noqa: E402
from clm.heads import default_checkpoint, make_head  # noqa: E402
from finetune_vision import fold_standardisation  # noqa: E402


def random_vector(text: str, dim: int) -> torch.Tensor:
    """A fixed random unit vector per text (seeded by the text), for the random-options control."""
    g = torch.Generator().manual_seed(int(hashlib.sha1(text.encode()).hexdigest()[:12], 16))
    return F.normalize(torch.randn(dim, generator=g), dim=0)


def option_vectors(texts: list[str], source: str, bank: dict, action_head, proj: int) -> torch.Tensor:
    """[K, proj] L2-normalised option vectors: released action head on Qwen3-8B embeddings, or random."""
    if source == "random":
        return torch.stack([random_vector(t, proj) for t in texts])
    with torch.no_grad():
        return F.normalize(action_head(torch.stack([bank[t] for t in texts]).float()), dim=-1)


def auroc(y: np.ndarray, p: np.ndarray, k: int, pos: int | None) -> float:
    """Binary AUROC on the positive option, or macro one-vs-rest AUROC over classes present."""
    if k == 2:
        return float(roc_auc_score(y == pos, p[:, pos]))
    present = [c for c in range(k) if 0 < (y == c).sum() < len(y)]
    return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in present]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", default=fracatlas_data.DATA)
    ap.add_argument("--options", default=None, help="default DATA/options_multiq.pt")
    ap.add_argument("--questions", nargs="+", default=None, help="training questions (default: all but --holdout)")
    ap.add_argument("--holdout", nargs="*", default=list(fq.HOLDOUT))
    ap.add_argument("--option-source", choices=["text", "random"], default="text")
    ap.add_argument("--init-ckpt", default=None, help="released checkpoint (action head, logit_scale, head cfg)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    train_q = a.questions or [q for q in fq.QUESTIONS if q not in a.holdout]
    if set(train_q) & set(a.holdout) and a.questions is None:
        raise SystemExit("a question cannot be both trained and held out")
    holdout = [q for q in a.holdout if q not in train_q]
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device(a.device)

    d = fracatlas_data.load_splits(a.encoder, ("train", "val"), a.data)
    tr, va = d["train"], d["val"]
    hidden = tr.x.shape[1]
    ob = torch.load(a.options or os.path.join(a.data, "options_multiq.pt"), map_location="cpu")
    bank = dict(zip(ob["texts"], ob["raw"]))

    ck_path = a.init_ckpt or default_checkpoint()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    c0 = ck["cfg"]
    head_cfg = {k: c0[k] for k in ("width", "depth", "activation", "layernorm", "residual")}
    proj = c0.get("projection_dim", 512)
    action_hidden = c0.get("hidden_size", 4096)
    ah = make_head(proj=proj, hidden=action_hidden, **head_cfg).eval()
    ah.load_state_dict(ck["action_head"])
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))

    qs = {}
    for q in train_q:
        keys, texts = fq.options(q, 0)
        ytr, yva = fq.labels(q, tr.rows), fq.labels(q, va.rows)
        counts = np.bincount(ytr, minlength=len(keys)).astype(np.float64)
        w = np.where(counts > 0, len(ytr) / (len(keys) * np.maximum(counts, 1)), 0.0)
        qs[q] = {"keys": keys, "texts": texts, "pos": fq.positive_index(q),
                 "A": option_vectors(texts, a.option_source, bank, ah, proj).to(dev),
                 "ytr": torch.as_tensor(ytr, device=dev), "yva": yva,
                 "w": torch.as_tensor(w, dtype=torch.float32, device=dev)}

    mu, sd = tr.x.mean(0), tr.x.std(0).clamp_min(1e-6)
    xtr, xva = ((tr.x - mu) / sd).to(dev), ((va.x - mu) / sd).to(dev)
    sh = make_head(proj=proj, hidden=hidden, **head_cfg).to(dev)
    opt = torch.optim.AdamW(sh.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n = len(tr.ids)
    print(f"[multiq] {tr.meta['encoder']} hidden {hidden} | train {n} val {len(va.ids)} | train questions "
          f"{train_q} | held out {holdout} | options {a.option_source} | scale {scale:.0f}", flush=True)

    @torch.no_grad()
    def evaluate():
        sh.eval()
        z = F.normalize(sh(xva), dim=-1)
        sh.train()
        out = {}
        for q, s in qs.items():
            p = torch.softmax(scale * z @ s["A"].t(), -1).cpu().numpy()
            out[q] = auroc(s["yva"], p, len(s["keys"]), s["pos"])
        return {"per_question": out, "mean_auroc": float(np.mean(list(out.values())))}

    def blob(epoch, metrics):
        head = make_head(proj=proj, hidden=hidden, **head_cfg)
        head.load_state_dict({k: v.detach().cpu() for k, v in sh.state_dict().items()})
        fold_standardisation(head, mu, sd)
        return {"state_head": head.state_dict(), "action_head": ck["action_head"],
                "logit_scale": torch.as_tensor(ck["logit_scale"]).clone(),
                "cfg": {**head_cfg, "projection_dim": proj, "hidden_size": hidden,
                        "action_hidden_size": action_hidden, "task": "vision_multiq",
                        "state_encoder": tr.meta["encoder"], "state_modality": "image",
                        "state_embedding": tr.meta, "action_encoder": ob["meta"]["model"],
                        "questions": {q: fq.wire(q, 0) for q in train_q}, "holdout": holdout,
                        "option_source": a.option_source, "init_ckpt": os.path.basename(ck_path),
                        "lr": a.lr, "weight_decay": a.weight_decay, "batch": a.batch, "seed": a.seed},
                "epoch": epoch, "metrics": metrics}

    m0 = evaluate()
    best, best_ep, bad, history = m0["mean_auroc"], 0, 0, [{"epoch": 0, "val": m0}]
    torch.save(blob(0, m0), os.path.join(a.out_dir, "best_head.pt"))
    g = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        tot = nb = 0
        for idx in torch.randperm(n, generator=g).split(a.batch):
            idx = idx.to(dev)
            z = F.normalize(sh(xtr[idx]), dim=-1)
            loss = torch.stack([F.cross_entropy(scale * z @ s["A"].t(), s["ytr"][idx], weight=s["w"])
                                for s in qs.values()]).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(sh.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        mv = evaluate()
        history.append({"epoch": ep, "train_loss": tot / nb, "val": mv})
        if mv["mean_auroc"] > best + 1e-6:
            best, best_ep, bad = mv["mean_auroc"], ep, 0
            torch.save(blob(ep, mv), os.path.join(a.out_dir, "best_head.pt"))
        else:
            bad += 1
        if ep % 10 == 0 or bad == 0:
            print(f"[multiq] epoch {ep:>3} loss {tot / nb:.4f} val mean auroc {mv['mean_auroc']:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
        if bad >= a.patience:
            break
    json.dump(history, open(os.path.join(a.out_dir, "history.json"), "w"), indent=1)
    best_metrics = history[best_ep]["val"]
    json.dump({"best_epoch": best_ep, "val_mean_auroc": best, "val_per_question": best_metrics["per_question"],
               "train_questions": train_q, "holdout": holdout, "minutes": round((time.time() - t0) / 60, 2),
               "args": vars(a)}, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
    print(f"[multiq] best epoch {best_ep}: val mean auroc {best:.4f} "
          + " ".join(f"{q} {v:.3f}" for q, v in best_metrics["per_question"].items()), flush=True)


if __name__ == "__main__":
    main()
