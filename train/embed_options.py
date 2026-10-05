#!/usr/bin/env python3
"""Option (action-side) embeddings for the FracAtlas question, through the released action head.

    python train/embed_options.py                  # -> data/fracatlas/options.pt

The action tower is unchanged: Qwen3-8B, last-token pooling of the final-norm hidden
state, L2-normalised, tokenised with ``embed_utils.Recipe.text_ids(keep="tail")`` exactly
as ``finetune.py --task choice`` does; then the released ``action_head`` (``clm.heads.HeadPair``,
the code ``clm-serve`` runs). Runs with ``transformers`` instead of vLLM (no vLLM on Apple
silicon), in bf16, the dtype the weights ship in and the one vLLM would serve them at.

Both wire forms are embedded, so a head trained on one can be served as either:
``choice`` (each option's description verbatim) and ``noul`` (``"true: ..."`` /
``"false: ..."``, see ``clm.schema.candidates``).

Writes ``{"sets": {form: {"keys", "texts", "raw" [K, hidden], "proj" [K, P]}}, "meta": {...}}``.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import embed_utils  # noqa: E402
from clm.heads import HeadPair, default_checkpoint  # noqa: E402
from clm.schema import build_pairs  # noqa: E402

INSTRUCTIONS = "Is there a bone fracture?"
OPTIONS = {"fracture": "Radiograph showing an acute bone fracture.",
           "no_fracture": "Radiograph of intact bones with no fracture."}
NOUL = {"true": OPTIONS["fracture"], "false": OPTIONS["no_fracture"]}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Text-only sanity states: the pipeline is right if the released heads pick the matching option.
SANITY = {"fracture": ["X-ray report: transverse fracture of the distal radius with dorsal angulation.",
                       "Radiograph shows a displaced fracture of the femoral neck."],
          "no_fracture": ["X-ray report: no acute fracture or dislocation. Bones are intact.",
                          "Radiograph of the hand: normal study, no fracture seen."]}


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LastToken:
    """Qwen3 backbone (no lm_head): final-norm hidden state of the last token, one text at a time."""

    def __init__(self, name: str, device: str, dtype: torch.dtype):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, dtype=dtype, device_map=device).eval()
        self.hidden = self.model.config.hidden_size
        self.device = device

    @torch.no_grad()
    def __call__(self, id_lists: list[list[int]]) -> np.ndarray:
        out = []
        for ids in id_lists:            # unpadded, so the last position is the last real token
            h = self.model(input_ids=torch.tensor([ids], device=self.device)).last_hidden_state
            out.append(h[0, -1].float().cpu().numpy())
        return embed_utils.l2(np.stack(out))

    @torch.no_grad()
    def batch(self, id_lists: list[list[int]], pad_id: int) -> np.ndarray:
        """One right-padded forward pass for all texts: the weights stream through memory once,
        which matters when they do not fit next to everything else. Same vectors as ``__call__``."""
        L = max(map(len, id_lists))
        ids = torch.full((len(id_lists), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(id_lists), L), dtype=torch.long)
        for i, t in enumerate(id_lists):
            ids[i, :len(t)] = torch.tensor(t); mask[i, :len(t)] = 1
        h = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
        last = mask.sum(1).to(h.device) - 1
        return embed_utils.l2(h[torch.arange(len(id_lists), device=h.device), last].float().cpu().numpy())


def embed_multiq(a) -> None:
    """Raw Qwen3-8B embeddings of every option text of the multi-question set (projected at training time)."""
    import fracatlas_questions
    out = a.out if a.out != "data/fracatlas/options.pt" else "data/fracatlas/options_multiq.pt"
    texts = json.load(open(a.texts)) if a.texts else fracatlas_questions.all_texts()
    recipe = embed_utils.Recipe(a.model, a.max_len)
    t0 = time.time()
    enc = LastToken(a.model, a.device, DTYPES[a.dtype])
    print(f"[options] {a.model} on {a.device} ({a.dtype}), loaded in {time.time() - t0:.0f}s; "
          f"{len(texts)} option texts", flush=True)
    pad = recipe.tok.pad_token_id if recipe.tok.pad_token_id is not None else recipe.tok.eos_token_id
    ids = [recipe.text_ids(t, keep="tail") for t in texts]
    raw = torch.from_numpy(np.concatenate([enc.batch(ids[i:i + a.chunk], pad) for i in range(0, len(ids), a.chunk)]))
    if not torch.isfinite(raw).all():
        raise SystemExit("non-finite option embedding")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"texts": texts, "raw": raw,
                "meta": {"model": a.model, "dtype": a.dtype, "pooling": "last_token",
                         "tokenization": "embed_utils.Recipe.text_ids(keep='tail')", "max_len": a.max_len,
                         "texts": a.texts or "train/fracatlas_questions.py"}}, out)
    print(f"[options] -> {out} {tuple(raw.shape)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", default=None, help="checkpoint holding the action head (default: released head)")
    ap.add_argument("--out", default="data/fracatlas/options.pt")
    ap.add_argument("--max-len", type=int, default=2048, help="token budget, as finetune.py --task choice")
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--no-sanity", action="store_true", help="skip the text-only check through both heads")
    ap.add_argument("--multiq", action="store_true",
                    help="embed every option wording of train/fracatlas_questions.py instead "
                         "(default --out data/fracatlas/options_multiq.pt)")
    ap.add_argument("--texts", default=None,
                    help="with --multiq: embed the texts of this JSON list instead (e.g. a text corpus)")
    ap.add_argument("--chunk", type=int, default=256, help="with --multiq: texts per padded forward pass")
    a = ap.parse_args()
    if a.multiq:
        embed_multiq(a)
        return

    ckpt = a.ckpt or default_checkpoint()
    if not ckpt:
        raise SystemExit("no checkpoint: run clm-download or pass --ckpt")
    heads = HeadPair("released", ckpt, device="cpu").ensure()

    pairs = build_pairs("", {"choice": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": OPTIONS},
                             "noul": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": NOUL}})
    sanity = [build_pairs(s, {"q": {"type": "noul", "instructions": INSTRUCTIONS}})["q"][0]
              for v in SANITY.values() for s in v]
    texts = list(dict.fromkeys([t for _, _, ts in pairs.values() for t in ts] + ([] if a.no_sanity else sanity)))

    recipe = embed_utils.Recipe(a.model, a.max_len)
    t0 = time.time()
    enc = LastToken(a.model, a.device, DTYPES[a.dtype])
    print(f"[options] {a.model} on {a.device} ({a.dtype}), hidden {enc.hidden}, loaded in {time.time() - t0:.0f}s",
          flush=True)
    if enc.hidden != heads.cfg.get("hidden_size", enc.hidden):
        raise SystemExit(f"encoder hidden {enc.hidden} != head hidden {heads.cfg['hidden_size']}")
    raw = dict(zip(texts, enc([recipe.text_ids(t, keep="tail") for t in texts])))
    del enc

    sets = {}
    for form, (_, keys, ts) in pairs.items():
        r = np.stack([raw[t] for t in ts])
        sets[form] = {"keys": keys, "texts": ts, "raw": torch.from_numpy(r),
                      "proj": heads.project_actions(r).cpu()}
        r, p = sets[form]["raw"], sets[form]["proj"]   # torch: numpy+Accelerate warns spuriously on matmul
        print(f"[options] {form}: {dict(zip(keys, ts))}\n          cos(raw) {float(r[0] @ r[1]):.4f}  "
              f"cos(proj) {float(p[0] @ p[1]):.4f}  proj {tuple(p.shape)}", flush=True)

    if not a.no_sanity:
        zs = heads.project_states(np.stack([raw[s] for s in sanity])).cpu()
        print(f"[options] sanity (text states, released state head, scale {heads.scale:.1f}):")
        hits = 0
        for (label, s), z in zip([(k, s) for k, v in SANITY.items() for s in v], zs):
            prob = torch.softmax(heads.scale * sets["choice"]["proj"] @ z, 0)
            pick = sets["choice"]["keys"][int(prob.argmax())]
            hits += pick == label
            print(f"          {label:<12} -> {pick:<12} p(fracture) {prob[0]:.3f}  {s[:60]}")
        print(f"[options] sanity {hits}/{len(sanity)} correct", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"sets": sets, "meta": {"model": a.model, "dtype": a.dtype, "pooling": "last_token",
                                       "tokenization": "embed_utils.Recipe.text_ids(keep='tail')",
                                       "max_len": a.max_len, "instructions": INSTRUCTIONS,
                                       "ckpt": os.path.basename(ckpt), "ckpt_md5": md5sum(ckpt),
                                       "logit_scale": float(torch.as_tensor(
                                           torch.load(ckpt, map_location="cpu")["logit_scale"]))}},
               a.out)
    print(f"[options] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
