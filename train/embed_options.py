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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ckpt", default=None, help="checkpoint holding the action head (default: released head)")
    ap.add_argument("--out", default="data/fracatlas/options.pt")
    ap.add_argument("--max-len", type=int, default=2048, help="token budget, as finetune.py --task choice")
    ap.add_argument("--dtype", choices=DTYPES, default="bf16")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--no-sanity", action="store_true", help="skip the text-only check through both heads")
    a = ap.parse_args()

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
