#!/usr/bin/env python3
"""Image embeddings for the FracAtlas splits, one L2-normalised vector per image.

    python train/embed_images.py --encoder Qwen/Qwen3-VL-4B-Instruct
    python train/embed_images.py --encoder google/medsiglip-448

Encoders live in ``clm.image_embedder`` so the serving engine embeds images identically.

* Qwen3-VL (``*-VL-*``): a user turn holding the image and ``--instruction``, rendered
  with the chat template (no generation prompt, like ``embed_utils.Recipe.state_ids``);
  the state is the final-norm hidden state of the last token.
* SigLIP / MedSigLIP: the pooled image embedding (``get_image_features``).

Writes ``{out}/<encoder slug>/{split}.pt`` = ``{"emb": {image_id: fp16 [hidden]}, "meta": {...}}``.
Progress is saved every ``--save-every`` images and a rerun skips images already done, so
the script is resumable. Runs on MPS (``PYTORCH_ENABLE_MPS_FALLBACK=1`` for missing ops).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.image_embedder import QwenVL, Siglip  # noqa: E402  (shared with the serving engine)

INSTRUCTION = "Radiograph for fracture assessment."
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def slug(encoder: str) -> str:
    return encoder.replace("/", "__")


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load_split(data: str, split: str) -> list[dict]:
    with open(os.path.join(data, "splits", f"{split}.csv")) as f:
        return list(csv.DictReader(f))


def save(path: str, emb: dict, meta: dict) -> None:
    tmp = path + ".tmp"
    torch.save({"emb": emb, "meta": meta}, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--data", default="data/fracatlas")
    ap.add_argument("--out", default=None, help="default DATA/emb")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--instruction", default=INSTRUCTION, help="Qwen3-VL: text after the image")
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024, help="Qwen3-VL: resize cap per image")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=128)
    ap.add_argument("--dtype", choices=DTYPES, default="fp16")
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N images per split (smoke test)")
    a = ap.parse_args()

    dtype = DTYPES[a.dtype]
    siglip = "siglip" in a.encoder.lower()
    t0 = time.time()
    enc = Siglip(a.encoder, a.device, dtype) if siglip else QwenVL(a.encoder, a.device, dtype, a.max_pixels,
                                                                     a.instruction)
    print(f"[embed] {a.encoder} on {a.device} ({a.dtype}), hidden {enc.hidden}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    meta = {"encoder": a.encoder, "hidden_size": enc.hidden, "dtype": a.dtype,
            "pooling": "siglip_pooled" if siglip else "last_token",
            **({} if siglip else {"instruction": a.instruction, "max_pixels": a.max_pixels,
                                  "chat_template": True, "add_generation_prompt": False})}

    out_dir = os.path.join(a.out or os.path.join(a.data, "emb"), slug(a.encoder))
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(a.data, "raw")
    for split in a.splits:
        rows = load_split(a.data, split)[:a.limit]
        path = os.path.join(out_dir, f"{split}.pt")
        emb: dict = {}
        if os.path.exists(path):
            got = torch.load(path, map_location="cpu")
            if {k: v for k, v in got["meta"].items() if k != "complete"} != meta:
                raise SystemExit(f"{path} was made with different settings {got['meta']}; delete it to redo")
            emb = got["emb"]
        todo = [r for r in rows if r["image_id"] not in emb]
        # similar sizes share a batch, so little compute goes to padding
        todo.sort(key=lambda r: (math.prod(Image.open(os.path.join(raw, r["path"])).size), r["image_id"]))
        print(f"[embed] {split}: {len(rows)} images, {len(emb)} cached, {len(todo)} to do", flush=True)
        t0, since = time.time(), 0
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            images = [Image.open(os.path.join(raw, r["path"])).convert("RGB") for r in chunk]
            z = enc(images)
            if not torch.isfinite(z).all():
                raise SystemExit(f"non-finite embedding at {chunk[0]['image_id']}; retry with --dtype bf16 or fp32")
            z = F.normalize(z, dim=-1).cpu().half()
            if a.device == "mps":
                torch.mps.empty_cache()                 # hand freed activations back: unified memory is shared
            emb.update({r["image_id"]: v for r, v in zip(chunk, z)})
            since += len(chunk)
            if since >= a.save_every or i + a.batch >= len(todo):
                save(path, emb, {**meta, "complete": len(emb) == len(rows)})
                done = i + len(chunk)
                rate = done / (time.time() - t0)
                print(f"[embed] {split} {len(emb)}/{len(rows)}  {rate:.2f} img/s  "
                      f"eta {(len(todo) - done) / rate / 60:.1f} min", flush=True)
                since = 0
        if not todo and os.path.exists(path):
            print(f"[embed] {split}: already complete", flush=True)
    print(f"[embed] -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
