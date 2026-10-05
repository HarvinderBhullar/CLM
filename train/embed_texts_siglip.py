#!/usr/bin/env python3
"""Embed the multi-question option descriptions with MedSigLIP's own text tower (zero-shot baseline).

    python train/embed_texts_siglip.py        # -> data/fracatlas/options_multiq_medsiglip.pt

MedSigLIP's image and text towers were trained together, so its text embeddings live in the same
space as the cached image embeddings (``train/embed_images.py``, pooled image features). Scoring an
image against each option's description needs no FracAtlas training at all: the baseline any
learned head should be compared with. Texts are padded to 64 tokens, as SigLIP was trained.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracatlas_questions as fq  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/medsiglip-448")
    ap.add_argument("--out", default="data/fracatlas/options_multiq_medsiglip.pt")
    ap.add_argument("--texts", default=None, help="embed the texts of this JSON list instead (e.g. a text corpus)")
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model, dtype=torch.float32).eval()
    import json
    texts = json.load(open(a.texts)) if a.texts else fq.all_descriptions()
    out = []
    for i in range(0, len(texts), a.chunk):
        x = tok(texts[i:i + a.chunk], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        with torch.no_grad():
            t = model.get_text_features(**x)
            out.append(F.normalize(t if torch.is_tensor(t) else t.pooler_output, dim=-1))
    t = torch.cat(out)
    torch.save({"texts": texts, "emb": t, "meta": {"model": a.model, "padding": "max_length 64",
                                                   "logit_scale": float(model.logit_scale.exp()),
                                                   "logit_bias": float(model.logit_bias)}}, a.out)
    print(f"[siglip-text] {len(texts)} descriptions -> {a.out} {tuple(t.shape)}; "
          f"logit_scale {float(model.logit_scale.exp()):.2f}, bias {float(model.logit_bias):.2f}")


if __name__ == "__main__":
    main()
