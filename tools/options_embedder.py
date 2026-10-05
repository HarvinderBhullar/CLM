#!/usr/bin/env python3
"""Serve the cached FracAtlas option embeddings as a tiny ``/v1/embeddings`` endpoint.

    python tools/options_embedder.py              # :8090, in place of vllm serve Qwen/Qwen3-8B
    clm-serve --model fracatlas-medsiglip=checkpoints/fracatlas-medsiglip.pt

A vision head only ever needs the Qwen3-8B embeddings of its option texts, and
``train/embed_options.py`` already computed them (Qwen3-8B, last token, bf16) into
``data/fracatlas/options.pt``. This serves exactly those vectors, so a vision head can be tried
locally without vLLM. Any other text is refused with HTTP 400: this is not a text encoder, and
``clm-latest`` (the text head) will not work against it.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--options", default="data/fracatlas/options.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    a = ap.parse_args()
    if not os.path.exists(a.options):
        sys.exit(f"{a.options} not found: run python train/embed_options.py first")

    opts = torch.load(a.options, map_location="cpu")
    vecs = {t: r.numpy().astype(np.float32) for s in opts["sets"].values() for t, r in zip(s["texts"], s["raw"])}
    app = FastAPI(title="cached option embeddings")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "qwen3-8b", "object": "model"}]}

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        body = await request.json()
        texts = body.get("input")
        texts = [texts] if isinstance(texts, str) else texts
        unknown = [t for t in texts if not isinstance(t, str) or t not in vecs]
        if unknown:
            raise HTTPException(400, f"not a cached option text: {str(unknown[0])[:80]!r}. This stand-in only "
                                     f"knows the {len(vecs)} texts in {a.options}; use vLLM for anything else")
        data = [{"object": "embedding", "index": i, "embedding": base64.b64encode(vecs[t].tobytes()).decode()}
                for i, t in enumerate(texts)]
        return {"object": "list", "data": data, "model": "qwen3-8b", "usage": {"prompt_tokens": 0, "total_tokens": 0}}

    print(f"[options-embedder] {len(vecs)} cached Qwen3-8B option embeddings from {a.options} "
          f"on http://{a.host}:{a.port}/v1/embeddings", flush=True)
    for t in vecs:
        print(f"  {t}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
