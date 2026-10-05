"""FracAtlas splits + cached image embeddings, shared by the vision trainer, baselines and eval.

Labels come from ``data/fracatlas/splits/{split}.csv`` (``tools/fracatlas_download.py``),
embeddings from ``data/fracatlas/emb/<encoder slug>/{split}.pt`` (``train/embed_images.py``).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import torch

DATA = "data/fracatlas"
SPLITS = ("train", "val", "test")
ENCODERS = {"qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct", "medsiglip": "google/medsiglip-448"}


def encoder_id(name: str) -> str:
    """Short alias (``qwen3-vl-4b``, ``medsiglip``) or a full HF id -> HF id."""
    return ENCODERS.get(name, name)


def emb_dir(encoder: str, data: str = DATA) -> str:
    return os.path.join(data, "emb", encoder_id(encoder).replace("/", "__"))


@dataclass
class Split:
    ids: list[str]
    x: torch.Tensor          # [n, hidden] float32, L2-normalised encoder embeddings
    y: torch.Tensor          # [n] long, 1 = fractured
    rows: list[dict]         # the split csv rows (region, view, path, ...)
    meta: dict               # embedding meta (encoder, hidden_size, instruction, ...)


def load_split(encoder: str, split: str, data: str = DATA) -> Split:
    with open(os.path.join(data, "splits", f"{split}.csv")) as f:
        rows = list(csv.DictReader(f))
    blob = torch.load(os.path.join(emb_dir(encoder, data), f"{split}.pt"), map_location="cpu")
    if not blob["meta"].get("complete"):
        raise SystemExit(f"{emb_dir(encoder, data)}/{split}.pt is incomplete: rerun train/embed_images.py")
    emb = blob["emb"]
    missing = [r["image_id"] for r in rows if r["image_id"] not in emb]
    if missing:
        raise SystemExit(f"{len(missing)} {split} images have no embedding, e.g. {missing[:3]}")
    x = torch.stack([emb[r["image_id"]] for r in rows]).float()
    y = torch.tensor([int(r["fractured"]) for r in rows])
    return Split([r["image_id"] for r in rows], x, y, rows, {k: v for k, v in blob["meta"].items() if k != "complete"})


def load_splits(encoder: str, splits=SPLITS, data: str = DATA) -> dict[str, Split]:
    return {s: load_split(encoder, s, data) for s in splits}
