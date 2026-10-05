#!/usr/bin/env python3
"""Download FracAtlas, inspect it and write fixed stratified image-level splits.

    python tools/fracatlas_download.py            # -> data/fracatlas/{raw,splits}

FracAtlas (Abedeen et al. 2023, CC BY 4.0) comes from figshare as one zip:
``FracAtlas/dataset.csv`` (one row per image: multi-hot body region ``hand``/``leg``/
``hip``/``shoulder``/``mixed``, multi-hot view ``frontal``/``lateral``/``oblique``,
``hardware``, ``multiscan``, ``fractured``, ``fracture_count``) and
``FracAtlas/images/{Fractured,Non_fractured}/``.

Images found in *both* class folders have an ambiguous label (the csv says fractured
but their YOLO box files are empty), so they are dropped from every split.

59 JPEGs (all non-fractured, IMG0004028-IMG0004347) are truncated in the source zip;
decoding them anyway fills the bottom rows with flat grey, a label shortcut. They are
removed *after* splitting, so every other image keeps its split.

There are no patient ids, so splits are image-level, stratified on label x body region,
70/15/15 with a fixed seed. Consecutive image ids often look like one study (several
views of the same limb), so a patient can land in more than one split: this leaks, and
test numbers are an optimistic estimate.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import zipfile

import requests
from PIL import Image

URL = "https://ndownloader.figshare.com/files/65518038"
MD5 = "fe9da2c7c285915ebee69dfdab8fd396"
REGIONS = ("hand", "leg", "hip", "shoulder")
VIEWS = ("frontal", "lateral", "oblique")
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest: str) -> None:
    if os.path.exists(dest) and md5sum(dest) == MD5:
        return
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    with requests.get(URL, stream=True, timeout=600, headers={"Range": f"bytes={have}-"} if have else {}) as r:
        r.raise_for_status()
        with open(part, "ab" if r.status_code == 206 else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if md5sum(part) != MD5:
        raise SystemExit(f"md5 mismatch for {part}; delete it and retry")
    os.replace(part, dest)


def region_of(row: dict) -> str:
    """One body region per image: the single flagged region, else ``mixed``."""
    hits = [r for r in REGIONS if row[r] == "1"]
    return hits[0] if len(hits) == 1 and row["mixed"] == "0" else "mixed"


def view_of(row: dict) -> str:
    return "+".join(v for v in VIEWS if row[v] == "1") or "none"


def load(root: str) -> tuple[list[dict], list[str]]:
    base = os.path.join(root, "FracAtlas")
    folders = {d: set(os.listdir(os.path.join(base, "images", d))) for d in ("Fractured", "Non_fractured")}
    ambiguous = sorted(folders["Fractured"] & folders["Non_fractured"])
    rows = []
    with open(os.path.join(base, "dataset.csv")) as f:
        for r in csv.DictReader(f):
            iid = r["image_id"]
            if iid in ambiguous:
                continue
            folder = "Fractured" if r["fractured"] == "1" else "Non_fractured"
            if iid not in folders[folder]:
                raise SystemExit(f"{iid}: label {r['fractured']} but not in images/{folder}")
            rows.append({"image_id": iid, "path": os.path.join("FracAtlas", "images", folder, iid),
                         "fractured": int(r["fractured"]), "region": region_of(r), "view": view_of(r),
                         "hardware": int(r["hardware"]), "multiscan": int(r["multiscan"]),
                         "fracture_count": int(r["fracture_count"])})
    return rows, ambiguous


def truncated(root: str, rows: list[dict]) -> list[str]:
    """Image ids whose JPEG stream ends early (PIL refuses to fully decode them)."""
    bad = []
    for r in rows:
        try:
            with Image.open(os.path.join(root, r["path"])) as im:
                im.load()
        except OSError:
            bad.append(r["image_id"])
    return bad


def split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    strata = collections.defaultdict(list)
    for r in rows:
        strata[(r["fractured"], r["region"])].append(r)
    rng = random.Random(seed)
    out = {k: [] for k in FRACTIONS}
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r["image_id"])
        rng.shuffle(group)
        n_val = round(FRACTIONS["val"] * len(group))
        n_test = round(FRACTIONS["test"] * len(group))
        out["val"] += group[:n_val]
        out["test"] += group[n_val:n_val + n_test]
        out["train"] += group[n_val + n_test:]
    return {k: sorted(v, key=lambda r: r["image_id"]) for k, v in out.items()}


def table(title: str, rows: list[dict], key: str) -> None:
    c = collections.Counter((r[key], r["fractured"]) for r in rows)
    print(f"\n{title:<16}{'n':>6}{'fract':>7}{'pos%':>7}")
    for k in sorted({r[key] for r in rows}, key=lambda k: -sum(c[(k, y)] for y in (0, 1))):
        n = c[(k, 0)] + c[(k, 1)]
        print(f"{k:<16}{n:>6}{c[(k, 1)]:>7}{100 * c[(k, 1)] / n:>6.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/fracatlas")
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()

    raw = os.path.join(a.out, "raw")
    zpath = os.path.join(a.out, "FracAtlas.zip")
    os.makedirs(a.out, exist_ok=True)
    download(zpath)
    if not os.path.exists(os.path.join(raw, "FracAtlas", "dataset.csv")):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(raw)

    rows, ambiguous = load(raw)
    pos = sum(r["fractured"] for r in rows)
    print(f"FracAtlas: {len(rows)} images, {pos} fractured ({100 * pos / len(rows):.1f}%), "
          f"{len(rows) - pos} non-fractured; dropped {len(ambiguous)} in both class folders: {ambiguous}")
    table("label", [{**r, "label": "fractured" if r["fractured"] else "non_fractured"} for r in rows], "label")
    table("body region", rows, "region")
    table("view", rows, "view")
    print(f"\nhardware {sum(r['hardware'] for r in rows)}, multiscan {sum(r['multiscan'] for r in rows)}")

    splits = split(rows, a.seed)
    broken = set(truncated(raw, rows))
    splits = {k: [r for r in v if r["image_id"] not in broken] for k, v in splits.items()}
    print(f"dropped {len(broken)} truncated JPEGs after splitting "
          f"({sum(r['fractured'] for r in rows if r['image_id'] in broken)} fractured)")
    sdir = os.path.join(a.out, "splits")
    os.makedirs(sdir, exist_ok=True)
    cols = list(rows[0])
    for name, rs in splits.items():
        with open(os.path.join(sdir, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rs)
    json.dump({"seed": a.seed, "fractions": FRACTIONS, "stratify": ["fractured", "region"],
               "zip_md5": MD5, "dropped_ambiguous": ambiguous,
               "dropped_truncated": sorted(broken),
               "counts": {k: {"n": len(v), "fractured": sum(r["fractured"] for r in v)} for k, v in splits.items()}},
              open(os.path.join(sdir, "split_meta.json"), "w"), indent=1)
    print()
    for k, v in splits.items():
        p = sum(r["fractured"] for r in v)
        print(f"split {k:<5} {len(v):>5} images, {p:>4} fractured ({100 * p / len(v):.1f}%)")
    print(f"-> {sdir}/{{train,val,test}}.csv (seed {a.seed})")


if __name__ == "__main__":
    main()
