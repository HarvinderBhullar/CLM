#!/usr/bin/env python3
"""Template text corpus of radiograph descriptions, for text-only alignment of a vision state head.

    python train/fracatlas_text_corpus.py      # -> data/fracatlas/corpus_{strict,general}.json

Sentences combine body parts, findings (normal, fracture types, other), orthopedic hardware, image
composition and fracture counts. Two versions (``research/protocol_alignment.md``):

* ``strict``: none of the view vocabulary (frontal, lateral, oblique, AP, PA, view, projection, side,
  angle, angled) appears anywhere, so the held-out view questions' concepts are never seen in training;
* ``general``: strict plus sentences that do mention views.

No sentence equals any option text of ``train/fracatlas_questions.py`` (checked). Deterministic.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fracatlas_questions as fq  # noqa: E402

VIEW_WORDS = re.compile(r"\b(frontal|lateral|oblique|ap|pa|views?|projections?|sides?|angles?|angled)\b", re.I)

PARTS = ["finger", "thumb", "hand", "wrist", "metacarpal", "scaphoid", "distal radius", "forearm", "elbow",
         "ulna", "humerus", "shoulder", "clavicle", "scapula", "hip", "pelvis", "femoral neck", "femur",
         "knee", "patella", "tibia", "fibula", "ankle", "heel", "foot", "toe", "metatarsal"]
NORMAL = ["no fracture", "intact bones", "normal bony alignment", "no acute bony abnormality",
          "no fracture or dislocation", "normal bone density and alignment"]
FRACTURE_TYPES = ["transverse", "spiral", "comminuted", "displaced", "non-displaced", "hairline", "avulsion",
                  "greenstick", "buckle", "stress", "impacted", "segmental", "intra-articular"]
OTHER = ["a dislocation", "osteoarthritis", "soft tissue swelling", "a joint effusion", "osteopenia",
         "degenerative changes"]
HARDWARE = ["a metal plate and screws", "an intramedullary nail", "surgical screws", "Kirschner wires",
            "an external fixator", "a joint replacement", "a cast", "a splint"]
COMPOSITION = ["a single radiograph", "two radiographs next to each other", "several radiographs in one image",
               "a composite of multiple radiographs", "one radiograph only"]
COUNTS = ["a single fracture", "one fracture", "two fractures", "multiple fractures", "several fractures"]
PREFIX = ["Radiograph", "X-ray", "Plain film", "Pediatric radiograph", "Adult radiograph"]
VIEWS = ["frontal view", "AP view", "PA view", "lateral view", "oblique view", "frontal and lateral views",
         "lateral projection", "oblique projection"]


def strict_sentences() -> list[str]:
    s = []
    for pre, part in itertools.product(PREFIX, PARTS):
        s += [f"{pre} of the {part} showing {f}." for f in NORMAL]
        s += [f"{pre} of the {part} showing a {t} fracture." for t in FRACTURE_TYPES]
        s += [f"{pre} of the {part} with {o}." for o in OTHER]
        s += [f"{pre} of the {part} with {h}." for h in HARDWARE]
        s += [f"{pre} of the {part} showing {c}." for c in COUNTS]
    for part in PARTS:
        s += [f"{c[0].upper()}{c[1:]} of the {part}." for c in COMPOSITION]
        s += [f"The {part} shows {f}." for f in NORMAL]
        s += [f"{t[0].upper()}{t[1:]} fracture of the {part}." for t in FRACTURE_TYPES]
        s += [f"{h[0].upper()}{h[1:]} fixing a fracture of the {part}." for h in HARDWARE[:6]]
        s += [f"Healed fracture of the {part} with {h}." for h in HARDWARE]
    return s


def view_sentences() -> list[str]:
    s = []
    for v, part in itertools.product(VIEWS, PARTS):
        s += [f"{v[0].upper()}{v[1:]} of the {part}.", f"Radiograph of the {part}, {v}, showing no fracture.",
              f"Radiograph of the {part}, {v}, showing a fracture."]
    return s


def build(seed: int = 0) -> dict[str, list[str]]:
    options = set(fq.all_texts()) | set(fq.all_descriptions())
    strict = sorted({t for t in strict_sentences() if t not in options})
    bad = [t for t in strict if VIEW_WORDS.search(t)]
    if bad:
        raise SystemExit(f"strict corpus contains view vocabulary: {bad[:3]}")
    general = sorted(set(strict) | {t for t in view_sentences() if t not in options})
    rng = random.Random(seed)
    rng.shuffle(strict); rng.shuffle(general)
    return {"strict": strict, "general": general}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/fracatlas")
    a = ap.parse_args()
    corpora = build()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, texts in corpora.items():
        path = os.path.join(a.out_dir, f"corpus_{name}.json")
        json.dump(texts, open(path, "w"), indent=0)
        print(f"[corpus] {name:<7} {len(texts):>5} sentences -> {path}; e.g. {texts[:2]}")


if __name__ == "__main__":
    main()
