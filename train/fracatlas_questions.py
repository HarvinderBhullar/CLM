"""Typed questions about a FracAtlas radiograph, with their labels (multi-question experiment).

Every question is a System One wire question (``choice`` / ``noul`` / ``score``) with three
wordings per option: wording 0 is used for training, wordings 1 and 2 only to test whether a
head's answers survive paraphrase. Candidate texts come from ``clm.schema.candidates``, so they
are exactly what the engine embeds when the question is served.

Labels are read from the split csvs written by ``tools/fracatlas_download.py``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.schema import candidates  # noqa: E402

N_WORDINGS = 3
HOLDOUT = ("view_frontal", "view_lateral", "view_oblique")   # default zero-shot questions

QUESTIONS: dict[str, dict] = {
    "fracture": {
        "type": "choice", "instructions": "Is there a bone fracture?",
        "options": {
            "fracture": ["Radiograph showing an acute bone fracture.", "X-ray with a broken bone.",
                         "A radiograph in which a bone is fractured."],
            "no_fracture": ["Radiograph of intact bones with no fracture.", "X-ray with no broken bones.",
                            "A radiograph in which all bones are intact."]},
        "label": lambda r: "fracture" if r["fractured"] == "1" else "no_fracture"},
    "region": {
        "type": "choice", "instructions": "Which body region is shown?",
        "options": {
            "hand": ["Radiograph of the hand and wrist.", "X-ray of the hand.",
                     "A radiograph showing fingers, hand or wrist."],
            "leg": ["Radiograph of the leg, knee, ankle or foot.", "X-ray of the leg.",
                    "A radiograph showing the lower limb."],
            "hip": ["Radiograph of the hip and pelvis.", "X-ray of the hip.",
                    "A radiograph showing the pelvis and hip joint."],
            "shoulder": ["Radiograph of the shoulder.", "X-ray of the shoulder joint.",
                         "A radiograph showing the shoulder and upper arm."],
            "several": ["Radiograph showing several body regions.", "X-ray covering more than one body part.",
                        "A radiograph of multiple anatomical regions."]},
        "label": lambda r: "several" if r["region"] == "mixed" else r["region"]},
    "view_frontal": {
        "type": "noul", "instructions": "Is this a frontal view?",
        "options": {
            "true": ["Frontal (AP or PA) projection radiograph.", "An X-ray taken from the front or back.",
                     "A frontal-view radiograph."],
            "false": ["Radiograph that is not a frontal projection.", "An X-ray not taken from the front or back.",
                      "A radiograph that is not a frontal view."]},
        "label": lambda r: "true" if "frontal" in r["view"].split("+") else "false"},
    "view_lateral": {
        "type": "noul", "instructions": "Is this a lateral view?",
        "options": {
            "true": ["Lateral projection radiograph.", "An X-ray taken from the side.", "A side-view radiograph."],
            "false": ["Radiograph that is not a lateral projection.", "An X-ray not taken from the side.",
                      "A radiograph that is not a side view."]},
        "label": lambda r: "true" if "lateral" in r["view"].split("+") else "false"},
    "view_oblique": {
        "type": "noul", "instructions": "Is this an oblique view?",
        "options": {
            "true": ["Oblique projection radiograph.", "An X-ray taken at an angle.", "An oblique-view radiograph."],
            "false": ["Radiograph that is not an oblique projection.", "An X-ray not taken at an angle.",
                      "A radiograph that is not an oblique view."]},
        "label": lambda r: "true" if "oblique" in r["view"].split("+") else "false"},
    "hardware": {
        "type": "noul", "instructions": "Is orthopedic hardware visible?",
        "options": {
            "true": ["Radiograph showing orthopedic hardware such as plates, screws or nails.",
                     "X-ray with a metal implant.", "A radiograph with surgical fixation hardware."],
            "false": ["Radiograph with no orthopedic hardware or implants.", "X-ray with no metal implant.",
                      "A radiograph without surgical fixation hardware."]},
        "label": lambda r: "true" if r["hardware"] == "1" else "false"},
    "multiscan": {
        "type": "noul", "instructions": "Does the image contain several scans?",
        "options": {
            "true": ["An image containing several radiographs side by side.", "Multiple X-ray views in one image.",
                     "A composite image of more than one radiograph."],
            "false": ["A single radiograph view.", "One X-ray view in the image.",
                      "An image of exactly one radiograph."]},
        "label": lambda r: "true" if r["multiscan"] == "1" else "false"},
    "fracture_count": {
        "type": "score", "instructions": "How many fractures are there?",
        "options": {
            "0": ["Radiograph with no fracture.", "X-ray with no broken bone.", "No fractures are visible."],
            "1": ["Radiograph with a single fracture.", "X-ray with one broken bone.",
                  "Exactly one fracture is visible."],
            "2": ["Radiograph with two or more fractures.", "X-ray with several broken bones.",
                  "Multiple fractures are visible."]},
        "label": lambda r: str(min(int(r["fracture_count"]), 2))},
}
POSITIVE = {"fracture": "fracture", "true": "true"}      # the positive key of a binary question


def wire(qid: str, wording: int = 0) -> dict:
    """The question in System One wire format, with option wording ``wording``."""
    q = QUESTIONS[qid]
    opts = {k: v[wording] for k, v in q["options"].items()}
    if q["type"] == "score":
        return {"type": "score", "instructions": q["instructions"], "criteria": [opts[k] for k in sorted(opts)]}
    return {"type": q["type"], "instructions": q["instructions"], "criteria": opts}


def options(qid: str, wording: int = 0) -> tuple[list[str], list[str]]:
    """(option keys, candidate texts) exactly as the engine builds them."""
    return candidates(wire(qid, wording))


def positive_index(qid: str) -> int | None:
    """Index of the positive option for a binary question, else None."""
    keys, _ = options(qid)
    if len(keys) != 2:
        return None
    return next(i for i, k in enumerate(keys) if k in POSITIVE)


def labels(qid: str, rows: list[dict]) -> np.ndarray:
    """Option index of each row's gold answer, in ``options(qid)`` key order."""
    keys, _ = options(qid)
    f = QUESTIONS[qid]["label"]
    return np.array([keys.index(f(r)) for r in rows])


def all_texts() -> list[str]:
    return list(dict.fromkeys(t for qid in QUESTIONS for w in range(N_WORDINGS) for t in options(qid, w)[1]))
