"""Post-hoc calibration of a vision head's two-option answer (``cfg["calibration"]``).

A vision head trained with ``logit_scale`` frozen at 100 ranks images well but is
overconfident: most raw probabilities sit near 0 or 1, and the cut that separates the
classes is far from 0.5. ``train/calibrate_vision.py`` fits, on val only, one entry per
option set the head is served with:

    {"texts": {"positive": <option text>, "negative": <option text>},
     "a": float, "b": float,      # Platt: p(positive) = sigmoid(a * (l_pos - l_neg) + b)
     "threshold": float,          # decision cut on the calibrated p(positive), picked on val
     ...provenance...}

The engine applies an entry only when a question's candidate texts are exactly that pair, so
any other question (and every text head) is answered as before. Platt scaling is monotone,
so it changes probabilities but never the ranking (AUROC). For a ``choice`` question the
answer's ``choice`` follows the threshold rather than the argmax, and the answer carries the
``threshold`` it used; a ``noul`` answer carries it as the recommended cut on ``noul``.
"""
from __future__ import annotations

from typing import Any


def find(cfg: dict, texts: list[str]) -> dict | None:
    """The calibration entry for exactly these two candidate texts, if the head has one."""
    if len(texts) != 2:
        return None
    for entry in cfg.get("calibration") or []:
        t = entry["texts"]
        if sorted(texts) == sorted((t["positive"], t["negative"])):
            return entry
    return None


def apply(entry: dict, texts: list[str], logits: list[float]) -> list[float]:
    """Raw logits (in ``texts`` order) -> calibrated logits whose softmax is the Platt probability."""
    pos = texts.index(entry["texts"]["positive"])
    z = entry["a"] * (logits[pos] - logits[1 - pos]) + entry["b"]
    out = [0.0, 0.0]
    out[pos] = z
    return out


def decide(entry: dict, keys: list[str], texts: list[str], answer: dict[str, Any]) -> dict[str, Any]:
    """Attach the threshold; for a choice, pick the option by threshold instead of argmax."""
    pos = texts.index(entry["texts"]["positive"])
    t = float(entry["threshold"])
    answer = {**answer, "threshold": t}
    if answer["type"] == "choice":
        answer["choice"] = keys[pos] if answer["probabilities"][keys[pos]] >= t else keys[1 - pos]
    return answer
