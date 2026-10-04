#!/usr/bin/env python3
"""Static figures for the FracAtlas vision extension (README + examples notebook).

    python tools/fracatlas_figures.py      # -> assets/clm_vision_arch.svg, assets/fracatlas_auroc.svg

The architecture diagram is fixed; the AUROC chart is drawn from ``results/fracatlas.json``
(``evaluation/vision_eval.py``). Both are plain SVG on a white ground, so they read the same
in GitHub's light and dark themes and in Jupyter.
"""
from __future__ import annotations

import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, monospace"
INK, INK2, MUTED, RULE = "#14202a", "#46555f", "#6f7d86", "#d6dde1"
TRAIN, TRAIN_FILL = "#c4501f", "#fbe9e0"
NEW, NEW_FILL = "#0f6b8a", "#e3f0f4"
FROZEN_FILL = "#eef1f3"
LR, CLM, REF = "#2a78d6", "#eb6834", "#8a959c"


def box(x, y, w, h, kind, lines):
    stroke, fill, sw = {"train": (TRAIN, TRAIN_FILL, 2.2), "new": (NEW, NEW_FILL, 1.5),
                        "frozen": (MUTED, FROZEN_FILL, 1.5), "plain": (RULE, "#ffffff", 1.5)}[kind]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>']
    ty = y + 26
    for i, (text, style) in enumerate(lines):
        size, weight, color, family = {"t": (14, 600, INK, FONT), "s": (12.5, 400, INK2, FONT),
                                       "m": (12, 400, INK2, MONO)}[style]
        out.append(f'<text x="{x + 14}" y="{ty}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
                   f'fill="{color}">{text}</text>')
        ty += 21 if i == 0 else 19
    return "\n".join(out)


def arrow(d, color=INK2, dashed=False, marker="a"):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.6"{dash} marker-end="url(#{marker})"/>'


def architecture() -> str:
    W, H = 1180, 520
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="{FONT}">',
         '<defs>'
         f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker>'
         f'<marker id="l" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{TRAIN}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    # legend
    lx = 20
    for label, stroke, fill in (("Trained here", TRAIN, TRAIN_FILL), ("New to CLM, frozen", NEW, NEW_FILL),
                                ("Released CLM, unchanged", MUTED, FROZEN_FILL)):
        p.append(f'<rect x="{lx}" y="14" width="14" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        p.append(f'<text x="{lx + 22}" y="26" font-size="13" fill="{INK2}">{label}</text>')
        lx += 22 + len(label) * 7.2 + 26
    y0 = 50
    p.append(f'<text x="440" y="{y0 + 24}" font-size="12.5" font-weight="500" fill="{TRAIN}">Training: class-weighted '
             f'cross-entropy over the two options; only state_head gets gradients</text>')
    p.append(arrow(f"M1045,{y0 + 168} V{y0 + 34} H665 V{y0 + 66}", TRAIN, dashed=True, marker="l"))
    lane = f'font-family="{MONO}" font-size="11.5" font-weight="500" letter-spacing="1" fill="{MUTED}"'
    p.append(f'<text x="20" y="{y0 + 56}" {lane}>STATE TOWER · NEW</text>')
    p.append(box(20, y0 + 70, 130, 96, "plain", [("X-ray image", "t"), ("JPEG / PNG,", "s"), ("path or base64", "s")]))
    p.append(box(180, y0 + 70, 240, 96, "new", [("Image encoder", "t"), ("Qwen3-VL-4B: image + fixed prompt,", "s"),
                                                ("last token, 2560-d", "s"), ("or MedSigLIP-448: pooled, 1152-d", "s")]))
    p.append(box(450, y0 + 92, 80, 52, "new", [("L2 norm", "t")]))
    p.append(box(560, y0 + 70, 230, 96, "train", [("state_head (new)", "t"), ("d → 1536 → 1536 → 512", "m"),
                                                  ("LayerNorm, GELU, random init", "s"), ("7.1M / 4.9M params", "s")]))
    p.append(box(820, y0 + 92, 100, 52, "plain", [("z_state", "t"), ("512-d, unit", "m")]))
    p.append(f'<text x="20" y="{y0 + 246}" {lane}>ACTION TOWER · UNCHANGED</text>')
    p.append(box(20, y0 + 260, 260, 96, "plain", [("Option texts", "t"), ("fracture: “…an acute bone fracture.”", "s"),
                                                  ("no_fracture: “…intact bones with", "s"), ("no fracture.”", "s")]))
    p.append(box(310, y0 + 260, 220, 96, "frozen", [("Qwen3-8B text encoder", "t"), ("last token, 4096-d, L2 norm", "s"),
                                                    ("embedded once, cached", "s")]))
    p.append(box(560, y0 + 260, 230, 96, "frozen", [("action_head (released)", "t"), ("4096 → 1536 → 1536 → 512", "m"),
                                                    ("CLM-v0.1-8B weights", "s")]))
    p.append(box(820, y0 + 282, 100, 52, "plain", [("z_option", "t"), ("2 × 512", "m")]))
    p.append(box(950, y0 + 168, 210, 90, "frozen", [("Score", "t"), ("100 × cosine similarity", "s"),
                                                    ("softmax → p(fracture)", "s")]))
    p.append(box(950, y0 + 292, 210, 64, "plain", [("Typed answer", "t"), ("Choice / Noul, same API as text", "s")]))
    for d in ("M150,{a} H176", "M420,{a} H446", "M530,{a} H556", "M790,{a} H816"):
        p.append(arrow(d.format(a=y0 + 118)))
    for d in ("M280,{b} H306", "M530,{b} H556", "M790,{b} H816"):
        p.append(arrow(d.format(b=y0 + 308)))
    p.append(arrow(f"M920,{y0 + 118} H935 V{y0 + 200} H946"))
    p.append(arrow(f"M920,{y0 + 308} H935 V{y0 + 228} H946"))
    p.append(arrow(f"M1055,{y0 + 258} V{y0 + 288}"))
    p.append(f'<rect x="20" y="{y0 + 392}" width="1140" height="62" rx="8" fill="none" stroke="{MUTED}" '
             f'stroke-dasharray="4 4"/>')
    p.append(f'<text x="36" y="{y0 + 418}" font-size="14" font-weight="600" fill="{INK}">Serving</text>')
    p.append(f'<text x="110" y="{y0 + 418}" font-size="12.5" fill="{INK2}">Engine reads cfg.state_modality: image heads '
             f'take {{"type": "image", "image": base64}} and run the head’s own image encoder;</text>')
    p.append(f'<text x="110" y="{y0 + 438}" font-size="12.5" fill="{INK2}">text heads (clm-latest) keep the original '
             f'path, text → Qwen3-8B → released state_head, unchanged. Mismatches are refused.</text>')
    p.append("</svg>")
    return "\n".join(p)


def auroc_chart(res: dict) -> str:
    m = res["methods"]
    def agg(name):
        runs = m[name]["runs"]
        return (statistics.mean(r["auroc"] for r in runs), statistics.mean(r["auroc_ci"][0] for r in runs),
                statistics.mean(r["auroc_ci"][1] for r in runs), len(runs))
    rows = [("Resolution only", "pixel count", "resolution only", REF),
            ("Linear probe", "Qwen3-VL-4B", "Linear probe (LR) · Qwen3-VL-4B", LR),
            ("CLM head", "Qwen3-VL-4B", "CLM head · Qwen3-VL-4B", CLM),
            ("Linear probe", "MedSigLIP-448", "Linear probe (LR) · MedSigLIP-448", LR),
            ("CLM head", "MedSigLIP-448", "CLM head · MedSigLIP-448", CLM)]
    W, L, R, T, rowH, B = 900, 190, 80, 64, 46, 40
    H = T + len(rows) * rowH + B
    x = lambda v: L + (v - 0.45) / 0.55 * (W - L - R)  # noqa: E731
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="0" y="22" font-size="16" font-weight="600" fill="{INK}">FracAtlas test AUROC with 95% bootstrap interval</text>']
    lx = 0
    for label, c in (("Linear probe (logistic regression)", LR), ("CLM head (mean of 3 seeds)", CLM), ("Reference", REF)):
        p.append(f'<circle cx="{lx + 6}" cy="42" r="6" fill="{c}"/>')
        p.append(f'<text x="{lx + 18}" y="47" font-size="13" fill="{INK2}">{label}</text>')
        lx += 18 + len(label) * 7 + 28
    t = 0.5
    while t <= 1.0001:
        p.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T}" y2="{H - B}" stroke="{RULE}" stroke-dasharray="2 3"/>')
        p.append(f'<text x="{x(t):.1f}" y="{H - B + 18}" font-family="{MONO}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="middle">{t:.1f}</text>')
        t += 0.1
    p.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="{RULE}"/>')
    for i, (name, sub, key, c) in enumerate(rows):
        v, lo, hi, k = agg(key)
        cy = T + i * rowH + rowH / 2
        p.append(f'<text x="0" y="{cy - 2}" font-size="13" fill="{INK}">{name}</text>')
        p.append(f'<text x="0" y="{cy + 14}" font-size="11.5" fill="{MUTED}">{sub}</text>')
        p.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{cy}" y2="{cy}" stroke="{c}" stroke-width="2" '
                 f'stroke-linecap="round"/>')
        p.append(f'<circle cx="{x(v):.1f}" cy="{cy}" r="6" fill="{c}" stroke="#ffffff" stroke-width="2"/>')
        p.append(f'<text x="{x(hi) + 10:.1f}" y="{cy + 4}" font-family="{MONO}" font-size="12" fill="{INK2}">{v:.3f}</text>')
    p.append(f'<text x="0" y="{H - 4}" font-size="11.5" fill="{MUTED}">{res["eval_split"]} split, evaluated once. '
             f'CLM intervals: mean of the three seeds’ intervals. Source: results/fracatlas.json</text>')
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    res = json.load(open(os.path.join(REPO, "results", "fracatlas.json")))
    if res["eval_split"] != "test":
        raise SystemExit("results/fracatlas.json is a dry run; run evaluation/vision_eval.py first")
    os.makedirs(os.path.join(REPO, "assets"), exist_ok=True)
    for name, svg in (("clm_vision_arch.svg", architecture()), ("fracatlas_auroc.svg", auroc_chart(res))):
        with open(os.path.join(REPO, "assets", name), "w") as f:
            f.write(svg + "\n")
        print(f"-> assets/{name}")


if __name__ == "__main__":
    main()
