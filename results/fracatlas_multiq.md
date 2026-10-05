# FracAtlas multi-question experiment: one CLM vision head, many typed questions

Research prototype, not a clinical tool. Encoder **medsiglip**, evaluated 2026-10-05 on the **test** split (600 images, 109 fractured), once; every choice (config, C, thresholds) was made on val.

One state head is trained on 5 questions at once (fracture, body region, hardware, several scans, fracture count); the 3 view questions (frontal view, lateral view, oblique view) are **held out**: never trained on and never used for selection, so the head answers them zero-shot from the option text alone. Best val config `lr3e-05_wd0.1`; CLM rows are mean ± sd over seeds 0, 1, 2.

## Summary

| macro AUROC | trained questions | held-out questions (zero-shot) |
|---|---:|---:|
| CLM multi-question head | 0.952 ± 0.004 | 0.488 ± 0.014 |
| same head, random option vectors | 0.953 ± 0.003 | 0.465 ± 0.114 |
| one CLM head per question | 0.957 ± 0.002 | 0.978 ± 0.004 (supervised) |
| one linear probe per question | 0.956 | 0.974 (supervised) |

- Paired bootstrap (2000 resamples), macro AUROC over trained questions, CLM multi (seed-mean probabilities) − linear probes: **-0.000 [-0.010, +0.008]**.
- Zero-shot macro AUROC on the held-out questions (seed-mean probabilities): **0.496 [0.478, 0.513]**; chance is 0.5. Rows marked *supervised* had labels for those questions and are an upper reference, not a comparison.

**Verdict: sharing one head across questions works, zero-shot does not, and the text tower's real
contribution is robustness to rewording.**

- **Trained questions: no gain, no loss.** One shared CLM head matches one linear probe per question
  (macro AUROC 0.952 vs 0.956; paired Δ −0.000 [−0.010, +0.008]) and one CLM head per question (0.957).
  The hardest question pays a little for sharing: fracture 0.888 ± 0.010 shared vs 0.905 ± 0.001 alone.
  Random option vectors do as well on the training wording (0.953), so there the text adds no accuracy.
- **Zero-shot fails.** The held-out view questions are answered at chance: macro AUROC 0.496
  [0.478, 0.513]; lateral is even below 0.5. A state head trained from scratch on five questions only
  learns where images sit relative to those questions' option vectors; nothing links the text "frontal
  view" to the image features that encode the view, although those features exist (a supervised probe
  reaches 0.987).
- **Paraphrase is where the text tower pays off.** Asked with option wordings it never saw, the CLM
  head keeps macro AUROC 0.919 / 0.845 on trained questions, while random option vectors fall to chance
  (0.449 / 0.460). For fracture: 0.862 / 0.842 vs 0.523 / 0.378. Robustness depends on the wording:
  region wording 2 ("lower limb", "pelvis and hip joint", ...) drops to about 0.64 for every head.

So the CLM head buys a typed-question interface that tolerates rewording of the questions it was
trained on, not answers to new questions. Zero-shot would need a state space aligned with text before
fine-tuning, e.g. pre-training the state head on radiograph–report pairs, or starting from an encoder
whose image and text towers are already aligned (MedSigLIP's own text tower is the natural baseline).

*(Written by hand after the test evaluation; `evaluation/vision_eval_multiq.py` inserts this file.)*

## Per question

AUROC (macro one-vs-rest for region and fracture count); BA = balanced accuracy (binary: val Youden threshold; otherwise argmax).

| question | positives / n | CLM multi AUROC | CLM multi BA | random options AUROC | single head AUROC | linear probe AUROC | linear probe BA |
|---|---|---:|---:|---:|---:|---:|---:|
| fracture | 109 / 600 | 0.888 ± 0.010 | 0.815 ± 0.008 | 0.895 ± 0.008 | 0.905 ± 0.001 | 0.898 | 0.815 |
| body region | 5 classes | 0.990 ± 0.004 | 0.912 ± 0.006 | 0.990 ± 0.002 | 0.993 ± 0.002 | 0.995 | 0.913 |
| frontal view (held out) | 366 / 600 | 0.546 ± 0.018 | 0.548 ± 0.015 | 0.425 ± 0.199 | 0.985 ± 0.002 | 0.987 | 0.944 |
| lateral view (held out) | 206 / 600 | 0.451 ± 0.029 | 0.510 ± 0.010 | 0.427 ± 0.107 | 0.992 ± 0.001 | 0.990 | 0.976 |
| oblique view (held out) | 68 / 600 | 0.468 ± 0.048 | 0.544 ± 0.027 | 0.543 ± 0.110 | 0.957 ± 0.011 | 0.946 | 0.858 |
| hardware | 11 / 600 | 1.000 | 0.995 ± 0.005 | 1.000 | 1.000 | 1.000 | 0.999 |
| several scans | 63 / 600 | 1.000 ± 0.000 | 0.993 ± 0.008 | 1.000 ± 0.000 | 1.000 | 0.999 | 0.960 |
| fracture count | 3 classes | 0.881 ± 0.011 | 0.649 ± 0.005 | 0.879 ± 0.007 | 0.889 ± 0.007 | 0.885 | 0.672 |

## Paraphrased options

The same heads, asked with option wordings 1 and 2, which no head saw during training. Random option vectors cannot follow a paraphrase (each wording is a new random vector), so they show what the text tower contributes.

| question | CLM multi: wording 0 / 1 / 2 | random options: 0 / 1 / 2 | single head: 0 / 1 / 2 |
|---|---|---|---|
| fracture | 0.888 / 0.862 / 0.842 | 0.895 / 0.523 / 0.378 | 0.905 / 0.898 / 0.903 |
| body region | 0.990 / 0.925 / 0.635 | 0.990 / 0.467 / 0.613 | 0.993 / 0.929 / 0.650 |
| frontal view (held out) | 0.546 / 0.484 / 0.559 | 0.425 / 0.441 / 0.603 | 0.985 / 0.979 / 0.985 |
| lateral view (held out) | 0.451 / 0.497 / 0.442 | 0.427 / 0.541 / 0.430 | 0.992 / 0.991 / 0.992 |
| oblique view (held out) | 0.468 / 0.427 / 0.440 | 0.543 / 0.570 / 0.478 | 0.957 / 0.931 / 0.957 |
| hardware | 1.000 / 0.999 / 1.000 | 1.000 / 0.465 / 0.299 | 1.000 / 0.998 / 1.000 |
| several scans | 1.000 / 0.937 / 0.899 | 1.000 / 0.382 / 0.523 | 1.000 / 0.994 / 0.998 |
| fracture count | 0.881 / 0.872 / 0.850 | 0.879 / 0.408 / 0.488 | 0.889 / 0.845 / 0.869 |
| **macro, trained** | 0.952 / 0.919 / 0.845 | 0.953 / 0.449 / 0.460 | 0.957 / 0.933 / 0.884 |

## Setup

- Same data, splits, cached image embeddings and released action head as the binary experiment (`results/fracatlas.md`). Questions and wordings: `train/fracatlas_questions.py`; option texts embedded once with Qwen3-8B (`train/embed_options.py --multiq`).
- One image embedding answers every question (the question text does not reach the image encoder).
- Loss: class-weighted cross-entropy per question over its own options, averaged over trained questions; `logit_scale` frozen at 100; early stopping on mean val AUROC over trained questions.
- Caveats: per-image splits (no patient ids); hardware has few positives (11 in test) and 97 of the 99 hardware images are fractured, so it overlaps with the fracture question; region and view labels are easy for every method.
