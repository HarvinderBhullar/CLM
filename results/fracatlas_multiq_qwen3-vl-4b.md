# FracAtlas multi-question experiment: one CLM vision head, many typed questions

Research prototype, not a clinical tool. Encoder **qwen3-vl-4b**, evaluated 2026-10-05 on the **test** split (600 images, 109 fractured), once; every choice (config, C, thresholds) was made on val.

One state head is trained on 5 questions at once (fracture, body region, hardware, several scans, fracture count); the 3 view questions (frontal view, lateral view, oblique view) are **held out**: never trained on and never used for selection, so the head answers them zero-shot from the option text alone. Best val config `lr1e-05_wd0.01`; CLM rows are mean ± sd over seeds 0, 1, 2.

## Summary

| macro AUROC | trained questions | held-out questions (zero-shot) |
|---|---:|---:|
| CLM multi-question head | 0.951 ± 0.004 | 0.495 ± 0.012 |
| same head, random option vectors | 0.946 ± 0.003 | 0.495 ± 0.062 |
| one CLM head per question | 0.950 ± 0.004 | 0.954 ± 0.002 (supervised) |
| one linear probe per question | 0.948 | 0.951 (supervised) |

- Paired bootstrap (2000 resamples), macro AUROC over trained questions, CLM multi (seed-mean probabilities) − linear probes: **+0.006 [-0.003, +0.017]**.
- Zero-shot macro AUROC on the held-out questions (seed-mean probabilities): **0.495 [0.477, 0.512]**; chance is 0.5. Rows marked *supervised* had labels for those questions and are an upper reference, not a comparison.

**Verdict: the MedSigLIP findings replicate on a second, very different encoder.**

- **Trained questions: no gain, no loss.** One shared CLM head 0.951 macro AUROC vs one linear probe per
  question 0.948 (paired Δ +0.006 [−0.003, +0.017]) and one CLM head per question 0.950.
- **Zero-shot fails.** Held-out view questions 0.495 [0.477, 0.512], at chance, although a supervised
  probe on the same embedding reaches 0.951. Qwen3-VL-4B has no separate text tower aligned with its image
  embedding here, so there is no zero-shot baseline to compare with (see the MedSigLIP report: 0.660).
- **Paraphrase robustness comes from the text tower.** Reworded options keep macro AUROC 0.915 / 0.844;
  random option vectors fall to 0.501 / 0.545. Region wording 2 again drops to about 0.64–0.68 for every head.

*(Written by hand after the test evaluation; `evaluation/vision_eval_multiq.py` inserts this file.)*

## Per question

AUROC (macro one-vs-rest for region and fracture count); BA = balanced accuracy (binary: val Youden threshold; otherwise argmax).

| question | positives / n | CLM multi AUROC | CLM multi BA | random options AUROC | single head AUROC | linear probe AUROC | linear probe BA |
|---|---|---:|---:|---:|---:|---:|---:|
| fracture | 109 / 600 | 0.887 ± 0.010 | 0.807 ± 0.006 | 0.874 ± 0.008 | 0.884 ± 0.009 | 0.890 | 0.833 |
| body region | 5 classes | 0.986 ± 0.005 | 0.885 ± 0.010 | 0.990 ± 0.002 | 0.990 ± 0.003 | 0.990 | 0.887 |
| frontal view (held out) | 366 / 600 | 0.546 ± 0.023 | 0.551 ± 0.012 | 0.536 ± 0.074 | 0.973 ± 0.001 | 0.973 | 0.912 |
| lateral view (held out) | 206 / 600 | 0.490 ± 0.014 | 0.503 ± 0.003 | 0.487 ± 0.095 | 0.984 ± 0.001 | 0.981 | 0.951 |
| oblique view (held out) | 68 / 600 | 0.450 ± 0.025 | 0.477 ± 0.014 | 0.461 ± 0.034 | 0.903 ± 0.006 | 0.899 | 0.840 |
| hardware | 11 / 600 | 1.000 ± 0.000 | 0.953 ± 0.045 | 1.000 ± 0.000 | 0.999 ± 0.000 | 1.000 | 0.987 |
| several scans | 63 / 600 | 1.000 ± 0.000 | 0.973 ± 0.012 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | 0.976 |
| fracture count | 3 classes | 0.883 ± 0.008 | 0.639 ± 0.015 | 0.867 ± 0.005 | 0.875 ± 0.007 | 0.862 | 0.680 |

## Paraphrased options

The same heads, asked with option wordings 1 and 2, which no head saw during training. Random option vectors cannot follow a paraphrase (each wording is a new random vector), so they show what the text tower contributes.

| question | CLM multi: wording 0 / 1 / 2 | random options: 0 / 1 / 2 | single head: 0 / 1 / 2 |
|---|---|---|---|
| fracture | 0.887 / 0.864 / 0.830 | 0.874 / 0.535 / 0.509 | 0.884 / 0.874 / 0.875 |
| body region | 0.986 / 0.906 / 0.641 | 0.990 / 0.511 / 0.566 | 0.990 / 0.911 / 0.676 |
| frontal view (held out) | 0.546 / 0.461 / 0.557 | 0.536 / 0.402 / 0.485 | 0.973 / 0.960 / 0.970 |
| lateral view (held out) | 0.490 / 0.474 / 0.454 | 0.487 / 0.487 / 0.443 | 0.984 / 0.983 / 0.983 |
| oblique view (held out) | 0.450 / 0.480 / 0.453 | 0.461 / 0.447 / 0.498 | 0.903 / 0.900 / 0.901 |
| hardware | 1.000 / 0.998 / 0.997 | 1.000 / 0.489 / 0.571 | 0.999 / 0.999 / 0.999 |
| several scans | 1.000 / 0.947 / 0.908 | 1.000 / 0.537 / 0.539 | 1.000 / 0.960 / 0.981 |
| fracture count | 0.883 / 0.858 / 0.845 | 0.867 / 0.431 / 0.541 | 0.875 / 0.821 / 0.853 |
| **macro, trained** | 0.951 / 0.915 / 0.844 | 0.946 / 0.501 / 0.545 | 0.950 / 0.913 / 0.877 |

## Setup

- Same data, splits, cached image embeddings and released action head as the binary experiment (`results/fracatlas.md`). Questions and wordings: `train/fracatlas_questions.py`; option texts embedded once with Qwen3-8B (`train/embed_options.py --multiq`).
- One image embedding answers every question (the question text does not reach the image encoder).
- MedSigLIP zero-shot: the cached image embedding scored against MedSigLIP's own text embedding of each option description (`train/embed_texts_siglip.py`), softmax over options; no FracAtlas training. The released MedSigLIP logit scale and bias equal SigLIP's initial values (10, −10); AUROC does not depend on the scale.
- Loss: class-weighted cross-entropy per question over its own options, averaged over trained questions; `logit_scale` frozen at 100; early stopping on mean val AUROC over trained questions.
- Caveats: per-image splits (no patient ids); hardware has few positives (11 in test) and 97 of the 99 hardware images are fractured, so it overlaps with the fracture question; region and view labels are easy for every method.
