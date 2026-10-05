# FracAtlas: keeping zero-shot ability in a CLM vision head (MedSigLIP)

Research prototype, not a clinical tool. Pre-registered protocol: `research/protocol_alignment.md`. Evaluated 2026-10-05 on the **test** split (600 images, 109 fractured), once, after all runs and selections.

Trained questions: fracture, body region, hardware, several scans, fracture count. Held out (never trained on, never used for selection): frontal view, lateral view, oblique view. Rows are mean ± sd over seeds 0, 1, 2; the CI is the 95% stratified bootstrap of the held-out macro AUROC of the seed-mean probabilities.

## Summary (macro AUROC)

| variant | image labels | trained questions | held-out questions | held-out 95% CI |
|---|---|---:|---:|---|
| A0 from scratch (multi-question head) | 5 trained questions | 0.952 ± 0.004 | 0.488 ± 0.014 | [0.478, 0.513] |
| A1 text-only, strict corpus | none | 0.765 ± 0.033 | 0.506 ± 0.016 | [0.469, 0.507] |
| A1 text-only, general corpus | none | 0.786 ± 0.020 | 0.502 ± 0.028 | [0.464, 0.503] |
| A2 A1-strict init, fine-tuned, λ = 0 | 5 trained questions | 0.953 ± 0.003 | 0.495 ± 0.011 | [0.464, 0.497] |
| A3 A1-strict init, fine-tuned, λ = 0.1 | 5 trained questions | 0.954 ± 0.003 | 0.499 ± 0.010 | [0.465, 0.500] |
| A3 A1-strict init, fine-tuned, λ = 0.3 | 5 trained questions | 0.954 ± 0.002 | 0.490 ± 0.013 | [0.458, 0.490] |
| A3 A1-strict init, fine-tuned, λ = 1 | 5 trained questions | 0.953 ± 0.002 | 0.493 ± 0.011 | [0.464, 0.496] |
| A3 A1-strict init, fine-tuned, λ = 3 | 5 trained questions | 0.955 ± 0.003 | 0.498 ± 0.016 | [0.466, 0.500] |
| A3g A1-general init, fine-tuned, λ = 1 | 5 trained questions | 0.952 ± 0.002 | 0.495 ± 0.015 | [0.476, 0.508] |
| MedSigLIP text tower, no training (reference) | none | 0.811 | 0.660 | [0.629, 0.690] |

## Pre-registered hypotheses

- **H1** (A1 strict, held-out CI lower bound > 0.5): lower bound 0.469 → **not supported**.
- **H2** (A3 strict λ = 1: held-out CI lower bound > 0.5 and trained macro ≥ 0.942): lower bound 0.464, trained 0.953 → **not supported**.

**Verdict: both pre-registered hypotheses are not supported. Text-only alignment transfers some concepts
to images, but not the held-out view questions.**

- **H1 not supported.** The text-only head (A1, strict corpus, no image labels) answers the held-out view
  questions at chance: macro 0.506, CI [0.469, 0.507]. With the general corpus, which does contain view
  sentences, it is no better (0.502). MedSigLIP's own text tower gets 0.660 [0.629, 0.690] on the same
  questions, so the view information that MedSigLIP's alignment carries is lost on the way through the
  learned text-to-text map into the CLM action space.
- **Alignment does transfer other concepts (secondary outcome).** With no image labels at all, A1 answers
  fracture at 0.741 (strict) / 0.782 (general), above MedSigLIP's own zero-shot 0.659; hardware 0.892 /
  0.883 and region 0.825 / 0.796. The corpus describes fractures, hardware and body parts in many ways; it
  describes views only as a minor attribute (none in strict, ~10% of sentences in general), and the map
  that best matches corpus sentences need not keep that direction.
- **H2 not supported.** Fine-tuning from the aligned head keeps trained-question accuracy (0.953 vs 0.952
  for A0) at every λ, but no λ in {0, 0.1, 0.3, 1, 3} lifts the held-out questions above chance (0.490–0.499;
  every CI includes or lies below 0.5). The alignment loss is minimised on text and does not constrain where
  images go along directions the corpus does not exercise.
- Paraphrase robustness on trained questions may improve slightly with the aligned start (λ = 1: 0.929 /
  0.849 vs A0 0.919 / 0.845), within seed noise; not claimed.

Not tested here, and suggested by the per-question pattern: corpora that vary the held-out attribute as
much as the others, or aligning on image–text pairs instead of text alone.

*(Written by hand after the test evaluation; `evaluation/vision_eval_align.py` inserts this file.)*

## Per question (wording 0)

| variant | fracture | body region | frontal view * | lateral view * | oblique view * | hardware | several scans | fracture count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 from scratch (multi-question head) | 0.888 | 0.990 | 0.546 | 0.451 | 0.468 | 1.000 | 1.000 | 0.881 |
| A1 text-only, strict corpus | 0.741 | 0.825 | 0.435 | 0.516 | 0.566 | 0.892 | 0.687 | 0.679 |
| A1 text-only, general corpus | 0.782 | 0.796 | 0.490 | 0.474 | 0.542 | 0.883 | 0.742 | 0.727 |
| A2 A1-strict init, fine-tuned, λ = 0 | 0.892 | 0.987 | 0.537 | 0.433 | 0.516 | 1.000 | 1.000 | 0.885 |
| A3 A1-strict init, fine-tuned, λ = 0.1 | 0.894 | 0.987 | 0.548 | 0.423 | 0.524 | 1.000 | 1.000 | 0.890 |
| A3 A1-strict init, fine-tuned, λ = 0.3 | 0.897 | 0.988 | 0.520 | 0.465 | 0.486 | 1.000 | 1.000 | 0.887 |
| A3 A1-strict init, fine-tuned, λ = 1 | 0.889 | 0.990 | 0.535 | 0.458 | 0.485 | 1.000 | 1.000 | 0.885 |
| A3 A1-strict init, fine-tuned, λ = 3 | 0.894 | 0.985 | 0.560 | 0.411 | 0.522 | 1.000 | 1.000 | 0.896 |
| A3g A1-general init, fine-tuned, λ = 1 | 0.889 | 0.985 | 0.575 | 0.410 | 0.500 | 1.000 | 1.000 | 0.884 |
| MedSigLIP text tower | 0.659 | 0.964 | 0.645 | 0.652 | 0.684 | 1.000 | 0.750 | 0.682 |

\* held out.

## Paraphrased options (macro AUROC, wording 1 / 2)

| variant | trained questions | held-out questions |
|---|---|---|
| A0 from scratch (multi-question head) | 0.919 / 0.845 | 0.470 / 0.480 |
| A1 text-only, strict corpus | 0.801 / 0.723 | 0.508 / 0.502 |
| A1 text-only, general corpus | 0.801 / 0.730 | 0.530 / 0.516 |
| A2 A1-strict init, fine-tuned, λ = 0 | 0.924 / 0.852 | 0.452 / 0.508 |
| A3 A1-strict init, fine-tuned, λ = 0.1 | 0.929 / 0.852 | 0.461 / 0.509 |
| A3 A1-strict init, fine-tuned, λ = 0.3 | 0.933 / 0.855 | 0.477 / 0.505 |
| A3 A1-strict init, fine-tuned, λ = 1 | 0.929 / 0.849 | 0.456 / 0.507 |
| A3 A1-strict init, fine-tuned, λ = 3 | 0.920 / 0.847 | 0.467 / 0.517 |
| A3g A1-general init, fine-tuned, λ = 1 | 0.925 / 0.846 | 0.468 / 0.507 |

## Setup

- Corpus (`train/fracatlas_text_corpus.py`): template radiograph descriptions. *Strict*: 6,156 sentences, none containing view vocabulary (checked by regex); *general*: strict + 648 sentences mentioning views. No sentence equals an option text. Embedded by MedSigLIP's text tower and by Qwen3-8B.
- A1: symmetric InfoNCE between state_head(MedSigLIP text) and the frozen CLM action embedding of the same sentence; lr and modality-gap shift chosen by mean val zero-shot AUROC over the trained questions.
- A2 / A3: fine-tuning on the trained questions (class-weighted CE per question) + λ · alignment loss on a random batch of 256 corpus sentences per step; early stopping on mean val AUROC over trained questions; λ not selected.
- References from the protocol: A0 trained macro 0.952 and MedSigLIP zero-shot held-out 0.66 (both test, multi-question experiment).
