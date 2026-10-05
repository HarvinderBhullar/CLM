# Pre-registered protocol: keeping zero-shot ability in a CLM vision head

Written and committed **before** any of these runs. Deviations, if any, are listed at the end with reasons.

## Motivation

In the multi-question experiment (`research/log.md`, 2026-10-05) a state head trained from scratch on
five questions answered the three held-out view questions at chance (test macro AUROC 0.496), while
MedSigLIP's own text tower, with no FracAtlas training, reached 0.660. The image features support
zero-shot; training the head from scratch discards the image–text alignment MedSigLIP was trained with.

MedSigLIP's image and text embeddings share one space. A state head that maps MedSigLIP **text**
embeddings of a description onto the CLM action embedding of the same description (Qwen3-8B + released
action head) should, applied to an **image** embedding, land near the action embeddings of descriptions
that match the image. This needs only text, no image labels.

## Hypotheses

- **H1 (text-only alignment).** A state head trained only on text pairs answers FracAtlas questions
  zero-shot through the CLM API, at a level comparable to MedSigLIP's own zero-shot (held-out views 0.660).
- **H2 (alignment-preserving fine-tuning).** Starting from that head and fine-tuning on the five trained
  questions with the text-alignment loss kept as a regulariser keeps held-out zero-shot above chance while
  trained-question accuracy stays close to the head trained from scratch.

## Fixed setup

- Encoder: MedSigLIP-448 only (the only encoder here with an aligned text tower). Cached image embeddings,
  splits, questions, wordings, option embeddings and the released action head are those of the
  multi-question experiment. Trained questions: fracture, region, hardware, multiscan, fracture count.
  Held out: frontal, lateral, oblique view.
- **Text corpus** (`train/fracatlas_text_corpus.py`): template-generated radiograph descriptions (body
  parts × findings × hardware × image composition × fracture count). Two versions:
  - *strict*: contains none of the words frontal, lateral, oblique, AP, PA, view, projection, side, angle,
    angled. The held-out concepts never appear in any training text.
  - *general*: strict plus sentences that mention views.
  No corpus sentence equals any option text of any question (checked in code).
  Each sentence is embedded by MedSigLIP's text tower and by Qwen3-8B (same recipe as the options).
- **Alignment loss**: symmetric InfoNCE over a batch of 256 corpus texts between
  `state_head(MedSigLIP_text)` and the frozen CLM action embedding of the same text; logit scale 100.
- **Modality gap**: MedSigLIP image and text embeddings occupy different regions. Inputs are standardised
  with the corpus text mean / std; *gap shift on* standardises images with the image mean instead of the
  text mean (a mean shift that removes the offset between modalities). Both are affine, so they fold into
  the head's first layer and the checkpoint keeps the CLM format.

## Variants (3 seeds each)

| id | training | image labels used |
|---|---|---|
| A0 | from scratch on the 5 questions (existing multi-question head) | 5 trained questions |
| A1 | text-only alignment; corpus ∈ {strict, general} | none |
| A2 | A1 (strict) init, fine-tuned on the 5 questions, no regulariser | 5 trained questions |
| A3 | A1 (strict) init, fine-tuned on the 5 questions + λ · alignment loss, λ ∈ {0.1, 0.3, 1, 3} | 5 trained questions |
| A3g | as A3 with λ = 1 and the general corpus | 5 trained questions |

## Selection (no held-out labels anywhere)

- A1: lr ∈ {3e-5, 1e-4} and gap shift ∈ {on, off} chosen by the **mean val zero-shot AUROC over the five
  trained questions** (their labels act as a development set; A1 never trains on them). Weight decay 0.1,
  batch 256, up to 100 epochs, early stopping on the same dev metric, patience 10.
- A2 / A3: lr 3e-5, weight decay 0.1 (the A0 config), early stopping on mean val AUROC over the trained
  questions, patience 20. λ is **not** selected: every λ is reported; **λ = 1 is the primary setting,
  fixed now**.
- The held-out view questions' labels are used only in the final test evaluation.

## Outcomes

- **Primary:** test macro AUROC on the held-out view questions for A3 (strict, λ = 1), with a 95%
  stratified bootstrap CI (2000 resamples), compared with A0 (0.496) and MedSigLIP zero-shot (0.660).
- **H1 supported** if A1 (strict) held-out macro AUROC's CI lower bound exceeds 0.5.
- **H2 supported** if A3 (strict, λ = 1) held-out CI lower bound exceeds 0.5 **and** its trained-question
  macro AUROC is within 0.01 of A0 (0.952).
- Secondary: trained-question macro AUROC for every variant; zero-shot on trained questions for A1;
  paraphrase robustness (wordings 1–2); the λ trade-off curve (reported, not selected on).
- Test evaluated once, after all runs and selections are complete (`evaluation/vision_eval_align.py`,
  dry-run on val first).

## Deviations

Recorded after the runs; none changed a variant, a selection or a setting.

1. The val dry run of `evaluation/vision_eval_align.py` printed held-out **val** AUROCs (the protocol
   reserves held-out labels for the test evaluation). They were seen before the test run but were not used:
   the sweep and its selections had already finished and were not changed.
2. The A1 learning rate selected for both corpora (1e-4) is the upper end of the pre-registered grid
   {3e-5, 1e-4}. Recorded, not extended.
3. The corpus sizes (6,156 strict, 6,804 general) were not fixed in the protocol; they follow from the
   templates committed in `30e6bf1`, before any run.
