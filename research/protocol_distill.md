# Pre-registered protocol: distilling MedSigLIP's zero-shot ability into a CLM vision head

Written and committed **before** any of these runs. Deviations, if any, are listed at the end with reasons.

## Motivation

The alignment experiment (`research/protocol_alignment.md`, results in `results/fracatlas_align_medsiglip.md`)
trained the state head on MedSigLIP **text** embeddings and applied it to **image** embeddings. It transferred
fracture, hardware and region to images without image labels, but not views: the text-only head (A1) answered the
held-out view questions at chance, even with a corpus that mentions views (A1 general, 0.502). MedSigLIP's own text
tower reaches 0.660 on the same questions. Two explanations remain:

1. **Modality gap.** A head fitted on text embeddings need not behave the same on image embeddings, which lie in a
   different region of MedSigLIP's space; directions that matter little in text (views) may not carry over.
2. **Fine-tuning distortion.** Fine-tuning on five questions moves the head away from any zero-shot solution
   (Kumar et al., 2022); weight-space ensembling of a zero-shot and a fine-tuned model can recover it (WiSE-FT,
   Wortsman et al., 2022).

A1 cannot test (2): it has no view ability to preserve. This experiment first gives the head that ability by
**distillation on real images**, which removes the modality gap, then tests whether fine-tuning keeps it.

## Distillation objective

For a batch of **unlabelled** training images *i* and a batch of corpus sentences *j*:

- teacher: `p_T(j | i) = softmax_j(T · cos(MedSigLIP_image(i), MedSigLIP_text(j)))`, MedSigLIP frozen;
- student: `p_S(j | i) = softmax_j(100 · cos(state_head(MedSigLIP_image(i)), action_head(Qwen3-8B(j))))`, the CLM
  scoring with the frozen released action tower and logit scale;
- loss: `KL(p_T || p_S)`, averaged over images.

The student sees only image embeddings, so it is trained where it is used. No image labels enter this loss.

## Fixed setup

- Encoder MedSigLIP-448 only (needs an aligned text tower). Cached image embeddings, splits, questions, wordings,
  option embeddings, released action head, corpora and their embeddings are those of the alignment experiment.
  Trained questions: fracture, region, hardware, multiscan, fracture count. Held out: frontal, lateral, oblique view.
- Corpora: *strict* (6,156 sentences, no view vocabulary) and *general* (6,804, includes 648 view sentences).
  No corpus sentence equals an option text.
- Images for distillation: the **train** split only (2,813), labels unused. Per step 64 images and 256 corpus
  sentences sampled uniformly. Image inputs standardised with the train image mean / std (folded into the head).
- Optimiser AdamW, weight decay 0.1, gradient clip 1.0.

## Variants (3 seeds each)

| id | training | image labels used |
|---|---|---|
| A0 | from scratch on the 5 questions (existing multi-question head) | 5 trained questions |
| D1 | distillation only; corpus ∈ {strict, general} | none |
| D2 | D1 (general) init, fine-tuned on the 5 questions, no distillation term | 5 trained questions |
| D3 | D1 (general) init, fine-tuned on the 5 questions + λ · distillation loss, λ ∈ {0.1, 0.3, 1, 3} | 5 trained questions |
| W  | weight interpolation `(1 − α) · θ_D1 + α · θ_D2` per seed, α ∈ {0.25, 0.5, 0.75} | 5 trained questions |

Seed *s* of D2 / D3 starts from seed *s* of the selected D1 (general) run; W interpolates seed *s* of D1 and D2
(same input standardisation, so weights live in the same coordinates).

## Selection (no held-out labels anywhere)

- D1: lr ∈ {3e-5, 1e-4} × teacher scale T ∈ {10, 30, 100}, chosen per corpus by the **mean val zero-shot AUROC over
  the five trained questions** (a development set; D1 never trains on their labels). Up to 100 epochs, early
  stopping on the same metric, patience 10.
- D2 / D3: lr 3e-5 (the A0 config), up to 200 epochs, early stopping on mean val AUROC over the trained questions,
  patience 20. λ is **not** selected: every λ is reported; **λ = 1 is primary**.
- W: α is **not** selected: every α is reported; **α = 0.5 is primary**.
- The held-out view questions' labels are used only in the final test evaluation. The val dry run of the evaluation
  script reports trained questions only (a lesson from deviation 1 of the alignment protocol).

## Estimates and outcomes

- **Point estimate and CI are computed on the same thing**: the AUROC of the seed-averaged probabilities (a
  three-seed ensemble), with a 95% stratified bootstrap CI (2,000 resamples). Seed mean ± sd is reported alongside.
  (The alignment report paired a seed-mean point estimate with an ensemble CI; this is fixed here.)
- **H3 (distillation gives zero-shot on images).** D1 (general) held-out macro AUROC CI lower bound > 0.5.
- **H4 (regularised fine-tuning keeps it).** D3 (λ = 1) held-out CI lower bound > 0.5 **and** trained-question
  macro AUROC ≥ 0.942 (within 0.01 of A0, 0.952).
- **H5 (weight interpolation keeps it).** W (α = 0.5) held-out CI lower bound > 0.5 **and** trained macro ≥ 0.942.
- Secondary, reported regardless: D1 (strict) held-out macro (prediction: chance, since its corpus never mentions
  views; above chance would mean view information reaches the student through other sentences); D1 zero-shot on
  trained questions vs A1 and vs MedSigLIP's text tower; the λ and α trade-off curves; per-question AUROC;
  paraphrase robustness (wordings 1–2).
- Test evaluated once, after all runs and selections are complete (`evaluation/vision_eval_distill.py`).

## Deviations

Recorded after the runs and before the test evaluation; none changed a variant, a selection or a setting.

1. The teacher scale selected for both corpora (T = 100) is the upper end of the pre-registered grid {10, 30, 100},
   and the learning rate (1e-4) the upper end of {3e-5, 1e-4}. Recorded, not extended.
2. The val dry run of `evaluation/vision_eval_distill.py` scored the trained questions only, as planned; no
   held-out label was read before the test evaluation.
