# FracAtlas: distilling MedSigLIP's zero-shot ability into a CLM vision head

Research prototype, not a clinical tool. Pre-registered protocol: `research/protocol_distill.md`. Evaluated 2026-10-21 on the **test** split (600 images, 109 fractured), once, after all runs and selections.

Trained questions: fracture, body region, hardware, several scans, fracture count. Held out (never trained on, never used for selection): frontal view, lateral view, oblique view. Estimates and 95% stratified bootstrap CIs are both computed on the seed-averaged probabilities (seeds 0, 1, 2); seed mean ± sd alongside.

## Summary (macro AUROC)

| variant | image labels | trained | held out | held-out 95% CI | held out, seed mean ± sd |
|---|---|---:|---:|---|---:|
| A0 from scratch (multi-question head) | 5 trained questions | 0.955 | 0.496 | [0.478, 0.513] | 0.488 ± 0.014 |
| D1 distillation only, strict corpus | none | 0.706 | 0.557 | [0.537, 0.575] | 0.562 ± 0.028 |
| D1 distillation only, general corpus | none | 0.692 | 0.531 | [0.508, 0.554] | 0.537 ± 0.019 |
| D2 D1 init, fine-tuned, λ = 0 | 5 trained questions | 0.959 | 0.539 | [0.520, 0.558] | 0.536 ± 0.014 |
| D3 D1 init, fine-tuned + distillation, λ = 0.1 | 5 trained questions | 0.958 | 0.559 | [0.540, 0.579] | 0.551 ± 0.015 |
| D3 D1 init, fine-tuned + distillation, λ = 0.3 | 5 trained questions | 0.959 | 0.556 | [0.537, 0.575] | 0.552 ± 0.027 |
| D3 D1 init, fine-tuned + distillation, λ = 1 | 5 trained questions | 0.956 | 0.542 | [0.521, 0.561] | 0.544 ± 0.022 |
| D3 D1 init, fine-tuned + distillation, λ = 3 | 5 trained questions | 0.958 | 0.537 | [0.515, 0.557] | 0.542 ± 0.026 |
| W interpolation D1 → D2, α = 0.25 | 5 trained questions | 0.914 | 0.536 | [0.512, 0.558] | 0.542 ± 0.015 |
| W interpolation D1 → D2, α = 0.5 | 5 trained questions | 0.952 | 0.546 | [0.525, 0.566] | 0.546 ± 0.015 |
| W interpolation D1 → D2, α = 0.75 | 5 trained questions | 0.958 | 0.541 | [0.522, 0.560] | 0.540 ± 0.017 |
| MedSigLIP text tower, no training (reference) | none | 0.811 | 0.660 | [0.629, 0.690] | – |

## Pre-registered hypotheses

- **H3** (D1 general, held-out CI lower bound > 0.5): lower bound 0.508 → **supported**.
- **H4** (D3 λ = 1: held-out CI lower bound > 0.5 and trained ≥ 0.942): lower bound 0.521, trained 0.956 → **supported**.
- **H5** (W α = 0.5: held-out CI lower bound > 0.5 and trained ≥ 0.942): lower bound 0.525, trained 0.952 → **supported**.

## Verdict

All three pre-registered hypotheses meet their criteria, but the effect is small and should be read as **above
chance, not as recovered zero-shot ability**.

- **Distillation on images transfers some view information without view labels (H3).** D1 general reaches held-out
  macro 0.531 [0.508, 0.554], against 0.484 for text-only alignment (A1 general) and 0.660 for MedSigLIP's own text
  tower. Training the head on image embeddings, where it is used, does better than training it on text embeddings.
- **Fine-tuning from the distilled head keeps it (H4, H5)** at full trained accuracy: D3 λ = 1 held-out 0.542
  [0.521, 0.561], trained 0.956; W α = 0.5 held-out 0.546 [0.525, 0.566], trained 0.952. A0 from scratch: 0.496.
- **The initialisation does the work, not the regulariser or the interpolation.** D2 (λ = 0, no distillation term)
  already keeps 0.539; no λ and no α is clearly better than another (all within 0.536–0.559, overlapping CIs).
  An alignment-initialised head (A2) did not (0.480).
- **Unexpected: the strict corpus, which never mentions views, transfers more than the general one** (D1 0.557 vs
  0.531). The protocol predicted chance for strict. View information reaches the student through other sentences
  (for example, composition and body-part descriptions that co-vary with view), so the gain is not evidence that the
  student learned the *concept* of a view.
- **The gain is fragile.** Per question it is uneven: frontal does not improve on A0 (0.48–0.58 vs 0.558), lateral
  moves from below chance (0.434) to about 0.50, and oblique gains most (up to 0.649), and with paraphrased held-out options every variant falls back to chance (0.42–0.54).
- **Cost on the trained questions:** none for the original wording (0.952–0.959 vs 0.955), but paraphrase
  robustness drops as λ grows (wording 1: 0.907 at λ = 0 → 0.840 at λ = 3; A0 0.935).

Bottom line: distillation on unlabelled images is the first variant in this study that keeps held-out questions above
chance after fine-tuning, but it recovers a small part of the teacher's zero-shot ability (about 0.04 of the 0.16
gap between chance and MedSigLIP) and that part does not survive rewording. Keeping zero-shot ability remains open.

## Per question (wording 0, ensemble)

| variant | fracture | body region | frontal view * | lateral view * | oblique view * | hardware | several scans | fracture count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 from scratch (multi-question head) | 0.897 | 0.991 | 0.558 | 0.434 | 0.496 | 1.000 | 1.000 | 0.889 |
| D1 distillation only, strict corpus | 0.637 | 0.850 | 0.479 | 0.612 | 0.579 | 0.859 | 0.514 | 0.672 |
| D1 distillation only, general corpus | 0.559 | 0.870 | 0.477 | 0.569 | 0.549 | 0.804 | 0.565 | 0.664 |
| D2 D1 init, fine-tuned, λ = 0 | 0.902 | 0.993 | 0.565 | 0.403 | 0.649 | 1.000 | 1.000 | 0.899 |
| D3 D1 init, fine-tuned + distillation, λ = 0.1 | 0.902 | 0.992 | 0.528 | 0.503 | 0.646 | 1.000 | 1.000 | 0.897 |
| D3 D1 init, fine-tuned + distillation, λ = 0.3 | 0.903 | 0.992 | 0.552 | 0.500 | 0.615 | 1.000 | 1.000 | 0.898 |
| D3 D1 init, fine-tuned + distillation, λ = 1 | 0.902 | 0.991 | 0.581 | 0.494 | 0.550 | 1.000 | 1.000 | 0.888 |
| D3 D1 init, fine-tuned + distillation, λ = 3 | 0.905 | 0.991 | 0.571 | 0.517 | 0.523 | 1.000 | 1.000 | 0.894 |
| W interpolation D1 → D2, α = 0.25 | 0.806 | 0.968 | 0.493 | 0.549 | 0.566 | 1.000 | 0.981 | 0.816 |
| W interpolation D1 → D2, α = 0.5 | 0.885 | 0.992 | 0.517 | 0.500 | 0.621 | 1.000 | 1.000 | 0.882 |
| W interpolation D1 → D2, α = 0.75 | 0.902 | 0.993 | 0.539 | 0.441 | 0.644 | 1.000 | 1.000 | 0.897 |
| MedSigLIP text tower | 0.659 | 0.964 | 0.645 | 0.652 | 0.684 | 1.000 | 0.750 | 0.682 |

\* held out.

## Paraphrased options (macro AUROC, ensemble, wording 1 / 2)

| variant | trained questions | held-out questions |
|---|---|---|
| A0 from scratch (multi-question head) | 0.935 / 0.853 | 0.458 / 0.471 |
| D1 distillation only, strict corpus | 0.713 / 0.705 | 0.455 / 0.524 |
| D1 distillation only, general corpus | 0.721 / 0.718 | 0.421 / 0.499 |
| D2 D1 init, fine-tuned, λ = 0 | 0.907 / 0.890 | 0.437 / 0.537 |
| D3 D1 init, fine-tuned + distillation, λ = 0.1 | 0.894 / 0.870 | 0.427 / 0.517 |
| D3 D1 init, fine-tuned + distillation, λ = 0.3 | 0.886 / 0.863 | 0.431 / 0.493 |
| D3 D1 init, fine-tuned + distillation, λ = 1 | 0.863 / 0.849 | 0.432 / 0.480 |
| D3 D1 init, fine-tuned + distillation, λ = 3 | 0.840 / 0.821 | 0.427 / 0.479 |
| W interpolation D1 → D2, α = 0.25 | 0.799 / 0.775 | 0.424 / 0.500 |
| W interpolation D1 → D2, α = 0.5 | 0.844 / 0.826 | 0.429 / 0.503 |
| W interpolation D1 → D2, α = 0.75 | 0.880 / 0.866 | 0.433 / 0.523 |

## Setup

- D1: KL(teacher ‖ student) on unlabelled train images; teacher softmax over 256 corpus sentences of T · cos(MedSigLIP image, MedSigLIP text), student the CLM scoring 100 · cos(state_head(image), action_head(Qwen3-8B text)). lr and T selected by mean val zero-shot AUROC over the trained questions.
- D2 / D3: fine-tuning on the trained questions + λ · distillation loss; λ not selected.
- W: per seed, (1 − α) · D1 general + α · D2 in weight space; α not selected.
