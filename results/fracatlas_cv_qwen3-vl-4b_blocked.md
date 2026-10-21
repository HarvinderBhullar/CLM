# FracAtlas cross-validation: qwen3-vl-4b, blocked folds

Post-hoc robustness analysis (not pre-registered), 2026-10-21. Research prototype, not a clinical tool. Script: `evaluation/fracatlas_cv.py`.

All 4,022 images pooled; 4 × 5-fold outer CV (20 folds, about 804 test images and 143 fractured per fold), folds built from blocks of 20 consecutive image ids (approximate study grouping). Inner val split for early stopping and probe C; hyperparameters fixed at the values selected on the original val split. Values are mean ± sd over folds.

## AUROC

| | mean ± sd over folds |
|---|---:|
| fracture, CLM head | 0.856 ± 0.023 |
| fracture, CLM head with random option vectors | 0.862 ± 0.021 |
| fracture, linear probe | 0.872 ± 0.025 |
| fracture without hardware images, CLM head | 0.835 ± 0.028 |
| fracture without hardware images, linear probe | 0.854 ± 0.028 |
| five trained questions (macro), one shared CLM head | 0.944 ± 0.007 |
| five trained questions (macro), one probe per question | 0.944 ± 0.010 |
| five trained questions, paraphrase 1 / 2 | 0.886 ± 0.029 / 0.827 ± 0.017 |
| three held-out view questions (macro, zero-shot) | 0.489 ± 0.017 |

## Differences

| comparison | mean | sd | folds > 0 | 95% CI (corrected resampled t) |
|---|---:|---:|---:|---|
| fracture: CLM − probe | -0.016 | 0.019 | 4/20 | [-0.037, +0.005] |
| fracture: CLM − random options | -0.007 | 0.021 | 9/20 | [-0.031, +0.017] |
| fracture, no hardware: CLM − probe | -0.019 | 0.022 | 4/20 | [-0.044, +0.007] |
| five questions: shared CLM − probes | +0.000 | 0.007 | 11/20 | [-0.008, +0.009] |
| held-out views: CLM − 0.5 | -0.011 | 0.017 | 5/20 | [-0.031, +0.009] |

Runtime 135.2 min on mps.
