# FracAtlas cross-validation: medsiglip, image folds

Post-hoc robustness analysis (not pre-registered), 2026-10-21. Research prototype, not a clinical tool. Script: `evaluation/fracatlas_cv.py`.

All 4,022 images pooled; 4 × 5-fold outer CV (20 folds, about 804 test images and 143 fractured per fold), folds per image. Inner val split for early stopping and probe C; hyperparameters fixed at the values selected on the original val split. Values are mean ± sd over folds.

## AUROC

| | mean ± sd over folds |
|---|---:|
| fracture, CLM head | 0.915 ± 0.011 |
| fracture, CLM head with random option vectors | 0.913 ± 0.013 |
| fracture, linear probe | 0.913 ± 0.011 |
| fracture without hardware images, CLM head | 0.902 ± 0.013 |
| fracture without hardware images, linear probe | 0.901 ± 0.012 |
| five trained questions (macro), one shared CLM head | 0.963 ± 0.005 |
| five trained questions (macro), one probe per question | 0.961 ± 0.005 |
| five trained questions, paraphrase 1 / 2 | 0.920 ± 0.021 / 0.858 ± 0.022 |
| three held-out view questions (macro, zero-shot) | 0.506 ± 0.025 |

## Differences

| comparison | mean | sd | folds > 0 | 95% CI (corrected resampled t) |
|---|---:|---:|---:|---|
| fracture: CLM − probe | +0.001 | 0.010 | 12/20 | [-0.010, +0.013] |
| fracture: CLM − random options | +0.002 | 0.011 | 12/20 | [-0.011, +0.015] |
| fracture, no hardware: CLM − probe | +0.001 | 0.011 | 12/20 | [-0.012, +0.015] |
| five questions: shared CLM − probes | +0.002 | 0.005 | 14/20 | [-0.004, +0.007] |
| held-out views: CLM − 0.5 | +0.006 | 0.025 | 10/20 | [-0.022, +0.034] |

Runtime 8.6 min on mps.
