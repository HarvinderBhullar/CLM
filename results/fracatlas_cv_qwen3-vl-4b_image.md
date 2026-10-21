# FracAtlas cross-validation: qwen3-vl-4b, image folds

Post-hoc robustness analysis (not pre-registered), 2026-10-21. Research prototype, not a clinical tool. Script: `evaluation/fracatlas_cv.py`.

All 4,022 images pooled; 4 × 5-fold outer CV (20 folds, about 804 test images and 143 fractured per fold), folds per image. Inner val split for early stopping and probe C; hyperparameters fixed at the values selected on the original val split. Values are mean ± sd over folds.

## AUROC

| | mean ± sd over folds |
|---|---:|
| fracture, CLM head | 0.876 ± 0.017 |
| fracture, CLM head with random option vectors | 0.876 ± 0.017 |
| fracture, linear probe | 0.881 ± 0.017 |
| fracture without hardware images, CLM head | 0.858 ± 0.020 |
| fracture without hardware images, linear probe | 0.865 ± 0.019 |
| five trained questions (macro), one shared CLM head | 0.951 ± 0.005 |
| five trained questions (macro), one probe per question | 0.948 ± 0.007 |
| five trained questions, paraphrase 1 / 2 | 0.898 ± 0.022 / 0.836 ± 0.016 |
| three held-out view questions (macro, zero-shot) | 0.491 ± 0.018 |

## Differences

| comparison | mean | sd | folds > 0 | 95% CI (corrected resampled t) |
|---|---:|---:|---:|---|
| fracture: CLM − probe | -0.006 | 0.010 | 7/20 | [-0.017, +0.006] |
| fracture: CLM − random options | -0.000 | 0.013 | 10/20 | [-0.015, +0.015] |
| fracture, no hardware: CLM − probe | -0.007 | 0.011 | 6/20 | [-0.019, +0.006] |
| five questions: shared CLM − probes | +0.004 | 0.006 | 15/20 | [-0.003, +0.010] |
| held-out views: CLM − 0.5 | -0.009 | 0.018 | 9/20 | [-0.030, +0.012] |

Runtime 12.1 min on mps.
