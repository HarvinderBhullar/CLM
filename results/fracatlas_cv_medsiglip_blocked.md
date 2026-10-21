# FracAtlas cross-validation: medsiglip, blocked folds

Post-hoc robustness analysis (not pre-registered), 2026-10-21. Research prototype, not a clinical tool. Script: `evaluation/fracatlas_cv.py`.

All 4,022 images pooled; 4 × 5-fold outer CV (20 folds, about 804 test images and 143 fractured per fold), folds built from blocks of 20 consecutive image ids (approximate study grouping). Inner val split for early stopping and probe C; hyperparameters fixed at the values selected on the original val split. Values are mean ± sd over folds.

## AUROC

| | mean ± sd over folds |
|---|---:|
| fracture, CLM head | 0.903 ± 0.019 |
| fracture, CLM head with random option vectors | 0.900 ± 0.019 |
| fracture, linear probe | 0.901 ± 0.021 |
| fracture without hardware images, CLM head | 0.890 ± 0.020 |
| fracture without hardware images, linear probe | 0.887 ± 0.022 |
| five trained questions (macro), one shared CLM head | 0.959 ± 0.007 |
| five trained questions (macro), one probe per question | 0.957 ± 0.008 |
| five trained questions, paraphrase 1 / 2 | 0.911 ± 0.022 / 0.849 ± 0.017 |
| three held-out view questions (macro, zero-shot) | 0.502 ± 0.026 |

## Differences

| comparison | mean | sd | folds > 0 | 95% CI (corrected resampled t) |
|---|---:|---:|---:|---|
| fracture: CLM − probe | +0.002 | 0.011 | 12/20 | [-0.011, +0.015] |
| fracture: CLM − random options | +0.003 | 0.008 | 12/20 | [-0.006, +0.012] |
| fracture, no hardware: CLM − probe | +0.002 | 0.013 | 12/20 | [-0.012, +0.017] |
| five questions: shared CLM − probes | +0.002 | 0.005 | 14/20 | [-0.004, +0.008] |
| held-out views: CLM − 0.5 | +0.002 | 0.026 | 10/20 | [-0.029, +0.032] |

Runtime 7.5 min on mps.
