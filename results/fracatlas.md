# FracAtlas fracture detection: CLM vision head vs linear probes

Research prototype, not a clinical tool. Evaluated 2026-10-04 on the **test** split (600 images, 109 fractured, 18.2%), once; every choice (configs, C, thresholds, Platt scaling) was made on val.

## Test results

CLM rows: mean ± sd over seeds 0, 1, 2 of the best val config. Single-model rows: AUROC with a 95% stratified bootstrap CI (2000 resamples). Threshold = max Youden J on val. Sens @ 90% spec is read off the test ROC. ECE: 15 equal-width bins on p(fracture); both trained methods use balanced class weights, which inflates raw p(fracture) by design, so ECE is also shown after Platt scaling fit on val.

| method | AUROC | sens @ val thr | spec @ val thr | sens @ 90% spec | ECE | ECE (val Platt) |
|---|---|---|---|---|---|---|
| resolution only | 0.528 [0.491, 0.566] | 0.147 | 0.916 | 0.147 | 0.317 | 0.003 |
| Linear probe (LR) · Qwen3-VL-4B | 0.890 [0.850, 0.927] | 0.817 | 0.849 | 0.725 | 0.159 | 0.036 |
| CLM head · Qwen3-VL-4B | 0.887 ± 0.012 | 0.820 ± 0.029 | 0.778 ± 0.032 | 0.697 ± 0.018 | 0.129 ± 0.020 | 0.043 ± 0.001 |
| Linear probe (LR) · MedSigLIP-448 | 0.898 [0.860, 0.931] | 0.771 | 0.859 | 0.743 | 0.118 | 0.020 |
| CLM head · MedSigLIP-448 | 0.898 ± 0.012 | 0.810 ± 0.023 | 0.838 ± 0.057 | 0.761 ± 0.016 | 0.088 ± 0.005 | 0.029 ± 0.006 |

### Does the CLM head add value over the linear probe?

Paired stratified bootstrap of AUROC(CLM, mean over seeds) − AUROC(linear probe), same encoder:

| encoder | ΔAUROC | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|
| Qwen3-VL-4B | -0.003 | [-0.026, +0.021] | 0.614 |
| MedSigLIP-448 | +0.000 | [-0.020, +0.020] | 0.491 |

**Verdict: no, not on this task.** On test the CLM head matches the plain linear probe within noise
on both encoders (ΔAUROC −0.003 on Qwen3-VL-4B, +0.000 on MedSigLIP; both CIs straddle 0 at about
±0.02). The +0.020 val edge on MedSigLIP did not hold up on test; it was most likely selection noise
from picking the best of 16 configs × 3 seeds on the same val split.

- **The text action tower carries no measurable signal here.** With a randomly initialised state head
  and only two frozen options, the options are just a fixed output direction. Swapping the Qwen3-8B
  option embeddings for random vectors gives 0.871 ± 0.011 (Qwen3-VL-4B) and 0.894 ± 0.007
  (MedSigLIP) vs 0.887 ± 0.012 and 0.898 ± 0.012: within about 1.5 sd with 3 seeds. The head is in
  effect a 7M-parameter MLP classifier, and on 2.8k training images that is no better than a linear model.
- **Calibration:** the CLM head's raw ECE is lower (0.129 vs 0.159; 0.088 vs 0.118), but after Platt
  scaling on val the linear probe is as good or better (0.036 vs 0.043; 0.020 vs 0.029). The frozen
  scale of 100 is not the problem: the trainable-scale ablation barely moves it (~95–99) or the metrics.
- **What the CLM head does buy is the interface.** The image becomes a typed `Choice` / `Noul` question
  in the same checkpoint format and API as text CLM. That is engineering value, not accuracy.
- **Encoder:** MedSigLIP-448 edges out Qwen3-VL-4B by about 0.01 AUROC with either method, inside the
  CIs, at a quarter of the embedding cost.

If accuracy is the goal, a linear probe on MedSigLIP is the simpler equivalent. For the CLM head to
earn its keep, the action side has to carry information: many or free-text options (e.g. fracture type
and location from report text), or pre-training the state head on image–report pairs so that the
options are more than a fixed direction.

*(Verdict written by hand after the single test run; rerunning `evaluation/vision_eval.py` regenerates
the tables and replaces this section with a placeholder.)*

## Ablations (CLM head, best config)

| method | AUROC | sens @ val thr | spec @ val thr | sens @ 90% spec | ECE | ECE (val Platt) |
|---|---|---|---|---|---|---|
| trainable logit_scale · Qwen3-VL-4B | 0.881 ± 0.003 | 0.780 ± 0.066 | 0.824 ± 0.073 | 0.688 ± 0.009 | 0.122 ± 0.025 | 0.043 ± 0.002 |
| no input standardisation · Qwen3-VL-4B | 0.890 ± 0.001 | 0.734 ± 0.009 | 0.893 ± 0.015 | 0.731 ± 0.005 | 0.234 ± 0.110 | 0.037 ± 0.008 |
| no class weights · Qwen3-VL-4B | 0.881 ± 0.012 | 0.789 ± 0.024 | 0.821 ± 0.034 | 0.706 ± 0.018 | 0.112 ± 0.010 | 0.042 ± 0.017 |
| random option vectors · Qwen3-VL-4B | 0.871 ± 0.011 | 0.780 ± 0.049 | 0.771 ± 0.068 | 0.670 ± 0.056 | 0.122 ± 0.009 | 0.036 ± 0.005 |
| trainable logit_scale · MedSigLIP-448 | 0.903 ± 0.008 | 0.792 ± 0.021 | 0.847 ± 0.011 | 0.761 ± 0.018 | 0.088 ± 0.015 | 0.032 ± 0.006 |
| no input standardisation · MedSigLIP-448 | 0.889 ± 0.007 | 0.777 ± 0.014 | 0.833 ± 0.038 | 0.722 ± 0.023 | 0.074 ± 0.010 | 0.036 ± 0.005 |
| no class weights · MedSigLIP-448 | 0.896 ± 0.003 | 0.792 ± 0.021 | 0.845 ± 0.041 | 0.758 ± 0.029 | 0.088 ± 0.015 | 0.040 ± 0.003 |
| random option vectors · MedSigLIP-448 | 0.894 ± 0.007 | 0.795 ± 0.043 | 0.845 ± 0.063 | 0.731 ± 0.026 | 0.104 ± 0.001 | 0.032 ± 0.007 |

## Per body region (test AUROC)

Regions with fewer than 5 fractured images are shown for completeness only.

| method | hand (n=190, 57+) | hip (n=27, 2+) | leg (n=310, 32+) | mixed (n=59, 16+) | shoulder (n=14, 2+) |
|---|---|---|---|---|---|
| Linear probe (LR) · Qwen3-VL-4B | 0.855 | 0.980* | 0.880 | 0.844 | 0.958* |
| CLM head · Qwen3-VL-4B | 0.818 | 1.000* | 0.891 | 0.869 | 0.931* |
| Linear probe (LR) · MedSigLIP-448 | 0.830 | 0.960* | 0.918 | 0.865 | 0.958* |
| CLM head · MedSigLIP-448 | 0.869 | 0.960* | 0.907 | 0.822 | 0.722* |

\* fewer than 5 positives: not interpretable.

## Setup

- **Data**: FracAtlas (figshare 22363012, CC BY 4.0). 4022 of 4083 images used: 2 dropped (present in both class folders, ambiguous label) and 59 dropped (truncated JPEGs, all non-fractured; decoding leaves a grey band, a label shortcut). Stratified (label × body region) image-level 70/15/15 split, seed 20261004: train 2813 (499+), val 609 (109+), test 600 (109+).
- **CLM head**: frozen image encoder → L2-normalised embedding → new `state_head` (make_head 1536×3, LayerNorm, GELU → 512, random init); options "Radiograph showing an acute bone fracture." / "Radiograph of intact bones with no fracture." embedded once by Qwen3-8B (last token, bf16) through the released `action_head`, both frozen; `logit_scale` frozen (100). Class-weighted CE over the two options, AdamW, early stopping on val AUROC. Config from a val-only grid (lr × weight decay × input dropout, 3 seeds): Qwen3-VL-4B `lr3e-05_weight_decay0.1_input_dropout0` (val 0.892); MedSigLIP-448 `lr3e-05_weight_decay0.1_input_dropout0` (val 0.918).
- **Encoders**: Qwen3-VL-4B-Instruct (fp16, MPS; chat turn = image ≤1024² px + "Radiograph for fracture assessment.", last-token final hidden state, 2560-d) and MedSigLIP-448 (pooled image embedding, 1152-d). The 24 GB Mac rules out Qwen3-VL-8B, so the released `state_head` cannot be used as a warm start (2560-d input vs 4096-d).
- **Linear probe**: standardise → logistic regression, balanced class weights, C picked on val.
- **Resolution only**: logistic regression on log(pixel count) — images > 1 MP are 29% fractured vs 16.5% for the rest, so image size alone leaks some label signal.

## Caveats

- **Leakage**: no patient ids; consecutive ids look like several views of one study, so the same patient can appear in train and test. Test numbers are optimistic for unseen patients.
- **Small test set**: 109 fractured images; single-model AUROC 95% CIs are ±0.037 wide on average, larger than most gaps between methods.
- **Val-selected configs**: CLM configs and C were chosen on val, so val numbers are optimistic; the test split was not used for any choice.
- **Small regions**: hip has 2 fractured / 27, shoulder has 2 fractured / 14 test images, too few for a per-region AUROC.
