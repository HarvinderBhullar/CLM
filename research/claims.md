# Claims ledger

Each claim a write-up could make, with its number, where it comes from and the commit that produced it.
All numbers are on the FracAtlas **test** split (600 images, 109 fractured) unless marked val. "CI" is a
95% stratified bootstrap interval (2000 resamples).

## Data

| # | claim | evidence | source | commit |
|---|---|---|---|---|
| D1 | 2 FracAtlas images appear in both class folders with empty fracture annotations | IMG0003375, IMG0003376 | `research/log.md`; `data/fracatlas/splits/split_meta.json` | `ebda7dc` |
| D2 | 59 JPEGs are truncated in the official zip (md5 verified), all non-fractured; decoding leaves a flat grey band of up to 54 rows | list in `split_meta.json` (`dropped_truncated`) | `tools/fracatlas_download.py` | `85f041e` |
| D3 | image size alone carries little label signal | resolution-only AUROC 0.528 [0.491, 0.566] | `results/fracatlas.md` | `768bde6` |
| D4 | 97 of 99 hardware images are fractured | dataset.csv | `research/log.md` | `7a9969c` |

## Binary fracture task

| # | claim | evidence | source | commit |
|---|---|---|---|---|
| B1 | a CLM vision head matches a linear probe on the same frozen embedding | ΔAUROC −0.003 [−0.026, +0.021] (Qwen3-VL-4B), +0.000 [−0.020, +0.020] (MedSigLIP) | `results/fracatlas.md` | `768bde6` |
| B2 | best test AUROC 0.898 (MedSigLIP, CLM head and probe) | CLM 0.898 ± 0.012; probe 0.898 [0.860, 0.931] | `results/fracatlas.md` | `768bde6` |
| B3 | with two fixed options the text tower adds no accuracy | random option vectors 0.894 ± 0.007 vs 0.898 ± 0.012 (MedSigLIP); 0.871 ± 0.011 vs 0.887 ± 0.012 (Qwen3-VL-4B) | `results/fracatlas.md` | `768bde6` |
| B4 | a val edge (+0.020, MedSigLIP) disappears on test | val 0.918 vs probe 0.898; test Δ +0.000 | `results/fracatlas.md`, `runs/.../sweep.json` | `768bde6` |
| B5 | with `logit_scale` frozen at 100 the raw head is overconfident and its plain answer misses fractures | MedSigLIP plain sens/spec 0.651 / 0.963; ~90% of val probabilities < 0.01 or > 0.99 | `results/fracatlas.md` (served heads) | `e2cf40b` |
| B6 | val-fit Platt scaling + threshold restores sensitivity and calibration | MedSigLIP 0.835 / 0.772, ECE 0.085 → 0.040; Qwen3-VL-4B 0.798 / 0.809, ECE 0.116 → 0.032 | `results/fracatlas.md` (served heads) | `e2cf40b` |

## Multi-question experiment

| # | claim | evidence | source | commit |
|---|---|---|---|---|
| M1 | one shared head answers five questions as well as one probe per question | macro 0.952 vs 0.956, Δ −0.000 [−0.010, +0.008] (MedSigLIP); 0.951 vs 0.948, Δ +0.006 [−0.003, +0.017] (Qwen3-VL-4B) | `results/fracatlas_multiq_*.md` | `7002778`, `1c0e655` |
| M2 | a head trained from scratch cannot answer held-out questions | held-out macro 0.496 [0.478, 0.513] (MedSigLIP), 0.495 [0.477, 0.512] (Qwen3-VL-4B) | same | same |
| M3 | the image features support zero-shot: MedSigLIP's own text tower answers the held-out questions | 0.660 (frontal 0.645, lateral 0.652, oblique 0.684), no training | `results/fracatlas_multiq_medsiglip.md` | `8fc425b` |
| M4 | the text tower makes answers robust to rewording | paraphrase macro 0.919 / 0.845 (CLM) vs 0.449 / 0.460 (random vectors), MedSigLIP; 0.915 / 0.844 vs 0.501 / 0.545, Qwen3-VL-4B | `results/fracatlas_multiq_*.md` | `7002778`, `1c0e655` |

## Alignment-preserving experiment

Pre-registered: `research/protocol_alignment.md` (commit `13c4e1d`, before any run). Source:
`results/fracatlas_align_medsiglip.md`.

| # | claim | evidence | status |
|---|---|---|---|
| A1 | text-only alignment (no image labels) does **not** give zero-shot on the held-out view questions | A1 strict 0.506 [0.469, 0.507]; general 0.502 [0.464, 0.503] | pre-registered H1, not supported |
| A2 | fine-tuning with an alignment regulariser keeps trained accuracy but not held-out zero-shot | A3 λ = 1: trained 0.953, held-out 0.493 [0.464, 0.496]; all λ 0.490–0.499 | pre-registered H2, not supported |
| A3 | text-only alignment transfers some concepts to images with no image labels | A1 fracture 0.741 / 0.782 vs MedSigLIP zero-shot 0.659; hardware 0.892 / 0.883; region 0.825 / 0.796 | pre-registered secondary outcome |
| A4 | an aligned start does not change trained-question accuracy | 0.953 (A2/A3) vs 0.952 (A0) | pre-registered secondary outcome |
