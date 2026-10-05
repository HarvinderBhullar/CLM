# Research log: CLM vision extension on FracAtlas

Chronological record of what was done, why, what went wrong and what was found. Times are CEST.
Commit hashes refer to branch `fracatlas-vision`. Numbers are copied from the result files named; those
files are the source of truth.

## Rules followed throughout

- Every choice (hyperparameters, logistic-regression C, decision thresholds, calibration) is made on the
  validation split. The test split is evaluated once per experiment, after the code and the selection are fixed.
- Evaluation scripts have a `--dry-run` mode that runs on val, used to debug before the single test run.
- Seeds 0, 1, 2 for every trained CLM head; results are mean ± sd over seeds.
- Data, embeddings, checkpoints stay out of git; everything is reproducible from the README commands.
- Research prototype, not a clinical tool.

## 2026-10-04 — binary fracture task

### Design (before any code)

- CLM = frozen LLM encoder (Qwen3-8B, last-token pooling) + projection heads. Plan: swap the state tower
  (frozen image encoder + new trainable `state_head`), keep the action tower (Qwen3-8B + released
  `action_head`, frozen).
- Loss: class-weighted cross-entropy over the fixed option embeddings, not in-batch InfoNCE (with two
  classes, same-label images would be treated as negatives).
- Hardware: 24 GB Apple Silicon Mac, so `Qwen/Qwen3-VL-4B-Instruct` (2560-d) instead of the 8B model, plus
  `google/medsiglip-448` (1152-d pooled) as a medical encoder. No vLLM; `transformers` on MPS.

### Data cleaning (commit `ebda7dc`, then `85f041e`)

| issue | images | action |
|---|---:|---|
| present in both `Fractured/` and `Non_fractured/`, csv says fractured, YOLO boxes empty | 2 (IMG0003375, IMG0003376) | dropped (ambiguous label) |
| JPEG truncated in the source zip (md5 verified), all non-fractured, IMG0004028–IMG0004347 | 59 | dropped after splitting: decoding fills the bottom with flat grey (up to 54 rows), a label shortcut |

Result: 4,022 images, 717 fractured (17.8%). Split train 2,813 (499+), val 609 (109+), test 600 (109+).
Image size correlates with the label (29.1% fractured above 1 MP vs 16.5% below); checked later with a
resolution-only reference (test AUROC 0.528, negligible).

### Embeddings (commits `85f041e`, `a9faaa8`)

- Qwen3-VL-4B: chat turn = image (≤1024² px) + "Radiograph for fracture assessment.", final-norm hidden
  state of the last token, L2-normalised, fp16. Batched vs single-image embeddings agree (cos ≥ 0.999996).
- MedSigLIP: pooled image embedding, L2-normalised, fp16.
- Option texts through Qwen3-8B (bf16, `transformers`, same token recipe as `train/embed_utils.py`) and the
  released action head. Sanity check: the released heads classify 4/4 text-only fracture reports correctly.
- Problem: a run slowed 10× from memory swapping (other apps open). Fixed by loading weights directly to
  MPS and emptying the MPS cache after each batch.

### Training, baselines, test evaluation (commits `6588d29`, `358abac`, `768bde6`)

- Val-only grid lr × weight decay × input dropout, 3 seeds; ablations: trainable `logit_scale`, no input
  standardisation, no class weights, random option vectors.
- Implementation details: input standardisation is folded into the first Linear so checkpoints keep the
  CLM format; the trainable-scale ablation starts just under the clamp (log 100) because `clamp` passes
  no gradient above it.
- Test (once), `results/fracatlas.md`: CLM head = linear probe within noise (ΔAUROC −0.003 Qwen3-VL-4B,
  +0.000 MedSigLIP, both CIs ±0.02). A +0.020 val edge (MedSigLIP) did not hold: selection noise from 48
  runs on one val split. Random option vectors match text options: with two fixed options the text tower is
  only a fixed output direction.

### Serving (commit `1cf3306`)

Engine routes image states (`{"type": "image", "image": <base64>}`) to the head's own image encoder;
mismatched head/state modalities are refused; the HTTP server never reads paths. Text path verified
byte-identical before/after.

### Calibration (commit `e2cf40b`)

- Finding (user question "why always predicting no fracture"): with `logit_scale` frozen at 100 the raw
  head is overconfident (~90% of probabilities < 0.01 or > 0.99); its plain `choice` (p > 0.5) missed
  35–40% of fractures. The earlier sensitivity numbers used a val-chosen threshold the API did not apply.
- Fix: Platt scaling + Youden threshold fit on val (`train/calibrate_vision.py`), stored in
  `cfg["calibration"]`, applied by the engine. Test, deployed seed-0 heads: MedSigLIP sens/spec
  0.651/0.963 → 0.835/0.772, ECE 0.085 → 0.040; Qwen3-VL-4B 0.596/0.937 → 0.798/0.809, ECE 0.116 → 0.032.
- Note: the released MedSigLIP `logit_scale` / `logit_bias` equal SigLIP's initial values (10, −10).

## 2026-10-05 — multi-question experiment (commits `7a9969c`, `7002778`, `8fc425b`, `1c0e655`)

### Design

- 8 typed questions from FracAtlas labels: fracture (choice), body region (choice, 5), frontal / lateral /
  oblique view (noul), orthopedic hardware (noul), several scans (noul), fracture count (score 0/1/2+).
  Three wordings per option: 0 for training, 1–2 only for paraphrase tests (`train/fracatlas_questions.py`).
- One state head for all questions; one image embedding answers every question. The three view questions
  are held out (never trained on, never used for selection) and asked zero-shot.
- Controls: random option vectors; one CLM head per question; one linear probe per question; MedSigLIP's
  own text tower (no training).
- Problem: embedding 60 option texts one by one thrashed swap for 15+ min (each text re-read 16 GB of
  weights from swap). Fixed with one right-padded batch (61 s; cos 0.99996 to per-text embeddings).
- Label notes: 97 of 99 hardware images are fractured; region / view / scans are easy for every method.

### Results (test, once)

| macro AUROC | MedSigLIP trained | MedSigLIP held out | Qwen3-VL-4B trained | Qwen3-VL-4B held out |
|---|---:|---:|---:|---:|
| one CLM head, all questions | 0.952 | 0.496 | 0.951 | 0.495 |
| random option vectors | 0.953 | 0.465 | 0.946 | 0.495 |
| one CLM head per question | 0.957 | 0.978 (supervised) | 0.950 | 0.954 (supervised) |
| one linear probe per question | 0.956 | 0.974 (supervised) | 0.948 | 0.951 (supervised) |
| MedSigLIP text tower, no training | 0.811 | 0.660 | – | – |
| CLM head, paraphrased options (w1 / w2) | 0.919 / 0.845 | | 0.915 / 0.844 | |
| random vectors, paraphrased (w1 / w2) | 0.449 / 0.460 | | 0.501 / 0.545 | |

Findings: sharing a head across questions costs nothing; zero-shot fails for a head trained from scratch,
although MedSigLIP's aligned text tower shows the image features support it (0.660); the text tower makes
answers robust to rewording. Sources: `results/fracatlas_multiq_medsiglip.md`,
`results/fracatlas_multiq_qwen3-vl-4b.md`.

### Selection notes

- Qwen3-VL-4B multi-question best lr 1e-5 is the grid edge; the next configs are within 0.0002 val AUROC.
- Two MedSigLIP probes picked the largest C (view_oblique, multiscan; val AUROC 0.97 / 1.00).

## 2026-10-05 — alignment-preserving experiment

Pre-registered in `research/protocol_alignment.md` before any run. Results are appended below once the
test evaluation has run.
