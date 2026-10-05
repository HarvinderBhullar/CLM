**Verdict: sharing one head across questions works, zero-shot does not, and the text tower's real
contribution is robustness to rewording.**

- **Trained questions: no gain, no loss.** One shared CLM head matches one linear probe per question
  (macro AUROC 0.952 vs 0.956; paired Δ −0.000 [−0.010, +0.008]) and one CLM head per question (0.957).
  The hardest question pays a little for sharing: fracture 0.888 ± 0.010 shared vs 0.905 ± 0.001 alone.
  Random option vectors do as well on the training wording (0.953), so there the text adds no accuracy.
- **Zero-shot fails for the CLM head, though it is possible with these image features.** The CLM head
  answers the held-out view questions at chance: macro AUROC 0.496 [0.478, 0.513]; lateral is even below
  0.5. MedSigLIP's own text tower, with no FracAtlas training at all, answers the same questions at 0.660
  (frontal 0.645, lateral 0.652, oblique 0.684). A state head trained from scratch on five questions only
  learns where images sit relative to those questions' option vectors, and so loses the image–text
  alignment MedSigLIP was pre-trained with. The view information is in the features: a supervised probe
  reaches 0.987.
- **Training is still worth a lot where labels exist.** On the trained questions MedSigLIP zero-shot
  reaches 0.811 macro (fracture 0.659, region 0.964, several scans 0.750) against 0.952 for the CLM head.
- **Paraphrase is where the text tower pays off.** Asked with option wordings it never saw, the CLM
  head keeps macro AUROC 0.919 / 0.845 on trained questions, while random option vectors fall to chance
  (0.449 / 0.460). For fracture: 0.862 / 0.842 vs 0.523 / 0.378. Robustness depends on the wording:
  region wording 2 ("lower limb", "pelvis and hip joint", ...) drops to about 0.64 for every head.

So the CLM head buys a typed-question interface that tolerates rewording of the questions it was
trained on, not answers to new questions. Keeping zero-shot ability would need a state space that stays
aligned with text: for instance initialising or regularising the state head so it maps MedSigLIP image
embeddings close to where Qwen3-8B + the action head put matching descriptions, or pre-training it on
radiograph–report pairs.

*(Written by hand after the test evaluation; `evaluation/vision_eval_multiq.py` inserts this file.)*
