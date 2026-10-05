**Verdict: sharing one head across questions works, zero-shot does not, and the text tower's real
contribution is robustness to rewording.**

- **Trained questions: no gain, no loss.** One shared CLM head matches one linear probe per question
  (macro AUROC 0.952 vs 0.956; paired Δ −0.000 [−0.010, +0.008]) and one CLM head per question (0.957).
  The hardest question pays a little for sharing: fracture 0.888 ± 0.010 shared vs 0.905 ± 0.001 alone.
  Random option vectors do as well on the training wording (0.953), so there the text adds no accuracy.
- **Zero-shot fails.** The held-out view questions are answered at chance: macro AUROC 0.496
  [0.478, 0.513]; lateral is even below 0.5. A state head trained from scratch on five questions only
  learns where images sit relative to those questions' option vectors; nothing links the text "frontal
  view" to the image features that encode the view, although those features exist (a supervised probe
  reaches 0.987).
- **Paraphrase is where the text tower pays off.** Asked with option wordings it never saw, the CLM
  head keeps macro AUROC 0.919 / 0.845 on trained questions, while random option vectors fall to chance
  (0.449 / 0.460). For fracture: 0.862 / 0.842 vs 0.523 / 0.378. Robustness depends on the wording:
  region wording 2 ("lower limb", "pelvis and hip joint", ...) drops to about 0.64 for every head.

So the CLM head buys a typed-question interface that tolerates rewording of the questions it was
trained on, not answers to new questions. Zero-shot would need a state space aligned with text before
fine-tuning, e.g. pre-training the state head on radiograph–report pairs, or starting from an encoder
whose image and text towers are already aligned (MedSigLIP's own text tower is the natural baseline).

*(Written by hand after the test evaluation; `evaluation/vision_eval_multiq.py` inserts this file.)*
