**Verdict: the MedSigLIP findings replicate on a second, very different encoder.**

- **Trained questions: no gain, no loss.** One shared CLM head 0.951 macro AUROC vs one linear probe per
  question 0.948 (paired Δ +0.006 [−0.003, +0.017]) and one CLM head per question 0.950.
- **Zero-shot fails.** Held-out view questions 0.495 [0.477, 0.512], at chance, although a supervised
  probe on the same embedding reaches 0.951. Qwen3-VL-4B has no separate text tower aligned with its image
  embedding here, so there is no zero-shot baseline to compare with (see the MedSigLIP report: 0.660).
- **Paraphrase robustness comes from the text tower.** Reworded options keep macro AUROC 0.915 / 0.844;
  random option vectors fall to 0.501 / 0.545. Region wording 2 again drops to about 0.64–0.68 for every head.

*(Written by hand after the test evaluation; `evaluation/vision_eval_multiq.py` inserts this file.)*
