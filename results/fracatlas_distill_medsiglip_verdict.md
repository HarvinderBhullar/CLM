## Verdict

All three pre-registered hypotheses meet their criteria, but the effect is small and should be read as **above
chance, not as recovered zero-shot ability**.

- **Distillation on images transfers some view information without view labels (H3).** D1 general reaches held-out
  macro 0.531 [0.508, 0.554], against 0.484 for text-only alignment (A1 general) and 0.660 for MedSigLIP's own text
  tower. Training the head on image embeddings, where it is used, does better than training it on text embeddings.
- **Fine-tuning from the distilled head keeps it (H4, H5)** at full trained accuracy: D3 λ = 1 held-out 0.542
  [0.521, 0.561], trained 0.956; W α = 0.5 held-out 0.546 [0.525, 0.566], trained 0.952. A0 from scratch: 0.496.
- **The initialisation does the work, not the regulariser or the interpolation.** D2 (λ = 0, no distillation term)
  already keeps 0.539; no λ and no α is clearly better than another (all within 0.536–0.559, overlapping CIs).
  An alignment-initialised head (A2) did not (0.480).
- **Unexpected: the strict corpus, which never mentions views, transfers more than the general one** (D1 0.557 vs
  0.531). The protocol predicted chance for strict. View information reaches the student through other sentences
  (for example, composition and body-part descriptions that co-vary with view), so the gain is not evidence that the
  student learned the *concept* of a view.
- **The gain is fragile.** Per question it is uneven: frontal does not improve on A0 (0.48–0.58 vs 0.558), lateral
  moves from below chance (0.434) to about 0.50, and oblique gains most (up to 0.649), and with paraphrased held-out options every variant falls back to chance (0.42–0.54).
- **Cost on the trained questions:** none for the original wording (0.952–0.959 vs 0.955), but paraphrase
  robustness drops as λ grows (wording 1: 0.907 at λ = 0 → 0.840 at λ = 3; A0 0.935).

Bottom line: distillation on unlabelled images is the first variant in this study that keeps held-out questions above
chance after fine-tuning, but it recovers a small part of the teacher's zero-shot ability (about 0.04 of the 0.16
gap between chance and MedSigLIP) and that part does not survive rewording. Keeping zero-shot ability remains open.
