**Verdict: both pre-registered hypotheses are not supported. Text-only alignment transfers some concepts
to images, but not the held-out view questions.**

- **H1 not supported.** The text-only head (A1, strict corpus, no image labels) answers the held-out view
  questions at chance: macro 0.506, CI [0.469, 0.507]. With the general corpus, which does contain view
  sentences, it is no better (0.502). MedSigLIP's own text tower gets 0.660 [0.629, 0.690] on the same
  questions, so the view information that MedSigLIP's alignment carries is lost on the way through the
  learned text-to-text map into the CLM action space.
- **Alignment does transfer other concepts (secondary outcome).** With no image labels at all, A1 answers
  fracture at 0.741 (strict) / 0.782 (general), above MedSigLIP's own zero-shot 0.659; hardware 0.892 /
  0.883 and region 0.825 / 0.796. The corpus describes fractures, hardware and body parts in many ways; it
  describes views only as a minor attribute (none in strict, ~10% of sentences in general), and the map
  that best matches corpus sentences need not keep that direction.
- **H2 not supported.** Fine-tuning from the aligned head keeps trained-question accuracy (0.953 vs 0.952
  for A0) at every λ, but no λ in {0, 0.1, 0.3, 1, 3} lifts the held-out questions above chance (0.490–0.499;
  every CI includes or lies below 0.5). The alignment loss is minimised on text and does not constrain where
  images go along directions the corpus does not exercise.
- Paraphrase robustness on trained questions may improve slightly with the aligned start (λ = 1: 0.929 /
  0.849 vs A0 0.919 / 0.845), within seed noise; not claimed.

Not tested here, and suggested by the per-question pattern: corpora that vary the held-out attribute as
much as the others, or aligning on image–text pairs instead of text alone.

*(Written by hand after the test evaluation; `evaluation/vision_eval_align.py` inserts this file.)*
