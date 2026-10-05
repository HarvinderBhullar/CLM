**Verdict: no, not on this task.** On test the CLM head matches the plain linear probe within noise
on both encoders (ΔAUROC −0.003 on Qwen3-VL-4B, +0.000 on MedSigLIP; both CIs straddle 0 at about
±0.02). The +0.020 val edge on MedSigLIP did not hold up on test; it was most likely selection noise
from picking the best of 16 configs × 3 seeds on the same val split.

- **The text action tower carries no measurable signal here.** With a randomly initialised state head
  and only two frozen options, the options are just a fixed output direction. Swapping the Qwen3-8B
  option embeddings for random vectors gives 0.871 ± 0.011 (Qwen3-VL-4B) and 0.894 ± 0.007
  (MedSigLIP) vs 0.887 ± 0.012 and 0.898 ± 0.012: within about 1.5 sd with 3 seeds. The head is in
  effect a 7M-parameter MLP classifier, and on 2.8k training images that is no better than a linear model.
- **Calibration:** with `logit_scale` frozen at 100 the head's raw probabilities are overconfident and its
  best cut sits far below 0.5, so its plain `choice` misses many fractures (see *Served heads* below).
  Letting the scale train does not fix this (it only drifts to ~95–99). Post-hoc Platt scaling and a
  threshold fit on val (`train/calibrate_vision.py`, applied by the engine) do. After Platt scaling the
  linear probe is as well or better calibrated (ECE 0.036 vs 0.043; 0.020 vs 0.029).
- **What the CLM head does buy is the interface.** The image becomes a typed `Choice` / `Noul` question
  in the same checkpoint format and API as text CLM. That is engineering value, not accuracy.
- **Encoder:** MedSigLIP-448 edges out Qwen3-VL-4B by about 0.01 AUROC with either method, inside the
  CIs, at a quarter of the embedding cost.

If accuracy is the goal, a linear probe on MedSigLIP is the simpler equivalent. For the CLM head to
earn its keep, the action side has to carry information: many or free-text options (e.g. fracture type
and location from report text), or pre-training the state head on image–report pairs so that the
options are more than a fixed direction.

*(Written by hand after the first test evaluation; `evaluation/vision_eval.py` inserts this file into the report.)*
