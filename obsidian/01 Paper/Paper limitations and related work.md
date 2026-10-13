---
tags: [paper, limitations, related-work]
---
# Limitations, failure cases and related work (from the paper)

## Failure cases and limits the paper itself reports

- **Tent can fail.** On MNIST → SVHN (a hard shift; source-only error 71.3%) Tent *increases* error to **79.8%**. For that shift the paper says success "presently requires joint optimization over source and target" (DIRT-T 45.5%, UDA-SS 38.7%).
- **Natural shifts without corruption.** On CIFAR-10.1 and ImageNetV2 (new test sets collected with the original procedure; error is higher, so shift exists) **Tent does not improve generalization**.
- **Needs batches.** Entropy needs batches for optimization and *"cannot update episodically on one point at a time"*. (TTT can, but only with a good proxy task.)
- **Errors remain.** The authors state that errors due to corruption and other shifts remain, so more adaptation is needed.
- **Unsupervised DA is still needed for the highest accuracy** and harder shifts (e.g. DIRT-T reaches 0.6% on SVHN→MNIST).
- **Cost.** Tent and TTT both require recomputing the model for each update because the loss depends on predictions.
- **Interaction with calibration.** The paper suggests entropy may interact with model calibration; better uncertainty estimates might drive better adaptation.

## Related work, in a table

| Family | Examples | How Tent differs |
|---|---|---|
| Domain adaptation | RG (adversarial), UDA-SS | They need source **and** target data jointly at adaptation time |
| Test-time training (TTT) | Sun et al. 2019 | TTT alters training and relies on a proxy task (e.g. rotation); Tent uses neither |
| Source-free adaptation | Li et al. 2020, Kundu et al. 2020, SHOT (Liang et al. 2020) | They use generative models or multiple losses/epochs and some alter training; Tent is online with one loss and an efficient parameterization |
| Test-time normalization | Schneider et al. 2020, Nado et al. 2020 | Tent builds on it (our **Norm**) and adds the entropy step |
| Entropy minimization | Grandvalet & Bengio 2005, semi-supervised and DA regularizers | Previously a **training-time regularizer** with other losses; Tent is the first to use entropy as the **sole loss at test time** |
| Feature modulation | BatchNorm, FiLM-like conditioning | Normally optimized in training and fixed at test; Tent optimizes it **at test time** with an unsupervised loss |

## Directions suggested in the discussion

More and harder shifts · more general parameters (the paper notes VisDA benefits from SHOT's parameterization) · more effective and efficient losses · adapting the input instead of the model.

## Why this matters for our experiment

- Our results cover only **CIFAR-10-C severity 5**. The paper's own failure cases are a reminder that **a gain on corruption benchmarks does not imply a gain on every shift** ([[Limitations]]).
- Tent needs batches → our batch size 200 is part of the method's operating conditions ([[Hyperparameters and protocol]]).

Back: [[Paper results]] · Home: [[00 Home]]
