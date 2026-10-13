---
tags: [paper, overview]
---
# The Tent paper at a glance

**Wang, Shelhamer, Liu, Olshausen, Darrell. "Tent: Fully Test-Time Adaptation by Entropy Minimization." ICLR 2021 (spotlight).** arXiv:2006.10726v3 · code: `github.com/DequanWang/tent`

## The problem

A model trained on one data distribution (the *source*) loses accuracy on shifted data (the *target*), for example when images are noisy, blurred or foggy. We often cannot fix this by re-training, because after deployment we may have **no source data and no target labels**.

## The idea in one paragraph

While the model is running on test data, let it **adapt itself using only its own predictions**. Tent measures how *uncertain* the predictions are (their **entropy**) and updates a **very small set of parameters** — the per-channel scale γ and shift β of the BatchNorm layers — to make the predictions more confident. It also re-estimates the BatchNorm statistics from each test batch. This needs just one extra gradient step per batch and does not change how the model was trained.

## Three building blocks

| Block | In one line | Note |
|---|---|---|
| Setting | *Fully test-time adaptation*: only the trained model + unlabeled target data are available | [[The setting - fully test-time adaptation]] |
| Objective | Minimize the Shannon entropy of the predictions | [[Entropy objective]] |
| Parameters | Update only channel-wise affine parameters of normalization layers (and use batch statistics) | [[Feature modulation - normalization and transformation]] |
| Algorithm | Online: predict, then update on that batch; the update affects the *next* batch | [[Tent algorithm]] |

## Headline results claimed in the paper

- **ImageNet-C:** 44.0% error (online) / 42.3% (offline) — a new state of the art at the time; test-time normalization alone gets 49.9%, source model 59.5%.
- **CIFAR-10-C (severity 5, R-26, offline):** 14.3% vs 17.3% for test-time normalization and 40.8% for the source model.
- Also: digit adaptation (SVHN → MNIST/MNIST-M/USPS), semantic segmentation (GTA → Cityscapes), VisDA-C.

Details: [[Paper results]] · Caveats and failure cases: [[Paper limitations and related work]]

## Where things are in the paper

| Section | Content |
|---|---|
| 1 | Introduction, contributions, Figures 1–2 (entropy vs error / corruption) |
| 2 | Setting (Table 1) |
| 3 | Method: objective (3.1), parameters (3.2), algorithm (3.3) |
| 4 | Experiments (4.1 corruptions, 4.2 domain adaptation, 4.3 analysis) |
| 5–6 | Related work, discussion |

## How this repository relates to the paper

We did **not** reproduce the paper's numbers. We reproduced the authors' public CIFAR-10-C example, which uses a different model and protocol: [[Protocol - paper vs public example]].
