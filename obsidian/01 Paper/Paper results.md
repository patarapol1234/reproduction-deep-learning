---
tags: [paper, results]
---
# What the paper reports

All numbers below are copied from the paper (ICLR 2021 version). They are **not** produced by this repository — see [[Results]] for our measurements, and [[Protocol - paper vs public example]] for why the two are not directly comparable.

## Paper setup (Section 4)

| Item | Paper |
|---|---|
| Datasets | CIFAR-10-C, CIFAR-100-C, ImageNet-C (corruptions); SVHN → MNIST / MNIST-M / USPS (digits); GTA → Cityscapes; VisDA-C |
| Corruption benchmarks | Test/validation sets duplicated and corrupted with **15 types at 5 severity levels** (Hendrycks & Dietterich, 2019) |
| Models | **R-26** (26-layer residual net) on CIFAR-10/100; R-50 on ImageNet |
| Optimizer | Adam on non-ImageNet datasets; SGD with momentum on ImageNet |
| Batch size / LR | **BS 128, LR 0.001** (non-ImageNet); BS 64, LR 0.00025 (ImageNet) |
| Order | Shuffled, with the same order shared across methods |
| Baselines | source, RG, UDA-SS, TTT, **BN** (= test-time normalization, our *Norm*), PL (pseudo-labeling) |

## Corruption robustness — Table 2 (severity 5, mean error %, offline)

| Method | Needs at test time | CIFAR-10-C | CIFAR-100-C |
|---|---|---:|---:|
| Source | – | 40.8 | 67.2 |
| RG (adversarial DA) | source + target, training | 18.3 | 38.9 |
| UDA-SS | source + target, training | 16.7 | 47.0 |
| TTT | altered training | 17.5 | 45.0 |
| BN (test-time norm) | target | 17.3 | 42.6 |
| PL (pseudo-label) | target | 15.7 | 41.2 |
| **Tent** | target | **14.3** | **37.3** |

## ImageNet-C (Section 4.1)

| Method | Error (%) |
|---|---:|
| Source | 59.5 |
| BN / Norm | 49.9 |
| ANT (adversarial noise training) | 50.2 |
| AugMix | 51.7 |
| ANT + SIN | 47.4 |
| **Tent, online** | **44.0** |
| **Tent, offline** | **42.3** |

Among fully test-time methods, Tent gives an **18% relative improvement** over test-time normalization; against Schneider et al.'s 49.3% for normalization, 14%. Tent reaches the least error on most corruption types without increasing the error on the original (uncorrupted) data.

## Source-free domain adaptation — digits (Table 3, error %)

| Method | MNIST | MNIST-M | USPS |
|---|---:|---:|---:|
| Source | 18.2 | 39.7 | 19.3 |
| RG | 15.0 | 33.4 | 18.9 |
| UDA-SS | 11.1 | 22.2 | 18.4 |
| BN | 15.7 | 39.7 | 18.0 |
| Tent (1 epoch) | 10.0 | 37.0 | 16.3 |
| Tent (10 epochs) | 8.2 | 36.8 | 14.4 |

Tent always beats BN and the source model and is best in 2 of 3 cases without using source data.

## Other benchmarks

- **Semantic segmentation (GTA → Cityscapes, IoU, higher is better):** source 28.8%, BN 31.4%, Tent 35.8% (offline, Adam); 36.4% when adapting to a single image with 10 episodic iterations.
- **VisDA-C (error):** source model 56.1%, Tent 45.6%; 39.6% when all layers except the final classifier are updated.
- **Alternative architectures on CIFAR-100-C (Table 4, error %, Source / BN / Tent):** SAN-10 (pair) 55.3 / 39.7 / 36.7 · SAN-10 (patch) 48.0 / 31.8 / 29.2 · MDEQ (large) 53.3 / 44.9 / 41.7.

## Analysis experiments (Section 4.3)

- **Tent reduces entropy and loss** on CIFAR-100-C (paper Figure 6).
- **Tent needs feature modulation:** see the ablation in [[Feature modulation - normalization and transformation]].
- **Adaptation generalizes across points:** adapting on target *train* data and testing on target *test* data lowers error (CIFAR-100-C 37.3% → 34.2%; SVHN→MNIST 8.2% → 6.5%), so the adapted modulation is not specific to the points used for the update.
- **Tent's effect differs from normalization:** adapted features move closer to an *oracle* trained with target labels, not just closer to the clean reference features.

Back: [[Tent algorithm]] · Next: [[Paper limitations and related work]]
