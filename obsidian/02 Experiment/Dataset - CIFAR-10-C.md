---
tags: [experiment, dataset]
---
# Dataset: CIFAR-10-C

## What it is

**CIFAR-10-C** (Hendrycks & Dietterich, ICLR 2019) takes the **10,000 test images of CIFAR-10** (32×32 RGB, 10 classes, 1,000 per class) and applies **15 types of corruption**, each at **5 severity levels** (1 = mild, 5 = most severe). The labels do not change, so we can measure classification error.

![[cifar10c-example.png]]

*The same test image (a cat) under 3 of the 15 corruptions × 5 severities, taken from the data files.*

## Counting

| Step | Count |
|---|---:|
| Original CIFAR-10 test images | 10,000 |
| × corruption types | × 15 |
| × severity levels | × 5 |
| **Whole dataset** | **750,000 corrupted images** |
| **Used here (severity 5 only)** | 15 × 10,000 = **150,000 per method** |

These 150,000 evaluations are **the same 10,000 images corrupted 15 different ways**, not 150,000 different images.

## File layout (verified)

- One `.npy` file per corruption, shape `(50000, 32, 32, 3)`, dtype `uint8`, **5 blocks × 10,000 images** (one block per severity level). Each of the 15 standard files is 153,600,128 bytes (= 50,000×32×32×3 + a 128-byte header; checked by `preflight.py`).
- `labels.npy` has 50,000 labels; the **same 10,000 labels repeat in all 5 blocks**.
- The data folder contains **19** corruption files: the 15 standard ones used here plus 4 extra ones that we do **not** use (speckle noise, Gaussian blur, spatter, saturate).
- Total size on disk ≈ 2.7 GB (`du` of a folder holding all 19 files). It is **not** stored in this repository; `slurm/prepare_assets.sh` downloads it through RobustBench.

## The 15 corruptions (in the official order)

| Group | Corruptions |
|---|---|
| Noise | gaussian_noise, shot_noise, impulse_noise |
| Blur | defocus_blur, glass_blur, motion_blur, zoom_blur |
| Weather | snow, frost, fog, brightness |
| Digital | contrast, elastic_transform, pixelate, jpeg_compression |

(The grouping is the dataset authors' own; the names and order are exactly those in `official/cfgs/*.yaml`.)

## How hard is severity 5? (Source model, our run)

The same model errs on only **9.31%** of brightness images but **72.92%** of impulse-noise images. Per-corruption numbers: [[Results]].

## Why severity 5

It is the level for which the authors publish numbers in their README. The data for severities 1–4 exists but we did not evaluate it ([[Limitations]]).

Related: [[Model and library]] · [[Experiment overview]]
