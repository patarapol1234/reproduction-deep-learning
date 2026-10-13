---
tags: [paper, theory, batchnorm]
---
# Feature modulation: normalization and transformation

## What Tent updates, and why only that

The paper (Section 3.2) argues against updating all model parameters θ at test time:

- θ is the **only representation of the source data** we have; changing it can make the model *diverge from its training*.
- The full model is nonlinear and high-dimensional, which makes test-time optimization **too sensitive and inefficient**.

So Tent updates only feature modulations that are **linear** (scales and shifts) and **low-dimensional** (channel-wise). Quote: *"For stability and efficiency, we instead only update feature modulations that are linear (scales and shifts), and low-dimensional (channel-wise)."* In practice these parameters are **under 1% of the model parameters**.

## Refresher: Batch Normalization (BatchNorm)

A BatchNorm layer does two steps per channel (paper Figure 4):

1. **Normalization:** x̄ = (x − μ) / σ, where μ and σ are the mean and standard deviation of the layer's features.
2. **Transformation:** x' = γ · x̄ + β, with a learnable scale γ and shift β.

In PyTorch notation (with a small ε for numerical stability):

**y = γ · (x − μ_B) / √(σ²_B + ε) + β**

- `μ_B`, `σ²_B` = mean and variance of the current batch (per channel)
- `ε` = 1e-5 in this experiment (`BN.EPS` in the official config)
- `γ`, `β` = learnable per-channel parameters

## Two different things happen in Tent

| Part | What changes | How |
|---|---|---|
| Statistics μ, σ | Re-estimated from the **target batch** during the forward pass | no gradient — just computed from the data |
| Parameters γ, β | **Optimized** to reduce entropy | gradient step after each batch |

The paper notes: *the statistics μ, σ are estimated from the data while the parameters γ, β are optimized by the loss.* The source model's stored statistics are **discarded**.

## Why this matters for our results

The first part (statistics) is exactly "test-time normalization", called **Norm** (or **BN** in the paper) — it needs no optimization. The second part (γ, β update) is what Tent adds. In our experiment, Norm alone delivers most of the improvement ([[Results]]).

The paper's ablation (Section 4.3) agrees with this decomposition:
- Not updating the normalization statistics **increases error**.
- **Not updating the transformation parameters reduces Tent to test-time normalization.**
- Updating only the last layer helps at first but degrades with more optimization.
- Updating the **full** parameters θ **never improves over the unadapted source model**.

## In our model

WRN-28-10 has **25 BatchNorm2d layers**; Tent optimizes **50 tensors** (25 × {γ, β}) — verified from the official log line `params for adaptation` ([[Methods as implemented]]).

Next: [[Tent algorithm]] · Back: [[Entropy objective]]
