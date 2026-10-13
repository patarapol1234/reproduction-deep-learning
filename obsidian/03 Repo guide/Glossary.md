---
tags: [repo-guide, glossary]
---
# Glossary

| Term | Meaning in this project |
|---|---|
| **Source** | The pretrained model used as is; BatchNorm uses statistics stored from training |
| **Target** | The (shifted) data seen at test time, without labels |
| **Distribution shift** | Test data looks different from training data (noise, blur, weather …) |
| **Test-time adaptation (TTA)** | Adjusting a model while it processes test data |
| **Fully test-time adaptation** | TTA that uses *only* the trained model and unlabeled target data — no source data, no labels ([[The setting - fully test-time adaptation]]) |
| **TTT** (test-time training) | A TTA method that changes training (extra self-supervised task); not Tent |
| **DA** (domain adaptation) | Adaptation that needs source *and* target data jointly |
| **Entropy** | Uncertainty of a prediction: −Σ p log p. Low = confident ([[Entropy objective]]) |
| **BatchNorm (BN)** | Layer that normalizes features per channel and then applies a learnable scale γ and shift β |
| **γ (gamma), β (beta)** | The learnable BatchNorm scale and shift — the only parameters Tent updates |
| **Norm** (called **BN** in the paper) | Test-time normalization: use the current batch's BN statistics, update nothing |
| **Tent** | Norm + entropy minimization over γ, β |
| **Online adaptation** | Update after every batch; predictions are scored before that batch's update |
| **Offline adaptation** | Adapt on the whole test set first, then predict again |
| **Episodic** | Reset the model for every batch (not used here) |
| **Corruption** | A synthetic image degradation (e.g. Gaussian noise, fog) |
| **Severity (level)** | Strength of a corruption: 1 (mild) to 5 (most severe). We use **5** |
| **CIFAR-10-C** | CIFAR-10 test set × 15 corruptions × 5 severities ([[Dataset - CIFAR-10-C]]) |
| **WRN-28-10** | Wide Residual Network, depth 28, width factor 10 — the model used |
| **R-26** | 26-layer residual network used in the paper's CIFAR experiments (not released) |
| **RobustBench** | Python library providing the checkpoint and the CIFAR-10-C loader ([[Model and library]]) |
| **pp** | Percentage points (a difference between two percentages) |
| **Δ** | measured − published |
| **Published (reference)** | The numbers in the authors' repository README (43.5 / 20.4 / 18.6), not the paper's Table 2 |
| **Slurm** | Job scheduler on the GPU cluster; `sbatch` submits a job, `sacct` shows its accounting |
| **Preflight** | `scripts/preflight.py`: sanity checks before spending GPU time |
| **Job 21071** | The Slurm job whose outputs are used for all results ([[Run history]]) |
