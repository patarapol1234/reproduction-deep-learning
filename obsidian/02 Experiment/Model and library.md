---
tags: [experiment, model, robustbench]
---
# Model and library

## The model: WRN-28-10 (`Standard`)

- **WRN-28-10** is a Wide Residual Network, 28 layers deep with width factor 10. In the RobustBench model zoo the `Standard` entry is the default CIFAR-10 model (the authors' README: *"the default model for RobustBench"*).
- It is **pretrained**; we did **not** train or fine-tune it. Checkpoint file: `cifar10/corruptions/Standard.pt` (≈ 292 MB), SHA-256 `6ed1c75dad63e8ebdbef365020dd1a7eee5e79cf8a1438bd3bdc3b9c7aab3d44`.
- It contains **25 BatchNorm2d layers**. Tent optimizes their γ and β: **50 tensors** (25 × {weight, bias}), confirmed by the official log line `params for adaptation`.

## The library: RobustBench

RobustBench (Croce et al., NeurIPS Datasets & Benchmarks 2021) is a Python library that provides a **model zoo** and **dataset loaders**. The official example uses three functions:

| Function | Used for |
|---|---|
| `robustbench.utils.load_model("Standard", ...)` | Load the pretrained WRN-28-10 checkpoint |
| `robustbench.data.load_cifar10c(...)` | Load CIFAR-10-C (we pass severity, number of examples, `shuffle=False`, one corruption) |
| `robustbench.utils.clean_accuracy` | Evaluate the model in batches and return accuracy (error = 1 − accuracy) |

**Version:** we installed **RobustBench 1.1**. The authors' `requirements.txt` pins RobustBench **v0.1** (a git tag), so this is a difference from the original setup. RobustBench 1.1 also needs two workarounds, recorded in `requirements-cluster.txt`:

- it imports `pkg_resources`, so `setuptools<81` is pinned (you will still see a harmless `UserWarning: pkg_resources is deprecated`);
- it imports `autoattack`, whose PyPI package is a different project, so it is installed from a pinned GitHub commit (`a39220048b3c9f2cca9a4d3a54604793c68eca7e`).

## One GPU requirement worth knowing

The official `cifar10c.py` calls `.cuda()` directly, so **a CUDA GPU is required**. `scripts/preflight.py` checks that a real CUDA computation works *before* starting an experiment.

Related: [[Dataset - CIFAR-10-C]] · [[Compute environment]]
