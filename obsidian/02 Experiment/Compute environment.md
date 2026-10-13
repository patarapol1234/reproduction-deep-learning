---
tags: [experiment, environment]
---
# Compute environment

Source: `results/job-21071/environment.json`, `pip-freeze.txt`, and the first lines of each `console.log`.

## Hardware (job 21071)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti |
| Node / Slurm partition | `gpu04` / `gpu4070ti` (cluster-specific names) |
| OS | Linux 7.0.0-38-generic, glibc 2.39 |
| Slurm request | 1 GPU, 4 CPU cores, 16 GB RAM, 1 h limit (`reproduce.sbatch`) |

## Software

| Package | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.8.0 (`+cu128`, CUDA build 12.8) |
| cuDNN | 9.10.2 (`91002`) |
| torchvision | 0.23.0 |
| RobustBench | 1.1 |
| NumPy | 2.0.2 |
| yacs / iopath | 0.1.8 / 0.1.8 (needed by the official `conf.py`) |
| timm | 1.0.9 |
| matplotlib | 3.9.4 |

The full list is `results/job-21071/pip-freeze.txt`. The install recipe is `requirements-cluster.txt` (+ `requirements-extra.txt`).

## Differences from the authors' environment

| | Authors (`official/requirements.txt`) | Ours |
|---|---|---|
| PyTorch | 1.8.1 | **2.8.0** |
| torchvision | 0.9.1 | 0.23.0 |
| RobustBench | v0.1 | **1.1** |

A modern GPU (RTX 30/40-series) needs a recent CUDA build, so PyTorch 1.8.1 was not an option. The official modules still import and run under PyTorch 2.8.0 (the `@torch.jit.script` entropy function compiles), and results agree with the published table at one decimal. We did **not** run the official code on the old stack, so we cannot say how much of the tiny residual differences ([[Results]], section 4) comes from the software versions.

## Data and checkpoint integrity

- Checkpoint `Standard.pt` SHA-256: `6ed1c75dad63e8ebdbef365020dd1a7eee5e79cf8a1438bd3bdc3b9c7aab3d44`
- `labels.npy` SHA-256: `e6d972b1238665d8ef54aae5affe8e292dda1eb88a6840bf0f5988cdb649da7b`
- 15 corruption files checked by `preflight.py`: shape `(50000, 32, 32, 3)`, `uint8`.
- `official/`: all 11 files verified against `provenance.json` before and after every run (`official_files_verified: true`).

## Non-determinism

The official `conf.py` warns: *"non-determinism is still present due to non-deterministic GPU ops"* (cuDNN benchmark mode is on). A second submission of the same command on a different GPU model agreed with this run to within 0.08 pp per corruption ([[Run history]]).

Related: [[Model and library]] · [[Run history]]
