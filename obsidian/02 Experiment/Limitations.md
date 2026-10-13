---
tags: [experiment, limitations]
---
# Limitations of this experiment

## What the evidence covers

- **One benchmark, one severity, one model:** CIFAR-10-C at severity 5 with WRN-28-10. We did not evaluate severities 1–4, CIFAR-100-C, ImageNet-C or any domain-adaptation task.
- **One protocol:** online adaptation, batch size 200, fixed image order, as in the authors' example.
- **One run:** a single run with seed 1 on one GPU model. There is **no variance estimate** across seeds, orders or hardware. (We expect the seed to matter little here because the image order is fixed, but we did not test this. A duplicate run on a different GPU model differed by ≤ 0.08 pp per corruption — see [[Run history]].)

## What the results do **not** show

- **Not a reproduction of the paper's Table 2** (R-26, batch 128, offline, shuffled): [[Protocol - paper vs public example]].
- **Not evidence that Tent helps on every shift.** Low entropy measures *confidence*, not correctness. The paper itself reports failures (MNIST → SVHN) and no gain on CIFAR-10.1 / ImageNetV2 ([[Paper limitations and related work]]).
- **No causal claim** about *why* the gain is larger where Norm leaves a high error (r = 0.93 over only 15 points).
- **The small residual differences from the published table** (up to 0.09 pp per corruption; 5 of 45 values beyond the ±0.05 rounding of the published numbers) are **not explained**: they could be run-to-run variation or the different software stack.

## Measurement limits

- The official program prints error with **two decimals**; means come from rounded values, so compare with the published table at **one** decimal.
- The official program records **only error %**. No entropy, loss curve, per-batch behavior, predictions, loop timing or GPU memory. Any such analysis needs extra instrumentation (which means editing the official code).
- Timing in [[Results]] is inferred from log timestamps and includes start-up, so it is not a benchmark.

## Setup differences from the authors

PyTorch 2.8.0 vs 1.8.1, RobustBench 1.1 vs v0.1, a different GPU ([[Compute environment]]). No RTX 4090 result exists ([[Run history]]).

## Natural follow-up experiments (not done)

| Experiment | How | Needs code changes? |
|---|---|---|
| All severities 5→1 | drop the `CORRUPTION.SEVERITY` override | no |
| Batch size 128 | `TEST.BATCH_SIZE 128` | no |
| Learning-rate sensitivity | `OPTIM.LR` | no |
| Episodic Tent / more steps | `MODEL.EPISODIC True`, `OPTIM.STEPS` | no |
| Another model | `MODEL.ARCH Hendrycks2020AugMix_WRN` | no (extra checkpoint download) |
| Shuffled image order | – | **yes** (order is hard-coded) |
| Other datasets / log entropy and loss | – | **yes** |

Back: [[Results]] · Home: [[00 Home]]
