---
tags: [experiment, hyperparameters]
---
# Hyperparameters and protocol

Every value below was printed by the official program in `results/job-21071/runs/b200/tent/console.log` (the `cfg` dump and the `optimizer for adaptation` line).

## Settings

| Setting | Value | Where it comes from |
|---|---|---|
| Optimizer | **Adam** | `official/cfgs/tent.yaml` (`OPTIM.METHOD`) |
| Learning rate | **0.001** | YAML (`OPTIM.LR: 1e-3`) |
| β1 | **0.9** | YAML (`OPTIM.BETA`) |
| β2 | **0.999** | hard-coded in `cifar10c.py`: `betas=(cfg.OPTIM.BETA, 0.999)` |
| Adam ε | 1e-8 | PyTorch default (printed in the log) |
| Weight decay | **0.0** | YAML (`OPTIM.WD`) |
| Steps per batch | 1 | YAML (`OPTIM.STEPS`) |
| Mode | online, no reset between batches | `MODEL.EPISODIC: False` (default) |
| Batch size | **200** | YAML (`TEST.BATCH_SIZE`); `conf.py` default is 128 |
| Images per corruption | 10,000 | `CORRUPTION.NUM_EX` (default) |
| Severity | **[5]** | our CLI override (the YAML lists `[5,4,3,2,1]`) |
| Corruptions | 15, official order | YAML (`CORRUPTION.TYPE`) |
| Random seed | 1 | `RNG_SEED` (default) |
| cuDNN benchmark | True | `CUDNN.BENCHMARK` (default) |
| BatchNorm ε / momentum | 1e-5 / 0.1 | `BN.EPS`, `BN.MOM` |

## Why these values?

We did **not** tune anything. They are the authors' defaults:
- The paper states that for non-ImageNet datasets it uses **Adam, batch size 128 and learning rate 0.001**; the public example keeps lr 0.001 and uses batch size 200.
- The code comments advise: *"start with a low learning rate (like 0.001)"* and *"try tuning the learning rate and batch size"* — so 0.001 is a safe starting point, not an optimum.
- β1 = 0.9, β2 = 0.999, ε = 1e-8 are the standard Adam defaults.

We never tried other values, so we cannot say whether they are better or worse.

## The command that was run (from `run.json`)

```text
python official/cifar10c.py --cfg official/cfgs/tent.yaml \
  DATA_DIR <data>  CKPT_DIR <models>  SAVE_DIR <run_dir>/official-output \
  CORRUPTION.SEVERITY [5]  CORRUPTION.NUM_EX 10000  TEST.BATCH_SIZE 200  RNG_SEED 1  CUDNN.BENCHMARK True
```
(`source.yaml` and `norm.yaml` for the other two methods.) All overrides equal the official defaults **except `CORRUPTION.SEVERITY [5]`**. This should not change the severity-5 numbers because the model is reset before every (corruption, severity) pair.

## Things you can change without editing official code

`TEST.BATCH_SIZE`, `OPTIM.LR`, `OPTIM.STEPS`, `OPTIM.METHOD` (Adam/SGD), `OPTIM.WD`, `MODEL.EPISODIC`, `RNG_SEED`, `CORRUPTION.SEVERITY`, `MODEL.ARCH` (e.g. `Hendrycks2020AugMix_WRN`, mentioned in the authors' README). These are natural follow-up experiments; none were run.

## Things you cannot change without editing official code

Image order (hard-coded `shuffle=False`), the dataset (the loader is hard-coded to CIFAR-10-C), and what is logged (only error %).

Related: [[Methods as implemented]] · [[Protocol - paper vs public example]]
