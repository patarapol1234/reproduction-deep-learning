---
tags: [experiment, overview]
---
# Experiment overview

## Goal

Check whether the authors' **public CIFAR-10-C example** of Tent can be reproduced, and whether the ranking **Source > Norm > Tent (error)** holds. Two concrete questions:

1. Does updating BatchNorm statistics from the test batch (**Norm**) reduce error on corrupted images?
2. Does Tent's entropy minimization give a **further** improvement on top of Norm?

## What was run

| Item | Value |
|---|---|
| Code | Authors' **unchanged** repository, commit `e9e926a668d85244c66a6d5c006efbd2b82e83e8` (`official/`) |
| Entry point | `official/cifar10c.py --cfg official/cfgs/{source,norm,tent}.yaml`, one process per method |
| Model | RobustBench `Standard` WRN-28-10, pretrained (not trained by us) — [[Model and library]] |
| Data | CIFAR-10-C, **15 corruptions at severity 5**, 10,000 images each = 150,000 per method — [[Dataset - CIFAR-10-C]] |
| Methods | Source, Norm, Tent — [[Methods as implemented]] |
| Tent settings | Adam, lr 0.001, betas (0.9, 0.999), weight decay 0, 1 step/batch, batch 200, online — [[Hyperparameters and protocol]] |
| Hardware | Slurm job 21071 on an NVIDIA GeForce RTX 4070 Ti — [[Compute environment]] |
| Output | Error (%) per corruption per method — [[Results]] |

## What we wrote (and what we did not)

- We wrote **no adaptation code**. All of Source/Norm/Tent comes from the authors' files `tent.py`, `norm.py` and `cifar10c.py`.
- We wrote only *wrappers*: environment setup, a preflight check (hash of `official/`, CUDA, data files), a runner that calls the official script and keeps its log, an aggregator that parses the log, and an analysis script. See [[v2 folder guide]].

## Pipeline

```text
slurm/setup_environment.sh   -> Python env (torch 2.8.0, robustbench 1.1, ...)
slurm/prepare_assets.sh      -> download CIFAR-10-C + checkpoint (RobustBench), run preflight
slurm/reproduce.sbatch (1 Slurm job):
    scripts/preflight.py         -> checks official/ hashes, CUDA works, data files OK -> environment.json
    scripts/run_experiment.py x3 -> official/cifar10c.py (source, norm, tent) -> console.log, run.json
    scripts/aggregate_results.py -> summary.csv, per_corruption.csv, audit.json, figure
analysis/make_report.py      -> derived statistics + tables/figure (local, no GPU)
```

## Headline result

Mean error over 15 corruptions: **43.517% (Source) → 20.436% (Norm) → 18.583% (Tent)**, matching the authors' published 43.5 / 20.4 / 18.6 at one decimal. Norm provides 92.6% of the total improvement. Details: [[Results]].

## Scope and honesty notes

- Only **severity 5** was run (the authors' YAML runs severities 5→1; we override to `[5]` to match the one published table).
- **One run**, seed 1, fixed image order. No variance estimate. See [[Limitations]].
- This is the authors' *explanatory example*, not an exact reproduction of the paper's Table 2: [[Protocol - paper vs public example]].
- Run history, including failed attempts: [[Run history]].

Home: [[00 Home]]
