# Tent on CIFAR-10-C — reproduction with the official code

A reproduction of the authors' public CIFAR-10-C example of **Tent** ("Tent: Fully Test-Time Adaptation by Entropy Minimization", Wang et al., ICLR 2021, [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)).

Tent adapts a trained image classifier **at test time**, without labels or source data: for every test batch it re-estimates BatchNorm statistics and updates only the BatchNorm scale/shift (γ, β) to reduce the entropy of the model's predictions.

We ran the authors' **unchanged** code (`official/`, commit [`e9e926a`](https://github.com/DequanWang/tent/tree/e9e926a668d85244c66a6d5c006efbd2b82e83e8)) on the pretrained RobustBench `Standard` WRN-28-10 model and **CIFAR-10-C at severity 5** (15 corruptions × 10,000 images), batch size 200, online adaptation.

## Result

Mean error over the 15 corruptions (%, lower is better; one run, Slurm job 21071, NVIDIA RTX 4070 Ti):

| Method | Published (authors' README) | This run |
|---|---:|---:|
| Source (no adaptation) | 43.5 | **43.517** |
| Norm (current-batch BatchNorm statistics) | 20.4 | **20.436** |
| Tent (Norm + entropy minimization) | 18.6 | **18.583** |

- Agreement with the published table: within 0.04 pp on the means, at most 0.09 pp per corruption.
- Norm gives **92.6%** of the total improvement; Tent adds 1.85 pp more and beats Norm on **all 15** corruptions.
- This reproduces the authors' *public example*. It is **not** a reproduction of the paper's Table 2 (different model, batch size, offline scoring and image order).
- Limits: one severity, one run, fixed image order, newer PyTorch than the authors' pin. See `obsidian/02 Experiment/Limitations.md`.

## Where to start

Open **`obsidian/`** as an [Obsidian](https://obsidian.md) vault (or read the Markdown files directly). Start with `obsidian/00 Home.md`. It contains

- the theory of the main paper (setting, entropy objective, BatchNorm modulation, algorithm, results, limitations),
- the experiment (dataset, model, the three methods as implemented in the official code, hyperparameters, environment, results, run history),
- a guide to every folder of the original working directory and what to focus on (`obsidian/03 Repo guide/`).

## Repository layout

```
official/            Unchanged snapshot of the authors' code (the code that ran)   <- focus
slurm/               setup_environment.sh, prepare_assets.sh, reproduce.sbatch    <- focus: reproduce.sbatch
scripts/             Wrappers: preflight, run_experiment, aggregate_results, ...   <- focus: run_experiment, aggregate_results
results/job-21071/   All outputs of the run that the results come from            <- focus
results/job-21066/   Failed attempt (CUDA error), evidence only
results/slurm-accounting.txt   Slurm accounting for all jobs
logs/                Slurm stdout/stderr (jobs 21057 and 21066 failed, 21071 used)
analysis/            make_report.py: statistics, tables and figure from the job outputs
report/              template.html, used only by the last step of reproduce.sbatch (ignore)
obsidian/            Documentation vault (English)
provenance.json      Commit + SHA-256 of every file in official/
experiment-plan.json Plan constants (methods, severity, corruptions, reference means)
requirements-*.txt   Install recipes (PyTorch 2.8.0 stack)
```

## Quick checks (no GPU needed)

```bash
python3 -c "import sys; sys.path.insert(0,'scripts'); from common import verify_official; print(verify_official()['commit'])"
python3 scripts/aggregate_results.py --results-dir results/job-21071     # needs matplotlib
STUDENT_ID="<your id>" python3 analysis/make_report.py                    # standard library only
```

To re-run the experiment on a Slurm GPU cluster, see `obsidian/03 Repo guide/How to reproduce.md`.

## What is not in this repository

- The **CIFAR-10-C data** (≈ 2.7 GB) and the **model checkpoint** (≈ 292 MB). They are downloaded by `slurm/prepare_assets.sh` through RobustBench.
- Outputs of other jobs (a duplicate submission is described in `obsidian/02 Experiment/Run history.md`).

## Notes on the raw files

`results/` and `logs/` are the unmodified outputs of the cluster run (their paths refer to the cluster where they ran). The `run.json` files store SHA-256 hashes of the logs and the analysis script checks them, so please do not edit these files.

## Licenses and attribution

`official/` is the authors' code under the **MIT license** (`official/LICENSE`); see `THIRD-PARTY.md`. The wrapper and analysis scripts were written for a course project; no separate license has been chosen. The pretrained model and CIFAR-10-C are obtained from RobustBench and the dataset authors under their own terms.

## References

1. D. Wang, E. Shelhamer, S. Liu, B. Olshausen, T. Darrell. *Tent: Fully Test-Time Adaptation by Entropy Minimization.* ICLR 2021.
2. Official Tent repository: <https://github.com/DequanWang/tent>
3. D. Hendrycks, T. Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.* ICLR 2019.
4. F. Croce et al. *RobustBench: a standardized adversarial robustness benchmark.* NeurIPS Datasets & Benchmarks 2021.
