---
tags: [repo-guide, howto]
---
# How to reproduce

Two levels: **(A)** verify and re-analyze the data in this repository on any laptop (no GPU); **(B)** re-run the experiment on a machine with an NVIDIA GPU.

## A. Verify and re-analyze (no GPU)

Needs Python 3.9+. Run from the repository root.

```bash
# 1. official/ is byte-for-byte what provenance.json says (11 files)
python3 -c "import sys; sys.path.insert(0,'scripts'); from common import verify_official; print(verify_official()['commit'])"

# 2. optional: compare with upstream yourself
curl -sSL -o /tmp/tent.tgz https://codeload.github.com/DequanWang/tent/tar.gz/e9e926a668d85244c66a6d5c006efbd2b82e83e8
shasum -a 256 /tmp/tent.tgz     # expected a6a177f4f7419bee6d48d6db17004902a72614c21f78a84fa51570ce78434396
tar -xzf /tmp/tent.tgz -C /tmp && diff -r /tmp/tent-e9e926a668d85244c66a6d5c006efbd2b82e83e8 official && echo IDENTICAL

# 3. recompute the summary from the raw logs (needs matplotlib; rewrites summary/ with identical content)
python3 scripts/aggregate_results.py --results-dir results/job-21071

# 4. regenerate the statistics, tables and a PDF report (standard library only; PDF needs PyMuPDF)
STUDENT_ID="<your id>" python3 analysis/make_report.py
pip install PyMuPDF==1.26.5 && python3 scripts/render_pdf.py --report-dir analysis
```

Step 4 stops with an assertion error if the data is inconsistent (wrong log order, hash mismatch, etc.). Generated files in `analysis/` are git-ignored.

## B. Re-run the experiment on a Slurm GPU cluster

**Requirements:** a Slurm cluster with an NVIDIA GPU whose driver supports the CUDA 12.8 build of PyTorch 2.8.0 · Python 3 (the recorded run used 3.12.3) · `git` and internet access for `pip` and for the downloads · ≈ 3 GB free disk. Run `prepare_assets.sh` on a machine with internet (we ran it on the login node; compute nodes may have none). See [[Compute environment]].

> [!warning] What has and has not been tested
> **Tested:** the unchanged `official/` hashes, re-aggregation of the raw logs, the analysis script, and that `requirements-cluster.txt` resolves to a consistent set of packages (a pip dry run on a Mac with Python 3.9 — not on Linux/CUDA).
> **Not tested:** `setup_environment.sh` building a fresh environment from scratch, and running `sbatch` from a clean clone on another cluster. The recorded run used a Python environment that **already existed** (created for an earlier experiment, plus `yacs` and `iopath`), so the "create new environment" branch of `setup_environment.sh` has never been exercised. Expect to fix small environment issues on a new cluster (CUDA driver vs wheel, no internet on compute nodes, module system, Python version).
> Running without Slurm is also untested: `reproduce.sbatch` reads `SLURM_JOB_ID`, but you can run its steps (`preflight.py`, `run_experiment.py` ×3, `aggregate_results.py`) by hand.

```bash
git clone <this repo> ~/deep_learning/src/RRv2 && cd ~/deep_learning/src/RRv2
bash slurm/setup_environment.sh      # creates ~/deep_learning/env_tent from requirements-cluster.txt
bash slurm/prepare_assets.sh         # downloads CIFAR-10-C (~2.7 GB) + checkpoint (~292 MB), runs preflight
mkdir -p logs
sbatch slurm/reproduce.sbatch        # EDIT the #SBATCH --partition line first
```

**What to adapt** (all in `slurm/reproduce.sbatch`):
- `#SBATCH --partition=gpu4090` is the author's cluster; change it to a GPU partition of yours.
- Default paths can be overridden with environment variables: `TENT_PROJECT_DIR` (default `~/deep_learning/src/RRv2`), `TENT_ENV_DIR`, `TENT_DATA_DIR`, `TENT_MODEL_DIR`.
- `TENT_RUN_BATCH128=1` adds a batch-size-128 run (not part of our results).

**What you should see** in the Slurm output within ~1 minute: `Inputs and official imports ready; CUDA=True`, then `Starting official source, B=200`. The whole job took ~6 minutes on an RTX 4070 Ti. When finished, `results/job-<ID>/summary/audit.json` must say `"status": "completed"`.

If the job stops in the preflight with a CUDA error, the GPU or driver of that node is unusable — try another node ([[Run history]] explains what happened to us).

## C. Running something different with the official code

The wrapper `scripts/run_experiment.py` only allows batch size 200 or 128 and severity `[5]` on purpose. To explore other settings, call the official script directly (from `official/`):

```bash
cd official
python cifar10c.py --cfg cfgs/tent.yaml DATA_DIR <data> CKPT_DIR <models> SAVE_DIR /tmp/out \
       TEST.BATCH_SIZE 128                 # any option from conf.py can be appended like this
```

Options you can change without editing code are listed in [[Hyperparameters and protocol]]; ideas in [[Limitations]].

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UserWarning: pkg_resources is deprecated` | Harmless (RobustBench 1.1). Keep `setuptools<81` |
| `ModuleNotFoundError: autoattack` | RobustBench needs it; install from the pinned GitHub commit (done by `setup_environment.sh`) |
| `Failed to initialize NVML: Driver/library version mismatch` | Broken GPU driver state on that node; use another node |
| `The unchanged official cifar10c.py requires a CUDA GPU` | Preflight found no working CUDA device |
| `Official source missing or modified` | A file in `official/` changed; restore it from git |

Home: [[00 Home]] · Folder map: [[v2 folder guide]]
