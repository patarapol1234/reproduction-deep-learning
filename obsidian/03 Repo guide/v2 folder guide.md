---
tags: [repo-guide]
---
# What is in each folder of the original `v2/` workspace — and what to focus on

This repository was extracted from a larger working folder called **`v2/`**. That folder also contains many temporary and historical items. This note explains **every top-level item of `v2/`**, says whether it was copied into this repository, and tells you how much attention it deserves.

**Legend:** ⭐⭐⭐ read this · ⭐⭐ useful reference · ⭐ low priority · – not in this repo

## Read these first (the 30-minute path)

1. [[Experiment overview]] (5 min)
2. `official/tent.py` and `official/cifar10c.py` + [[Methods as implemented]] (10 min) — *this is the actual method*
3. `results/job-21071/summary/per_corruption.csv` + [[Results]] (5 min) — *this is the actual outcome*
4. `slurm/reproduce.sbatch` + `scripts/run_experiment.py` (5 min) — *how the run was launched*
5. [[Limitations]] (5 min)

## Folder-by-folder

### Code that was executed

| `v2/` item | What it is | In this repo | Focus |
|---|---|:-:|:-:|
| `official/` | **Unchanged snapshot of the authors' repository** (11 files, commit `e9e926a`). The code that actually ran the experiment. Most important files: `cifar10c.py` (evaluation loop), `tent.py` (Tent), `norm.py` (Norm), `conf.py` (default config), `cfgs/{source,norm,tent}.yaml` (per-method config). The rest (`README.md`, `LICENSE`, `requirements.txt`, `.gitignore`) is upstream material | ✅ `official/` | ⭐⭐⭐ |
| `slurm/reproduce.sbatch` | **The single Slurm job** that ran everything: preflight → Source, Norm, Tent → aggregate → short auto-report | ✅ | ⭐⭐⭐ |
| `slurm/setup_environment.sh`, `slurm/prepare_assets.sh` | One-time setup: create the Python environment; download CIFAR-10-C and the checkpoint, then run the preflight | ✅ | ⭐⭐ |
| `scripts/run_experiment.py` | Thin wrapper: verifies `official/` hashes, builds the command line for `official/cifar10c.py`, runs it, saves `console.log` + `run.json` | ✅ | ⭐⭐⭐ |
| `scripts/aggregate_results.py` | Parses the official logs into `summary.csv`, `per_corruption.csv`, `audit.json` and the bar-chart figure; refuses to run on incomplete data | ✅ | ⭐⭐⭐ |
| `scripts/preflight.py` | Checks before any experiment: `official/` hashes, `import cifar10c`, a real CUDA computation, data/checkpoint files → `environment.json` | ✅ | ⭐⭐ |
| `scripts/common.py` | Shared helpers (`verify_official`, `parse_errors`, hashing) | ✅ | ⭐⭐ |
| `scripts/prepare_assets.py` | Triggers the RobustBench download of the checkpoint and data | ✅ | ⭐ |
| `scripts/build_report.py` + `report/template.html` | Writes a short HTML summary at the **end of the Slurm job** from a fixed template. Included only because `reproduce.sbatch` calls it (the student id in the template was replaced by a placeholder). **Not the analysis we use — ignore** | ✅ | ⭐ |
| `scripts/render_pdf.py` | Turns an HTML report into PDF with PyMuPDF | ✅ | ⭐ |
| `scripts/package_cluster.py`, `scripts/package_results.sh` | Made tarballs to move files to/from the cluster. Not needed when you use git | – | – |
| `provenance.json` | Commit hash and SHA-256 of every file in `official/` (used by `verify_official`) | ✅ | ⭐⭐ |
| `experiment-plan.json` | Constants of the plan: methods, severity 5, the 15 corruptions, seed, published reference means | ✅ | ⭐⭐ |
| `requirements-cluster.txt`, `requirements-extra.txt`, `requirements-report.txt` | Install recipe (PyTorch 2.8.0 stack), extra packages for an existing env, and the PDF renderer | ✅ | ⭐⭐ / ⭐ / ⭐ |
| `THIRD-PARTY.md` | Attribution for the authors' MIT-licensed code | ✅ | ⭐ |

### Data produced by the run

| `v2/` item | What it is | In this repo | Focus |
|---|---|:-:|:-:|
| `results/job-21071/` | **All outputs of the run used for every number in this repo.** `environment.json` (GPU, versions, hashes), `pip-freeze.txt`, `runs/b200/{source,norm,tent}/` (`console.log` = the official log, `run.json` = command + timing + log hash, `official-output/*.txt` = the log the official program wrote itself), `summary/` (CSV + `audit.json`), `figures/` | ✅ | ⭐⭐⭐ |
| `results/job-21066/` | A **failed** attempt. Only `environment.json` (status `failed`: CUDA error) | ✅ that one file | ⭐ |
| `results/job-21057/` | A failed attempt that produced no experiment output (locally it only holds an auto-generated "interrupted" report) | – | – |
| `results/job-21072/` | An **accidental duplicate** submission on a different GPU (see [[Run history]]) | – | – |
| `results/preparation.json` | Record of the asset preparation on the login node | – | – |
| `logs/` | Slurm stdout/stderr per job. Kept for 21057, 21066 (failure evidence) and 21071 | ✅ those three | ⭐ |
| `slurm-accounting-all.txt` | `sacct` output for all jobs of the day | ✅ as `results/slurm-accounting.txt` | ⭐ |

### Analysis (what turns raw logs into the tables)

| `v2/` item | What it is | In this repo | Focus |
|---|---|:-:|:-:|
| `report-21071/make_report.py` | Computes every statistic in [[Results]] from the job-21071 files (means, per-corruption table, Tent-vs-Norm gain, correlation, comparison with the published table) and checks the data first | ✅ moved to **`analysis/make_report.py`** (student id now read from the `STUDENT_ID` environment variable) | ⭐⭐ |

### Intermediate and historical items (not copied — you can ignore them)

| `v2/` item | What it is | Why it was left out |
|---|---|---|
| `report/`, `report-final/`, `report-new/`, `report-21071/` (outputs), `report-interrupted/` | Several **versions** of a generated PDF/HTML report from different moments (the first one even documents a failed attempt) | Superseded drafts; the analysis script is kept instead |
| `dist/` | Tarball used to upload code to the cluster | Generated artifact; git replaces it |
| `cluster-download/`, `cluster-download-final/`, `cluster-download-final2/` | Temporary folders where results downloaded from the cluster were unpacked | Duplicates of data already in `results/` |
| `README.md`, `CLUSTER-COMMANDS.md`, `RECOVERY-21057.md`, `V2-HANDOFF.md`, `verification.json` | The author's internal notes and command lists (Thai; some outdated) | Replaced by this vault and the top-level `README.md` |

## What is new in this repository (not in `v2/`)

| Item | Purpose |
|---|---|
| `README.md` | Entry point |
| `obsidian/` | This vault |
| `analysis/` | The analysis script (see above) |
| `.gitignore` | Keeps data, caches and generated outputs out of git |

## Not included on purpose

- **CIFAR-10-C data** (≈ 2.7 GB) and the **checkpoint** (≈ 292 MB): downloaded by `slurm/prepare_assets.sh`.
- Any other experiment of the author (an earlier independent re-implementation exists in the author's own workspace, but it is not part of this repository).

Next: [[How to reproduce]] · Home: [[00 Home]]
