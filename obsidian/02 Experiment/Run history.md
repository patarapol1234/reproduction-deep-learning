---
tags: [experiment, runs, cluster]
---
# Run history

Source: `results/slurm-accounting.txt` (the `sacct` output) and `logs/`. Times are Slurm elapsed times.

| Job | Node (partition) | State | Elapsed | What happened |
|---|---|---|---:|---|
| 21057 | gpu01 | **FAILED** (exit 18) | 0:00:00 | The script called `nvidia-smi`, which printed `Failed to initialize NVML: Driver/library version mismatch` (NVML library 580.178); with `set -e` the job stopped at once |
| 21066 | gpu01 (`gpu4090`) | **FAILED** (exit 1) | 0:00:43 | Script fixed so `nvidia-smi` is only a warning and a real CUDA check runs. The check failed: PyTorch reported `Error 804: forward compatibility was attempted on non supported HW`; **no method started** |
| 21069 | – | cancelled | – | Submitted with `--exclude=gpu01…` and cancelled while still pending; never ran |
| **21071** | `gpu04` (`gpu4070ti`) | **COMPLETED** | 0:06:02 | **The run used for all results in this repository** (RTX 4070 Ti) |
| 21072 | `gpu05` (`gpu3080`) | COMPLETED | 0:06:08 | An **accidental duplicate submission** of the same command on an RTX 3080. Its outputs are **not included** here and not used. For transparency: its means were 43.520 / 20.435 / 18.589%, and its per-corruption errors differed from job 21071 by at most 0.08 pp |

## The `gpu01` problem (not a code bug)

- The same node `gpu01` (an RTX 4090 node) ran an earlier, different experiment successfully weeks before, with the same partition, GPU request and Python environment.
- The scripts never touch CUDA environment variables or drivers.
- So the failure is a **node-level driver problem** (NVML and the CUDA runtime failed together). The root cause was **not confirmed** by the cluster administrators. Our guess — a driver update without reloading the kernel module — is **unverified**.
- Because of this there is **no RTX 4090 result** for this experiment.

## Practical lessons (useful if you re-run on a shared cluster)

- Check file hashes on the cluster before `sbatch`.
- A node listed as `idle` is not necessarily healthy; the preflight check (`scripts/preflight.py`) catches a broken GPU in ~40 s.
- Slurm node names may be fully-qualified (e.g. `gpu01.<domain>`); `--exclude` needs the exact name.
- Command-line flags of `sbatch` override the `#SBATCH` lines in the script (we used this to list several partitions).

Related: [[Compute environment]] · [[How to reproduce]]
