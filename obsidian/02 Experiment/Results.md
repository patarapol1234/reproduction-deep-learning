---
tags: [experiment, results]
---
# Results (Slurm job 21071)

> [!summary]
> Official code, CIFAR-10-C, **severity 5**, batch size 200: mean error **43.517% (Source) → 20.436% (Norm) → 18.583% (Tent)**. This agrees with the authors' published table (43.5 / 20.4 / 18.6) to one decimal. Norm provides 92.6% of the total improvement; Tent adds 1.85 pp more and beats Norm on all 15 corruptions.

All numbers below are computed from `results/job-21071/` (`console.log`, `summary/*.csv`, `audit.json`). The official program prints each error with **two decimals**, so means are computed from rounded values and should only be compared to the published table at **one decimal**.

## 1. Mean error over the 15 corruptions

| Method | Published (README) | This run | Δ (pp) |
|---|---:|---:|---:|
| Source | 43.5 | **43.517** | +0.017 |
| Norm | 20.4 | **20.436** | +0.036 |
| Tent | 18.6 | **18.583** | -0.017 |

Δ = measured − published. "pp" = percentage points.

| Quantity | Value |
|---|---:|
| Source → Norm | −23.081 pp |
| Norm → Tent | −1.853 pp (9.1% relative) |
| Source → Tent (total) | −24.933 pp (57.3% relative) |
| Share of the total gain that Norm already provides | 92.6% |

**Take-away:** most of the improvement comes from simply using the *current batch's* BatchNorm statistics (Norm). Tent's entropy step gives a smaller, but consistent, extra gain. See [[Methods as implemented]].

## 2. Per-corruption error (%), severity 5

Format: *our value (Δ vs published)*. Last two columns are computed from our values.

| Corruption | Source | Norm | Tent | Norm − Tent | Source − Norm |
|---|---|---|---|---:|---:|
| Gaussian noise | 72.32 (+0.02) | 28.05 (−0.05) | 24.85 (+0.05) | 3.20 | 44.27 |
| shot noise | 65.72 (+0.02) | 26.12 (+0.02) | 23.50 (+0.00) | 2.62 | 39.60 |
| impulse noise | 72.92 (+0.02) | 36.28 (−0.02) | 33.03 (+0.03) | 3.25 | 36.64 |
| defocus blur | 46.95 (+0.05) | 12.79 (−0.01) | 11.96 (−0.04) | 0.83 | 34.16 |
| glass blur | 54.32 (+0.02) | 35.30 (+0.00) | 31.83 (+0.03) | 3.47 | 19.02 |
| motion blur | 34.75 (−0.05) | 14.16 (−0.04) | 13.65 (−0.05) | 0.51 | 20.59 |
| zoom blur | 42.01 (+0.01) | 12.11 (+0.01) | 10.77 (−0.03) | 1.34 | 29.90 |
| snow | 25.09 (−0.01) | 17.27 (−0.03) | 15.91 (+0.01) | 1.36 | 7.82 |
| frost | 41.32 (+0.02) | 17.42 (+0.02) | 16.11 (−0.09) | 1.31 | 23.90 |
| fog | 26.00 (+0.00) | 15.23 (−0.07) | 13.71 (+0.01) | 1.52 | 10.77 |
| brightness | 9.31 (+0.01) | 8.41 (+0.01) | 7.88 (−0.02) | 0.53 | 0.90 |
| contrast | 46.71 (+0.01) | 12.68 (+0.08) | 12.17 (+0.07) | 0.51 | 34.03 |
| elastic transform | 26.58 (−0.02) | 23.78 (−0.02) | 21.97 (−0.03) | 1.81 | 2.80 |
| pixelate | 58.45 (−0.05) | 19.63 (−0.07) | 17.26 (−0.04) | 2.37 | 38.82 |
| jpeg compression | 30.30 (+0.00) | 27.31 (+0.01) | 24.15 (−0.05) | 3.16 | 2.99 |

Visual version: `results/job-21071/figures/per_corruption.png`

![[per_corruption.png]]

## 3. Observations (all derived from the table above)

- **Tent < Norm on all 15 corruptions.** Largest gains: glass blur (3.47 pp), impulse noise (3.25), Gaussian noise (3.20). Smallest: motion blur and contrast (0.51 pp each).
- **The gain from Tent is larger where Norm still leaves a high error:** Pearson r = 0.93, Spearman ρ = 0.90 between "error after Norm" and "Norm − Tent" over the 15 corruptions. This is a correlation over 15 points, not proof of a mechanism.
- **Where Norm helps most (Source → Norm):** Gaussian noise (−44.27 pp), shot noise (−39.60), pixelate (−38.82), then impulse noise (−36.64). It helps least on brightness (−0.90), elastic transform and jpeg compression.
- **Hardest after Tent:** impulse noise (33.03%) and glass blur (31.83%). **Easiest:** brightness (7.88%).

## 4. Agreement with the published per-corruption table

- Largest difference over the 45 values (3 methods × 15 corruptions): **0.09 pp**.
- The published values are rounded to 0.1, so differences up to 0.05 pp are expected from rounding alone. **5 of 45** values are larger than that: Norm fog (−0.07), Norm contrast (+0.08), Norm pixelate (−0.07), Tent frost (−0.09), Tent contrast (+0.07).
- With a single run we cannot say whether these small differences come from run-to-run variation or from the different software stack (see [[Compute environment]]).

## 5. Time (from log timestamps; not a benchmark)

| Method | Whole process | Median per corruption |
|---|---:|---:|
| Source | 88.2 s | ~5 s |
| Norm | 73.8 s | ~5 s |
| Tent | 151.8 s | ~9 s |

Whole-process time includes start-up and data loading (the first corruption is slower). The official program does not record loop timings or GPU memory.

## 6. What the official program records

Only **one metric**: the classification error (%) per corruption (two decimals). It does **not** log entropy, loss, per-batch curves or predictions. See [[Limitations]].

Related: [[Hyperparameters and protocol]] · [[Run history]] · [[Protocol - paper vs public example]]
