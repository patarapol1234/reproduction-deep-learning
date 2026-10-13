---
tags: [experiment, protocol]
---
# Protocol: paper Table 2 vs the public example

> [!important]
> Our target is the authors' **public executable example** (WRN-28-10), **not** the numbers in the paper's Table 2. The paper's model (R-26) and full training recipe were not released.

| | Paper, Table 2 | Public example (**our target**) |
|---|---|---|
| Model | R-26 (26-layer residual net) | RobustBench `Standard` **WRN-28-10** |
| Batch size | 128 | **200** (set in the YAML; the code default in `conf.py` is 128) |
| Adaptation mode | **Offline** (Section 4.1) | **Online**, predictions scored before each update |
| Image order | Shuffled, shared across methods | **Fixed** (`shuffle=False` in the official loader call) |
| Optimizer | Adam, lr 0.001 | Adam, lr 0.001, betas (0.9, 0.999), weight decay 0 |
| Source / BN / Tent error (%) | 40.8 / 17.3 / 14.3 | **43.5 / 20.4 / 18.6** |

The authors' own README says: *"The purpose of the example is explanation, not reproduction: exact details of the model architecture, optimization settings, etc. may differ from the paper."*

## Consequences

- The "published" numbers we compare against are **the README table** (severity 5, WRN-28-10), never the paper's 40.8 / 17.3 / 14.3.
- A different model, offline vs online scoring and shuffled vs fixed order all change the numbers; we did **not** isolate their individual effects.
- Matching 43.5 / 20.4 / 18.6 shows that the *public example* is reproducible. It does **not** show that the paper's R-26 numbers are reproducible.

## What the README's table reports

Severity 5 only, mean over the 15 corruptions, Source / Norm / Tent = 43.5 / 20.4 / 18.6, plus per-corruption columns. The README links a wandb report for "full results"; we did not open it. Because only severity 5 has published numbers in the repo, we ran only severity 5.

Back: [[Experiment overview]] · See also: [[Tent algorithm]] (online vs offline)
