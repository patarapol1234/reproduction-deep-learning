---
tags: [home, moc]
---
# Tent on CIFAR-10-C — reproduction notes

Notes for teammates about a reproduction of **Tent: Fully Test-Time Adaptation by Entropy Minimization** (Wang et al., ICLR 2021), using the authors' own code on CIFAR-10-C.

> [!tip] How to open
> In Obsidian choose **Open folder as vault** and select the `obsidian/` folder of this repository. All notes are plain Markdown and also readable on GitHub.

## The result in three numbers

Mean classification error over 15 corruptions, severity 5 (lower is better):

| Source | Norm | Tent |
|---:|---:|---:|
| 43.517% | 20.436% | 18.583% |

The authors' published values are 43.5 / 20.4 / 18.6. **Most of the gain comes from Norm** (using the current batch's BatchNorm statistics); Tent adds a smaller extra gain and is better than Norm on all 15 corruptions. → [[Results]]

## What to read

**If you have 10 minutes:** [[Experiment overview]] → [[Results]]

**If you want the theory (paper):**
1. [[Tent paper at a glance]]
2. [[The setting - fully test-time adaptation]]
3. [[Entropy objective]]
4. [[Feature modulation - normalization and transformation]]
5. [[Tent algorithm]]
6. [[Paper results]] and [[Paper limitations and related work]]

**If you want the experiment:**
1. [[Experiment overview]] and [[Protocol - paper vs public example]] — what exactly we reproduced (and what not)
2. [[Dataset - CIFAR-10-C]] · [[Model and library]]
3. [[Methods as implemented]] — the three methods in the official code, line by line
4. [[Hyperparameters and protocol]] · [[Compute environment]]
5. [[Results]] · [[Run history]] · [[Limitations]]

**If you want to use the repository:**
- [[v2 folder guide]] — what each folder is and what to focus on
- [[How to reproduce]] · [[Glossary]]

## Three things to keep in mind

1. **We ran the authors' unchanged code** (`official/`, commit `e9e926a`). We wrote only wrappers around it.
2. **We reproduced the public example, not the paper's Table 2.** The paper used a different model (R-26), batch size, offline scoring and shuffled order ([[Protocol - paper vs public example]]).
3. **Only severity 5, one run.** The conclusions are limited accordingly ([[Limitations]]).

## Conventions

*pp* = percentage points · *Δ* = measured − published · *Norm* = test-time normalization (called *BN* in the paper) · all errors are in %. More terms: [[Glossary]].
