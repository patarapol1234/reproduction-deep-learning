---
tags: [paper, theory, setting]
---
# The setting: fully test-time adaptation

## Plain-language version

Imagine you ship a trained classifier to a customer. On day one it sees images that look different from your training images. You cannot ask the customer for labels, and you may not even have your training data any more. All you have is **the model and the incoming unlabeled images**. Can the model still improve? That is the *fully test-time adaptation* setting.

## Notation (from the paper)

- A model f_θ(x) with parameters θ is trained on **source** data (x^s, y^s).
- It is tested on **target** data x^t that has shifted.
- During adaptation the model may use only θ and x^t.

## How it differs from other settings (paper Table 1)

| Setting | Source data | Target data | Train-time loss | Test-time loss |
|---|---|---|---|---|
| Fine-tuning | – | x^t, y^t (labeled) | L(x^t, y^t) | – |
| Domain adaptation (DA) | x^s, y^s | x^t | L(x^s, y^s) + L(x^s, x^t) | – |
| Test-time training (TTT) | x^s, y^s | x^t | L(x^s, y^s) + L(x^s) *(self-supervised)* | L(x^t) |
| **Fully test-time adaptation (Tent)** | – | x^t | – | L(x^t) |

Key differences:
- **Fine-tuning** needs target *labels*.
- **DA** needs the *source data* at adaptation time.
- **TTT** needs the training procedure to be *changed* (an extra self-supervised task trained jointly with the main task).
- **Tent** needs none of these: it is independent of the training data and loss, given θ.

## Why the paper says this setting matters

1. **Availability** — a model may be distributed without its source data (bandwidth, privacy, profit).
2. **Efficiency** — re-processing source data at test time may be too expensive.
3. **Accuracy** — the model may be too inaccurate without adaptation.

## What the model must satisfy

The model must be (paper Section 3): **trained** for the supervised task (no supervision exists at test time), **probabilistic** (entropy needs a distribution over predictions), and **differentiable** (to use gradients). Typical deep classifiers satisfy all three.

Next: [[Entropy objective]] · Back: [[Tent paper at a glance]]
