---
tags: [paper, theory, entropy]
---
# The entropy objective

## Intuition first

Two students answer a 4-option multiple-choice question:

- Student A: "a 97%, b 1%, c 1%, d 1%" → confident.
- Student B: "25%, 25%, 25%, 25%" → clueless.

We do not know yet who is right, but we can tell who is **uncertain**. *Entropy* is the number that measures this uncertainty. Tent asks the model to become **less uncertain** on the test data.

## Definition

For a model prediction ŷ = f_θ(x^t), with p(ŷ_c) the probability of class c (after softmax), the Shannon entropy is

**H(ŷ) = − Σ_c p(ŷ_c) · log p(ŷ_c)**

The Tent loss on a batch X is the **average entropy over the batch**:

**L(X; θ) = −(1/|X|) Σ_{x∈X} Σ_c p_θ(c|x) · log p_θ(c|x)**

Symbols: `X` = the unlabeled test batch; `|X|` = batch size; `p_θ(c|x)` = predicted probability of class `c` for image `x`; `log` = natural log (nats).

## A tiny worked example (4 classes, natural log)

| Prediction | Entropy |
|---|---:|
| [0.97, 0.01, 0.01, 0.01] | ≈ 0.168 |
| [0.25, 0.25, 0.25, 0.25] | ≈ 1.386 (= ln 4, the maximum for 4 classes) |

For CIFAR-10 (10 classes) the maximum is ln 10 ≈ 2.303.

## Why entropy? (paper, Section 1)

- **Entropy is related to error:** predictions with lower entropy tend to have lower error (paper Figure 1, on corrupted CIFAR-100-C).
- **Entropy is related to shift:** stronger corruption → more entropy, with a strong rank correlation to the loss as the corruption level increases (paper Figure 2).
- **No labels needed:** the loss depends only on the model's own predictions.
- **Directly tied to the task:** unlike a self-supervised *proxy* task (e.g. predicting image rotation, used by TTT), entropy is defined by the supervised task itself, so no proxy needs to be designed or balanced.

## A subtle point: why batches?

Minimizing entropy of a *single* prediction has a trivial solution: put all probability on the most likely class. The paper avoids this by **optimizing batched predictions over parameters shared across the batch**. This is also why Tent needs batches and cannot update on one image at a time (see [[Paper limitations and related work]]).

## A caution

**Low entropy means "confident", not "correct".** A model can be confidently wrong. Tent works when confidence and correctness are aligned, which the paper shows for the shifts it tests; it is not a guarantee for every shift.

In the code this is a one-liner (official `tent.py`):

```python
-(x.softmax(1) * x.log_softmax(1)).sum(1)   # x = logits of one batch; entropy per image
```

Next: [[Feature modulation - normalization and transformation]] · Back: [[The setting - fully test-time adaptation]]
