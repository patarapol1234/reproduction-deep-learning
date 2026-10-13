---
tags: [paper, theory, algorithm]
---
# The Tent algorithm (paper Section 3.3)

## Three phases

**Initialization.** The optimizer collects the affine parameters {γ_{l,k}, β_{l,k}} for each normalization layer *l* and channel *k*. All other parameters θ \ {γ, β} are **fixed**. The normalization statistics {μ, σ} stored from the source data are **discarded**.

**Iteration.** Each step handles one batch:
- The normalization statistics are estimated for each layer, during the **forward pass**.
- γ, β are updated by the gradient of the prediction entropy ∇H(ŷ), during the **backward pass**.
- *"Note that the transformation update follows the prediction for the current batch, and so it only affects the next batch (unless forward is repeated)."* This needs **one gradient per point** of extra computation, so it is the default.

**Termination.** *Online:* no termination — keep going as long as test data arrives. *Offline:* update on the data first, then run inference again (possibly for several epochs).

## Pseudocode

```text
init:   collect γ, β of every BatchNorm layer; freeze everything else
        discard the stored source statistics
for each test batch x:                       # online
    logits = model(x)                        # BN uses statistics of THIS batch
    output logits  -> predictions are scored here (before the update)
    loss = mean over the batch of entropy(softmax(logits))
    loss.backward()                          # gradients only reach γ, β
    optimizer.step()                         # e.g. Adam, 1 step
    optimizer.zero_grad()
    # next batch uses the updated γ, β
```

## Online vs offline vs episodic

| Mode | Meaning | Used in |
|---|---|---|
| **Online** | Update after every batch; predictions are made *before* that batch's update | Our experiment and the authors' public CIFAR-10-C example |
| **Offline** | Adapt on the whole test set, then infer again | Paper Table 2 (CIFAR-10/100-C) |
| **Episodic** | Reset the model for every batch/point | Not used here (`MODEL.EPISODIC False`) |

Note the paper's Table 2 is **offline**, while the public example (which we reproduced) is **online** — one reason the numbers cannot be compared directly ([[Protocol - paper vs public example]]).

## Complexity

Tent needs one forward and one backward pass per batch and updates only ~<1% of the parameters. In our logs, Tent took about 9 s per corruption vs about 5 s for Source/Norm ([[Results]]).

## Where this lives in the code

`official/tent.py` — see [[Methods as implemented]] for the line-by-line mapping.

Back: [[Feature modulation - normalization and transformation]] · Next: [[Paper results]]
