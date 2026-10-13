---
tags: [experiment, code, methods]
---
# Methods as implemented in the official code

All code below is from the authors' repository (MIT license), commit `e9e926a`, **unchanged**. Files: `official/cifar10c.py`, `official/tent.py`, `official/norm.py`, `official/conf.py`.

## The three methods

| Method | BatchNorm statistics | γ, β updated? | Code |
|---|---|---|---|
| **Source** | stored, from training | no | `setup_source`: just `model.eval()` |
| **Norm** | **current batch** | no | `norm.Norm` → `configure_model` |
| **Tent** | **current batch** | **yes**, to minimize entropy | `tent.Tent` → `configure_model`, `collect_params`, `forward_and_adapt` |

## Source (`cifar10c.py`)

```python
def setup_source(model):
    model.eval()
    return model
```
It has no `reset()` method, so the official loop logs `not resetting model` (×15 in our log). Nothing needs resetting.

## Norm (`norm.py`)

```python
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.train()              # use batch-wise statistics in forward
        m.eps = eps            # 1e-5
        m.momentum = momentum  # 0.1
```
Putting each BatchNorm layer in *train mode* makes it normalize with the **current batch's** statistics. No parameter is optimized. (The layers still keep updating their running averages in the background, but those are not used to normalize the current batch.) The log line `stats for adaptation` lists **75** entries = 25 layers × (`running_mean`, `running_var`, `num_batches_tracked`).

## Tent (`tent.py`)

**Configuration — only BatchNorm is trainable, and it always uses batch statistics:**

```python
def configure_model(model):
    model.train()
    model.requires_grad_(False)                 # freeze everything
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)              # except BatchNorm γ, β
            m.track_running_stats = False       # force batch statistics
            m.running_mean = None
            m.running_var = None
    return model
```

**Which parameters are optimized:** `collect_params` returns `weight` (γ, scale) and `bias` (β, shift) of every `nn.BatchNorm2d` → 50 tensors in WRN-28-10.

**One adaptation step — note that the logits are returned *before* the update takes effect:**

```python
@torch.enable_grad()
def forward_and_adapt(x, model, optimizer):
    outputs = model(x)                              # forward (BN uses this batch)
    loss = softmax_entropy(outputs).mean(0)         # mean entropy over the batch
    loss.backward()                                 # gradients reach only γ, β
    optimizer.step()                                # Adam step
    optimizer.zero_grad()
    return outputs                                  # predictions made BEFORE this update
```

**Entropy:** `-(x.softmax(1) * x.log_softmax(1)).sum(1)` (see [[Entropy objective]]).

**Reset:** `Tent.reset()` reloads the saved initial model and optimizer state (`copy_model_and_optimizer` is called once in the constructor). With `episodic=False` (our setting) Tent never resets *between batches*.

## The evaluation loop (`cifar10c.py`)

```python
for severity in cfg.CORRUPTION.SEVERITY:          # we pass [5]
    for corruption_type in cfg.CORRUPTION.TYPE:   # 15 types, fixed order
        try:   model.reset()                      # fresh model + optimizer for each corruption
        except: logger.warning("not resetting model")
        x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX, severity, cfg.DATA_DIR, False, [corruption_type])
        x_test, y_test = x_test.cuda(), y_test.cuda()
        acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)   # batches of 200, in order
        logger.info(f"error % [{corruption_type}{severity}]: {1. - acc:.2%}")
```

Consequences you can read directly from this loop:
- **Online:** within a corruption, Tent keeps adapting from batch to batch (no reset), so 10,000 images ÷ 200 = **50 adaptation steps per corruption**.
- **Reset between corruptions:** each corruption starts from the *source* weights and a fresh Adam state, so one corruption cannot leak into the next.
- **Fixed order:** the loader is called with `shuffle=False` (hard-coded; there is no config switch).
- **Error is printed with two decimals** (`:.2%`), which limits the precision of every number we report.

Related: [[Tent algorithm]] · [[Hyperparameters and protocol]] · [[v2 folder guide]]
