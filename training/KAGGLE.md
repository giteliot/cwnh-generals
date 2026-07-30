# Training on a free Kaggle GPU

30 GPU-hours a week, 12 hours per session, T4 or P100. Enough to be useful now
that checkpoints carry full training state and resume cleanly.

> **Not verified end-to-end.** I can't run Kaggle from this machine, so the
> commands below are right in principle but expect to fix a version pin or two
> on the first go. Everything marked *measured* was measured locally.

---

## Step 0 — push the code (required)

**Nothing in `training/` is committed yet.** `git clone` on Kaggle would give
you a repo with no trainer in it. Check:

```bash
git status --porcelain
```

If `training/`, `tools/watch.py`, `tools/play_human.py` and the `tests/test_training_*`
files show as untracked, commit and push them first. Ask me and I'll do it.

If your GitHub repo is **private**, Kaggle can't clone it without credentials.
Either make it public, or skip git entirely and upload the repo as a Kaggle
Dataset (Datasets → New Dataset → upload a zip); it mounts read-only at
`/kaggle/input/<name>/`, so copy it to `/kaggle/working/` before installing.

---

## Step 1 — notebook settings

Create a new Notebook, then in the right-hand panel:

| setting | value | why |
|---|---|---|
| **Accelerator** | GPU T4 x2 (or P100) | the whole point |
| **Internet** | **On** | required for `pip` and `git`; off by default |
| **Persistence** | Files only | keeps `/kaggle/working` between sessions |

The model is tiny (~200K params at the config below), so a single T4 is plenty.
Don't wait around for a P100.

---

## Step 2 — setup cell

```python
!git clone -q https://github.com/giteliot/cwnh-generals.git /kaggle/working/repo
%cd /kaggle/working/repo
!pip install -q -e ".[train]"
!pip install -q -U "jax[cuda12]"

import jax
print(jax.devices())        # must print [CudaDevice(id=0), ...]
```

If that prints `CpuDevice`, stop — you'd be training slower than your laptop.
Usually it means the Accelerator is still None, or `jax[cuda12]` got overridden
by Kaggle's preinstalled JAX. Retry the install and restart the kernel.

`[train]` pulls flax + optax. You don't need `[dev]` unless you want to run the
tests, and you don't need pygame to work — the trainer never imports it
(*measured*).

---

## Step 3 — training cell

```python
!cd /kaggle/working/repo && python -u -m training.train \
  --stage medium --updates 20000 \
  --envs 1024 --steps 32 \
  --channels 32 --blocks 4 --groups 8 \
  --minibatches 8 --epochs 2 \
  --pool-size 8192 --gamma 0.997 \
  --anneal-frac 10 --checkpoint-every 50 \
  --resume /kaggle/working/ckpt.pkl \
  --out    /kaggle/working/ckpt.pkl \
  2>&1 | tail -n 2000
```

**Checkpoint to `/kaggle/working/`.** It's the only directory that survives the
session. Anything written inside `repo/` is lost.

`--resume` on a missing file starts fresh, so this exact cell is both the first
run and every continuation — no editing between sessions.

### Why this config and not the Mac one

The settings tuned for your laptop (`--channels 8 --blocks 2 --envs 128`) are
deliberately starved: on 8 CPU cores the gradient step was *98% of wall-clock*
(*measured*), so I shrank the network until it fit. A GPU inverts that. Scale up
or you're renting a card to run a 15K-parameter model on a 16×16 board.

The startup line prints a VRAM estimate — check it before walking away:

```
batch=32768 minibatch=4096 | est VRAM: buffer 0.3 GB + activations 1.3 GB
```

~1.7 GB against a T4's 16 GB, so there's room to push `--channels 64 --blocks 8`
if throughput allows. **`--minibatches` is the setting that OOMs you**: it
divides the batch, so lowering it makes each minibatch bigger. At `--envs 1024`
the default of 8 is already the difference between 1.3 GB and 10 GB.

---

## Step 4 — actually leave it running

This is the part that catches people. An **interactive** session stops when you
close the tab or idle out — you'll lose hours.

Use **Save Version → Save & Run All (Commit)**. That runs the whole notebook
headlessly for up to 12 h and saves `/kaggle/working` as that version's output.
Close the browser; it keeps going.

### Chaining sessions

Kaggle gives 12 h; a real run wants more. To continue:

1. New notebook version (or a copy).
2. **Add Data → Notebook Output →** pick the previous version. It mounts at
   `/kaggle/input/<notebook-name>/`.
3. Before the training cell:
   ```python
   !cp /kaggle/input/<notebook-name>/ckpt.pkl /kaggle/working/ckpt.pkl
   ```
4. Run the same training cell unchanged.

It prints what it recovered:

```
resumed /kaggle/working/ckpt.pkl: update 675/20000, bank 10, optimizer restored
```

Weights, Adam moments, the update counter, the opponent league and the RNG key
all carry over, and the shaping schedule picks up where it left off instead of
resetting. **Keep `--updates` identical across sessions** — it's the denominator
of the anneal schedule.

Not carried over: in-flight episodes. Env states restart, so you lose partially
played games. Noise against a full run.

---

## Step 5 — check it's learning

```python
!grep -E "^upd" /kaggle/working/train.log | awk 'NR%200==0'
```

In priority order:

1. **`v` under ~1.** This is the one that has bitten twice. In `ppo_medium_v1`
   it ran `0.005 → 3226 → 12,964,842` and the policy collapsed. The value head
   is now soft-clipped at ±20 so it can't diverge, but if `v` climbs into double
   digits something else is wrong — stop and look.
2. **Land count**, from a quick eval. Both previous runs ended holding **1 tile**,
   passing every turn. That's the failure mode to watch for, and win rate won't
   show it.
3. `W/L/D` shifting toward W; `H` (entropy) falling.

---

## Step 6 — get the model out

`/kaggle/working/ckpt.pkl` is downloadable from the version's Output tab. Then
locally:

```bash
.venv/bin/python tools/watch.py ppo:ckpt.pkl expander --stage medium --games 20 --no-gui
.venv/bin/python tools/play_human.py --agent ppo --checkpoint ckpt.pkl --stage medium
.venv/bin/python -m training.export ckpt.pkl      # -> competition/agents/ppo_bot/weights.npz
```

---

## Gotchas

| symptom | cause |
|---|---|
| `jax.devices()` shows CPU | Accelerator not set, or Kaggle's JAX shadowing `jax[cuda12]`. Restart kernel after installing. |
| `git clone` hangs / fails | Internet is off in settings, or the repo is private. |
| Checkpoint gone next session | Wrote inside `repo/` instead of `/kaggle/working/`. |
| OOM | `--minibatches` too low for the batch. Raise it first; it costs nothing statistically. |
| `IndexError: axis 0 with size 0` | `--pool-size` below the number of board-size combos (16 for `--stage full`). |
| Session died at ~40 min | Interactive session idled out. Use Save & Run All (Commit). |
| Very slow first update | XLA compiling the fused rollout. One-time per shape; far quicker on GPU than the 10+ min it takes on CPU. |
