# Training a PPO agent

Implementation of [PPO_AGENT.md](../PPO_AGENT.md). Read that for *why*; this file is *how*.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev,train]"
```

`train` pulls in flax + optax. JAX is already a base dependency.

Verify before you spend GPU hours — these tests are the safety net for the whole pipeline:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests -q          # 79 tests
```

## Start training

The curriculum is **map size only** — `truncation` stays at 1200 because reaching turn 1200 is a
genuine draw under the competition ruleset, not an artificial cutoff. Smaller boards put the
generals closer together, so games end by decapitation instead of timing out, which is what makes
the terminal reward dense enough to learn from.

```bash
# 1. Sanity check — should beat nothing but must not NaN, and entropy must fall
.venv/bin/python -m training.train --stage tiny --updates 200 --envs 64 --steps 128

# 2. Small boards: the real first milestone
.venv/bin/python -m training.train --stage small --updates 3000 --envs 256 --steps 128 \
    --out ckpt_small.pkl

# 3. Medium, warm-started
.venv/bin/python -m training.train --stage medium --updates 3000 --envs 256 --steps 128 \
    --resume ckpt_small.pkl --out ckpt_medium.pkl

# 4. Full competition ruleset (18-21 padded to 22, fog, build-castles, deathtouch@800)
.venv/bin/python -m training.train --stage full --updates 20000 --envs 512 --steps 128 \
    --resume ckpt_medium.pkl --out ckpt_full.pkl
```

Stages (`--stage`):

| | board | notes |
|---|---|---|
| `tiny` | 8–9, pad 10 | fastest loop, use for debugging |
| `small` | 10–12, pad 12 | first real learning signal |
| `medium` | 14–16, pad 16 | |
| `full` | `mode="competition"` | 18–21 pad 22, the actual ruleset |

The network is fully convolutional with global pooling and no flatten, so a checkpoint transfers
across board sizes unchanged — that's what makes `--resume` across stages work.

### Reading the log

```
upd   120 | loss -0.412 v 0.031 pi -0.008 H 4.21 | eps  312 W/L/D 141/138/ 33 | alpha 0.94 | sps  48213 |   0.7s
```

- `H` — policy entropy. Starts near `log(#valid actions)` (~8) and should fall steadily. Flat high
  entropy = not learning; entropy at ~0 by update 500 = collapsed, raise `ENT_COEF`.
- `W/L/D` — from the **learner's** seat. A climbing `D` is the classic shaping bug: the agent is
  farming potential and stalling. Check it every time you change the reward.
- `alpha` — shaping weight, annealed 1 → 0 over `--anneal-frac` of the run. The final policy must
  optimize the true win/loss objective.
- `eps` — episodes finished this update. If this is 0 for many updates on `full`, games are running
  to truncation and you have almost no terminal signal; drop to a smaller stage.

### Opponents

Per update, the opponent is sampled: `--expander-prob` (default 0.2) of the time it's
`ExpanderAgent`, otherwise the current policy (`--self-prob`, default 0.5) or a random past
checkpoint from a bank of `--bank-size` snapshots taken every `--checkpoint-every` updates. The
scripted anchor matters — pure self-play cycles.

Seats are split 50/50 across envs automatically (P0 wins move-order ties, so the seats are not
symmetric and a P0-only agent plays measurably worse as P1).

## Hardware reality — read this before planning

**This needs a GPU. Measured on this Mac (CPU JAX): ~35 environment steps/second.** At that rate
10⁸ steps would take about a month. Use the local machine to validate correctness only; run real
training on CUDA:

```bash
pip install -U "jax[cuda12]"
```

Budget on the order of 10⁸–10⁹ environment steps for a competitive agent, and plan in GPU-days.

### Where to run

The model is tiny (652K params, 22×22 boards) — this is **not** a big-GPU problem. A T4 or 3090 is
plenty; an A100 buys almost nothing. What you actually need is *many uninterrupted hours*.

- **Kaggle Notebooks** — best free option: 30 h/week GPU (T4/P100), 12 h max per session, and the
  quota is predictable. Sessions die at the limit, so checkpoint to `/kaggle/working` and chain runs
  with `--resume`.
- **Google Colab free** — usable but the least reliable; disconnects are arbitrary. Point `--out` at
  mounted Drive or you lose the checkpoint.
- **TPU Research Cloud** — free TPU v3/v4 for research, application-based. Best *fit* for a JAX
  codebase, but the rollout would need pmap/sharding work first. Worth it only for a long project.
- **Paid spot, the pragmatic choice** — Vast.ai RTX 4090 from ~$0.29/hr, RunPod ~$0.34/hr. Three
  GPU-days ≈ **$20–25**, with no session limits and no babysitting. Cheaper than your time spent
  restarting Kaggle sessions.

Because free tiers cut sessions off, always run with checkpointing on and resume:

```bash
.venv/bin/python -m training.train --stage full --envs 512 --steps 128 --minibatches 64 \
    --checkpoint-every 25 --out /kaggle/working/ckpt.pkl --resume /kaggle/working/ckpt.pkl
```

### Stopping and resuming

Checkpoints are **full training state**: weights, Adam moments, the update counter, the checkpoint
league, and the RNG key. Resuming continues the run rather than restarting it — in particular the
shaping anneal picks up where it left off instead of snapping `alpha` back to 1.0.

```bash
# stop
kill $(cat train.pid)

# continue — same --updates total, same file for --resume and --out
.venv/bin/python -u -m training.train --stage tiny --updates 2000 \
  --envs 128 --steps 32 --channels 16 --blocks 3 --minibatches 4 --epochs 2 \
  --pool-size 4096 --checkpoint-every 25 \
  --resume ckpt_overnight.pkl --out ckpt_overnight.pkl
```

It prints what it recovered:

```
resumed ckpt_overnight.pkl: update 675/2000, bank 10, optimizer restored
```

Writes go through a `.tmp` + atomic `os.replace`, so killing mid-write can't corrupt the file.
`--resume` on a non-existent path starts fresh, so the same command works for both the first launch
and every continuation.

**Keep `--updates` the same across resumes.** It's the denominator of the anneal schedule
(`progress = update / (updates × anneal_frac)`) — lowering it on a resume fast-forwards `alpha` to 0.

One thing that does *not* carry over: in-flight episodes. Env states and beliefs restart from the
pool head, so partially-played games are lost on resume. Costs a few hundred wasted turns, which is
noise against a full run — not worth persisting.

**Expect a long first update — it's XLA compilation, not a hang.** The whole rollout (env step,
feature extraction with 6 BFS passes, two network forwards) is fused into one `jit`, so the graph
is large. At `--envs 32 --steps 16 --channels 32 --blocks 4` the first update took 20 s and later
ones 14 s. At production settings (`--steps 128`, `--blocks 8`) compilation on CPU exceeded 10
minutes before I killed it. Compile cost grows with `--steps`, `--blocks`, and `--channels`; it is
paid once per shape configuration, and it is dramatically better on GPU. If you're iterating on
code, use `--stage tiny --steps 16 --blocks 2 --channels 16` to keep the loop tight.

`eps 0` in the log for the first while is normal: with `--steps 16` a rollout is 16 turns and games
last hundreds, so no episode finishes inside one update. Episodes still complete across updates —
state carries over.

### VRAM sizing — set `--minibatches` before you launch

The startup line prints an estimate:

```
batch=65536 minibatch=1024 | est VRAM: buffer 1.3 GB + activations 2.3 GB
```

**Activation memory scales with minibatch size, and the default `--minibatches 8` is too small a
divisor at production scale.** At `--envs 512 --steps 128` it gives a 8192-sample minibatch needing
~18 GB of activations — an instant OOM on a 16 GB card. Raise the divisor:

| envs | steps | `--minibatches` | minibatch | est. total VRAM | fits |
|---|---|---|---|---|---|
| 512 | 128 | 8 (default) | 8192 | ~19.6 GB | A100 only |
| 512 | 128 | **64** | 1024 | **~3.6 GB** | T4 / any |
| 256 | 128 | 32 | 1024 | ~2.9 GB | T4 / any |
| 128 | 64 | 16 | 512 | ~1.3 GB | anything |

Keeping the *batch* large while making each *minibatch* small costs nothing statistically — PPO
takes more gradient steps per update, which is generally an improvement. So on a free T4:

```bash
.venv/bin/python -m training.train --stage full --envs 512 --steps 128 --minibatches 64 ...
```

Tunable without editing code: `--minibatches --epochs --lr --gamma --ent-coef`.

## Export and validate the submission

```bash
.venv/bin/python -m training.export ckpt_full.pkl          # -> competition/agents/ppo_bot/weights.npz

SDL_VIDEODRIVER=dummy .venv/bin/python tools/localmatch.py \
    competition/agents/ppo_bot/run.sh \
    competition/agents/expander_python/run.sh --games 20 --swap
```

Always validate with `tools/localmatch.py`, **never** `competition/matchup.py` — the latter bypasses
`env.step` and so applies neither build-castles nor deathtouch (see ARCHITECTURE.md, *Traps*).

The submitted bot runs **pure NumPy**, no JAX. Measured on this machine for the default
64-channel/8-block network (651,871 params):

| | measured | budget |
|---|---|---|
| Weight load (cold start) | 6 ms | 10 s first move |
| Full `act()` on 22×22 | 6.8 ms p50, 7.0 ms max | 150 ms/turn |
| 2400-turn match | 0 faults, 0 over budget | 50 faults = forfeit |

~22× headroom, so there is room to grow the network if it helps. `tests/test_training_parity.py`
asserts the NumPy feature extraction, action mask, and forward pass match the JAX ones exactly —
run it after **any** change to `training/features.py`, or train/inference will silently diverge.

Submission is a zip of `competition/agents/ppo_bot/` (must contain `run.sh` and `weights.npz`).

## Layout

| file | role |
|---|---|
| `features.py` | `Observation` + `Belief` → 32 planes + 20 globals. Size-agnostic. |
| `potential.py` | Income-denominated potential Φ and the PBRS shaping term. |
| `masking.py` | Legal-action mask over the flat action space. |
| `actions.py` | Flat index ↔ `[pass,row,col,dir,split]`. |
| `network.py` | Flax fully-conv actor-critic. |
| `ppo_core.py` | PPO update, adapted from PureJaxRL's `ppo.py`. |
| `train.py` | Self-play rollout, league, curriculum, entry point. |
| `export.py` | Checkpoint → `weights.npz`. |

## Gotchas that cost real time

- **`pool_size` must be ≥ the number of board-size combinations** (16 for `full`: 18–21 × 18–21).
  Integer division floors to zero per combo and you get an empty pool with a confusing
  `IndexError: index is out of bounds for axis 0 with size 0`.
- **`env.step` auto-resets**, and `timestep.observation` is the observation of the *reset* state.
  Rewards and bootstrapping must use `timestep.last_state`. Handled in `train.py`; don't undo it.
- **flax `GroupNorm` on a rank-3 `(H,W,C)` input treats `H` as the batch axis** and normalizes per
  row. `network.py` adds an explicit batch axis (`y[None]`) to avoid this. It's silent if you get it
  wrong — the parity test catches it.
- Turn 0 has exactly **one** legal action (pass): generals start with 1 army, and a move needs 2.
