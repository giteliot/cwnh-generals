# training

PPO agent for generals.io. Reward and observation: [REWARD_AND_OBSERVATION.md](REWARD_AND_OBSERVATION.md). GPU: [KAGGLE.md](KAGGLE.md).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,train]"
```

## Test a model

```bash
# vs a bot: expander | hunter | random | harvester
.venv/bin/python tools/watch.py ppo:ckpt.pkl hunter --stage medium --games 20 --no-gui

# same, then open a replay of the last game
.venv/bin/python tools/watch.py ppo:ckpt.pkl hunter --stage medium

# two checkpoints against each other
.venv/bin/python tools/watch.py ppo:new.pkl ppo:old.pkl --stage medium --games 20 --no-gui

# play it yourself
.venv/bin/python tools/play_human.py --agent ppo --checkpoint ckpt.pkl --stage medium
```

`--stage` must match what it trained on. Read the **land** column, not just the winner.

Replay: `Space` play/pause · `←/→` step · `R` restart · `Q` quit
Play: arrows move cursor · `Enter` pick source · arrow to move (`Shift` = split) · `B` build · `Space` pass · `Q` quit

## Train

```bash
.venv/bin/python -m training.train --stage medium --updates 4000 \
  --envs 2048 --steps 32 --channels 32 --blocks 8 --groups 8 \
  --minibatches 32 --epochs 2 --pool-size 8192 \
  --gamma 0.997 --anneal-frac 10 --checkpoint-every 50 \
  --resume ckpt.pkl --out ckpt.pkl
```

Stages: `tiny` 8–9 · `small` 10–12 · `medium` 14–16 · `full` 18–21 (competition).

`--resume` on a missing file starts fresh, so the same line works every time. Keep `--updates` fixed across resumes.

## Reading the log

```
upd 120 | loss -0.41 v 0.03 pi -0.01 H 4.2 | eps 312 W/L/D 141/138/33 | alpha 0.94 | sps 48213 | 0.7s
```

- `v` must stay under ~1. If it climbs into the thousands, stop.
- `W/L/D` from the learner's seat. Rising `D` means it's stalling.
- `H` entropy, should fall.

## Submit

```bash
.venv/bin/python -m training.export ckpt.pkl        # -> competition/agents/ppo_bot/weights.npz
SDL_VIDEODRIVER=dummy .venv/bin/python tools/localmatch.py \
    competition/agents/ppo_bot/run.sh competition/agents/expander_python/run.sh --games 20 --swap
```

Zip `competition/agents/ppo_bot/`.

## Tests

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests -q
```

Run after any change to `features.py` — the parity test catches train/inference drift.

## Gotchas

- Changing observation channels or `--blocks` breaks old checkpoints. Changing reward doesn't.
- OOM: lower `--envs` (buffer) or raise `--minibatches` (activations).
- `--pool-size` must be ≥ 16 on `--stage full`.
