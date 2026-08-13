# Plan: Add JAX PPO Training to the generals.bot Challenge Repository

## Objective

Starting from the existing `generals.bot` challenge repository, add a clean PPO training layer using **JAX**, while preserving the existing game environment.

The goal is to make the following components independently configurable:

1. Raw environment observations
2. Observation processing / feature engineering
3. Reward calculation
   - terminal reward
   - potential-based shaping reward
4. Opponent selection
5. PPO actor/critic training
6. Checkpointing and evaluation

Do **not** rewrite or substantially modify the existing game environment unless required to expose an interface.

---

## Phase 1 — Clone and understand the repository

Clone the `generals.bot` challenge repository.

Before implementing anything:

- Identify the existing game/environment implementation.
- Identify how a game is initialized.
- Identify how actions are submitted.
- Identify how observations are generated.
- Identify how the game determines termination.
- Identify how the opponent is currently selected.
- Identify whether the environment already has a Gym/Gymnasium-compatible interface.
- Identify dependencies and Python version requirements.

Do not immediately modify the environment.

Create a short architectural map of the existing codebase.

---

## Phase 2 — Define the RL interface

Create a thin adapter around the existing environment if necessary.

The desired conceptual interface is:

```text
reset() -> raw_observation

step(action) ->
    raw_observation,
    game_result / transition information,
    done,
    info
```

The existing environment remains responsible for:

- game state
- legal actions
- game transitions
- terminal conditions
- opponent interaction

The RL layer should be responsible for:

- observation processing
- reward construction
- PPO
- opponent policy selection
- rollout collection

Avoid putting PPO-specific logic into the environment.

---

## Phase 3 — Observation processing layer

Implement a dedicated observation-processing component.

Conceptually:

```text
environment
    ↓
raw observation
    ↓
ObservationProcessor
    ↓
processed observation
    ↓
actor / critic
```

The processor should be independently configurable.

For example:

```python
processed_obs = observation_processor(raw_obs)
```

It should be possible to experiment with different representations without modifying PPO or the environment.

Potential features may include:

- spatial board representation
- player-relative features
- army counts
- territory/control information
- distances
- income
- turn information
- normalized scalar features
- masks
- other engineered features

Keep the representation modular so that a future version could replace handcrafted features with a more direct board representation.

### JAX requirement

Prefer making the processing function a pure function:

```text
raw observation → processed observation
```

where possible.

The eventual goal is to make it compatible with:

- `jax.jit`
- `jax.vmap`

so observation processing can be performed efficiently across many parallel environments.

---

## Phase 4 — Reward layer

Keep reward calculation separate from the environment.

Implement a reward component that supports both:

### Terminal reward

For example:

```text
win  → +1
loss → -1
draw → 0
```

The exact values should be configurable.

### Potential-based shaping

Support:

```text
r = terminal_reward + α * (γ * Φ(s') - Φ(s))
```

where:

- `Φ(s)` is the potential function
- `α` controls shaping strength
- `γ` is the PPO discount factor

The potential function should be independently replaceable.

For example:

```python
potential(state)
```

The reward module should therefore expose something conceptually equivalent to:

```text
RewardFunction
    ├── terminal_reward
    └── potential_shaping
```

Do not mix reward computation into the PPO implementation.

---

## Phase 5 — Opponent abstraction

Create an opponent interface independent from PPO.

The training system should be able to play against:

```text
Opponent
├── scripted/random bot
├── current PPO policy
├── fixed checkpoint
└── future population/self-play opponents
```

Conceptually:

```python
opponent.act(observation) -> action
```

The PPO training configuration should determine which opponent is used.

Eventually support opponent sampling such as:

```text
current policy     70%
historical model   20%
scripted bot       10%
```

But start with a simple deterministic opponent implementation.

Do not hard-code opponent logic into the PPO algorithm.

---

## Phase 6 — PPO implementation

Use **JAX** for PPO.

Use an existing, minimal PPO implementation such as **CleanRL's JAX PPO implementation as the conceptual/reference starting point**, rather than blindly copying a large RL framework.

The PPO implementation should contain:

### Actor

Maps processed observations to an action distribution.

```text
observation → policy distribution → action
```

### Critic

Maps processed observations to:

```text
V(s)
```

### Rollout collection

Store at least:

```text
observation
action
reward
done
value
log_probability
```

### GAE

Calculate advantages from the rollout.

Support configurable:

```text
gamma
lambda
```

### PPO update

Implement:

- probability ratio
- clipped policy objective
- value loss
- entropy bonus
- minibatches
- multiple optimization epochs
- gradient clipping if desired

All major PPO hyperparameters should live in configuration rather than being hard-coded.

---

## Phase 7 — Network architecture

Start simple.

Use an actor and critic with either:

### Option A — shared encoder

```text
processed observation
        ↓
    shared NN
      ↙   ↘
 actor     critic
```

or:

### Option B — separate networks

```text
processed observation
      ↙       ↘
 actor       critic
```

Prefer shared representations initially unless the observation structure strongly suggests otherwise.

Make the architecture configurable.

Do not over-engineer the network before verifying that PPO training works.

---

## Phase 8 — Action representation

The existing Generals environment may have a large/discrete action space.

Explicitly determine:

- total action count
- how actions are encoded
- whether invalid actions exist
- whether action masking is required

If invalid actions exist, implement action masking at the policy-distribution level rather than allowing PPO to repeatedly sample impossible actions.

Keep action encoding separate from the neural network.

---

## Phase 9 — Rollout architecture

Implement parallel environments if the existing environment allows it.

Target architecture:

```text
             ┌── Env 1 ──┐
             ├── Env 2 ──┤
PPO policy ──┼── Env 3 ──┼── rollout buffer
             ├── ... ────┤
             └── Env N ──┘
```

Use JAX `vmap` where appropriate.

The goal is to make large batches of experience efficient.

However, first make **one environment work correctly**, then vectorize.

---

## Phase 10 — Training loop

The high-level training loop should be:

```text
initialize PPO
initialize opponent
initialize environments

while training:

    collect rollout

        observation
            ↓
        ObservationProcessor
            ↓
        PPO policy
            ↓
        action
            ↓
        Environment
            ↓
        raw observation + game transition
            ↓
        RewardFunction
            ↓
        rollout buffer

    compute GAE

    PPO update

    periodically:
        evaluate
        save checkpoint
        update opponent
```

The training loop should not contain game-specific feature engineering or reward logic.

---

## Phase 11 — Configuration

Create a central configuration system.

At minimum expose:

### PPO

```text
learning_rate
gamma
gae_lambda
clip_coef
entropy_coef
value_coef
num_epochs
minibatch_size
rollout_length
num_envs
max_grad_norm
```

### Reward

```text
terminal_win_reward
terminal_loss_reward
terminal_draw_reward
shaping_alpha
```

### Observation

```text
observation representation
feature toggles
normalization
```

### Opponent

```text
opponent type
checkpoint path
opponent sampling probabilities
```

### Training

```text
seed
total timesteps
checkpoint frequency
evaluation frequency
```

Do not scatter these values throughout the codebase.

---

## Phase 12 — Checkpointing

Checkpoint:

- actor parameters
- critic parameters
- optimizer state
- training step
- configuration
- RNG state if necessary
- opponent state/configuration if necessary

Support:

```text
train from scratch
resume training
load policy for evaluation
```

Ensure that resuming a checkpoint restores the optimizer state rather than only the neural-network weights.

---

## Phase 13 — Evaluation

Implement evaluation separately from training.

Evaluation should:

- disable exploration where appropriate
- play a configurable number of games
- report win/loss/draw
- optionally report average reward
- optionally report episode length
- optionally evaluate against multiple opponent types

Example:

```text
Evaluation
----------
Games: 1000
Wins:  XXX
Losses: XXX
Draws:  XXX
Win rate: XX.X%
```

Training metrics and evaluation metrics should not be conflated.

---

## Phase 14 — Validation milestones

Do not immediately launch a massive training run.

Validate incrementally.

### Milestone 1

Environment adapter works.

```text
reset → observation
step → transition
```

### Milestone 2

Observation processor produces valid tensors.

Check:

- shapes
- dtypes
- NaNs
- normalization
- batch dimensions

### Milestone 3

Actor produces valid actions.

### Milestone 4

Critic produces finite values.

### Milestone 5

One complete rollout works.

### Milestone 6

GAE produces finite advantages.

### Milestone 7

One PPO update works.

### Milestone 8

Run PPO against a simple/static opponent.

### Milestone 9

Verify that the agent learns something measurable.

### Milestone 10

Enable parallel environments and JAX vectorization.

### Milestone 11

Introduce self-play/checkpoint opponents.

---

## Important architectural constraint

Keep these components independent:

```text
                 ┌─────────────────────┐
                 │    Environment      │
                 └──────────┬──────────┘
                            │
                     raw observation
                            │
                            ▼
                 ┌─────────────────────┐
                 │ ObservationProcessor│
                 └──────────┬──────────┘
                            │
                   processed observation
                            │
                            ▼
                 ┌─────────────────────┐
                 │        PPO          │
                 │                     │
                 │ Actor + Critic      │
                 │ GAE                 │
                 │ PPO optimization    │
                 └──────────┬──────────┘
                            │
                          action
                            │
                            ▼
                 ┌─────────────────────┐
                 │    Environment      │
                 └─────────────────────┘

Reward calculation should be a separate component.

Opponent selection should be a separate component.
```

The key objective is that I should be able to change:

```text
observation representation
reward shaping
opponent
network architecture
PPO hyperparameters
```

**independently**, without rewriting the rest of the system.

---

## Implementation priority

Implement in this order:

1. Understand existing repository
2. Environment adapter
3. ObservationProcessor
4. RewardFunction
5. Simple opponent interface
6. JAX actor/critic
7. Rollout buffer
8. GAE
9. PPO update
10. Single-environment training
11. Checkpointing
12. Evaluation
13. Parallel environments / `vmap`
14. Self-play opponent management

At every stage, keep the implementation minimal and testable.

Do not optimize prematurely.
Do not rewrite the existing environment.
Do not introduce a heavyweight RL framework unless there is a concrete need for it.