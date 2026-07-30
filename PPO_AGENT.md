# PPO Agent — Design Document

How to train a PPO agent for the Generals Competition ruleset. Complements
[ARCHITECTURE.md](ARCHITECTURE.md), which describes the engine; this file is about the learning
system. The two sections that matter most — and that decide whether this works at all — are
[Observations](#3-observations) and [Rewards](#4-rewards).

> **Implemented.** This design is built out under [training/](training/) — see
> [training/README.md](training/README.md) for how to run it. 79 tests pass, including an exact
> mask-vs-engine equivalence check, the PBRS telescoping identity, and JAX↔NumPy parity for the
> submission bot. Measured: 651,871 params, **6.8 ms per turn** in pure NumPy against the 150 ms
> budget, 0 faults over a 2400-turn match. Two things changed during implementation and are
> reflected below: truncation is treated as a genuine terminal ([§4.6](#46-the-auto-reset-trap)),
> and the curriculum varies map size only ([§4.5](#45-annealing-and-curriculum)).

---

## 0. Executive summary

| Decision | Choice |
|---|---|
| PPO implementation | **Vendored from [PureJaxRL](https://github.com/luchris429/purejaxrl)**, not Unity ML-Agents — see [§1](#1-which-ppo-framework) |
| Observation | **~32 spatial planes @ 22×22 + ~20 global scalars**, uint8-quantized in the rollout buffer |
| Reward | **Terminal ±1 + potential-based shaping** on an *income-denominated* potential, annealed to 0 |
| Action space | Flat masked categorical over `22·22·9 + 1 = 4357` |
| Network | Fully-convolutional resnet, ~650K params, size-agnostic (global pooling, no flatten) |
| Discount | **γ = 0.999** (1200-turn episodes; γ=0.99 cannot span them) |
| Opponent | Self-play against a **checkpoint league** + scripted bots |
| Deployment | Export weights to `.npz`, **pure-NumPy forward pass** in the submitted bot |

The three things most likely to sink this project, in order: the reward being sparse over
1200-turn episodes, fog requiring explicit memory, and the 150 ms inference budget at
submission time. Each gets its own section.

---

## 1. Which PPO framework

You asked for Unity ML-Agents. I'd advise against it here, and the reason is specific rather
than stylistic.

**ML-Agents is the wrong shape for this repo.** Its trainer (`mlagents-learn`) is built to
drive a Unity binary over gRPC — `UnityEnvironment` is the first-class (effectively only)
env interface, and the object model it assumes (Behaviors, `DecisionRequester`, side channels)
has no analogue here. Its PPO is PyTorch. This engine is pure JAX and `vmap`-able to thousands
of concurrent games entirely on-device; routing every step through a gRPC hop to a PyTorch
trainer would mean a device→host→device round-trip per step and would throw away the ~10⁶
steps/sec throughput that is the entire reason the engine was written this way. You'd be
adopting a framework to avoid writing PPO, and paying for it with a 100–1000× throughput loss
plus an adapter layer larger than PPO itself.

The good news is that your actual goal — *don't hand-write PPO, take a proven implementation* —
is better served by a JAX-native one.

### Recommendation: vendor PureJaxRL's `ppo.py`

[PureJaxRL](https://github.com/luchris429/purejaxrl) is a single-file, end-to-end-jittable PPO
(Flax + optax + distrax). The entire train loop — rollout, GAE, minibatch epochs — compiles
into one `jax.jit`, so nothing leaves the device. It expects a gymnax-style env interface:

```
env.reset(key, params)               -> (obs, state)
env.step(key, state, action, params) -> (obs, state, reward, done, info)
```

`GeneralsEnv` is already 90% of the way there; the adapter is ~60 lines (see [§6](#6-code-layout)).
Vendor the file into `training/ppo_core.py` and edit it directly rather than importing it —
you need to change action masking, the two-player step, and truncation bootstrapping anyway,
and it's a single file designed to be forked.

### Alternatives, ranked

| Framework | Backend | Verdict |
|---|---|---|
| **PureJaxRL** | JAX | **Recommended.** Single file, fully jittable, trivial to fork. Minimal, so you own the changes. |
| [rejax](https://github.com/keraJLi/rejax) | JAX | Library-ified PureJaxRL — more algorithms, checkpointing, cleaner config. Take this if you want batteries over hackability. |
| [Stoix](https://github.com/EdanToledo/Stoix) | JAX | Best engineering of the three (Anakin/podracer layouts, multi-device). Heavier abstraction; good if you scale to multi-GPU. |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | PyTorch | Excellent single-file reference and the best-documented PPO anywhere. Use it to *read* the algorithm; using it to run means host round-trips. |
| Stable-Baselines3 | PyTorch | Most batteries-included, worst fit. Gym vector-env API forces the env off-device. Masking needs `sb3-contrib`'s `MaskablePPO`. |
| **Unity ML-Agents** | PyTorch/gRPC | Not recommended, per above. |

If you want PyTorch for ecosystem reasons despite the cost, take CleanRL's `ppo.py` over SB3 —
you'll need to modify masking and self-play anyway, and CleanRL is far easier to modify.

> I can't verify these repos' current APIs from here. Check the exact `make_train` / config
> signature against upstream when you vendor; the shapes described above are the stable parts.

---

## 2. What's already in the repo, and what to do with it

`examples/_experimental/ppo/` is a 4×4-grid prototype. Treat it as a sketch to read, not a base
to build on. Concretely:

- [train.py](examples/_experimental/ppo/train.py) drives `game.step` **directly**, so neither
  `build_castles` nor `deathtouch` applies — it trains on a different game than the competition
  runs. (Same class of bug as `matchup.py`, documented in ARCHITECTURE.md's *Traps*.)
- [train2.py](examples/_experimental/ppo/train2.py) calls `jax.vmap(env.step)(states, actions)`
  — the old two-argument signature. `env.step` now takes `pool` as a third argument, so this
  is broken as written.
- [train2.py:71](examples/_experimental/ppo/train2.py#L71) computes the shaped reward from
  `timesteps.observation`. **That is the post-auto-reset observation.** On any step where the
  episode ended, the "next obs" is the first frame of a *fresh game*, so the reward for that
  transition is garbage. See [§4.6](#46-the-auto-reset-trap) — this one is easy to reproduce
  in the new code and worth a test.
- [network.py:51](examples/_experimental/ppo/network.py#L51) has
  `Linear(grid_size * grid_size * 4, 64)` in the value head — hardcodes the board size and
  blocks any size curriculum. Replace with global pooling.
- `network.py` uses 9 input channels and `obs.cities` (deprecated alias), and has no build
  action in its 9-way policy head — so it cannot express the competition ruleset's central
  strategic move.

What *is* reusable:

- [rewards.py](generals/core/rewards.py) — the functions are all "difference of features between
  consecutive observations", which is potential-based shaping with γ=1. Right instinct; see
  [§4.2](#42-potential-based-shaping) for the correction.
- [compute_valid_move_mask](generals/core/action.py#L29) — exactly the move mask the policy head
  needs, already `(H, W, 4)` and jitted.
- [build_cost_grid](generals/modifiers/build_castles.py#L45) — gives the build mask *and* one of
  the best observation channels ([§3.4](#34-build-economics-channels)).
- `HunterAgent`'s BFS-by-`fori_loop`-dilation is the pattern for the distance channels in
  [§3.5](#35-derived-geometry-channels), and the scripted bots are the evaluation panel.

---

## 3. Observations

The single highest-leverage part of this design. `Observation.as_tensor()` gives 14 planes and
is a reasonable *debug* view, but it is not a good network input: army counts are unnormalized
(they reach the thousands), five of the fourteen planes are constant broadcasts wasting
convolution, there is no memory across the fog, and nothing encodes the padding, the build
economy, or the turn-800 cliff.

Proposal: a `features.py` module producing `(C, 22, 22)` float planes plus a `(G,)` global
vector, built from `Observation` + a small persistent belief state.

### 3.1 Normalization

Everything must land roughly in `[-1, 1]`.

- Armies: `sign(x) · log1p(|x|) / 7`. `log1p(1000)/7 ≈ 0.99`, so the scale covers realistic
  late-game stacks without saturating early-game single units into indistinguishability.
- Counts (land, army totals): same log scaling.
- Distances: `d / 22`, with unreachable encoded as `1.0` **plus a separate reachability plane**
  (don't overload one channel with a sentinel).
- Booleans: `{0, 1}`.

Never feed raw army counts. A conv net given inputs spanning 0–3000 will be dominated by the
largest stack on the board for the first several million steps.

### 3.2 Direct-observation planes (13)

Straight from `Observation`, minus the broadcast scalars:

| # | Plane | Notes |
|---|---|---|
| 0 | `log_armies` own | armies masked to own cells, log-scaled |
| 1 | `log_armies` enemy | visible enemy armies only |
| 2 | `log_armies` neutral | visible neutral armies |
| 3 | `owned_cells` | |
| 4 | `opponent_cells` | visible only |
| 5 | `neutral_cells` | visible only |
| 6 | `mountains` | visible |
| 7 | `castles` | visible, either owner |
| 8 | `castles & owned` | own castles — the income engine |
| 9 | `castles & opponent` | |
| 10 | `generals` | visible generals (own always, enemy only if seen) |
| 11 | `fog_cells` | |
| 12 | `structures_in_fog` | mountain-or-castle in fog; the game never disambiguates these |

Splitting armies into own/enemy/neutral planes (rather than one army plane + ownership planes)
costs two channels and saves the first conv layer from having to learn the multiplication.

### 3.3 Padding and playable area (2)

Maps are 18–21 per side, always padded to 22×22 with mountains anchored top-left
(ARCHITECTURE.md, *Padding*). Give the net this explicitly — it's free and it stops the network
from spending capacity on a region where every action is a no-op:

| # | Plane |
|---|---|
| 13 | `in_bounds` — 1 inside the real `H×W`, 0 in the padding |
| 14 | `border_distance` — normalized distance to the nearest real-board edge |

`in_bounds` is inferable from the mountain border, but only by a network with a 22-cell
receptive field. Handing it over costs one channel.

### 3.4 Build-economics channels (3)

Under the competition ruleset the build action is the strategic core, and — this is the
convenient part — **the build price is exactly computable client-side**. The price depends only
on *your own* structures ([build_cost_grid](generals/modifiers/build_castles.py#L45)), and you
always see your own cells. No fog problem.

| # | Plane |
|---|---|
| 15 | `build_cost` — normalized `build_cost_grid(state, me) / 64` |
| 16 | `build_affordable` — `armies >= cost & owned & ~general & ~castle` (identical to the action mask) |
| 17 | `build_surplus` — `log1p(max(0, armies − cost)) / 7`; how much army survives the build |

`build_surplus` matters because building with *exactly* the cost leaves a 0-army castle that any
single enemy unit captures — the network needs to see that cliff, not infer it from two channels.

### 3.5 Derived geometry channels (5)

Cheap BFS products. Conv nets approximate distance transforms badly at this radius, and these
carry most of the tactical structure:

| # | Plane |
|---|---|
| 18 | `dist_to_own_general` |
| 19 | `dist_to_enemy_general` — to the last known position; zeroed and flagged if never seen |
| 20 | `dist_to_own_army_mass` — BFS from cells with army > threshold |
| 21 | `frontier` — own cell orthogonally adjacent to a passable non-own cell |
| 22 | `reachable` — connected to own territory through passable cells |

Reuse the `fori_loop` dilation from `HunterAgent`; a 22×22 BFS is ~22 iterations of a 4-neighbour
max, trivially jittable and `vmap`-able.

### 3.6 Belief / memory channels (6) — the fog solution

**This is what makes or breaks a fog-of-war agent.** A feed-forward net on the current frame
alone is amnesiac: it re-explores territory it cleared 40 turns ago, and it forgets where the
enemy general is the moment it steps out of vision.

Three options: frame stacking (weak — 4 frames buys 4 turns of memory in a 1200-turn game), an
RNN (expensive, and makes the net learn map memory from scratch), or **explicit belief channels
maintained by a wrapper**. Recommend the third. It's a handful of `jnp.where` updates per step,
fully jittable, and it hands the network what an RNN would spend tens of millions of steps
learning to approximate.

Carry a `Belief` NamedTuple alongside the env state, updated each step from the current
observation:

| # | Plane | Update rule |
|---|---|---|
| 23 | `ever_seen` | `ever_seen \| visible` |
| 24 | `staleness` | `visible ? 0 : min(staleness+1, 200) / 200` — how out-of-date this cell is |
| 25 | `last_known_enemy` | enemy ownership at last sighting, persists through fog |
| 26 | `last_known_army` | log-scaled army at last sighting |
| 27 | `enemy_general_prior` | see below |
| 28 | `last_known_enemy_castle` | enemy castles persist once seen — they don't move |

`enemy_general_prior` is a normalized distribution over where the enemy general can be. Initialize
uniform over cells at least `min_generals_distance` from your own general, then zero out every
cell you've ever seen that didn't contain it. It collapses to a point mass on sighting. This is
cheap Bayesian bookkeeping that a network would struggle to learn and that directly drives the
endgame.

Reset the belief on `done` — and note this interacts with auto-reset ([§4.6](#46-the-auto-reset-trap)).

### 3.7 Turn-phase channels (3)

The competition ruleset has three hard temporal structures, and the current observation exposes
only a raw `timestep` integer:

| # | Plane (broadcast constants) |
|---|---|
| 29 | `time / 1200` — game progress |
| 30 | `time % 2` — structure growth parity; decides whether a capture lands before or after +1 |
| 31 | `deathtouch_proximity` — `clip((time − 700) / 100, 0, 1)`, saturating at 1 from turn 800 |

Broadcasting three scalars as planes is slightly wasteful but lets every convolution condition on
phase locally, which matters for `time % 2` tactics. The rest of the scalars go in the global
vector.

`deathtouch_proximity` deserves emphasis: at turn 800 the game changes character completely —
before it, your general is protected by army count; after it, by position and tempo only. A ramp
starting at 700 gives the policy 100 turns to reposition rather than discovering a discontinuity.

### 3.8 Global scalar vector (~20)

Fed via concatenation into the value head and injected into the trunk (broadcast-add after the
stem, or FiLM-style per-block modulation):

- `log` own/enemy army and land totals, and their differences
- own/enemy **income** (see [§4.3](#43-the-potential-function)) and income difference
- own castle count; estimated enemy castle count
- **`unaccounted_enemy_army` = `opponent_army_count − Σ visible enemy armies`**
- `explored_fraction`, `visible_fraction`
- `time`, `time % 50` (land-growth phase), `turns_to_deathtouch`
- `is_player_0` — P0 wins move-order ties, so the seat is genuinely part of the state

`unaccounted_enemy_army` is the sharpest feature in the whole set. Opponent totals are *never*
fogged ([get_observation](generals/core/game.py#L143) leaves them as true globals), so a large
total with no visible presence means a stack is in motion somewhere. ARCHITECTURE.md notes the
reference bots ignore this entirely. It's a free edge.

### 3.9 Budget

**32 planes + 20 scalars.** Start here, then ablate — several of these (the geometry group
especially) are candidates for removal once the belief channels are working.

### 3.10 Rollout buffer memory — a real constraint

`32 × 22 × 22 × 4 bytes = 62 KB` per observation. At 512 envs × 256 steps that's **8 GB** of
observations per iteration, before advantages, log-probs, and masks.

Mitigations, in order of preference:

1. **Store as `uint8`** (quantize to 256 levels; every channel is already in `[-1,1]`), dequantize
   in the loss pass. 8 GB → 2 GB. The precision loss is irrelevant for conv inputs.
2. Store the compact `Observation` + `Belief` and **recompute derived planes** in the update pass.
   Trades FLOPs for memory; the geometry BFS is the expensive part.
3. Shrink the rollout: 256 envs × 128 steps = 32K transitions still gives PPO plenty per update.

Do (1) unconditionally. It's ~20 lines and buys 4×.

---

## 4. Rewards

### 4.1 The problem

`env.step` returns bare ±1 on general capture. Over a 1200-turn episode that is **one bit of
signal per ~1200 actions**, and the action space is 4357-wide. Worse, two untrained policies
essentially never decapitate each other — they truncate to a draw, so the reward is *identically
zero* for the entire early phase of training. There is nothing for PPO to bootstrap from.

Shaping is not optional here. The question is how to shape without corrupting the objective.

### 4.2 Potential-based shaping

Use **potential-based reward shaping** (Ng, Harada & Russell 1999): for a potential function
`Φ(s)`, the shaping term

```
F(s, s') = γ · Φ(s') − Φ(s)
```

provably leaves the optimal policy unchanged. That's the property you want — it lets you shape
aggressively and specifically without the agent learning to farm the shaping instead of winning.

Two corrections to the existing [rewards.py](generals/core/rewards.py):

1. Its terms are `Φ(s') − Φ(s)` with **no γ**. At γ=0.999 over 1200 steps the discrepancy
   compounds; include the γ.
2. `win_lose_reward_fn` adds `split_bonus = 0.0015` when `action[4] == 1`
   ([rewards.py:112](generals/core/rewards.py#L112)). That is a function of the *action*, not a
   potential difference, so it is **not** policy-invariant — it pays the agent 0.0015 per split
   move, up to ~1.8 over a full game, i.e. comparable to winning. Drop it. If splitting is good,
   the potential should say so.

Total per-step reward:

```
r_t = terminal_t + α · (γ·Φ(s_{t+1}) − Φ(s_t))
```

with `terminal_t` = ±1 on capture, 0 on draw, and `α` annealed 1.0 → 0.0 over training
([§4.5](#45-annealing-and-curriculum)).

### 4.3 The potential function

Denominate the potential in **income**, because the game's economy is the strategy. From
[global_update](generals/core/game.py#L260): structures grow +1 every 2 ticks, land +1 every 50
ticks. So per-tick income is

```
income(p) = 0.5 · structures(p) + 0.02 · land(p)
```

A castle is worth **25 land tiles** of income — that ratio *is* the game, and expressing the
potential in these units means the agent's economic incentives are correct by construction rather
than by weight-tuning.

```
Φ(s) = w_econ   · tanh( (income_me − income_opp) / 4 )
     + w_mat    · tanh( (log1p(army_me) − log1p(army_opp)) / 2 )
     + w_terr   · tanh( (land_me − land_opp) / 60 )
     + w_safe   · safety(s)
     + w_threat · threat(s)
     + w_explore· explored_fraction(s)
```

- `safety(s)` — normalized `min(dist(enemy known army → my general), 12) / 12`, multiplied by a
  ramp that switches on near turn 800. Before deathtouch, army count defends you; after it, only
  distance does.
- `threat(s)` — the mirror: `1 − min(dist(my army mass → enemy general), 12) / 12`, gated on the
  enemy general being known. Without this term a pure-economy potential produces a bot that
  out-farms the opponent and then draws at turn 1200, which scores half a win.
- `explored_fraction` — small, and annealed out first. Fog makes exploration a genuine
  chicken-and-egg problem early; this is the cheapest fix.

Every `tanh` is there deliberately: it bounds each term, so no single component can dominate, and
it flattens the gradient once you're far ahead — the agent stops being paid for running up the
score in a won position and starts caring about actually closing it out.

Suggested starting weights: `w_econ=1.0, w_mat=0.4, w_terr=0.3, w_safe=0.5, w_threat=0.4,
w_explore=0.2`. These are guesses; the ablation in [§8](#8-evaluation) is how you find the real
ones.

### 4.4 Failure modes to design against

| Failure | Cause | Fix |
|---|---|---|
| **Turtling** — sits on general hoarding army | Rewarding raw army | Income formulation: hoarding has near-zero income. `w_terr` and `w_threat` also push out. |
| **Draw-seeking** | Draw = 0 > loss = −1, so a losing agent prefers stalling | `w_threat` + γ<1 makes earlier wins strictly better. Do **not** add a per-step survival bonus. |
| **Build-thrash** — builds, loses castle, rebuilds | Rewarding the build *action* | Potential is over castle *count*, so losing one immediately refunds the gain. Never reward the action. |
| **Suicidal rushing** | `w_threat` too high | `tanh` bounds it; keep `w_threat < w_econ`. |
| **Shaping farmed over winning** | Non-potential terms | Strict PBRS + anneal α → 0. |

The general rule: **reward states, never actions.** Every non-potential term in the existing
`rewards.py` is a bug waiting to happen.

### 4.5 Annealing and curriculum

- **α: 1.0 → 0.0** over the first ~60% of training. The final policy must optimize the true
  win/loss objective; shaping is scaffolding.
- **`w_explore` → 0 first** (~20% of training). It's the most distortionary term.
- **Map-size curriculum.** Start at `grid_dims=(10,10)`. Smaller boards put the generals closer
  together, so games end by decapitation instead of timing out — that's what makes the terminal
  signal dense. Graduate to the competition preset (18–21, padded 22). This requires the network to
  be **size-agnostic** — global pooling in the value head, no flatten. Design for it from day one;
  retrofitting means retraining.
- **Keep `truncation` at 1200 throughout** — *not* a curriculum knob. Reaching turn 1200 is a
  genuine draw under the competition ruleset, so it is a real terminal outcome, not an artificial
  cutoff. Shortening it would teach the agent that stalling to an early cutoff is neutral. Using
  map size as the only curriculum axis gets the same density benefit without corrupting the
  objective, and it removes the need for truncation bootstrapping entirely
  (see [§4.6](#46-the-auto-reset-trap)).

### 4.6 The auto-reset trap

`env.step` auto-resets from the pool, and **`timestep.observation` is computed from the
post-reset state** ([env.py:322-328](generals/core/env.py#L322)). On any step where the episode
ended, `timestep.observation` is the first frame of a *new game*.

This breaks three things if unhandled:

1. **Shaped reward** — `Φ(s')` is evaluated on an unrelated fresh board. Random garbage, at
   exactly the transitions that matter most. This is the live bug in
   [train2.py:71](examples/_experimental/ppo/train2.py#L71).
2. **GAE bootstrapping** — the value of the next state is the value of a new game.
3. **Belief state** — must reset in lockstep with the env, or the new episode inherits the old
   one's memory.

`timestep.last_state` is the pre-reset state and is provided for exactly this reason. So:

```python
obs_next = get_observation(timestep.last_state, me)
shaped = jnp.where(done, 0.0, gamma * phi(obs_next) - phi(obs_prev))
reward = terminal + alpha * shaped
belief = jax.tree.map(lambda b: jnp.where(done, 0, b), belief)
```

**Truncation vs termination.** The general rule is that these need different bootstrapping —
`terminated` → `next_value = 0`, `truncated` → `next_value = V(last_state)`, because a game cut
off by a time limit was not actually lost.

**But under this ruleset truncation at 1200 is a real terminal state**: the competition scores it
as a draw. So both flags are genuine episode ends, `next_value = 0` is correct for both, and
`Φ(s_T) = 0` applies to both. This is only valid because the curriculum never shortens
`truncation` ([§4.5](#45-annealing-and-curriculum)) — if you ever do shorten it for training
convenience, you must restore proper truncation bootstrapping or the agent learns that stalling to
the cutoff is free.

With `Φ(terminal) = 0`, the discounted shaping over a whole episode telescopes to exactly
`−Φ(s_0)` — a constant independent of the policy, which is the policy-invariance guarantee. That
identity is worth asserting directly in a test; `tests/test_training_potential.py` does.

### 4.7 Asymmetric actor-critic

Worth doing, and cheap: give the **critic the unfogged state** (`get_full_observation`) while the
**actor sees only the fogged observation**. Training-time privileged information is legitimate —
the critic is discarded at deployment — and it substantially reduces value-function variance in
partially observable games, because the critic no longer has to guess at hidden state to explain
outcomes.

The same applies to the potential: `Φ` may use true state. It only shapes the reward, and the
policy never sees it.

---

## 5. Action space and network

### 5.1 Action space

Flat masked categorical over **`H·W·9 + 1 = 4357`**:

- indices `0 .. H·W·4−1` — move from cell, direction `d`, `split=0`
- indices `H·W·4 .. H·W·8−1` — same, `split=1`
- indices `H·W·8 .. H·W·9−1` — **build** at cell (`pass=2`)
- index `4356` — pass

Decode to the engine's `[pass, row, col, direction, split]`. Alternatives (autoregressive
cell-then-direction) are more parameter-efficient but need sequential sampling and complicate the
PPO log-prob; flat-and-masked is simpler and PPO handles 4357 categories comfortably.

**Masking** (add `−1e9` to invalid logits before the softmax — never sample-and-reject, it
destroys the log-prob):

- moves: `compute_valid_move_mask(armies, owned_cells, mountains)` → `(H,W,4)`, tiled across both
  split variants (both require `army > 1`, so the mask is identical)
- builds: `owned & ~generals & ~castles & (armies >= build_cost_grid)`
- pass: always valid

Masking is not just an efficiency measure. Unmasked, the vast majority of the 4357 actions are
no-ops at any given state, and the policy spends its early training learning legality instead of
strategy.

### 5.2 Network

```
input (32, 22, 22)  +  globals (20,)
  ├─ stem: conv3x3 → 64, + broadcast Linear(globals → 64)
  ├─ 8 × residual block [conv3x3 64 → GN → ReLU → conv3x3 64 → GN → +skip → ReLU]
  ├─ policy head: conv1x1 → 9 planes → flatten (4356)
  │                + pass logit from pooled features        → concat → 4357 → mask → softmax
  └─ value head:  global mean-pool ‖ max-pool (128) ‖ globals (20) → MLP 256 → 1
```

~650K parameters, ~2.6 MB in float32. Notes:

- **No flatten anywhere except the policy head**, which is spatial by construction. This is what
  makes the size curriculum possible.
- GroupNorm over BatchNorm — no running statistics to sync, and it behaves identically at
  train and inference time.
- Don't bound the value output with `tanh`. With shaping, returns exceed ±1. Use return
  normalization instead.
- The policy head being a 1×1 conv means it's translation-equivariant: "move this stack right" is
  the same computation everywhere on the board, which is exactly the symmetry the game has.

### 5.3 Hyperparameters

| Param | Value | Why |
|---|---|---|
| `γ` | **0.999** | 1200-turn episodes. γ=0.99 has a ~100-step horizon and cannot connect an early castle to a turn-900 win. |
| `λ` | 0.95 | |
| clip | 0.2 | |
| lr | 3e-4, cosine → 0 | |
| envs | 512–2048 | Tune to VRAM against [§3.10](#310-rollout-buffer-memory--a-real-constraint) |
| rollout | 128–256 | |
| epochs | 4 | |
| minibatches | 8 | |
| entropy coef | 0.01 → 0.001 | Anneal; 4357 actions need early exploration pressure |
| value coef | 0.5 | |
| max grad norm | 0.5 | |

γ is the one to get right. It interacts with the value function's scale — at γ=0.999 returns
accumulate over ~1000 steps, so **normalize returns** (running mean/std, or PopArt) or the value
loss will swamp the policy loss.

---

## 6. Code layout

Keep everything out of `competition/`, which stays exactly as upstream ships it.

```
training/
  features.py       Observation → (32,22,22) planes + (20,) globals; Belief update
  potential.py      Φ(s), the shaping term, α annealing
  masking.py        move + build + pass masks → (4357,) bool
  actions.py        flat index ↔ [pass,row,col,dir,split]
  network.py        Flax resnet, policy + value heads
  selfplay.py       opponent pool, sampling, checkpoint league
  env_adapter.py    GeneralsEnv → gymnax-style API for PureJaxRL
  ppo_core.py       vendored PureJaxRL ppo.py, modified
  train.py          entry point
  export.py         Flax params → .npz for the submission bot
  eval.py           win-rate vs the scripted panel + past checkpoints

competition/agents/ppo_bot/
  run.sh
  main.py           stdio loop (copy from expander_python, unmodified)
  agent.py          NumPy forward pass + belief tracking
  weights.npz

tests/
  test_features.py    channel shapes, ranges, belief reset on done
  test_potential.py   PBRS identity: Σ shaping telescopes to γ^T Φ(s_T) − Φ(s_0)
  test_masking.py     every masked-in action is a legal engine action, and vice versa
  test_autoreset.py   reward and belief are correct across episode boundaries
```

`test_masking.py` is the highest-value test: enumerate all 4357 actions on random states, apply
each through `env.step`, and assert that mask ⟺ state actually changed. A masking bug is silent,
looks like slow learning, and can cost weeks.

---

## 7. Self-play

Naive self-play (current policy vs current policy) cycles — strategies chase each other around a
rock-paper-scissors loop without net improvement. Use a league:

- **40%** current policy (fresh, sharp)
- **40%** uniform over past checkpoints, saved every N iterations (prevents forgetting)
- **20%** scripted bots — `HunterAgent`, `ExpanderAgent`, `RandomAgent` (anchors absolute skill;
  without these, a league can drift to being strong only against itself)

Only the learner's trajectories go into the buffer; the opponent's parameters are frozen during
rollout. Collect from **both seats equally** — P0 wins move-order ties
([_determine_move_order](generals/core/game.py#L291)), so the seats are not symmetric and a
P0-only agent will be measurably worse as P1.

Once basic play works, consider prioritized fictitious self-play (sample opponents in proportion
to how often they beat you) — it concentrates compute on the checkpoints you actually struggle
against.

---

## 8. Evaluation

Fixed panel, run every N iterations, 200 games each with seats swapped:

| Opponent | Meaning |
|---|---|
| `RandomAgent` | sanity; should be ~100% within the first hour |
| `ExpanderAgent` | basic economic competence |
| `HunterAgent` | the real bar — it garrisons, pushes a stack, and decapitates |
| checkpoint −10, −50 | monotonic improvement check |

Track separately: **win / loss / draw**, not just win rate. A rising draw rate is the
characteristic signature of a shaping bug — the agent has learned to accumulate potential and
stall. Also log mean game length, castles built per game, and the fraction of games decided
before turn 800.

Validate the final agent with [tools/localmatch.py](tools/localmatch.py), never `matchup.py` —
per ARCHITECTURE.md's *Traps*, `matchup.py` bypasses `env.step` and therefore applies neither
modifier, so a `--mode competition` match there is a castle-less, deathtouch-less, untimed game.

**Ablations worth running**, since the observation set above is a hypothesis: belief channels off;
geometry channels off; `unaccounted_enemy_army` off; shaping off; γ=0.99 vs 0.999.

---

## 9. Deployment — the 150 ms problem

Easy to defer and expensive to get wrong. The competition runs your bot as a subprocess over
stdio with **150 ms per turn**, a **10 s budget for the first move**, and **50 faults or one crash
forfeits the game**.

**Don't ship JAX.** Import plus JIT compilation costs seconds and the first-call latency is
unpredictable across machines. Instead:

1. `export.py` writes the trained weights to `weights.npz`.
2. The submitted `agent.py` implements the forward pass in **pure NumPy** (im2col + GEMM for the
   convs). Deterministic, no compilation, no GPU assumption, ~2.6 MB of weights in the zip.
3. Budget: 8 residual blocks × 2 convs × (3·3·64·64·484) ≈ 430 MFLOPs. At typical single-threaded
   BLAS throughput that's tens of milliseconds — comfortable, but **measure it on the first day
   you have a network, not the week before submission**. If it's tight, ONNX Runtime is the
   fallback; if it's very tight, drop to 6 blocks × 48 channels.
4. Use the 10 s first-move budget to warm up: import, load weights, run one dummy forward pass.
5. Sample with `argmax`, not from the distribution — no reason to add variance at inference.

**One deliberate exception to the no-safety-checks rule.** In the local tooling, letting things
break loudly is right. In the submitted bot it is not: an uncaught exception forfeits the game
outright. Wrap the inference call so that any failure falls back to a legal move (or a pass),
and keep the belief update inside the same guard. This is the rules' explicit design constraint —
"robustness beats cleverness at the margin" — not defensive habit. Flagging it because it cuts
against the style you set for the rest of the repo; your call.

---

## 10. Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 1 | `features.py`, `masking.py`, `actions.py` + tests | Masks provably match engine legality |
| 2 | `env_adapter.py` + vendored `ppo_core.py`; 10×10, no shaping, vs `RandomAgent` | Beats Random > 90% |
| 3 | `potential.py`, annealing; still 10×10 | Beats `ExpanderAgent` > 70% |
| 4 | Belief channels; fog turned on properly | Measurable ablation win vs no-belief |
| 5 | Full competition preset, size + truncation curriculum | Beats `HunterAgent` > 60% |
| 6 | League self-play | Monotonic Elo vs checkpoint panel |
| 7 | `export.py` + NumPy bot + `localmatch.py` validation | < 150 ms measured, 0 faults over 50 games |

Phase 7's latency measurement should be pulled forward to the end of Phase 2 as a spike — it
constrains the network size, and network size constrains everything else.

---

## 11. Open questions

- **Draw value.** Terminal is win=+1 / draw=0 / loss=−1, which is right if the league scores a
  draw as half a win. Worth confirming against the competition scoring before tuning the endgame.
- **RNN.** If the belief channels prove insufficient, a small GRU over the pooled trunk features
  is the next step — but try belief channels first; they're strictly cheaper and more debuggable.
- **Build-placement quality.** The proximity surcharge (`35 + Σ max(0, 10−2d)`) makes placement a
  real optimization. If the learned policy clusters castles badly, a placement-specific potential
  term is the lever.
- **Opponent modelling.** `unaccounted_enemy_army` gives a scalar; a learned spatial prior over
  *where* that army is would be strictly better, and could be trained as an auxiliary
  prediction head against the true state (which training has access to).
