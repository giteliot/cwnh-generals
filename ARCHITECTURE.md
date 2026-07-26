# ARCHITECTURE

Orientation notes for working in this repo. Complements [README.md](README.md), which is
user-facing; this file is about how the pieces fit and where the sharp edges are.

## What this repo is

Two things sharing one engine:

1. **A JAX simulator of [generals.io](https://generals.io)** for RL research — pure-functional,
   JIT-compiled, `vmap`-able to thousands of concurrent games.
2. **The reference implementation of the Generals Competition ruleset** — one preset
   (`mode="competition"`) pins the format, and [competition/](competition/) holds a
   language-agnostic stdio protocol plus starter bots in Python/C++/Rust.

Everything is 2-player. Player indices are `0` and `1` throughout the engine; the stdio
protocol re-frames them as *me* / *opp* so bots never care which side they are.

## Repo map

| Path | Role |
|---|---|
| [generals/core/game.py](generals/core/game.py) | The engine. `GameState`, `step`, move resolution, army growth, fog. |
| [generals/core/env.py](generals/core/env.py) | `GeneralsEnv` — RL wrapper, rule presets, auto-reset pool. |
| [generals/core/grid.py](generals/core/grid.py) | Map generation (JIT, guaranteed-valid). |
| [generals/core/observation.py](generals/core/observation.py) | `Observation` NamedTuple + `as_tensor()` for networks. |
| [generals/core/action.py](generals/core/action.py) | Action format, `compute_valid_move_mask`. |
| [generals/core/rewards.py](generals/core/rewards.py) | Shaped reward functions (RL only; competition ignores these). |
| [generals/modifiers/](generals/modifiers/) | Optional rules layered *over* the base game: `build_castles`, `deathtouch`. |
| [generals/agents/](generals/agents/) | In-process JAX agents: Random, Expander, Hunter, Harvester. |
| [generals/gui/](generals/gui/) | pygame renderer + scrubable replay window. |
| [generals/remote/](generals/remote/) | socket.io client for playing on live generals.io servers. |
| [competition/protocol.py](competition/protocol.py) | Engine-side stdio wire format. |
| [competition/matchup.py](competition/matchup.py) | Local match runner driving two subprocess bots. |
| [competition/agents/](competition/agents/) | Starter bots (`expander_python`, `expander_cpp`, `expander_rust`). |
| [tools/localmatch.py](tools/localmatch.py) | **Ours.** Competition-faithful local driver — routes through `env.step` so the modifiers apply, and times replies against the 150 ms budget. |
| [tests/](tests/) | 43 tests. Heaviest coverage on the two modifiers. |

## The engine

### State

`GameState` ([game.py:34](generals/core/game.py#L34)) is an immutable `NamedTuple` of
`(H, W)` arrays: `armies`, `ownership` (shape `(2, H, W)`), `ownership_neutral`, `generals`,
`castles`, `mountains`, `passable`, plus `general_positions (2,2)`, `time`, `winner`
(`-1` = ongoing), `pool_idx`.

Everything is fixed-shape — that's what makes `jax.vmap` over thousands of games work, and
it's why maps of different sizes get padded (see *Padding* below).

`castles` were renamed from `cities`; `.cities` deprecated aliases survive on both
`GameState` and `Observation`.

### One tick

[`game.step(state, actions)`](generals/core/game.py#L324), `actions` shape `(2, 5)`:

1. **Decide move order** ([`_determine_move_order`](generals/core/game.py#L291)) —
   chasing beats reinforcing beats larger source army; ties go to player 0. "Chasing" means
   your destination is the opponent's *source* tile.
2. **Execute first mover, then second mover.** Sequential, so the second move sees the
   first's result — this is the whole basis of the deathtouch defense.
3. **Increment `time`** (unless the game was already over).
4. **Growth**, or transfer of the loser's cells to the winner if someone just won.

### Action format

`[pass, row, col, direction, split]`, all ints.

- `pass`: `0` move, `1` skip, `2` **build a castle** (only meaningful with the build-castles modifier).
- `direction`: `0` up, `1` down, `2` left, `3` right.
- `split`: `1` sends `army // 2`, `0` sends `army - 1`. One unit always stays behind.

**Invalid actions are silently no-ops.** A move is valid iff source in bounds, destination in
bounds, you own the source, `army_to_move > 0`, and the destination is passable.

Careful: `execute_action` only treats `pass == 1` as a skip. Any other value falls through to
the move path — so a build action reaching the raw engine (rather than the modifier) is
executed as a **move in direction 0**. See *Traps*.

### Growth

[`global_update`](generals/core/game.py#L260):

- Generals and castles: **+1 on every even tick** (`time % 2 == 0`).
- All owned land: **+1 every 50 ticks**.

The even-tick phase is deliberate and load-bearing — it was tuned to replay scraped
generals.io games move-for-move. Don't "fix" it.

Structures are ~25× the income of plain land. That asymmetry is the strategic core of the game
and the reason the build-castles modifier exists.

### Fog of war

[`get_visibility`](generals/core/game.py#L143) — you see the 3×3 neighbourhood of every cell
you own. `get_observation` masks spatial fields by that; `get_full_observation` is the
perfect-info variant used when `perfect_info=True`.

Two things are **not** fogged, and matter tactically:

- `opponent_land_count` / `opponent_army_count` are always the true global totals. You always
  know how big your opponent is, just not where.
- `structures_in_fog` reveals mountains and castles inside fog. You know the terrain skeleton
  from turn 0; you don't know who owns what or how big it is.

## Modifiers

Both live in [generals/modifiers/](generals/modifiers/) and wrap the base step as a pre-step
transform. `generals.core.game` is never modified by them.

### build_castles

No neutral castles spawn (`strip_neutral_castles` scrubs generated grids). Instead you build
them: action `[2, row, col, _, _]`.

Cost ([`build_cost_grid`](generals/modifiers/build_castles.py#L45)):

```
35 + Σ over your own structures max(0, 10 − 2·manhattan_distance)
```

So 35 in fresh territory, 43 adjacent to your general, nothing added by structures ≥5 cells
away. Enemy structures never affect your price. A castle you capture immediately raises your
prices near it. The cost is paid out of the army standing on that cell, and the remainder
stays — building with exactly the cost leaves a 0-army castle any single unit can take.

Builds resolve **before** either move, then are rewritten to passes so the base step never
sees `pass == 2`. The two players' builds can't interact, so resolution order is irrelevant.

### deathtouch

From `deathtouch_turn` (competition: 800), a move that *executes* onto the enemy general's
tile wins instantly, regardless of army counts.

The subtlety worth internalizing ([deathtouch.py](generals/modifiers/deathtouch.py)): "executes"
means valid **at its slot in the resolution order**. Counter-moving onto the attacker's source
tile is a chase, so it resolves first — capture the source outright and the touch never
happens. Leave 2+ army there and at least one unit still moves, so the touch lands anyway.
Both players touching on the same tick is a **draw** (`is_done` with `winner == -1`).

Builds are rewritten to passes before the touch test, so building next to the enemy general is
never a touch.

## The competition ruleset

One preset pins everything ([env.py:57](generals/core/env.py#L57)). `mode=` is authoritative —
it overrides the matching constructor arguments:

```python
env = GeneralsEnv(mode="competition")
```

| | |
|---|---|
| Map | rectangular, each side drawn independently in **18–21**, padded to **22** |
| Mountains | density 0.24–0.26 |
| Castles | **none neutral** — you build them |
| Generals | ≥14 apart (pool-wide floor; the eval driver scales to 0.8·min(h,w)) |
| Fog of war | **on** |
| Deathtouch | from turn **800** |
| Truncation | **1200** turns → draw |

Verified against the published rules at <https://www.generals.bot/docs> — build pricing
(`35 + Σ max(0, 10−2d)`, surcharges stacking), even-turn structure growth, 50-turn land growth,
defender-resolves-first, mutual touch as a draw. The engine is faithful to the spec.

### Operational constraints (from the official docs, not enforced anywhere in this repo)

| | |
|---|---|
| Per-turn budget | **150 ms wall clock**; the **first move gets 10 s** so a model can load |
| Faults | **50 faults in one game forfeits it**; a crash forfeits immediately |
| Malformed-but-parseable moves | silent pass, no penalty |
| Submission | a **zip containing `run.sh`** (required) and optionally `build.sh` |
| Timeline | Sprint checkpoint **Aug 8**, then Marathon. Same ruleset for both. |

The 150 ms budget is a first-class design constraint, and nothing local measures it —
`matchup.py` asks each bot for a move and blocks indefinitely.

### Padding

This is the single most important practical detail for an agent. Maps are 18–21 per side but
every array is **22×22**, anchored top-left, with the overhang filled with mountains
(`grid.py` step 7). The wire protocol always ships 22 rows of 22 ints. An agent that assumes
the whole grid is playable will waste search over a mountain wall; one that infers real
dimensions from the mountain border gets slightly better heuristics but gains nothing
correctness-wise, since moves into mountains are no-ops.

## Map generation

[`generate_grid`](generals/core/grid.py#L18) is fully JIT'd and *guaranteed valid by
construction* — no rejection sampling. Order: place generals first (so the distance constraint
is satisfied trivially) → mountains → **ensure connectivity by carving an L-path if needed**
→ castles by converting mountains (so the carved path is never blocked) → re-assert generals →
pad.

Grid encoding: `-2` mountain, `0` empty, `1` P0 general, `2` P1 general, `>2` castle with that
army value.

## Two ways to write an agent

### 1. In-process JAX agent — `generals.agents.Agent`

Subclass and implement `act(observation, key) -> (5,) int32 array`. Used by the RL loop and
the examples. Reference implementations, in ascending order of strength:

- `RandomAgent` — uniform over valid moves.
- `ExpanderAgent` — greedy territory grab, scores `src_army × 10(expansion) × 2(vs opponent)`.
- `HunterAgent` — garrisons its general, pushes surplus out as one advancing stack via BFS,
  decapitates. Beats Expander because a timeout is a draw and Expander never threatens.
- `HarvesterAgent` — Hunter plus detours to bank affordable neutral castles for the income.
  (Note: castle-seizing is dead logic under the competition preset, which has no neutral castles.)

`HunterAgent` is the best-documented example of writing non-trivial JAX-friendly logic:
BFS via `fori_loop` dilation, priority selection via `argmax` over masked scores, no Python
control flow on traced values. Read it before writing a JAX agent.

### 2. Subprocess agent over stdio — `competition/agents/<name>/`

This is what the competition actually runs. Any language. A directory with `run.sh`
(and optionally `build.sh`, run automatically by the matchup runner).

Protocol ([competition/protocol.py](competition/protocol.py)), line-delimited ASCII ints:

```
handshake (once):   <player_id> <H> <W>
per turn (in):      <turn> <my_land> <my_army> <opp_land> <opp_army>
                    H lines × W ints   type:  0 fog, 1 plain, 2 mountain, 3 castle, 4 general, 5 structure-in-fog
                    H lines × W ints   owner: 0 neutral/unknown, 1 me, 2 opp
                    H lines × W ints   army:  0 if not visible
per turn (out):     <pass> <row> <col> <dir> <split>
game over:          engine closes stdin — treat EOF as "exit cleanly"
```

Owner is perspective-relative. **Flush stdout every turn** or both sides deadlock.

The Python starter splits the work usefully: [main.py](competition/agents/expander_python/main.py)
is untouched scaffolding (parse frame → `Agent.act` → write action);
[agent.py](competition/agents/expander_python/agent.py) is the only file you edit.

Run a match:

```bash
python competition/matchup.py --mode competition                     # two expanders
python competition/matchup.py path/to/a/run.sh path/to/b/run.sh --mode competition
python competition/matchup.py --mode competition --gui               # record, then scrub the replay
```

## RL surface

`GeneralsEnv` is a **stateless config bag**. `reset(key)` returns `(pool, init_state)` where
`pool` is a batched `GameState` of `pool_size` pre-generated games; `step(state, actions, pool)`
takes the pool explicitly and auto-resets from it on termination by indexing `pool_idx`. This
keeps the pool out of the jit closure so refreshing it doesn't trigger recompilation.

`env.step` is the layer that applies modifiers, computes rewards, and returns a `TimeStep`
(`observation`, `reward`, `terminated`, `truncated`, `info`, `last_state`). `last_state` is the
pre-reset state — needed for bootstrap values.

Reward from `env.step` is bare win/lose (±1). The shaped functions in
[rewards.py](generals/core/rewards.py) are opt-in and are applied by the training code, not the
env. Unfinished PPO experiments live in [examples/_experimental/](examples/_experimental/).

## GUI and remote play

`ReplayGUI` takes logs of states+infos and plays them back with scrubbing (SPACE, ←/→ or H/L,
R, Q). The renderer talks to a legacy `Channels`-shaped interface, bridged by
`JaxGameAdapter` in [core/rendering.py](generals/core/rendering.py). `matchup.py --gui` records
the whole match headless first, then opens the replay — match speed is decoupled from frame rate.

[generals/remote/](generals/remote/) has a socket.io client for the real generals.io bot server
(`autopilot(agent, user_id, lobby_id)`). Independent of the competition path.

## Dev workflow

The project needs its own venv — the ambient shell python has no jax.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests -q     # 43 tests, ~12s

# Validate a competition agent with THIS, not matchup.py (see Traps):
.venv/bin/python tools/localmatch.py A/run.sh B/run.sh --games 10 --swap
```

[tools/localmatch.py](tools/localmatch.py) speaks the identical wire protocol to the identical
`run.sh` subprocesses, but steps through `env.step` under `mode="competition"` so both
modifiers apply, samples rectangular board dims per seed, times every reply against the 150 ms
budget, and counts faults. `--swap` alternates seats to halve seat bias; `--gui` replays the
last game. Startup is ~15 s (JIT + pool), then ~5 s per game.

It imports `competition/protocol.py` read-only rather than vendoring a copy, so the frame
format can't drift — but writes nothing under `competition/`, which stays exactly as upstream
ships it.

`SDL_VIDEODRIVER=dummy` keeps pygame from opening a display; CI
([.github/workflows/ci.yml](.github/workflows/ci.yml)) sets it globally and runs `pytest -q` on
every push and PR. Ruff config lives in [pyproject.toml](pyproject.toml) (line length 120,
`tests/` and `examples/` excluded) but is not wired into CI.

## Traps

Verified as of this writing.

**`matchup.py` does not apply the competition modifiers.** It drives
[`game.step` directly](competition/matchup.py#L188) — deliberately, to skip the auto-reset pool
machinery — but that bypasses `env.step`, which is where `build_castles` and `deathtouch` are
applied. Under `--mode competition` the map generation is right (no neutral castles, correct
dims), but:

- A build action `[2, r, c, 0, 0]` is not a build. The raw engine sees `pass != 1`, falls into
  the move path, and executes a **move upward** from `(r, c)`. Confirmed: 60 army on `(3,3)`
  → no castle, 59 army now on `(2,3)`.
- Deathtouch never fires, so turn 800 passes with no endgame pressure.
- No per-turn time limit is enforced, so a bot that would blow the 150 ms budget looks fine.

Net effect: a local `--mode competition` match is a castle-less, deathtouch-less, untimed game,
which is why two expanders draw at turn 1200. Anything tuned against this runner is tuned
against the wrong rules.

This is a bug in their harness, not intentional: the official docs say to add `--mode
competition` for "exact ruleset conditions". **But `matchup.py` is only a local testing tool —
it is not the grader.** Submission is a zip of `run.sh`/`build.sh` and matches are run
server-side by an eval driver that isn't in this repo (see the references to it in the
[preset comments](generals/core/env.py#L59): "the eval driver samples exact dims per seed",
"the eval driver scales 0.8×min(h,w)"). So nothing done to `competition/` locally can affect
scoring — but the zero-risk move is still to leave it untouched and drive validation matches
through `env.step` from a separate local driver.

**`bench.py` is broken.** It calls `state = env.reset(key)` (which now returns a tuple) and then
`env._pool` — an attribute that no longer exists since the env became stateless. `make bench`
fails with `AttributeError` at [bench.py:50](bench.py#L50).

**`HarvesterAgent`'s castle logic is dead under the competition preset** — there are no neutral
castles to seize, so it degrades to `HunterAgent` with extra steps.

**`sample_valid_action`'s pass probability** is a hardcoded 10%, unrelated to `RandomAgent`'s
configurable `idle_prob`.

## Starting points for a competition agent

The ruleset shapes strategy more than raw search does. Four things to reason about first:

0. **The 150 ms turn budget.** It bounds everything else. A 22×22 board is 484 cells × 4
   directions — trivial for a heuristic sweep, tight for anything resembling search in pure
   Python. Measure early; the local runner won't tell you. A crash or 50 faults forfeits the
   game outright, so robustness beats cleverness at the margin.


1. **Castle economics.** Structures are +1/2 ticks; land is +1/50 ticks. A castle at 35 army
   pays for itself in 70 ticks and compounds after. In a 1200-turn game, early castles
   probably dominate — but the proximity surcharge punishes clustering, so placement is a real
   optimization, and a freshly built castle sitting at low army is a free capture for a raider.
2. **Turn 800.** Before it, generals are protected by army count; after it, by *position and
   tempo* only. Everything before 800 is preparation for a regime where one reachable path to
   your general loses the game. Also note the chase-defense: keeping 1 army on your attacker's
   likely source tiles is not enough, but capturing them outright is.
3. **Fog asymmetry.** You always know the opponent's totals, never their location. A large
   army total with no visible presence means a stack is moving somewhere — that's actionable
   information the reference bots completely ignore.

Read `HunterAgent` for the JAX idioms, and
[competition/agents/expander_python/agent.py](competition/agents/expander_python/agent.py)
for the stdio contract. Validate against `env.step` (not `matchup.py`) until the runner's
modifier gap is closed.
