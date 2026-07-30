# Reward and observation

The two things you actually design. Everything else — PPO, the network, the
rollout — is machinery that doesn't change. This is where behaviour comes from.

Both are now **lists of named functions**. Adding a behaviour or a channel means
writing one function and adding one line. Nothing else needs updating: counts,
CLI flags and tests all derive from the lists.

---

## 1. Reward — [potential.py](potential.py)

### The shape of it

```
reward = terminal_outcome  +  alpha * (gamma * PHI(next_state) - PHI(prev_state))
```

**Terminal outcome dominates.** `WIN=+10, LOSS=-6, DRAW=-2` (plus up to +/-1 on a
draw depending on who was ahead). The whole episode's shaping sums to exactly
`-alpha*PHI(s0)`, bounded by `alpha*PHI_BOUND ~= 1.4` -- so winning outweighs
everything shaping can ever pay, by about 4x. A test enforces this.

Why not bigger numbers? PPO normalises advantages per minibatch, so only the
*ratio* of terminal to shaping does anything: multiplying every reward by 100
changes training not at all. What a huge terminal *does* do is force the critic
to predict a rare enormous spike, which is exactly how `ppo_medium_v1`'s value
function diverged to 12.9M. 4:1 gets the intent with a stable critic.

`|LOSS| < |WIN|` on purpose: it makes a coin-flip fight better in expectation
than a certain draw. `DRAW < 0` because 78% of self-play games end in one, and
scoring them 0 is what made turtling rational.

`PHI` scores *how good a position is*. The reward is the **change** in PHI, so a
move is paid exactly when it makes your position better.

This is **potential-based shaping**, and the reason to use it is a theorem: shaping
of the form `γΦ(s') − Φ(s)` provably cannot change which policy is optimal
(Ng, Harada & Russell 1999). You can add terms freely without corrupting the
objective. `tests/test_training_potential.py::test_shaping_telescopes` verifies
this empirically — the whole episode's shaping sums to a constant `−Φ(s₀)`,
independent of what the agent did.

What shaping *does* change is which behaviours are easy to stumble into. That's
the entire point, and it's what you're tuning.

### The terms

Each is a function `(me, opp) -> scalar in [-1, 1]`, listed in `TERMS`:

| term | weight | what it makes the agent do |
|---|---|---|
| `material` | 0.5 | Have more army — i.e. **kill their soldiers, don't lose yours**. See below. |
| `territory` | 0.4 | Hold more tiles. |
| `econ` | 0.4 | Out-produce them. In army-per-tick, so a castle is correctly worth 25 tiles — the only term that values castles at all. |
| `explore` | 0.1 | See more board. Under fog you can't attack what you can't find. |
| `threat` | **0.0** | Move army toward the enemy general. Off: costs a BFS and is indirect. Turn on once the basics work. |
| `safety` | **0.0** | Keep enemy away from your general. Off: it pays the agent to turtle — see below. |

### "Reward kills, penalise losses" is `material`

That behaviour is already here, written as a *level* rather than an event.
`material` is `tanh((my_army − opp_army)/50)`, and since reward is the change in
PHI:

```
γΦ(next) − Φ(prev)  ≈  (Δmy_army − Δopp_army) / 50
```

Their army drops because you killed it → **positive reward**. Yours drops →
**negative reward**. Exactly the intent.

Writing it as a level is what makes it unfarmable. An event bonus (`+1 per kill`)
can be gamed by trading armies back and forth; a level cannot, because the
potential returns to where it started. It's linear inside the `tanh` (not log),
so 10 soldiers killed is worth the same at army 100 or 1000.

**Known limitation — revisit.** Using the *difference* forces kills and losses
to be weighted equally, so an even 10-for-10 trade scores ~0, which is
indistinguishable from never engaging. Those are very different games: an even
trade favours whoever is ahead on economy. Two fixes, both still valid
potentials:

```python
# 1. split it, so the two weights can differ
Term("kills",  0.6, lambda me, opp: -jnp.tanh(opp.army / 50)),
Term("losses", 0.4, lambda me, opp:  jnp.tanh(me.army / 50)),
#    an even trade now scores (0.4-0.6)*10/50 != 0
#    kills > losses -> aggressive trader;  losses > kills -> hoarder

# 2. or use a ratio, which rewards even trades for whoever is ahead
#    (100v50 -> 90v40 improves the ratio but not the difference)
Term("material", 0.5, lambda me, opp: jnp.tanh(jnp.log1p(me.army) - jnp.log1p(opp.army))),
```

Option 2 costs the linearity above — 10 kills would read differently at army 100
vs 1000. The two properties trade off; pick deliberately.

`me` and `opp` are `Side` objects with `.land .army .structures .income .cells
.armies .stacks .general .visible .passable` already computed, so terms stay
one-liners.

### Adding a behaviour

```python
def hold_castles(me, opp):
    """Keep the castles you build instead of trading them away."""
    return jnp.tanh((me.structures - opp.structures) / 3.0)

TERMS = (
    ...,
    Term("hold_castles", 0.5, hold_castles),
)
```

That's it. `--w-hold_castles 0.8` works immediately, `PHI_BOUND` updates, and
`test_every_term_is_bounded` starts checking it.

### Two rules

**Keep each term inside `[-1, 1]`** — wrap anything unbounded in `tanh`. An
unbounded term drowns out the others, and the critic has to predict the sum.
There's a test for this.

**Reward states, not actions.** `me.land - opp.land` is a property of the
position: fine. "+0.1 for every move" is a property of the *action* — it isn't
potential-based, it breaks the theorem, and it will be farmed. The old
`win_lose_reward_fn` in `generals/core/rewards.py` has exactly this bug (a
`+0.0015` split bonus worth ~1.8 over a game, comparable to winning).

### Why `safety` is off

It measures *distance from enemy armies to your general* — maximised by never
leaving home. It paid a turtling policy **+0.5 to sit on one tile and pass for
1200 turns**, which is exactly what `ppo_medium_v1` and `v2` learned to do.
It may be worth reviving near the turn-800 deathtouch rule, where position is
the only thing protecting your general. Measured directly: removing it dropped
a turtle's PHI from −0.18 to −0.68.

### The draw problem

**78% of self-play games end in a draw at turn 1200.** The engine scores a draw
as `0`, so the most common outcome used to carry no gradient whatsoever. Worse,
for an agent that loses more than it wins, turtling to a `0` genuinely beats
playing to a `−1`. Turtling was *correct*.

Fixed two ways, both on by default:
`DRAW = -2.0` makes timing out actively bad, and `DRAW_TIEBREAK = 1.0` adds
`±1` on top depending on who was ahead on income — so a 1-tile turtle lands
near `-2.4` and a 100-tile expander near `-1.6`. Tune with
`--draw --draw-tiebreak`.

This deliberately deviates from competition scoring (where a draw is just a
draw). It's a training device to break a degenerate equilibrium, not a claim
about the rules.

### Don't anneal `alpha` to zero

`--anneal-frac 0.6` decays shaping to 0 by 60% of training. That was a mistake:
PBRS is policy-invariant, so there's nothing to anneal *away*, and turning it
off deletes the only dense signal. In `v2` the value loss read `0.000` after the
anneal completed — no reward variance, nothing to learn from. Use
`--anneal-frac 10` to keep `alpha ≈ 0.9` throughout.

---

## 2. Observation — [features.py](features.py)

Each turn the network gets `PLANES` — one `(H, W)` grid per channel — and
`GLOBALS`, a flat vector. Both are built from a `View` that bundles the raw
fogged observation, remembered belief, and everything derived.

### What's in there

**On the board now** (13): own/enemy/neutral army, own/enemy/neutral territory,
mountains, castles, own castles, enemy castles, generals, fog, fog-structure.

**Arena shape** (2): `in_bounds` (maps are 18–21 padded to 22 with mountains —
handed over rather than making the net infer it), `border`.

**Castle economics** (3): `build_cost`, `can_build`, `build_surplus`. The price
depends only on *your* structures and you always see your own tiles, so this is
exactly computable under fog. `build_surplus` matters because paying exactly the
cost leaves a 0-army castle anyone can walk into.

**Geometry** (5): BFS distances to your general, the enemy general, your army
mass; plus `frontier` and `reachable`. A 3×3 conv approximates board-scale
distance badly, so it's given directly.

**Memory** (6): `ever_seen`, `staleness`, `last_enemy`, `last_enemy_army`,
`last_enemy_castle`, `general_prior`. This is the fog answer — a feed-forward net
on the current frame is amnesiac. `general_prior` is a distribution over where
the enemy general can still be: uniform at first, erased wherever you look and
don't find it, collapsing to a point when you do.

**Clock** (3): `progress`, `even_tick` (structures grow on even ticks),
`deathtouch` (ramps up before turn 800).

**Globals** (20) include the sharpest feature in the set: **`hidden_army`** =
`opponent_army_count − visible enemy army`. Opponent totals are *never* fogged,
so a large value means a stack is moving somewhere unseen. The reference bots
ignore this entirely.

### Adding a channel

```python
Plane("enemy_adjacent",
      lambda v: v.own & (_neighbour_min(jnp.where(v.enemy, 0, FAR)) == 0),
      "your tiles touching the enemy — where fights happen"),
```

Then **mirror it in [competition/agents/ppo_bot/agent.py](../competition/agents/ppo_bot/agent.py)**
with the same name in the same position, and run:

```bash
SDL_VIDEODRIVER=dummy .venv/bin/python -m pytest tests/test_training_parity.py -q
```

If you forget, the test tells you exactly what broke:

```
AssertionError: plane 'frontier' (index 21) differs by 1 between JAX and NumPy
```

The duplication exists because the submitted bot runs pure NumPy (6.8 ms/turn,
no JAX import) while training runs JAX. The parity test is what makes it safe.

### Two rules

**Keep values in `[-1, 1]`.** Raw army counts reach the thousands and would
swamp everything. Use `log_scale()` for counts, divide distances by their max.
The rollout buffer quantises to `int8`, so anything out of range clips.

**Don't add what the net can already see.** Stacked 3×3 convs learn local
patterns cheaply. What they *can't* cheaply compute — board-scale distances,
anything remembered across turns, anything about the *rules* like build cost —
is what belongs here.

### Changing channels invalidates checkpoints

Adding or removing a plane changes the input width, so **old checkpoints won't
load**. Copy any bot you want to keep before editing. Editing `potential.py` is
safe by contrast — reward only affects training, never inference, so old
checkpoints keep playing identically.

---

## 3. Where the evidence lives

Don't take the tuning above on faith; these reproduce it:

```bash
# does it actually play, or turtle? land count is the tell
.venv/bin/python tools/watch.py ppo:CKPT.pkl expander --stage medium --games 6 --no-gui

# watch a game
.venv/bin/python tools/watch.py ppo:CKPT.pkl hunter --stage medium

# play it yourself (B builds a castle)
.venv/bin/python tools/play_human.py --agent ppo --checkpoint CKPT.pkl --stage medium

# outcome mix over a training run — draws hide failure
grep -E "^upd" RUN.log | sed 's/.*W\/L\/D *//' | sed 's/|.*//' | tr '/' ' ' \
  | awk '{w+=$1;l+=$2;d+=$3} END {printf "W %d  L %d  D %d\n", w,l,d}'
```
