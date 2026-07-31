"""What the agent is rewarded for.

    reward = terminal_outcome + alpha * (gamma * PHI(next) - PHI(prev))

PHI scores how good a position is. Reward comes from the *change* in PHI, so a
move is paid exactly when it improves the position. This is potential-based
shaping: it provably cannot change which policy is optimal, so terms can be
added freely without corrupting the objective. What it does change is which
behaviours are easy to discover -- that is the point, and that is what you tune.

TO ADD A BEHAVIOUR
    1. Write a function (me, opp) -> scalar, roughly in [-1, 1].
    2. Add it to TERMS with a weight.
That is all. Bounds, CLI overrides (--w-<name>) and tests follow automatically.

KEEP EACH TERM BOUNDED. Use tanh on anything unbounded. An unbounded term lets
one behaviour drown out the rest, and the value function has to predict its sum.

REWARD LEVELS, NOT ACTIONS. `me.land - opp.land` is a level: good. "+0.1 for
moving" is an action bonus, is not potential-based, and will be farmed.
"""
from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp

from generals.core.game import get_visibility

from .features import bfs_distance

STRUCTURE_INCOME = 0.5      # a general or castle makes +1 army every 2 ticks
LAND_INCOME = 0.02          # a plain tile makes +1 army every 50 ticks
PROXIMITY_CAP = 12          # distances beyond this are "far"; keeps BFS cheap
PROXIMITY_ITERS = 16


class Side(NamedTuple):
    """One player's position, precomputed so terms stay one-liners."""

    cells: jnp.ndarray        # (H, W) bool  -- tiles this player owns
    armies: jnp.ndarray       # (H, W) int   -- armies, zero off-territory
    stacks: jnp.ndarray       # (H, W) bool  -- tiles with a movable army (>1)
    general: jnp.ndarray      # (2,) int     -- general position
    land: jnp.ndarray         # scalar       -- tiles owned
    army: jnp.ndarray         # scalar       -- total army
    structures: jnp.ndarray   # scalar       -- general + castles owned
    income: jnp.ndarray       # scalar       -- army produced per tick
    visible: jnp.ndarray      # (H, W) bool  -- tiles this player can see
    passable: jnp.ndarray     # (H, W) bool  -- shared terrain


def side(state, p) -> Side:
    cells = state.ownership[p]
    structures = jnp.sum((state.castles | state.generals) & cells).astype(jnp.float32)
    land = jnp.sum(cells).astype(jnp.float32)
    return Side(
        cells=cells,
        armies=state.armies * cells,
        stacks=cells & (state.armies > 1),
        general=state.general_positions[p],
        land=land,
        army=jnp.sum(state.armies * cells).astype(jnp.float32),
        structures=structures,
        income=STRUCTURE_INCOME * structures + LAND_INCOME * land,
        visible=get_visibility(cells),
        passable=state.passable,
    )


def _reach(seeds, passable, target):
    """Normalised distance from `seeds` to `target`. 0 = adjacent, 1 = far."""
    d = bfs_distance(seeds, passable, PROXIMITY_ITERS)
    return jnp.minimum(d[target[0], target[1]], PROXIMITY_CAP) / PROXIMITY_CAP


# ---------------------------------------------------------------- behaviours

def economy(me, opp):
    """Out-produce the opponent. Denominated in army-per-tick, so one castle
    is correctly worth 25 plain tiles rather than whatever a weight says."""
    return jnp.tanh((me.income - opp.income) / 4.0)


def material(me, opp):
    """Have more army than the opponent.

    This is "reward killing enemy soldiers, penalise losing your own" written
    as a level instead of an event. The reward is the change in PHI, so
        gamma*PHI' - PHI  ~=  (delta_my_army - delta_opp_army) / 50
    Their army drops because you killed it -> positive. Yours drops -> negative.
    Writing it as a level is what makes it unfarmable: an event bonus like
    "+1 per kill" can be gamed by trading armies back and forth, a level cannot
    because the potential returns to where it started.

    Linear inside the tanh (not log), so 10 soldiers killed is worth the same
    whether you have 100 or 1000. tanh only saturates the blowout cases.

    KNOWN LIMITATION -- revisit. Using the *difference* forces killing and
    being killed to be weighted equally, so an even 10-for-10 trade scores ~0,
    identical to not engaging at all. Those are very different games: an even
    trade favours whoever is ahead on economy. Two ways out, both still valid
    potentials:

      1. Split the term, so the two weights can differ:
             Term("kills",  0.6, lambda me, opp: -jnp.tanh(opp.army / 50))
             Term("losses", 0.4, lambda me, opp:  jnp.tanh(me.army / 50))
         An even trade now scores (0.4 - 0.6) * 10/50 != 0. kills > losses
         makes an aggressive trader, losses > kills makes a hoarder.

      2. Go back to a ratio: tanh(log1p(me.army) - log1p(opp.army)) rewards
         even trades for whoever is ahead (100v50 -> 90v40 improves the ratio
         but not the difference). Costs the linearity above -- pick one.
    """
    return jnp.tanh((me.army - opp.army) / 50.0)


def territory(me, opp):
    """Hold more tiles. Cheap early signal -- land is how you reach anything."""
    return jnp.tanh((me.land - opp.land) / 60.0)


def aggression(me, opp):
    """Get your army near the enemy general. Without this, a pure-economy
    agent out-farms the opponent and then draws at turn 1200."""
    return 1.0 - _reach(me.stacks, me.passable, opp.general)


def defence(me, opp):
    """Keep enemy army away from your general.

    WARNING: this is maximised by never leaving home. It paid a turtling
    policy +0.5 to sit on one tile and pass for 1200 turns. Default 0.
    Only worth reviving near the turn-800 deathtouch rule, where position
    is the only thing protecting your general.
    """
    return _reach(opp.stacks, me.passable, me.general)


def exploration(me, opp):
    """See more of the board. Under fog you cannot attack what you cannot
    find, so this is a prerequisite for `aggression` doing anything."""
    return jnp.mean(me.visible.astype(jnp.float32))


class Term(NamedTuple):
    name: str
    weight: float
    fn: Callable[[Side, Side], jnp.ndarray]


TERMS = (
    Term("material", 0.5, material),      # kills and losses -- the fastest signal
    Term("territory", 0.4, territory),    # land, the other thing you see immediately
    Term("econ", 0.4, economy),           # income; the only term that values castles
    Term("explore", 0.1, exploration),    # you can't attack what you can't find
    Term("threat", 0.0, aggression),      # off: costs a BFS, indirect. Turn on later.
    Term("safety", 0.0, defence),         # off: pays the agent to turtle. See docstring.
)

DEFAULT_WEIGHTS = {t.name: t.weight for t in TERMS}
PHI_BOUND = sum(abs(t.weight) for t in TERMS)


def phi(state, p, w_explore=None, weights=None):
    """Score the position from player p's point of view."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if w_explore is not None:
        w["explore"] = w_explore
    me, opp = side(state, p), side(state, 1 - p)
    return sum(w[t.name] * t.fn(me, opp) for t in TERMS)


def shaping(prev_state, next_state, done, p, gamma, w_explore=None, weights=None):
    """The per-step shaped reward. PHI(terminal) is 0, which is what makes the
    whole episode's shaping telescope to a constant -- see the test."""
    phi_prev = phi(prev_state, p, w_explore, weights)
    phi_next = jnp.where(done, 0.0, phi(next_state, p, w_explore, weights))
    return gamma * phi_next - phi_prev


def draw_score(state, p):
    """Who was ahead when the game timed out. In [-1, 1], on income."""
    return jnp.tanh((side(state, p).income - side(state, 1 - p).income) / 4.0)


# --------------------------------------------------------- terminal outcomes
#
# Winning has to dominate everything else. The whole episode's shaping sums to
# exactly -alpha*PHI(s0), so it is bounded by alpha*PHI_BOUND (~1.4 here).
# Keeping WIN well above that is what makes the objective "win the game" rather
# than "accumulate potential".
#
# The numbers are not huge on purpose. PPO normalises advantages per minibatch,
# so only the RATIO of terminal to shaping does anything -- multiplying every
# reward by 100 changes nothing. What a huge terminal DOES do is force the
# critic to predict a rare enormous spike, which is how the value function
# diverged to 12.9M in ppo_medium_v1. 4:1 over shaping, well-conditioned critic.
#
# LOSS is smaller in magnitude than WIN on purpose: it makes a coin-flip fight
# better in expectation than a certain draw, which is the opposite of the
# turtling the agent found on its own.

WIN = 10.0
LOSS = -9.0
DRAW = -2.0
DRAW_TIEBREAK = 1.0     # a draw while ahead on income hurts less than one behind


class Outcome(NamedTuple):
    win: float = WIN
    loss: float = LOSS
    draw: float = DRAW
    tiebreak: float = DRAW_TIEBREAK


def terminal_reward(raw, terminated, truncated, state, p, outcome=Outcome()):
    """Map the engine's +1/-1/0 onto the shaped terminal value.

    raw is +1 win, -1 loss, 0 otherwise. A terminated game with raw == 0 is a
    mutual deathtouch, which is a genuine draw. Non-final steps return 0.
    """
    decided = jnp.where(raw > 0, outcome.win, outcome.loss)
    drawn = outcome.draw + outcome.tiebreak * draw_score(state, p)
    ended = jnp.where(raw == 0, drawn, decided)
    return jnp.where(truncated, drawn, jnp.where(terminated, ended, 0.0))


REWARD_BOUND = max(abs(WIN), abs(LOSS), abs(DRAW) + abs(DRAW_TIEBREAK)) + PHI_BOUND
