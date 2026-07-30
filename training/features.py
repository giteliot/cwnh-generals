"""What the agent sees.

Each turn the network gets PLANES (one (H, W) grid per channel) and GLOBALS
(a flat vector of scalars). Both are built from `View`, which bundles the raw
fogged observation, the remembered belief, and everything derived from them.

TO ADD A CHANNEL
    1. Add a Plane(name, fn, "why it helps") to PLANES.
       fn takes a View and returns an (H, W) array in roughly [-1, 1].
    2. Mirror it in competition/agents/ppo_bot/agent.py (same name, same order).
    3. Run tests/test_training_parity.py -- it names any channel that diverges.
NUM_PLANES is derived, so nothing else needs touching.

KEEP VALUES IN [-1, 1]. Raw army counts reach the thousands and would swamp
every other channel. Use log_scale() for counts, and divide distances by their
maximum. The rollout buffer also quantises to int8, so out-of-range clips.

DON'T ADD WHAT THE NET CAN SEE. A 3x3 conv stacked 2-3 deep sees a small
neighbourhood, so local patterns are cheap for it to learn and not worth a
channel. Things it cannot cheaply compute -- distances across the whole board,
anything remembered from previous turns, anything about the *rules* like build
cost -- are exactly what belongs here.
"""
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from generals.modifiers.build_castles import BASE_COST, PROXIMITY_DECAY, PROXIMITY_PENALTY

ARMY_SCALE = 7.0        # log1p(1000)/7 ~= 1, so a 1000-stack maps to ~1.0
MAX_STALENESS = 200     # turns after which "not seen recently" saturates
TRUNCATION = 1200.0     # the competition's draw-at-turn-N rule
DEATHTOUCH_TURN = 800.0
DEATHTOUCH_RAMP = 100.0 # start warning the agent 100 turns before deathtouch
COST_SCALE = 64.0
FAR = 1 << 14           # BFS "unreachable" sentinel
_RADIUS = (PROXIMITY_PENALTY - 1) // PROXIMITY_DECAY


class Belief(NamedTuple):
    """Memory carried between turns. Fog means the current frame is not enough:
    without this the agent re-explores cleared ground and forgets where the
    enemy general was the moment it leaves vision."""

    ever_seen: jnp.ndarray
    staleness: jnp.ndarray
    last_enemy: jnp.ndarray
    last_enemy_army: jnp.ndarray
    last_enemy_castle: jnp.ndarray
    general_prior: jnp.ndarray


def init_belief(shape):
    h, w = shape
    return Belief(
        ever_seen=jnp.zeros(shape, bool),
        staleness=jnp.zeros(shape, jnp.int32),
        last_enemy=jnp.zeros(shape, bool),
        last_enemy_army=jnp.zeros(shape, jnp.int32),
        last_enemy_castle=jnp.zeros(shape, bool),
        general_prior=jnp.full(shape, 1.0 / (h * w), jnp.float32),
    )


def reset_belief_where(belief, done):
    fresh = init_belief(belief.ever_seen.shape)
    return jax.tree.map(lambda f, b: jnp.where(done, f, b), fresh, belief)


def update_belief(belief, obs):
    """Fold this turn's sighting into memory. Call once per turn, before extract."""
    visible = ~(obs.fog_cells | obs.structures_in_fog)
    enemy = obs.opponent_cells
    enemy_general = obs.generals & enemy

    # Where can the enemy general be? Start uniform, erase everywhere we look
    # and don't find it, and collapse to a point the moment we do.
    prior = jnp.where(visible & ~enemy_general, 0.0, belief.general_prior)
    prior = jnp.where(obs.mountains | obs.castles | obs.structures_in_fog, 0.0, prior)
    prior = jnp.where(jnp.any(enemy_general), enemy_general.astype(jnp.float32), prior)

    return Belief(
        ever_seen=belief.ever_seen | visible,
        staleness=jnp.where(visible, 0, jnp.minimum(belief.staleness + 1, MAX_STALENESS)),
        last_enemy=jnp.where(visible, enemy, belief.last_enemy),
        last_enemy_army=jnp.where(visible, obs.armies * enemy, belief.last_enemy_army),
        last_enemy_castle=jnp.where(visible, obs.castles & enemy, belief.last_enemy_castle),
        general_prior=prior / jnp.maximum(jnp.sum(prior), 1e-6),
    )


# ------------------------------------------------------------------ helpers

def log_scale(x):
    """Squash a count into ~[-1, 1], sign-preserving."""
    x = x.astype(jnp.float32)
    return jnp.sign(x) * jnp.log1p(jnp.abs(x)) / ARMY_SCALE


def _neighbour_min(d):
    padded = jnp.pad(d, 1, constant_values=FAR)
    h, w = d.shape
    return jnp.minimum(
        jnp.minimum(padded[0:h, 1:w + 1], padded[2:h + 2, 1:w + 1]),
        jnp.minimum(padded[1:h + 1, 0:w], padded[1:h + 1, 2:w + 2]),
    )


def bfs_distance(seeds, passable, iters):
    """Shortest-path distance from `seeds`, by repeated dilation. A conv net
    approximates this badly at board scale, which is why it is handed over."""
    d = jnp.where(seeds, 0, FAR).astype(jnp.int32)

    def step(_, d):
        return jnp.where(passable, jnp.minimum(d, _neighbour_min(d) + 1), FAR)

    return jax.lax.fori_loop(0, iters, step, d)


def build_cost(own_structures):
    """Castle price per tile: 35 + sum over your own structures of
    max(0, 10 - 2*distance). Depends only on YOUR structures, and you always
    see your own tiles, so the agent can compute this exactly under fog."""
    h, w = own_structures.shape
    padded = jnp.pad(own_structures.astype(jnp.int32), _RADIUS)
    cost = jnp.full((h, w), BASE_COST, jnp.int32)
    for di in range(-_RADIUS, _RADIUS + 1):
        for dj in range(-_RADIUS, _RADIUS + 1):
            surcharge = PROXIMITY_PENALTY - PROXIMITY_DECAY * (abs(di) + abs(dj))
            if surcharge > 0:
                cost += surcharge * padded[_RADIUS + di:_RADIUS + di + h, _RADIUS + dj:_RADIUS + dj + w]
    return cost


def playable_dims(obs):
    """Real board size. Maps are 18-21 a side but padded to 22 with mountains,
    so the bottom/right overhang is solid rock; count it off."""
    solid = obs.mountains | obs.castles | obs.structures_in_fog
    rows = jnp.all(solid, axis=1).astype(jnp.int32)
    cols = jnp.all(solid, axis=0).astype(jnp.int32)
    return (rows.shape[0] - jnp.sum(jnp.cumprod(rows[::-1])),
            cols.shape[0] - jnp.sum(jnp.cumprod(cols[::-1])))


class View(NamedTuple):
    """Everything a channel might want, computed once per turn."""

    obs: object
    belief: Belief
    player: object
    height: int
    width: int
    real_h: jnp.ndarray
    real_w: jnp.ndarray
    in_bounds: jnp.ndarray
    passable: jnp.ndarray
    visible: jnp.ndarray
    own: jnp.ndarray
    enemy: jnp.ndarray
    own_castles: jnp.ndarray
    own_general: jnp.ndarray
    own_structures: jnp.ndarray
    cost: jnp.ndarray
    affordable: jnp.ndarray
    dist_own_general: jnp.ndarray
    dist_enemy_general: jnp.ndarray
    dist_army_mass: jnp.ndarray
    reach_iters: int
    time: jnp.ndarray
    ones: jnp.ndarray


def build_view(obs, belief, player) -> View:
    height, width = obs.armies.shape
    reach_iters = height + width
    real_h, real_w = playable_dims(obs)
    i_idx, j_idx = jnp.arange(height)[:, None], jnp.arange(width)[None, :]
    in_bounds = (i_idx < real_h) & (j_idx < real_w)

    # Fogged structures are treated as walls: the game reveals mountains and
    # castles through fog, so this is the true terrain skeleton from turn 0.
    passable = ~(obs.mountains | obs.structures_in_fog) & in_bounds

    own, enemy = obs.owned_cells, obs.opponent_cells
    own_castles, own_general = obs.castles & own, obs.generals & own
    own_structures = own_castles | own_general
    cost = build_cost(own_structures)

    seen = jnp.any(obs.generals & enemy)
    target = jnp.where(seen, obs.generals & enemy, belief.general_prior > 0)

    return View(
        obs=obs, belief=belief, player=player, height=height, width=width,
        real_h=real_h, real_w=real_w, in_bounds=in_bounds, passable=passable,
        visible=~(obs.fog_cells | obs.structures_in_fog),
        own=own, enemy=enemy, own_castles=own_castles, own_general=own_general,
        own_structures=own_structures, cost=cost,
        affordable=own & ~obs.generals & ~obs.castles & (obs.armies >= cost),
        dist_own_general=bfs_distance(own_general, passable, reach_iters),
        dist_enemy_general=bfs_distance(target, passable, reach_iters),
        dist_army_mass=bfs_distance(own & (obs.armies > 2), passable, reach_iters),
        reach_iters=reach_iters, time=obs.timestep.astype(jnp.float32),
        ones=jnp.ones((height, width), jnp.float32),
    )


def _norm_dist(v, d):
    return jnp.minimum(d.astype(jnp.float32), v.reach_iters) / v.reach_iters


class Plane(NamedTuple):
    name: str
    fn: Callable[[View], jnp.ndarray]
    why: str


PLANES = (
    # --- what is on the board right now -------------------------------------
    Plane("own_army",      lambda v: log_scale(v.obs.armies * v.own),      "your stacks"),
    Plane("enemy_army",    lambda v: log_scale(v.obs.armies * v.enemy),    "visible enemy stacks"),
    Plane("neutral_army",  lambda v: log_scale(v.obs.armies * v.obs.neutral_cells), "cost to capture"),
    Plane("own",           lambda v: v.own,                                "your territory"),
    Plane("enemy",         lambda v: v.enemy,                              "visible enemy territory"),
    Plane("neutral",       lambda v: v.obs.neutral_cells,                  "free land"),
    Plane("mountains",     lambda v: v.obs.mountains,                      "walls"),
    Plane("castles",       lambda v: v.obs.castles,                        "any castle"),
    Plane("own_castles",   lambda v: v.own_castles,                        "your income engines"),
    Plane("enemy_castles", lambda v: v.obs.castles & v.enemy,              "targets worth 25 tiles"),
    Plane("generals",      lambda v: v.obs.generals,                       "win/lose tiles"),
    Plane("fog",           lambda v: v.obs.fog_cells,                      "unseen"),
    Plane("fog_structure", lambda v: v.obs.structures_in_fog,              "wall-or-castle under fog"),

    # --- the shape of the arena ---------------------------------------------
    Plane("in_bounds",     lambda v: v.in_bounds,                          "real board vs padding"),
    Plane("border",        lambda v: _border(v),                           "distance to edge"),

    # --- castle economics (exactly computable under fog) ---------------------
    Plane("build_cost",    lambda v: v.cost.astype(jnp.float32) / COST_SCALE, "price here"),
    Plane("can_build",     lambda v: v.affordable,                         "affordable right now"),
    Plane("build_surplus", lambda v: log_scale(jnp.where(v.affordable, v.obs.armies - v.cost, 0)),
          "army left after building; 0 leaves a castle anyone can walk into"),

    # --- geometry the conv net cannot cheaply compute -----------------------
    Plane("d_own_general",   lambda v: _norm_dist(v, v.dist_own_general),   "how far from home"),
    Plane("d_enemy_general", lambda v: _norm_dist(v, v.dist_enemy_general), "how far to the target"),
    Plane("d_army_mass",     lambda v: _norm_dist(v, v.dist_army_mass),     "where your force is"),
    Plane("frontier",        lambda v: v.own & (_neighbour_min(jnp.where(~v.own & v.passable, 0, FAR)) == 0),
          "your tiles that touch expandable ground"),
    Plane("reachable",       lambda v: v.dist_own_general < FAR,            "connected to you"),

    # --- memory (the fog answer) --------------------------------------------
    Plane("ever_seen",       lambda v: v.belief.ever_seen,                  "explored"),
    Plane("staleness",       lambda v: v.belief.staleness.astype(jnp.float32) / MAX_STALENESS,
          "how out of date this tile is"),
    Plane("last_enemy",      lambda v: v.belief.last_enemy,                 "enemy seen here once"),
    Plane("last_enemy_army", lambda v: log_scale(v.belief.last_enemy_army), "how big it was"),
    Plane("general_prior",   lambda v: v.belief.general_prior / jnp.maximum(jnp.max(v.belief.general_prior), 1e-6),
          "where the enemy general still could be"),
    Plane("last_enemy_castle", lambda v: v.belief.last_enemy_castle,        "castles don't move"),

    # --- the clock -----------------------------------------------------------
    Plane("progress",   lambda v: v.ones * (v.time / TRUNCATION),           "how close to the draw"),
    Plane("even_tick",  lambda v: v.ones * (v.time % 2),                    "structures grow on even ticks"),
    Plane("deathtouch", lambda v: v.ones * jnp.clip(
        (v.time - (DEATHTOUCH_TURN - DEATHTOUCH_RAMP)) / DEATHTOUCH_RAMP, 0.0, 1.0),
        "from turn 800 any touch on a general wins"),
)


def _border(v):
    i_idx, j_idx = jnp.arange(v.height)[:, None], jnp.arange(v.width)[None, :]
    d = jnp.minimum(jnp.minimum(i_idx, v.real_h - 1 - i_idx), jnp.minimum(j_idx, v.real_w - 1 - j_idx))
    return jnp.maximum(d, 0) / jnp.maximum(v.real_h, v.real_w)


class Global(NamedTuple):
    name: str
    fn: Callable[[View], jnp.ndarray]
    why: str


def _income(structures, land):
    return 0.5 * structures + 0.02 * land


def _own_struct(v):
    return jnp.sum(v.own_structures).astype(jnp.float32)


def _opp_struct(v):
    return jnp.sum(v.belief.last_enemy_castle | (v.obs.castles & v.enemy)).astype(jnp.float32) + 1.0


def _area(v):
    return jnp.maximum(v.real_h * v.real_w, 1).astype(jnp.float32)


GLOBALS = (
    Global("own_army",   lambda v: log_scale(v.obs.owned_army_count), "your total army"),
    Global("opp_army",   lambda v: log_scale(v.obs.opponent_army_count), "NEVER fogged: always true"),
    Global("own_land",   lambda v: log_scale(v.obs.owned_land_count), "your total land"),
    Global("opp_land",   lambda v: log_scale(v.obs.opponent_land_count), "NEVER fogged: always true"),
    Global("army_diff",  lambda v: log_scale(v.obs.owned_army_count - v.obs.opponent_army_count), "who is ahead"),
    Global("land_diff",  lambda v: log_scale(v.obs.owned_land_count - v.obs.opponent_land_count), "who is ahead"),
    Global("own_income", lambda v: _income(_own_struct(v), v.obs.owned_land_count) / 8.0, "army per tick"),
    Global("opp_income", lambda v: _income(_opp_struct(v), v.obs.opponent_land_count) / 8.0, "estimated"),
    Global("income_diff", lambda v: (_income(_own_struct(v), v.obs.owned_land_count)
                                     - _income(_opp_struct(v), v.obs.opponent_land_count)) / 8.0,
           "the number that decides long games"),
    Global("own_castles", lambda v: _own_struct(v) / 8.0, "your structure count"),
    Global("opp_castles", lambda v: _opp_struct(v) / 8.0, "what you have seen of theirs"),
    Global("hidden_army", lambda v: log_scale(jnp.maximum(
        v.obs.opponent_army_count - jnp.sum(v.obs.armies * v.enemy), 0.0)),
        "enemy army you CANNOT see: totals are never fogged, so a big number "
        "here means a stack is moving somewhere. Reference bots ignore this."),
    Global("explored",    lambda v: jnp.sum(v.belief.ever_seen).astype(jnp.float32) / _area(v), "map covered"),
    Global("visible",     lambda v: jnp.sum(v.visible).astype(jnp.float32) / _area(v), "seen right now"),
    Global("progress",    lambda v: v.time / TRUNCATION, "turn / 1200"),
    Global("land_tick",   lambda v: (v.time % 50) / 50.0, "land grows every 50 ticks"),
    Global("to_deathtouch", lambda v: jnp.clip((DEATHTOUCH_TURN - v.time) / DEATHTOUCH_TURN, 0.0, 1.0), "countdown"),
    Global("is_player_0", lambda v: jnp.asarray(v.player == 0, jnp.float32),
           "player 0 wins move-order ties, so the seat is part of the state"),
    Global("board_fill",  lambda v: _area(v) / (v.height * v.width), "real area vs padded"),
    Global("general_found", lambda v: jnp.any(v.obs.generals & v.enemy).astype(jnp.float32), "target acquired"),
)

NUM_PLANES = len(PLANES)
NUM_GLOBALS = len(GLOBALS)
PLANE_NAMES = [p.name for p in PLANES]
GLOBAL_NAMES = [g.name for g in GLOBALS]


def extract(obs, belief, player_idx):
    """-> (planes (H, W, NUM_PLANES), globals (NUM_GLOBALS,))"""
    v = build_view(obs, belief, player_idx)
    planes = jnp.stack([p.fn(v) for p in PLANES], axis=-1).astype(jnp.float32)
    globals_ = jnp.stack([g.fn(v) for g in GLOBALS]).astype(jnp.float32)
    return planes, globals_
