"""Self-play sanity: an agent playing an identical copy of itself must not win.

A 98% training win-rate against `self` once hid a bug where both sides were
handed the SAME belief and only the learner's was carried forward, so the
opponent played every rollout with its enemy's map memory and move trail.
"""
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import get_observation
from training.actions import decode
from training.features import extract, init_belief, record_move, update_belief
from training.masking import action_mask
from generals.core.env import GeneralsEnv
from training.network import ActorCritic, sample


@partial(jax.jit, static_argnums=(0, 4))
def _act(net, params, obs, belief, player, key):
    belief = update_belief(belief, obs)
    planes, globals_ = extract(obs, belief, player)
    logits, _ = net.apply(params, planes, globals_, action_mask(obs))
    move = decode(sample(logits, key), *obs.armies.shape)
    return record_move(belief, move), move


def test_identical_policies_split_games():
    """Same weights on both seats, each with its own belief -> near 50/50."""
    # Small board, adjacent generals, early deathtouch: random-init policies
    # only decide a game by blundering into a general, so the board has to be
    # tiny for enough of them to finish. On `tiny` (8x8, deathtouch 800) a
    # 200-turn cap decided nothing at all and the test measured luck.
    env = GeneralsEnv(
        grid_dims=(6, 6),
        truncation=80,
        min_generals_distance=2,
        build_castles=True,
        deathtouch_turn=15,
        pool_size=64,
    )
    pool, _ = env.reset(jrandom.PRNGKey(0))
    net = ActorCritic(channels=8, blocks=2, groups=4)
    shape = pool.armies.shape[1:]
    params = net.init(
        jrandom.PRNGKey(1), jnp.zeros((*shape, 32 + 2)), jnp.zeros((20,)),
        jnp.ones((shape[0] * shape[1] * 9 + 1,), bool),
    )
    wins = {0: 0, 1: 0, -1: 0}
    for g in range(64):
        state = jax.tree.map(lambda x: x[g], pool)
        beliefs = [init_belief(shape), init_belief(shape)]
        key = jrandom.PRNGKey(100 + g)
        for _ in range(85):
            key, k0, k1 = jrandom.split(key, 3)
            moves = []
            for p, k in ((0, k0), (1, k1)):
                beliefs[p], mv = _act(net, params, get_observation(state, p), beliefs[p], p, k)
                moves.append(mv)
            timestep, state = env.step(state, jnp.stack(moves), pool)
            if bool(timestep.terminated | timestep.truncated):
                wins[int(timestep.info.winner)] += 1
                break
    decided = wins[0] + wins[1]
    assert decided >= 10, f"too few decisive games to judge: {wins}"
    share = wins[0] / decided
    assert 0.15 < share < 0.85, (
        f"identical policies should split games, got p0 {share:.0%} ({wins}). "
        "One side is likely being handed the other's belief."
    )
