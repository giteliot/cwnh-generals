import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from training.actions import decode, num_actions
from training.features import (
    NUM_GLOBALS,
    NUM_PLANES,
    extract,
    init_belief,
    playable_dims,
    reset_belief_where,
    update_belief,
)
from training.masking import action_mask


def _random_action(state, player, key):
    obs = game.get_observation(state, player)
    mask = action_mask(obs)
    h, w = obs.armies.shape
    return decode(jrandom.choice(key, num_actions(h, w), p=mask / mask.sum()), h, w)


def advance(env, pool, steps, seed):
    state = jax.tree.map(lambda x: x[seed % pool.armies.shape[0]], pool)
    key = jrandom.PRNGKey(seed)
    for _ in range(steps):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([_random_action(state, 0, k0), _random_action(state, 1, k1)])
        _, state = env.step(state, actions, pool)
    return state


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_playable_dims_recovers_real_board(seed):
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = jax.tree.map(lambda x: x[seed], pool)
    obs = game.get_observation(state, 0)
    h, w = (int(x) for x in playable_dims(obs))

    assert 18 <= h <= 21 and 18 <= w <= 21
    assert bool(jnp.all(state.mountains[h:, :]))
    assert bool(jnp.all(state.mountains[:, w:]))
    assert not bool(jnp.all(state.mountains[h - 1, :w]))
    assert not bool(jnp.all(state.mountains[:h, w - 1]))


@pytest.mark.parametrize("dims", [(10, 10), (13, 16), (22, 22)])
def test_features_are_size_agnostic(dims):
    env = GeneralsEnv(grid_dims=dims, build_castles=True, deathtouch_turn=800, pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    obs = game.get_observation(state, 0)
    belief = update_belief(init_belief(obs.armies.shape), obs)
    planes, globals_ = extract(obs, belief, 0)

    assert planes.shape == state.armies.shape + (NUM_PLANES,)
    assert globals_.shape == (NUM_GLOBALS,)
    assert not bool(jnp.isnan(planes).any())
    assert not bool(jnp.isnan(globals_).any())
    assert float(jnp.abs(planes).max()) <= 1.0 + 1e-5


def test_belief_accumulates_and_resets():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)

    belief = init_belief(state.armies.shape)
    obs = game.get_observation(state, 0)
    belief = update_belief(belief, obs)
    seen_first = int(belief.ever_seen.sum())
    assert seen_first > 0

    state = advance(env, pool, 30, 0)
    obs = game.get_observation(state, 0)
    belief = update_belief(belief, obs)
    assert int(belief.ever_seen.sum()) >= seen_first
    assert int(belief.staleness.max()) > 0

    cleared = reset_belief_where(belief, jnp.bool_(True))
    assert int(cleared.ever_seen.sum()) == 0
    assert int(cleared.staleness.max()) == 0

    kept = reset_belief_where(belief, jnp.bool_(False))
    assert int(kept.ever_seen.sum()) == int(belief.ever_seen.sum())


def test_general_prior_collapses_when_enemy_general_seen():
    env = GeneralsEnv(grid_dims=(8, 8), min_generals_distance=2, perfect_info=True, pool_size=8)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    obs = game.get_full_observation(state, 0)
    belief = update_belief(init_belief(obs.armies.shape), obs)

    enemy_general = obs.generals & obs.opponent_cells
    assert int(enemy_general.sum()) == 1
    assert float(belief.general_prior[enemy_general].sum()) == pytest.approx(1.0)


def test_autoreset_observation_is_post_reset_while_last_state_is_not():
    env = GeneralsEnv(
        grid_dims=(8, 8), truncation=6, min_generals_distance=3, build_castles=True, pool_size=16
    )
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    key = jrandom.PRNGKey(0)

    for _ in range(6):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([_random_action(state, 0, k0), _random_action(state, 1, k1)])
        timestep, state = env.step(state, actions, pool)
        if bool(timestep.terminated | timestep.truncated):
            break

    assert bool(timestep.truncated)
    assert int(timestep.last_state.time) == 6
    assert int(timestep.observation.timestep[0]) == 0
    assert int(game.get_observation(timestep.last_state, 0).timestep) == 6
