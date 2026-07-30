import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.modifiers import build_castles
from training.actions import decode, encode, num_actions
from training.features import build_cost
from training.masking import action_mask


def advance(env, pool, steps, seed):
    state = jax.tree.map(lambda x: x[seed % pool.armies.shape[0]], pool)
    key = jrandom.PRNGKey(seed)
    for _ in range(steps):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([
            _random_action(state, 0, k0),
            _random_action(state, 1, k1),
        ])
        _, state = env.step(state, actions, pool)
    return state


def _random_action(state, player, key):
    obs = game.get_observation(state, player)
    mask = action_mask(obs)
    h, w = obs.armies.shape
    idx = jrandom.choice(key, num_actions(h, w), p=mask / mask.sum())
    return decode(idx, h, w)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_action_codec_roundtrip(seed):
    h, w = 22, 22
    key = jrandom.PRNGKey(seed)
    indices = jrandom.randint(key, (200,), 0, num_actions(h, w))
    for idx in indices:
        action = decode(idx, h, w)
        assert int(encode(action, h, w)) == int(idx)


def test_build_cost_matches_engine():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = advance(env, pool, 40, 0)
    for player in (0, 1):
        obs = game.get_observation(state, player)
        own_structures = (obs.castles | obs.generals) & obs.owned_cells
        expected = build_castles.build_cost_grid(state, player)
        assert jnp.array_equal(build_cost(own_structures), expected)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_masked_in_actions_are_legal(seed):
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = advance(env, pool, 60, seed)
    h, w = state.armies.shape

    for player in (0, 1):
        obs = game.get_observation(state, player)
        mask = action_mask(obs)
        for idx in jnp.flatnonzero(mask):
            action = decode(idx, h, w)
            head = int(action[0])
            if head == 1:
                continue
            if head == 2:
                after = build_castles._apply_one(state, player, action)
                assert not jnp.array_equal(after.castles, state.castles), f"build {action} was a no-op"
            else:
                after = game.execute_action(state, player, action)
                assert not jnp.array_equal(after.armies, state.armies), f"move {action} was a no-op"


@pytest.mark.parametrize("seed", [0, 1])
def test_mask_is_exact_without_fog(seed):
    env = GeneralsEnv(mode="competition", pool_size=64, perfect_info=True)
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = advance(env, pool, 60, seed)
    h, w = state.armies.shape
    player = 0
    obs = game.get_full_observation(state, player)
    mask = action_mask(obs)

    for idx in range(num_actions(h, w)):
        action = decode(idx, h, w)
        head = int(action[0])
        if head == 1:
            assert bool(mask[idx])
            continue
        if head == 2:
            changed = not jnp.array_equal(
                build_castles._apply_one(state, player, action).castles, state.castles
            )
        else:
            changed = not jnp.array_equal(
                game.execute_action(state, player, action).armies, state.armies
            )
        assert bool(mask[idx]) == changed, f"mask disagrees with engine at {action}"


def test_pass_is_always_valid():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    obs = game.get_observation(state, 0)
    mask = action_mask(obs)
    assert bool(mask[-1])
    assert int(mask.sum()) == 1
