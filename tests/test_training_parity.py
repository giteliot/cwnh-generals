import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from competition import protocol
from generals.core import game
from generals.core.env import GeneralsEnv
from training.actions import decode as jax_decode
from training.actions import num_actions
from training.features import GLOBAL_NAMES, PLANE_NAMES, init_belief, record_move, update_belief
from training.features import NUM_GLOBALS, NUM_PLANES
from training.features import extract as jax_extract
from training.masking import action_mask as jax_mask
from training.network import ActorCritic

BOT_DIR = Path(__file__).resolve().parents[1] / "competition" / "agents" / "ppo_bot"


def load_bot():
    spec = importlib.util.spec_from_file_location("ppo_bot_agent", BOT_DIR / "agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ppo_bot_agent"] = module
    spec.loader.exec_module(module)
    return module


bot = load_bot()


def wire_roundtrip(obs):
    lines = protocol.encode_observation(obs).strip().split("\n")
    h, w = obs.armies.shape
    scalars = [int(x) for x in lines[0].split()]
    grids = []
    at = 1
    for _ in range(3):
        grids.append([[int(x) for x in lines[at + r].split()] for r in range(h)])
        at += h
    return SimpleNamespace(
        H=h, W=w, turn=scalars[0], my_land=scalars[1], my_army=scalars[2],
        opp_land=scalars[3], opp_army=scalars[4],
        type_grid=grids[0], owner_grid=grids[1], army_grid=grids[2],
    )


def advance(env, pool, steps, seed):
    state = jax.tree.map(lambda x: x[seed % pool.armies.shape[0]], pool)
    key = jrandom.PRNGKey(seed)
    for _ in range(steps):
        key, k0, k1 = jrandom.split(key, 3)
        moves = []
        for p, k in ((0, k0), (1, k1)):
            obs = game.get_observation(state, p)
            m = jax_mask(obs)
            h, w = obs.armies.shape
            moves.append(jax_decode(jrandom.choice(k, num_actions(h, w), p=m / m.sum()), h, w))
        _, state = env.step(state, jnp.stack(moves), pool)
    return state


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("steps", [0, 45])
def test_numpy_features_match_jax(seed, steps):
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = advance(env, pool, steps, seed) if steps else jax.tree.map(lambda x: x[seed], pool)

    for player in (0, 1):
        obs = game.get_observation(state, player)
        belief = update_belief(init_belief(obs.armies.shape), obs)
        belief = record_move(belief, jnp.array([0, 3, 4, 1, 0], jnp.int32))
        planes_jax, globals_jax = jax_extract(obs, belief, player)
        mask_jax = jax_mask(obs)

        frame = bot.parse_frame(wire_roundtrip(obs))
        np_belief = bot.Belief(obs.armies.shape)
        np_belief.update(frame)
        np_belief.record_move((0, 3, 4, 1, 0))
        planes_np, globals_np = bot.extract(frame, np_belief, player)
        mask_np = bot.action_mask(frame)

        assert bot.PLANE_NAMES == PLANE_NAMES, (
            "bot PLANES drifted from training/features.py PLANES:\n"
            f"  training: {PLANE_NAMES}\n  bot:      {bot.PLANE_NAMES}"
        )
        assert bot.GLOBAL_NAMES == GLOBAL_NAMES, "bot GLOBALS drifted from training/features.py GLOBALS"

        pj, pn = np.asarray(planes_jax), planes_np
        for i, name in enumerate(PLANE_NAMES):
            worst = float(np.abs(pj[:, :, i] - pn[:, :, i]).max())
            assert worst < 1e-5, f"plane {name!r} (index {i}) differs by {worst:.3g} between JAX and NumPy"
        for i, name in enumerate(GLOBAL_NAMES):
            worst = float(abs(np.asarray(globals_jax)[i] - globals_np[i]))
            assert worst < 1e-5, f"global {name!r} (index {i}) differs by {worst:.3g}"
        np.testing.assert_array_equal(np.asarray(mask_jax), mask_np)


def test_numpy_forward_matches_flax(tmp_path):
    height = width = 22
    channels, blocks = 16, 2
    network = ActorCritic(channels=channels, blocks=blocks, groups=8, value_hidden=32)
    key = jrandom.PRNGKey(0)
    planes = jrandom.normal(key, (height, width, NUM_PLANES)) * 0.3
    globals_ = jrandom.normal(jrandom.PRNGKey(1), (NUM_GLOBALS,)) * 0.3
    mask = jrandom.bernoulli(jrandom.PRNGKey(2), 0.3, (num_actions(height, width),))
    mask = mask.at[-1].set(True)

    params = network.init(jrandom.PRNGKey(3), planes, globals_, mask)
    logits_jax, _ = network.apply(params, planes, globals_, mask)

    flat = {}

    def walk(tree, prefix=""):
        for k, v in tree.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}/")
            else:
                flat[f"{prefix}{k}"] = np.asarray(v, np.float32)

    walk(jax.tree.map(np.asarray, params)["params"])
    flat["meta/blocks"] = np.asarray(blocks, np.float32)
    flat["meta/channels"] = np.asarray(channels, np.float32)
    path = tmp_path / "weights.npz"
    np.savez(path, **flat)

    net_np = bot.Network(str(path))
    logits_np = net_np(np.asarray(planes), np.asarray(globals_), np.asarray(mask))

    valid = np.asarray(mask)
    np.testing.assert_allclose(
        np.asarray(logits_jax)[valid], logits_np[valid], atol=2e-3, rtol=2e-3
    )
    assert int(np.argmax(np.asarray(logits_jax))) == int(np.argmax(logits_np))


@pytest.mark.parametrize("seed", [0, 1])
def test_numpy_decode_matches_jax(seed):
    height, width = 22, 22
    indices = np.asarray(jrandom.randint(jrandom.PRNGKey(seed), (300,), 0, num_actions(height, width)))
    for idx in indices:
        expected = tuple(int(x) for x in jax_decode(int(idx), height, width))
        assert bot.decode(int(idx), height, width) == expected
