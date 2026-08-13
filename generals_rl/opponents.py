"""Phase 14: self-play opponent management.

Each parallel env is assigned one opponent "kind" for the whole rollout:
kinds 0-3 are the scripted bots, kind 4 is a policy checkpoint sampled from
a small pool of past training snapshots (mirrors the pool-based self-play
approach in arXiv:2507.06825, including greedy/argmax opponent play to
avoid teaching the learner to exploit a stochastic opponent).
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.expander_agent import ExpanderAgent
from generals.agents.harvester_agent import HarvesterAgent
from generals.agents.hunter_agent import HunterAgent
from generals.agents.random_agent import RandomAgent
from generals.core.observation import Observation
from generals_rl.network import ActorCritic, select_greedy_action
from generals_rl.observation_processor import process_observation

# Kind indices: 0-3 scripted bots, 4 = policy checkpoint from the pool.
_SCRIPTED_AGENTS = [RandomAgent(), ExpanderAgent(), HunterAgent(), HarvesterAgent()]
KIND_NAMES = ["random", "expander", "hunter", "harvester", "pool"]
POLICY_KIND = 4


def sample_opponent_assignment(
    key: jnp.ndarray,
    num_envs: int,
    pool_weight: float,
    scripted_weights: dict[str, float],
    num_checkpoints: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (kinds, checkpoint_idx), each shape (num_envs,).

    kinds: 0-3 scripted bot index, 4 = pool checkpoint.
    checkpoint_idx: which pool slot to use (only meaningful where kind == 4).
    """
    weights = jnp.array(
        [scripted_weights.get(name, 0.0) for name in KIND_NAMES[:4]] + [pool_weight]
    )
    probs = weights / jnp.sum(weights)

    kind_key, checkpoint_key = jrandom.split(key)
    kinds = jrandom.choice(kind_key, len(KIND_NAMES), shape=(num_envs,), p=probs)
    checkpoint_idx = jrandom.randint(checkpoint_key, (num_envs,), 0, num_checkpoints)
    return kinds, checkpoint_idx


def _per_env_action(
    model: ActorCritic,
    checkpoint_params,
    kind: jnp.ndarray,
    observation: Observation,
    key: jnp.ndarray,
) -> jnp.ndarray:
    branches = [
        lambda o, k: _SCRIPTED_AGENTS[0].act(o, k),
        lambda o, k: _SCRIPTED_AGENTS[1].act(o, k),
        lambda o, k: _SCRIPTED_AGENTS[2].act(o, k),
        lambda o, k: _SCRIPTED_AGENTS[3].act(o, k),
        lambda o, k: select_greedy_action(model, checkpoint_params, process_observation(o)),
    ]
    return jax.lax.switch(kind, branches, observation, key)


def mixed_opponent_action(
    model: ActorCritic,
    pool_params_stack,
    kinds: jnp.ndarray,
    checkpoint_idx: jnp.ndarray,
    observations: Observation,
    keys: jnp.ndarray,
) -> jnp.ndarray:
    """Batched opponent action over num_envs, dispatching per env by `kinds`.

    observations: raw (fog-limited) Observation for player 1, batched over num_envs.
    pool_params_stack: param pytree with a leading (pool_size,) axis.
    """
    own_checkpoint_params = jax.tree.map(lambda x: x[checkpoint_idx], pool_params_stack)
    return jax.vmap(_per_env_action, in_axes=(None, 0, 0, 0, 0))(
        model, own_checkpoint_params, kinds, observations, keys
    )

