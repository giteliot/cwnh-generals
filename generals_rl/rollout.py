"""Rollout collection via jax.lax.scan, vectorized over num_envs with jax.vmap.

num_envs=1 is just the degenerate case of the same batched code path.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import GameState
from generals_rl.env_adapter import AdapterState, RLEnvAdapter
from generals_rl.network import ActorCritic, get_value, select_action
from generals_rl.observation_processor import process_observation
from generals_rl.opponents import mixed_opponent_action
from generals_rl.parallel import step_parallel


class Rollout(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    bootstrap_values: jnp.ndarray


def collect_rollout(
    adapter: RLEnvAdapter,
    model: ActorCritic,
    params,
    adapter_state: AdapterState,
    observation: jnp.ndarray,
    pool: GameState,
    opponent_pool_params,
    opponent_kinds: jnp.ndarray,
    opponent_checkpoint_idx: jnp.ndarray,
    key: jnp.ndarray,
    rollout_length: int,
) -> tuple[Rollout, AdapterState, jnp.ndarray, jnp.ndarray]:
    """Collect `rollout_length` steps for a batch of num_envs environments.

    adapter_state/observation are batched over num_envs (leading axis); pool
    is shared across envs (not batched). Every field of the returned Rollout
    has shape (rollout_length, num_envs, ...).

    opponent_kinds/opponent_checkpoint_idx assign one opponent per env, held
    fixed for the whole rollout (see generals_rl.opponents).

    Returns (rollout, final_adapter_state, final_observation, final_key).
    """
    num_envs = observation.shape[0]

    def step_fn(carry, _):
        adapter_state, observation, key = carry
        key, action_key, opponent_key = jrandom.split(key, 3)
        action_keys = jrandom.split(action_key, num_envs)
        opponent_keys = jrandom.split(opponent_key, num_envs)

        action, log_prob, value = jax.vmap(select_action, in_axes=(None, None, 0, 0))(
            model, params, observation, action_keys
        )

        opponent_observation = jax.vmap(lambda state: adapter._get_obs(state, 1))(adapter_state.state)
        opponent_action = mixed_opponent_action(
            model, opponent_pool_params, opponent_kinds, opponent_checkpoint_idx, opponent_observation, opponent_keys
        )

        next_observation, info, done, next_adapter_state = step_parallel(
            adapter, adapter_state, action, opponent_action, pool
        )

        bootstrap_obs = jax.vmap(lambda state: process_observation(adapter._get_obs(state, 0)))(info["last_state"])
        bootstrap_value = jax.vmap(get_value, in_axes=(None, None, 0))(model, params, bootstrap_obs)

        transition = (
            observation,
            action,
            log_prob,
            value,
            info["reward"],
            info["terminated"],
            info["truncated"],
            bootstrap_value,
        )
        return (next_adapter_state, next_observation, key), transition

    (final_adapter_state, final_observation, final_key), transitions = jax.lax.scan(
        step_fn, (adapter_state, observation, key), None, length=rollout_length
    )

    rollout = Rollout(*transitions)
    return rollout, final_adapter_state, final_observation, final_key
