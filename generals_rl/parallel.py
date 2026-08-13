"""Phase 9: run num_envs games in parallel via jax.vmap.

The state pool (used for cheap auto-reset) is generated once and shared
across all environments — each tracks its own pool_idx independently — so
memory cost doesn't scale with num_envs.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.game import GameState
from generals_rl.env_adapter import AdapterState, RLEnvAdapter


def reset_parallel(
    adapter: RLEnvAdapter, key: jnp.ndarray, num_envs: int
) -> tuple[jnp.ndarray, AdapterState, GameState]:
    """Returns (observations, adapter_states, pool), all batched over num_envs
    (pool is shared, not batched)."""
    pool_key, init_key = jrandom.split(key)
    pool, _ = adapter.env.reset(pool_key)

    init_keys = jrandom.split(init_key, num_envs)
    observations, adapter_states = jax.vmap(adapter.reset_with_pool, in_axes=(0, None))(init_keys, pool)
    return observations, adapter_states, pool


def step_parallel(
    adapter: RLEnvAdapter,
    adapter_states: AdapterState,
    actions: jnp.ndarray,
    opponent_actions: jnp.ndarray,
    pool: GameState,
) -> tuple[jnp.ndarray, dict, jnp.ndarray, AdapterState]:
    """vmap of adapter.step over num_envs; pool is shared (not batched)."""
    return jax.vmap(adapter.step, in_axes=(0, 0, 0, None))(adapter_states, actions, opponent_actions, pool)
