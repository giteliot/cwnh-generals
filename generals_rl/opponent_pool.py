"""Checkpoint pool management with win-rate-gated promotion (arXiv:2507.06825).

The pool is a plain Python list of param pytrees, managed at the outer
training-loop level (not inside any jitted/scanned code) since promotion
checks only happen every `pool_update_frequency` iterations. It's stacked
into a single batched param pytree (for vmap'd opponent dispatch) on demand.
"""
import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.core.env import GeneralsEnv
from generals_rl.env_adapter import RLEnvAdapter
from generals_rl.network import ActorCritic, select_greedy_action
from generals_rl.observation_processor import process_observation


def create_pool(initial_params) -> list:
    return [initial_params]


def stack_pool(pool: list):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pool)


def _play_policy_vs_policy(
    env: GeneralsEnv, model: ActorCritic, params_a, params_b, key: jnp.ndarray
) -> int:
    """One greedy-vs-greedy game, params_a as player 0. Returns winner (0, 1, or -1 for draw)."""
    adapter = RLEnvAdapter(env)
    observation, adapter_state, pool = adapter.reset(key)

    done = False
    info = None
    while not done:
        action_a = select_greedy_action(model, params_a, observation)
        raw_obs_b = adapter._get_obs(adapter_state.state, 1)
        action_b = select_greedy_action(model, params_b, process_observation(raw_obs_b))
        observation, info, done, adapter_state = adapter.step(adapter_state, action_a, action_b, pool)

    return int(info["game_info"].winner)


def win_rate_vs_pool(
    env: GeneralsEnv, model: ActorCritic, candidate_params, pool: list, num_games: int, key: jnp.ndarray
) -> float:
    """Candidate's win rate playing greedy games against opponents sampled uniformly from the pool."""
    wins = 0
    for _ in range(num_games):
        key, opponent_key, game_key = jrandom.split(key, 3)
        opponent_params = pool[int(jrandom.randint(opponent_key, (), 0, len(pool)))]
        winner = _play_policy_vs_policy(env, model, candidate_params, opponent_params, game_key)
        if winner == 0:
            wins += 1
    return wins / num_games


def maybe_promote(
    env: GeneralsEnv,
    model: ActorCritic,
    pool: list,
    candidate_params,
    pool_size: int,
    win_rate_threshold: float,
    num_eval_games: int,
    key: jnp.ndarray,
) -> tuple[list, bool, float]:
    """Evaluate candidate against the pool; if it wins >= threshold, add it (evicting the oldest if full).

    Returns (new_pool, promoted, win_rate).
    """
    win_rate = win_rate_vs_pool(env, model, candidate_params, pool, num_eval_games, key)
    if win_rate < win_rate_threshold:
        return pool, False, win_rate

    new_pool = pool + [candidate_params]
    if len(new_pool) > pool_size:
        new_pool = new_pool[1:]
    return new_pool, True, win_rate
