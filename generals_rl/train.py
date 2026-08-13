"""Training loop (Phase 10): collect rollout -> GAE -> PPO update -> repeat.

Runs config.ppo.num_envs environments in parallel via jax.vmap (Phase 9), and
mixes self-play against a checkpoint pool with scripted bots (Phase 14).
"""
import time
from collections import deque

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from generals.core.env import GeneralsEnv
from generals_rl import checkpoint as checkpoint_lib
from generals_rl.config import Config
from generals_rl.env_adapter import RLEnvAdapter
from generals_rl.gae import compute_gae
from generals_rl.network import ActorCritic
from generals_rl.opponent_pool import create_pool, maybe_promote, stack_pool
from generals_rl.opponents import KIND_NAMES, POLICY_KIND, sample_opponent_assignment
from generals_rl.parallel import reset_parallel
from generals_rl.ppo import PPOConfig, create_train_state, update
from generals_rl.rollout import Rollout, collect_rollout


def _ppo_config_from(config: Config) -> PPOConfig:
    return PPOConfig(
        learning_rate=config.ppo.learning_rate,
        clip_coef=config.ppo.clip_coef,
        entropy_coef=config.ppo.entropy_coef,
        value_coef=config.ppo.value_coef,
        num_epochs=config.ppo.num_epochs,
        minibatch_size=config.ppo.minibatch_size,
        max_grad_norm=config.ppo.max_grad_norm,
    )


def _train_step(
    adapter,
    model,
    ppo_config,
    rollout_length,
    gamma,
    gae_lambda,
    train_state,
    adapter_state,
    observation,
    pool,
    opponent_pool_params,
    opponent_kinds,
    opponent_checkpoint_idx,
    key,
):
    """One rollout-collection + GAE + PPO-update step. adapter/model/ppo_config/rollout_length
    must be static (hashable, constant across calls) so this compiles once and is reused —
    without jax.jit here, every iteration would fully retrace and recompile from scratch."""
    rollout_key, update_key = jrandom.split(key)

    rollout, adapter_state, observation, _ = collect_rollout(
        adapter, model, train_state.params, adapter_state, observation, pool,
        opponent_pool_params, opponent_kinds, opponent_checkpoint_idx,
        rollout_key, rollout_length,
    )
    advantages, returns = compute_gae(rollout, gamma, gae_lambda)

    flatten = lambda x: x.reshape(-1, *x.shape[2:])
    train_state, metrics = update(
        train_state,
        model,
        flatten(rollout.observations),
        flatten(rollout.valid_masks),
        flatten(rollout.actions),
        flatten(rollout.log_probs),
        flatten(advantages),
        flatten(returns),
        update_key,
        ppo_config,
    )

    return train_state, adapter_state, observation, rollout, metrics


_jit_train_step = jax.jit(_train_step, static_argnums=(0, 1, 2, 3))


def _opponent_breakdown(rollout: Rollout, opponent_kinds: jnp.ndarray) -> dict[str, tuple[int, int, int]]:
    """Win/loss/draw counts from this rollout, broken down by opponent kind."""
    done = np.asarray(rollout.terminated | rollout.truncated)  # (T, N)
    winner = np.asarray(rollout.winner)  # (T, N)
    kinds = np.broadcast_to(np.asarray(opponent_kinds), done.shape)  # (T, N)

    stats = {}
    for kind_idx, name in enumerate(KIND_NAMES):
        mask = done & (kinds == kind_idx)
        total = int(mask.sum())
        if total == 0:
            continue
        wins = int((mask & (winner == 0)).sum())
        losses = int((mask & (winner == 1)).sum())
        draws = total - wins - losses
        stats[name] = (wins, losses, draws)
    return stats


def _pool_game_outcomes(rollout: Rollout, opponent_kinds: jnp.ndarray) -> list[float]:
    """1.0/0.0 (win/not-win) for each pool-opponent game that finished this rollout."""
    done = np.asarray(rollout.terminated | rollout.truncated)
    winner = np.asarray(rollout.winner)
    kinds = np.broadcast_to(np.asarray(opponent_kinds), done.shape)

    mask = done & (kinds == POLICY_KIND)
    wins = int((mask & (winner == 0)).sum())
    non_wins = int(mask.sum()) - wins
    return [1.0] * wins + [0.0] * non_wins


def train(
    env: GeneralsEnv,
    config: Config,
    checkpoint_dir: str | None = None,
    resume_from: int | None = None,
    num_iterations: int | None = None,
):
    """Train. `num_iterations`, if given, overrides config.training.total_timesteps-derived
    iteration count with an explicit count run from start_iteration — used for curriculum
    stages, where each stage runs a fixed number of additional iterations rather than
    training up to a fixed cumulative timestep count."""
    if resume_from is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required when resume_from is set")

    adapter = RLEnvAdapter(
        env,
        terminal_win_reward=config.reward.terminal_win_reward,
        terminal_loss_reward=config.reward.terminal_loss_reward,
        terminal_draw_reward=config.reward.terminal_draw_reward,
        shaping_alpha=config.reward.shaping_alpha,
        gamma=config.ppo.gamma,
        shaping_max_ratio=config.reward.shaping_max_ratio,
    )

    key = jrandom.PRNGKey(config.training.seed)
    key, reset_key, init_key = jrandom.split(key, 3)
    observation, adapter_state, pool = reset_parallel(adapter, reset_key, config.ppo.num_envs)

    model = ActorCritic(num_channels=config.network.num_channels)
    params = model.init(init_key, observation[0])
    ppo_config = _ppo_config_from(config)
    train_state = create_train_state(model, params, ppo_config)

    start_iteration = 0
    if resume_from is not None:
        train_state, key = checkpoint_lib.restore(checkpoint_dir, resume_from, train_state, key)
        start_iteration = resume_from + 1

    # Note: the checkpoint pool itself isn't persisted/restored (Phase 12 only
    # saves params/opt_state/key) — resuming restarts the pool from the
    # resumed params alone, losing prior pool history.
    opponent_pool = create_pool(train_state.params)
    pool_outcomes = deque(maxlen=config.opponent.pool_win_rate_window)

    steps_per_iteration = config.ppo.rollout_length * config.ppo.num_envs
    if num_iterations is not None:
        end_iteration = start_iteration + num_iterations
    else:
        end_iteration = config.training.total_timesteps // steps_per_iteration

    for iteration in range(start_iteration, end_iteration):
        iteration_start_time = time.time()
        key, rollout_key, opponent_assign_key = jrandom.split(key, 3)

        opponent_pool_params = stack_pool(opponent_pool)
        opponent_kinds, opponent_checkpoint_idx = sample_opponent_assignment(
            opponent_assign_key,
            config.ppo.num_envs,
            config.opponent.pool_checkpoint_weight,
            config.opponent.scripted_bot_weights,
            len(opponent_pool),
        )

        train_state, adapter_state, observation, rollout, metrics = _jit_train_step(
            adapter, model, ppo_config, config.ppo.rollout_length,
            config.ppo.gamma, config.ppo.gae_lambda,
            train_state, adapter_state, observation, pool,
            opponent_pool_params, opponent_kinds, opponent_checkpoint_idx,
            rollout_key,
        )
        pool_outcomes.extend(_pool_game_outcomes(rollout, opponent_kinds))

        if iteration % config.training.log_frequency == 0:
            # JAX dispatches asynchronously; block on the actual result before timing.
            metrics = jax.block_until_ready(metrics)
            elapsed = time.time() - iteration_start_time
            steps_per_sec = steps_per_iteration / elapsed

            num_episodes = int(jnp.sum(rollout.terminated | rollout.truncated))
            mean_reward = float(jnp.mean(rollout.rewards))
            breakdown = _opponent_breakdown(rollout, opponent_kinds)
            breakdown_str = " | ".join(f"{name} {w}/{l}/{d}" for name, (w, l, d) in breakdown.items())

            print(
                f"iter {iteration} | steps {iteration * steps_per_iteration} | "
                f"{elapsed:.2f}s ({steps_per_sec:.0f} steps/s) | "
                f"episodes {num_episodes} | mean_reward {mean_reward:.4f} | "
                + " | ".join(f"{k} {float(v):.4f}" for k, v in metrics.items())
            )
            if breakdown_str:
                print(f"  W/L/D by opponent | {breakdown_str}")

        if checkpoint_dir is not None and (iteration + 1) % config.training.checkpoint_frequency == 0:
            checkpoint_lib.save(checkpoint_dir, iteration, train_state, key, config)

        if (iteration + 1) % config.opponent.pool_update_frequency == 0:
            win_rate = sum(pool_outcomes) / len(pool_outcomes) if pool_outcomes else 0.0
            opponent_pool, promoted = maybe_promote(
                opponent_pool,
                train_state.params,
                config.opponent.pool_size,
                config.opponent.pool_win_rate_threshold,
                win_rate,
            )
            print(
                f"pool check | win_rate {win_rate:.2f} (n={len(pool_outcomes)}) | "
                f"promoted {promoted} | pool_size {len(opponent_pool)}"
            )

        # Evaluation (Phase 13) goes here, gated on config.training.evaluation_frequency.
        # Not yet wired into the loop — see generals_rl/evaluate.py for the standalone version.

    if checkpoint_dir is not None and end_iteration > start_iteration:
        # Always checkpoint at the end of a run/stage, regardless of checkpoint_frequency
        # alignment, so a caller (e.g. curriculum training) has a well-defined step to
        # resume_from for the next stage.
        checkpoint_lib.save(checkpoint_dir, end_iteration - 1, train_state, key, config)

    return train_state
