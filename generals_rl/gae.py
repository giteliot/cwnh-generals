"""Generalized Advantage Estimation, with correct truncation handling.

For every step t we have `bootstrap_values[t]`: the critic's value of the
real next state (before any pool auto-reset), regardless of whether the
episode ended there. This lets truncation and termination be treated
differently:

- termination (true game end): the value chain stops, next value is 0.
- truncation (time limit, episode continues logically): the value chain
  still bootstraps from the real next state's value, but the advantage
  recursion still breaks there, since the *stored* value at t+1 belongs to
  a different (reset) episode.
"""
import jax
import jax.numpy as jnp

from generals_rl.rollout import Rollout


def compute_gae(rollout: Rollout, gamma: float, gae_lambda: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    terminated = rollout.terminated.astype(jnp.float32)
    episode_end = jnp.logical_or(rollout.terminated, rollout.truncated).astype(jnp.float32)

    deltas = rollout.rewards + gamma * rollout.bootstrap_values * (1.0 - terminated) - rollout.values
    chain_continue = 1.0 - episode_end

    def body(next_advantage, x):
        delta, chain = x
        advantage = delta + gamma * gae_lambda * chain * next_advantage
        return advantage, advantage

    _, advantages_reversed = jax.lax.scan(
        body, jnp.zeros_like(deltas[0]), (deltas[::-1], chain_continue[::-1])
    )
    advantages = advantages_reversed[::-1]
    returns = advantages + rollout.values
    return advantages, returns
