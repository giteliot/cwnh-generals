"""Checkpoint pool management with win-rate-gated promotion (arXiv:2507.06825).

The pool is a plain Python list of param pytrees, managed at the outer
training-loop level (not inside any jitted/scanned code) since promotion
checks only happen every `pool_update_frequency` iterations. It's stacked
into a single batched param pytree (for vmap'd opponent dispatch) on demand.

Win rate against the pool is estimated from a rolling window of outcomes
already produced by normal training rollouts (see train.py's pool_outcomes
buffer) — not from a dedicated evaluation loop. A dedicated loop would play
games one unbatched step at a time with no vmap/jit, which is drastically
slower than the vectorized rollout collection already running every
iteration and produces no information the rollouts don't already have.
"""
import jax
import jax.numpy as jnp


def create_pool(initial_params) -> list:
    return [initial_params]


def stack_pool(pool: list):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pool)


def maybe_promote(
    pool: list,
    candidate_params,
    pool_size: int,
    win_rate_threshold: float,
    win_rate: float,
) -> tuple[list, bool]:
    """If win_rate clears the threshold, add candidate_params to the pool
    (evicting the oldest if full). Returns (new_pool, promoted)."""
    if win_rate < win_rate_threshold:
        return pool, False

    new_pool = pool + [candidate_params]
    if len(new_pool) > pool_size:
        new_pool = new_pool[1:]
    return new_pool, True
