import jax.numpy as jnp

from generals.core.action import compute_valid_move_mask

from .features import build_cost


def action_mask(obs) -> jnp.ndarray:
    blocked = obs.mountains | obs.structures_in_fog
    moves = compute_valid_move_mask(obs.armies, obs.owned_cells, blocked)

    own_structures = (obs.castles | obs.generals) & obs.owned_cells
    builds = (
        obs.owned_cells
        & ~obs.generals
        & ~obs.castles
        & (obs.armies >= build_cost(own_structures))
    )

    per_cell = jnp.concatenate([moves, moves, builds[:, :, None]], axis=-1)
    flat = jnp.transpose(per_cell, (2, 0, 1)).reshape(-1)
    return jnp.concatenate([flat, jnp.ones((1,), bool)])
