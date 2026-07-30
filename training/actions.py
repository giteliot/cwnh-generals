import jax.numpy as jnp

NUM_CELL_ACTIONS = 9
MOVE_KINDS = 8
BUILD_KIND = 8


def num_actions(height, width):
    return height * width * NUM_CELL_ACTIONS + 1


def decode(index, height, width):
    cells = height * width
    is_pass = index == cells * NUM_CELL_ACTIONS
    kind = index // cells
    cell = index % cells
    row = cell // width
    col = cell % width
    is_build = kind == BUILD_KIND
    split = (kind >= 4) & (kind < MOVE_KINDS)
    direction = jnp.where(is_build, 0, kind % 4)
    head = jnp.where(is_pass, 1, jnp.where(is_build, 2, 0))
    return jnp.stack([
        head,
        jnp.where(is_pass, 0, row),
        jnp.where(is_pass, 0, col),
        direction,
        jnp.where(split, 1, 0),
    ]).astype(jnp.int32)


def encode(action, height, width):
    head, row, col, direction, split = action
    cells = height * width
    kind = jnp.where(head == 2, BUILD_KIND, direction + 4 * split)
    return jnp.where(head == 1, cells * NUM_CELL_ACTIONS, kind * cells + row * width + col)
