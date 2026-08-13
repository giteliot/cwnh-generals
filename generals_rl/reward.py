"""Reward layer: terminal reward + potential-based shaping.

Potential function follows "Artificial Generals Intelligence: Mastering
Generals.io with Reinforcement Learning" (arXiv:2507.06825):

    Phi(s) = 0.3 * Phi_land(s) + 0.3 * Phi_army(s) + 0.4 * Phi_castle(s)
    Phi_x(s) = log((x_agent + eps) / (x_enemy + eps)) / log(max_ratio)

    r_shaped = r_terminal + alpha * (gamma * Phi(s') - Phi(s))

Uses ground-truth GameState (not the fog-limited Observation) so shaping is
not blinded by fog of war; only the policy's input respects fog.
"""
import jax.numpy as jnp

from generals.core.game import GameState


def terminal_reward(
    winner: jnp.ndarray,
    player: int,
    win: float = 1.0,
    loss: float = -1.0,
    draw: float = 0.0,
) -> jnp.ndarray:
    opponent = 1 - player
    return jnp.where(winner == player, win, jnp.where(winner == opponent, loss, draw))


def _log_ratio(agent_count: jnp.ndarray, enemy_count: jnp.ndarray, max_ratio: float, eps: float) -> jnp.ndarray:
    return jnp.log((agent_count + eps) / (enemy_count + eps)) / jnp.log(max_ratio)


def potential(
    state: GameState,
    player: int,
    max_ratio: float = 10.0,
    eps: float = 1.0,
) -> jnp.ndarray:
    opponent = 1 - player

    agent_land = jnp.sum(state.ownership[player])
    enemy_land = jnp.sum(state.ownership[opponent])

    agent_army = jnp.sum(state.armies * state.ownership[player])
    enemy_army = jnp.sum(state.armies * state.ownership[opponent])

    agent_castles = jnp.sum(state.castles & state.ownership[player])
    enemy_castles = jnp.sum(state.castles & state.ownership[opponent])

    phi_land = _log_ratio(agent_land, enemy_land, max_ratio, eps)
    phi_army = _log_ratio(agent_army, enemy_army, max_ratio, eps)
    phi_castle = _log_ratio(agent_castles, enemy_castles, max_ratio, eps)

    return 0.3 * phi_land + 0.3 * phi_army + 0.4 * phi_castle


def shaped_reward(
    state: GameState,
    next_state: GameState,
    winner: jnp.ndarray,
    player: int,
    alpha: float,
    gamma: float,
    terminal_win_reward: float = 1.0,
    terminal_loss_reward: float = -1.0,
    terminal_draw_reward: float = 0.0,
    max_ratio: float = 10.0,
    eps: float = 1.0,
) -> jnp.ndarray:
    r_terminal = terminal_reward(
        winner, player, terminal_win_reward, terminal_loss_reward, terminal_draw_reward
    )
    phi_s = potential(state, player, max_ratio, eps)
    phi_s_next = potential(next_state, player, max_ratio, eps)
    return r_terminal + alpha * (gamma * phi_s_next - phi_s)
