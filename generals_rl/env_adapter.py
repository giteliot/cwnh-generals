"""Single-agent RL adapter around the JAX GeneralsEnv.

Exposes player 0 as the learning agent. Player 1's action is supplied
externally by the caller (step() takes it as an argument) — opponent
selection is a separate component (generals_rl.opponents), not the
adapter's concern.
"""
import jax
import jax.numpy as jnp
from flax import struct

from generals.core import game
from generals.core.env import GeneralsEnv
from generals.core.game import GameState
from generals.core.observation import Observation
from generals_rl.observation_processor import process_observation
from generals_rl.reward import shaped_reward


@struct.dataclass
class AdapterState:
    state: GameState


class RLEnvAdapter:
    def __init__(
        self,
        env: GeneralsEnv,
        terminal_win_reward: float = 1.0,
        terminal_loss_reward: float = -1.0,
        terminal_draw_reward: float = 0.0,
        shaping_alpha: float = 0.0,
        gamma: float = 0.99,
        shaping_max_ratio: float = 10.0,
    ):
        self.env = env
        # Scale for the timestep observation channel (see observation_processor).
        self.max_timestep = float(env.truncation)
        self.terminal_win_reward = terminal_win_reward
        self.terminal_loss_reward = terminal_loss_reward
        self.terminal_draw_reward = terminal_draw_reward
        self.shaping_alpha = shaping_alpha
        self.gamma = gamma
        self.shaping_max_ratio = shaping_max_ratio

    def _get_obs(self, state: GameState, player: int) -> Observation:
        get_obs = game.get_full_observation if self.env.perfect_info else game.get_observation
        return get_obs(state, player)

    def reset(self, key: jnp.ndarray) -> tuple[jnp.ndarray, AdapterState, GameState]:
        """Generate a fresh pool and a single initial state. Returns (observation, adapter_state, pool).

        The pool must be kept by the caller and passed to step(); for
        multiple parallel environments, generate the pool once and use
        reset_with_pool() per environment so they share it (see parallel.py).
        """
        pool, init_state = self.env.reset(key)
        observation = process_observation(self._get_obs(init_state, 0), self.max_timestep)
        return observation, AdapterState(state=init_state), pool

    def reset_with_pool(self, key: jnp.ndarray, pool: GameState) -> tuple[jnp.ndarray, AdapterState]:
        """Generate a single initial state using an existing (shared) pool."""
        init_state = self.env.init_state(key)
        observation = process_observation(self._get_obs(init_state, 0), self.max_timestep)
        return observation, AdapterState(state=init_state)

    def step(
        self,
        adapter_state: AdapterState,
        action: jnp.ndarray,
        opponent_action: jnp.ndarray,
        pool: GameState,
    ) -> tuple[jnp.ndarray, dict, jnp.ndarray, AdapterState]:
        actions = jnp.stack([action, opponent_action])
        timestep, new_state = self.env.step(adapter_state.state, actions, pool)

        observation = process_observation(
            jax.tree.map(lambda x: x[0], timestep.observation), self.max_timestep
        )
        done = timestep.terminated | timestep.truncated
        reward = shaped_reward(
            state=adapter_state.state,
            next_state=timestep.last_state,
            winner=timestep.info.winner,
            player=0,
            alpha=self.shaping_alpha,
            gamma=self.gamma,
            terminal_win_reward=self.terminal_win_reward,
            terminal_loss_reward=self.terminal_loss_reward,
            terminal_draw_reward=self.terminal_draw_reward,
            max_ratio=self.shaping_max_ratio,
        )
        info = {
            "reward": reward,
            "game_info": timestep.info,
            "terminated": timestep.terminated,
            "truncated": timestep.truncated,
            "last_state": timestep.last_state,
        }

        return observation, info, done, AdapterState(state=new_state)
