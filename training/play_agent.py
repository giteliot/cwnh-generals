import pickle

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.agent import Agent

from .actions import decode
from .features import PLANE_NAMES, extract, init_belief, record_move, update_belief
from .masking import action_mask
from .network import ActorCritic


class PPOAgent(Agent):
    def __init__(self, checkpoint, id="PPO", player_idx=0, temperature=0.0):
        super().__init__(id)
        with open(checkpoint, "rb") as fh:
            blob = pickle.load(fh)
        saved = blob.get("planes")
        if saved is not None and saved != PLANE_NAMES:
            raise SystemExit(
                f"{checkpoint} was trained with {len(saved)} observation planes, the code now "
                f"builds {len(PLANE_NAMES)}. Observation changes break old checkpoints -- "
                "`git stash` back to the code it was trained with, or retrain."
            )
        self.params = blob["params"]
        self.network = ActorCritic(channels=blob["channels"], blocks=blob["blocks"], groups=blob.get("groups", 8))
        self.player_idx = player_idx
        self.temperature = temperature
        self.belief = None

        # Jit the WHOLE turn, not just the network. update_belief/extract/
        # action_mask are plain jnp, and extract runs three BFS passes as
        # lax.fori_loop -- eagerly that is ~130 dispatches per BFS, every turn.
        # A 10-game eval was taking >20 minutes with the GPU idle because it was
        # all launch overhead. Fused, it is one kernel launch per turn.
        def one_turn(params, obs, belief, player, key):
            belief = update_belief(belief, obs)
            planes, globals_ = extract(obs, belief, player)
            logits, _ = self.network.apply(params, planes, globals_, action_mask(obs))
            # temperature is fixed at construction, so branch in Python
            index = (jrandom.categorical(key, logits / self.temperature)
                     if self.temperature > 0 else jnp.argmax(logits))
            move = decode(index, *obs.armies.shape)
            return record_move(belief, move), move

        self._act = jax.jit(one_turn)

    def reset(self):
        self.belief = None

    def act(self, observation, key):
        if self.belief is None:
            self.belief = init_belief(observation.armies.shape)
        self.belief, move = self._act(
            self.params, observation, self.belief, self.player_idx, key)
        return move
