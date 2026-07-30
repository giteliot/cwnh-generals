import pickle

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.agent import Agent

from .actions import decode
from .features import extract, init_belief, update_belief
from .masking import action_mask
from .network import ActorCritic


class PPOAgent(Agent):
    def __init__(self, checkpoint, id="PPO", player_idx=0, temperature=0.0):
        super().__init__(id)
        with open(checkpoint, "rb") as fh:
            blob = pickle.load(fh)
        self.params = blob["params"]
        self.network = ActorCritic(channels=blob["channels"], blocks=blob["blocks"], groups=blob.get("groups", 8))
        self.player_idx = player_idx
        self.temperature = temperature
        self.belief = None
        self._forward = jax.jit(self.network.apply)

    def reset(self):
        self.belief = None

    def act(self, observation, key):
        if self.belief is None:
            self.belief = init_belief(observation.armies.shape)
        self.belief = update_belief(self.belief, observation)

        planes, globals_ = extract(observation, self.belief, self.player_idx)
        mask = action_mask(observation)
        logits, _ = self._forward(self.params, planes, globals_, mask)

        if self.temperature > 0:
            index = jrandom.categorical(key, logits / self.temperature)
        else:
            index = jnp.argmax(logits)

        height, width = observation.armies.shape
        return decode(index, height, width)
