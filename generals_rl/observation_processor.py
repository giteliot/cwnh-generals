"""Observation processing layer: raw Observation -> tensor for the actor/critic.

Thin wrapper around Observation.as_tensor() for now. Kept as a separate,
pure function so the representation can be swapped later (normalization,
feature toggles, a different encoding) without touching the env adapter or PPO.
"""
import jax.numpy as jnp

from generals.core.observation import Observation


def process_observation(observation: Observation) -> jnp.ndarray:
    return observation.as_tensor()
