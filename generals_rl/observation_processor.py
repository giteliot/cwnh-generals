"""Observation processing layer: raw Observation -> tensor for the actor/critic.

Kept as a separate, pure function so the representation can be swapped later
without touching the env adapter or PPO.

Normalization is not cosmetic here, it is required for stability. The raw
observation is unnormalized: army/land counts run to the hundreds and the
timestep channel runs to the truncation limit (1200 in competition mode),
while the critic's target is a discounted terminal reward bounded by |V| <= 1.
A network fed inputs of magnitude ~10^2-10^3 produces outputs on that same
scale, so each optimizer step moves the value prediction far past its target
and the critic diverges instead of converging. Counts are therefore
log-compressed (they span orders of magnitude) and the timestep is mapped to
roughly [0, 1]. The boolean mask channels are already 0/1 and pass through
unchanged.

Channel order is defined by Observation.as_tensor().
"""
import jax.numpy as jnp

from generals.core.observation import Observation

# armies, owned/opponent land and army counts — unbounded, span orders of magnitude.
_COUNT_CHANNELS = jnp.array([0, 9, 10, 11, 12])
_TIMESTEP_CHANNEL = 13


def process_observation(observation: Observation, max_timestep: float) -> jnp.ndarray:
    tensor = observation.as_tensor().astype(jnp.float32)
    tensor = tensor.at[_COUNT_CHANNELS].set(jnp.log1p(tensor[_COUNT_CHANNELS]))
    tensor = tensor.at[_TIMESTEP_CHANNEL].set(tensor[_TIMESTEP_CHANNEL] / max_timestep)
    return tensor
