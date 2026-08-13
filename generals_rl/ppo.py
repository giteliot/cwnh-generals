"""PPO loss and minibatch update."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct

from generals_rl.network import ActorCritic, evaluate_action


@struct.dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5


class TrainState(NamedTuple):
    params: object
    opt_state: object


def create_train_state(model: ActorCritic, params, config: PPOConfig) -> TrainState:
    return TrainState(params=params, opt_state=_optimizer(config).init(params))


def _optimizer(config: PPOConfig):
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )


def ppo_loss(
    params,
    model: ActorCritic,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    config: PPOConfig,
) -> tuple[jnp.ndarray, dict]:
    log_probs, entropy, values = jax.vmap(evaluate_action, in_axes=(None, None, 0, 0))(
        model, params, observations, actions
    )

    ratio = jnp.exp(log_probs - old_log_probs)
    normalized_advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    unclipped = ratio * normalized_advantages
    clipped = jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * normalized_advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

    value_loss = jnp.mean((values - returns) ** 2)
    entropy_loss = -jnp.mean(entropy)

    loss = policy_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": -entropy_loss,
    }
    return loss, metrics


def _update_minibatch(train_state: TrainState, batch, model: ActorCritic, config: PPOConfig) -> tuple[TrainState, dict]:
    observations, actions, old_log_probs, advantages, returns = batch

    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    (_, metrics), grads = grad_fn(
        train_state.params, model, observations, actions, old_log_probs, advantages, returns, config
    )

    updates, new_opt_state = _optimizer(config).update(grads, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    return TrainState(params=new_params, opt_state=new_opt_state), metrics


def update(
    train_state: TrainState,
    model: ActorCritic,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    key: jnp.ndarray,
    config: PPOConfig,
) -> tuple[TrainState, dict]:
    """Run num_epochs passes over the rollout in minibatches. Returns (train_state, metrics)."""
    num_samples = observations.shape[0]
    num_minibatches = num_samples // config.minibatch_size

    def epoch_fn(carry, epoch_key):
        train_state = carry
        perm = jax.random.permutation(epoch_key, num_samples)

        def minibatch_fn(train_state, minibatch_idx):
            batch_indices = jax.lax.dynamic_slice_in_dim(perm, minibatch_idx * config.minibatch_size, config.minibatch_size)
            batch = (
                observations[batch_indices],
                actions[batch_indices],
                old_log_probs[batch_indices],
                advantages[batch_indices],
                returns[batch_indices],
            )
            return _update_minibatch(train_state, batch, model, config)

        train_state, metrics = jax.lax.scan(minibatch_fn, train_state, jnp.arange(num_minibatches))
        return train_state, metrics

    epoch_keys = jax.random.split(key, config.num_epochs)
    train_state, all_metrics = jax.lax.scan(epoch_fn, train_state, epoch_keys)

    mean_metrics = jax.tree.map(jnp.mean, all_metrics)
    return train_state, mean_metrics
