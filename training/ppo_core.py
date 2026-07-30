from typing import NamedTuple

import jax
import jax.numpy as jnp

from .network import entropy, log_probs


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    planes: jnp.ndarray
    globals_: jnp.ndarray
    mask: jnp.ndarray


def quantize(planes):
    return jnp.round(jnp.clip(planes, -1.0, 1.0) * 127.0).astype(jnp.int8)


def dequantize(planes):
    return planes.astype(jnp.float32) / 127.0


def compute_gae(traj, last_value, gamma, lam):
    def body(carry, transition):
        gae, next_value = carry
        alive = 1.0 - transition.done
        delta = transition.reward + gamma * next_value * alive - transition.value
        gae = delta + gamma * lam * alive * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        body,
        (jnp.zeros_like(last_value), last_value),
        traj,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj.value


def make_update(apply_fn, config):
    def loss_fn(params, traj, gae, targets):
        logits, value = jax.vmap(apply_fn, in_axes=(None, 0, 0, 0))(
            params, dequantize(traj.planes), traj.globals_, traj.mask
        )
        lp = jax.vmap(log_probs)(logits)
        log_prob = jnp.take_along_axis(lp, traj.action[:, None], axis=1)[:, 0]

        clipped_value = traj.value + jnp.clip(
            value - traj.value, -config["CLIP_EPS"], config["CLIP_EPS"]
        )
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - targets), jnp.square(clipped_value - targets)
        ).mean()

        ratio = jnp.exp(log_prob - traj.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        actor_loss = -jnp.minimum(
            ratio * gae,
            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
        ).mean()

        ent = jax.vmap(entropy)(logits).mean()
        total = actor_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * ent
        return total, (value_loss, actor_loss, ent)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    def update_minibatch(train_state, batch):
        traj, advantages, targets = batch
        (total, aux), grads = grad_fn(train_state.params, traj, advantages, targets)
        return train_state.apply_gradients(grads=grads), (total, *aux)

    def update_epoch(carry, _):
        train_state, traj, advantages, targets, key = carry
        key, subkey = jax.random.split(key)
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        permutation = jax.random.permutation(subkey, batch_size)
        batch = (traj, advantages, targets)
        batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
        batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
        minibatches = jax.tree.map(
            lambda x: x.reshape((config["NUM_MINIBATCHES"], -1) + x.shape[1:]), batch
        )
        train_state, info = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (train_state, traj, advantages, targets, key), info

    return update_epoch
