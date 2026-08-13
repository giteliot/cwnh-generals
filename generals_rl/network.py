"""Actor/critic network and action sampling.

Shared conv encoder feeding an actor head (move logits over the H*W*4 grid,
masked to valid moves, plus a split head) and a critic head (V(s)).

The valid-move mask is passed in rather than derived from the processed
observation: the processor log-compresses army counts, while
compute_valid_move_mask thresholds raw armies (`armies > 1`), so the mask must
be computed from the raw Observation by the caller. This also keeps action
encoding independent of the observation representation.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax


class ActorCritic(nn.Module):
    num_channels: int = 32

    @nn.compact
    def __call__(self, processed_obs: jnp.ndarray):
        # processed_obs: (C, H, W) -> flax conv expects channels-last.
        x = jnp.moveaxis(processed_obs, -3, -1).astype(jnp.float32)

        x = nn.relu(nn.Conv(self.num_channels, (3, 3), padding="SAME")(x))
        x = nn.relu(nn.Conv(self.num_channels, (3, 3), padding="SAME")(x))
        shared = nn.relu(nn.Conv(self.num_channels, (3, 3), padding="SAME")(x))

        move_logits = nn.Conv(4, (1, 1), padding="SAME")(shared)  # (H, W, 4)

        pooled = jnp.mean(shared, axis=(-3, -2))
        split_logit = nn.Dense(1)(pooled).squeeze(-1)
        value = nn.Dense(1)(pooled).squeeze(-1)

        return move_logits, split_logit, value


def get_value(model: ActorCritic, params, processed_obs: jnp.ndarray) -> jnp.ndarray:
    _, _, value = model.apply(params, processed_obs)
    return value


def _masked_move_logits(move_logits: jnp.ndarray, valid_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (safe_logits, should_pass). valid_mask is (H, W, 4)."""
    flat_mask = valid_mask.reshape(-1)
    should_pass = jnp.sum(flat_mask) == 0

    flat_logits = move_logits.reshape(-1)
    masked_logits = jnp.where(flat_mask, flat_logits, -jnp.inf)
    # When no valid moves, masked_logits is all -inf; substitute a dummy
    # uniform distribution so softmax/categorical stay finite. The resulting
    # move is discarded anyway since `should_pass` forces a pass action.
    safe_logits = jnp.where(should_pass, jnp.zeros_like(masked_logits), masked_logits)
    return safe_logits, should_pass


def _bernoulli_entropy(logit: jnp.ndarray) -> jnp.ndarray:
    p = jax.nn.sigmoid(logit)
    log_p = jax.nn.log_sigmoid(logit)
    log_1_minus_p = jax.nn.log_sigmoid(-logit)
    return -(p * log_p + (1.0 - p) * log_1_minus_p)


def select_greedy_action(
    model: ActorCritic,
    params,
    processed_obs: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Deterministic action: argmax move, split by thresholding at 0.5. No exploration."""
    move_logits, split_logit, _ = model.apply(params, processed_obs)
    H, W = processed_obs.shape[-2:]

    safe_logits, should_pass = _masked_move_logits(move_logits, valid_mask)
    move_idx = jnp.argmax(safe_logits)
    row, col, direction = jnp.unravel_index(move_idx, (H, W, 4))
    split = (jax.nn.sigmoid(split_logit) > 0.5).astype(jnp.int32)

    return jnp.array(
        [should_pass.astype(jnp.int32), row, col, direction, split],
        dtype=jnp.int32,
    )


def select_action(
    model: ActorCritic,
    params,
    processed_obs: jnp.ndarray,
    valid_mask: jnp.ndarray,
    key: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample an action. Returns (action, log_prob, value)."""
    move_logits, split_logit, value = model.apply(params, processed_obs)
    H, W = processed_obs.shape[-2:]

    safe_logits, should_pass = _masked_move_logits(move_logits, valid_mask)

    move_key, split_key = jrandom.split(key)
    move_idx = jrandom.categorical(move_key, safe_logits)
    move_log_prob = jax.nn.log_softmax(safe_logits)[move_idx]

    row, col, direction = jnp.unravel_index(move_idx, (H, W, 4))

    split = jrandom.bernoulli(split_key, jax.nn.sigmoid(split_logit)).astype(jnp.int32)
    split_log_prob = -optax.sigmoid_binary_cross_entropy(split_logit, split.astype(jnp.float32))

    log_prob = jnp.where(should_pass, 0.0, move_log_prob + split_log_prob)

    action = jnp.array(
        [should_pass.astype(jnp.int32), row, col, direction, split],
        dtype=jnp.int32,
    )

    return action, log_prob, value


def evaluate_action(
    model: ActorCritic,
    params,
    processed_obs: jnp.ndarray,
    valid_mask: jnp.ndarray,
    action: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Re-evaluate a stored action under the current policy.

    Returns (log_prob, entropy, value).
    """
    move_logits, split_logit, value = model.apply(params, processed_obs)
    H, W = processed_obs.shape[-2:]

    safe_logits, should_pass = _masked_move_logits(move_logits, valid_mask)
    log_probs = jax.nn.log_softmax(safe_logits)
    probs = jnp.exp(log_probs)

    row, col, direction, split = action[1], action[2], action[3], action[4]
    move_idx = jnp.ravel_multi_index((row, col, direction), (H, W, 4), mode="clip")
    move_log_prob = log_probs[move_idx]
    # log_probs is -inf at masked-out entries; guard against 0 * -inf = nan
    # in both the forward computation and its gradient.
    safe_log_probs = jnp.where(jnp.isfinite(log_probs), log_probs, 0.0)
    categorical_entropy = -jnp.sum(probs * safe_log_probs)

    split_target = split.astype(jnp.float32)
    split_log_prob = -optax.sigmoid_binary_cross_entropy(split_logit, split_target)
    split_entropy = _bernoulli_entropy(split_logit)

    log_prob = jnp.where(should_pass, 0.0, move_log_prob + split_log_prob)
    entropy = jnp.where(should_pass, 0.0, categorical_entropy + split_entropy)

    return log_prob, entropy, value
