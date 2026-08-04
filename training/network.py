import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.linen.initializers import orthogonal

from .actions import NUM_CELL_ACTIONS

NEG_INF = -1e9
VALUE_SCALE = 20.0       # > max |return| (terminal ~10 + shaping ~1.4); see potential.REWARD_BOUND
NUM_OPPONENT_KINDS = 4   # expander, hunter, harvester, bank


class ResBlock(nn.Module):
    channels: int
    groups: int

    @nn.compact
    def __call__(self, x):
        y = nn.Conv(self.channels, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)))(x)
        y = nn.GroupNorm(num_groups=self.groups)(y[None])[0]
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)))(y)
        y = nn.GroupNorm(num_groups=self.groups)(y[None])[0]
        return nn.relu(x + y)


class ActorCritic(nn.Module):
    channels: int = 64
    blocks: int = 8
    groups: int = 8
    value_hidden: int = 256

    @nn.compact
    def __call__(self, planes, globals_, mask, opponent=None):
        """opponent: one-hot over NUM_OPPONENT_KINDS, fed ONLY to the value head.

        Episode return swings ~12 points with who we are playing (+9.8 vs a
        weak bot, -2.6 vs a banked checkpoint) while the board looks the
        same, so a critic that cannot see the opponent can only learn a blend
        and carries irreducible error -- which poisons the advantages and
        collapsed two runs at update ~4080. Asymmetric actor-critic: the policy
        path never sees this, so inference is unchanged and the submitted NumPy
        bot (which never computes the value head) needs no mirror.
        """
        x = nn.Conv(self.channels, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)))(planes)
        x = x + nn.Dense(self.channels, kernel_init=orthogonal(jnp.sqrt(2)))(globals_)
        x = nn.relu(x)
        for _ in range(self.blocks):
            x = ResBlock(self.channels, self.groups)(x)

        pooled = jnp.concatenate([x.mean(axis=(0, 1)), x.max(axis=(0, 1)), globals_])
        if opponent is None:
            opponent = jnp.zeros((NUM_OPPONENT_KINDS,), pooled.dtype)

        spatial = nn.Conv(NUM_CELL_ACTIONS, (1, 1), kernel_init=orthogonal(0.01))(x)
        spatial = jnp.transpose(spatial, (2, 0, 1)).reshape(-1)
        pass_logit = nn.Dense(1, kernel_init=orthogonal(0.01))(pooled)
        logits = jnp.where(mask, jnp.concatenate([spatial, pass_logit]), NEG_INF)

        v = nn.Dense(self.value_hidden, kernel_init=orthogonal(jnp.sqrt(2)))(
            jnp.concatenate([pooled, opponent]))
        v = nn.relu(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0))(v)[0]

        return logits, VALUE_SCALE * jnp.tanh(v / VALUE_SCALE)


def log_probs(logits):
    return logits - jax.nn.logsumexp(logits)


def sample(logits, key):
    return jrandom.categorical(key, logits)


def entropy(logits):
    lp = log_probs(logits)
    return -jnp.sum(jnp.exp(lp) * lp)
