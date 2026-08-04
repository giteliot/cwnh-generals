import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from generals.core import game
from generals.core.env import GeneralsEnv
from training.actions import decode, num_actions
from training.masking import action_mask
from training.potential import PHI_BOUND, TERMS, phi, shaping

GAMMA = 0.999


def _random_action(state, player, key):
    obs = game.get_observation(state, player)
    mask = action_mask(obs)
    h, w = obs.armies.shape
    return decode(jrandom.choice(key, num_actions(h, w), p=mask / mask.sum()), h, w)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shaping_telescopes(seed):
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = jax.tree.map(lambda x: x[seed], pool)
    key = jrandom.PRNGKey(seed + 100)

    horizon = 40
    phi_0 = float(phi(state, 0))
    total = 0.0
    discount = 1.0

    for _ in range(horizon):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([_random_action(state, 0, k0), _random_action(state, 1, k1)])
        timestep, next_state = env.step(state, actions, pool)
        done = bool(timestep.terminated | timestep.truncated)
        assert not done, "horizon must stay inside one episode for this identity"
        total += discount * float(shaping(state, timestep.last_state, False, 0, GAMMA))
        discount *= GAMMA
        state = next_state

    expected = discount * float(phi(state, 0)) - phi_0
    assert total == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("seed", [0, 1])
def test_shaping_over_full_episode_is_policy_independent(seed):
    env = GeneralsEnv(
        grid_dims=(8, 8),
        truncation=120,
        min_generals_distance=3,
        build_castles=True,
        deathtouch_turn=50,
        pool_size=16,
    )
    pool, _ = env.reset(jrandom.PRNGKey(seed))
    state = jax.tree.map(lambda x: x[seed], pool)
    phi_0 = float(phi(state, 0))

    key = jrandom.PRNGKey(seed)
    total = 0.0
    discount = 1.0
    for _ in range(130):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([_random_action(state, 0, k0), _random_action(state, 1, k1)])
        timestep, next_state = env.step(state, actions, pool)
        done = bool(timestep.terminated | timestep.truncated)
        total += discount * float(shaping(state, timestep.last_state, done, 0, GAMMA))
        discount *= GAMMA
        state = next_state
        if done:
            break
    else:
        pytest.skip("episode did not finish inside the cap")

    assert total == pytest.approx(-phi_0, abs=1e-2)


def test_phi_is_antisymmetric_in_material_terms():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    assert float(phi(state, 0, w_explore=0.0)) == pytest.approx(
        float(phi(state, 1, w_explore=0.0)), abs=1e-4
    )


def test_every_term_is_bounded():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    from training.potential import side
    for seed in range(6):
        state = jax.tree.map(lambda x: x[seed], pool)
        me, opp = side(state, 0), side(state, 1)
        for term in TERMS:
            v = float(term.fn(me, opp))
            assert -1.001 <= v <= 1.001, f"term {term.name!r} returned {v}, must stay in [-1, 1]"


def test_potential_is_bounded():
    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    bound = PHI_BOUND
    for seed in range(8):
        state = jax.tree.map(lambda x: x[seed], pool)
        for p in (0, 1):
            assert abs(float(phi(state, p))) <= bound + 1e-4


def test_value_head_cannot_diverge():
    from training.actions import num_actions
    from training.features import NUM_GLOBALS, NUM_PLANES
    from training.network import VALUE_SCALE, ActorCritic

    h = w = 16
    net = ActorCritic(channels=8, blocks=2, groups=4)
    params = net.init(
        jrandom.PRNGKey(0),
        jnp.zeros((h, w, NUM_PLANES)),
        jnp.zeros((NUM_GLOBALS,)),
        jnp.ones((num_actions(h, w),), bool),
    )
    for scale in (1.0, 1e3, 1e6):
        planes = jnp.full((h, w, NUM_PLANES), scale)
        globals_ = jnp.full((NUM_GLOBALS,), scale)
        _, value = net.apply(params, planes, globals_, jnp.ones((num_actions(h, w),), bool))
        assert jnp.isfinite(value)
        assert abs(float(value)) <= VALUE_SCALE


def test_stall_scores_the_same_as_a_draw():
    """A turn-1200 timeout and a mutual deathtouch are both draws and must be
    valued identically, and flat -- nothing about HOW a game timed out may score
    better than anything else, or a draw-heavy rollout optimises that instead of
    winning."""
    from training.potential import Outcome, terminal_reward

    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    o = Outcome()
    T, F = jnp.bool_(True), jnp.bool_(False)

    stall = float(terminal_reward(jnp.float32(0), F, T, state, 0, o))
    mutual = float(terminal_reward(jnp.float32(0), T, F, state, 0, o))
    assert stall == mutual, f"stall {stall} must equal mutual-kill draw {mutual}"
    assert float(terminal_reward(jnp.float32(0), F, F, state, 0, o)) == 0.0

    for seed in range(6):
        for p in (0, 1):
            other = jax.tree.map(lambda x: x[seed], pool)
            assert float(terminal_reward(jnp.float32(0), F, T, other, p, o)) == stall


def test_decisive_outcomes_are_zero_sum():
    """win + loss must be 0. Asymmetric terminals make self-play non-zero-sum,
    paying BOTH seats jointly for a decisive game -- a known route to degenerate
    cooperative equilibria."""
    from training.potential import LOSS, WIN
    assert WIN + LOSS == 0, f"WIN {WIN} + LOSS {LOSS} != 0"


def test_stalling_never_beats_fighting_for_a_competent_agent():
    """A draw must sit strictly between loss and win, so stalling is never a
    refuge but throwing the game is never attractive either."""
    from training.potential import DRAW, LOSS, WIN
    assert LOSS < DRAW < 0 < WIN
    breakeven = (DRAW - LOSS) / (WIN - LOSS)
    assert 0.1 < breakeven < 0.4, f"fight-vs-stall breakeven {breakeven:.0%} is out of range"


def test_terminal_reward_dominates_shaping():
    """Winning must be worth more than everything shaping can ever pay out."""
    from training.potential import PHI_BOUND, WIN, Outcome, terminal_reward
    assert abs(WIN) > 2 * PHI_BOUND, (
        f"WIN={WIN} is not clearly above the total shaping an episode can earn "
        f"({PHI_BOUND}); the agent would optimise potential over winning"
    )

    env = GeneralsEnv(mode="competition", pool_size=64)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    state = jax.tree.map(lambda x: x[0], pool)
    o = Outcome()
    T, F = jnp.bool_(True), jnp.bool_(False)

    win = float(terminal_reward(jnp.float32(1), T, F, state, 0, o))
    loss = float(terminal_reward(jnp.float32(-1), T, F, state, 0, o))
    drawn = float(terminal_reward(jnp.float32(0), F, T, state, 0, o))
    mutual = float(terminal_reward(jnp.float32(0), T, F, state, 0, o))
    ongoing = float(terminal_reward(jnp.float32(0), F, F, state, 0, o))

    assert win == o.win and loss == o.loss
    assert ongoing == 0.0, "a non-final step must carry no terminal reward"
    assert mutual == drawn, "mutual deathtouch is a draw"
    assert loss < drawn < 0 < win, f"need loss < draw < 0 < win, got {loss} {drawn} {win}"
    assert abs(loss) == abs(win), "decisive outcomes must stay zero-sum"
