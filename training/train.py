import argparse
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from flax.training.train_state import TrainState

from generals.agents import ExpanderAgent, HunterAgent
from generals.agents.harvester_agent import HarvesterAgent
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

from . import anchors
from .actions import decode, num_actions
from .features import (
    GLOBAL_NAMES,
    NUM_GLOBALS,
    NUM_PLANES,
    PLANE_NAMES,
    extract,
    init_belief,
    record_move,
    reset_belief_where,
    update_belief,
)
from .masking import action_mask
from .network import NUM_OPPONENT_KINDS, ActorCritic, log_probs, sample
from .potential import DRAW, DRAW_TIEBREAK, LOSS, WIN, Outcome, shaping, terminal_reward
from .ppo_core import Transition, compute_gae, make_update, quantize

STAGES = {
    "tiny": dict(min_grid_size=8, max_grid_size=9, pad_to=10, truncation=1200),
    "small": dict(min_grid_size=10, max_grid_size=12, pad_to=12, truncation=1200),
    "medium": dict(min_grid_size=14, max_grid_size=16, pad_to=16, truncation=1200),
    "full": dict(mode="competition"),
}

DEFAULT_CONFIG = dict(
    LR=1e-4,
    GAMMA=0.9999,
    GAE_LAMBDA=0.95,
    CLIP_EPS=0.2,
    VF_COEF=0.5,
    ENT_COEF=0.01,
    MAX_GRAD_NORM=0.5,
    UPDATE_EPOCHS=4,
    NUM_MINIBATCHES=8,
)


def make_env(stage, pool_size):
    kwargs = dict(STAGES[stage])
    if "mode" not in kwargs:
        kwargs.update(
            mountain_density_range=(0.24, 0.26),
            num_castles_range=(9, 11),
            min_generals_distance=max(4, kwargs["min_grid_size"] // 2),
            build_castles=True,
            deathtouch_turn=800,
            perfect_info=False,
        )
    return GeneralsEnv(pool_size=pool_size, **kwargs)


def pick(flag, a, b):
    return jax.tree.map(lambda x, y: jnp.where(flag, y, x), a, b)


def to_numpy(tree):
    return jax.tree.map(np.asarray, tree)


def to_jax(tree):
    return jax.tree.map(jnp.asarray, tree)


def save(path, train_state, update, bank, league, key, args):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({
            "params": to_numpy(train_state.params),
            "opt_state": to_numpy(train_state.opt_state),
            "step": int(train_state.step),
            "update": update,
            "bank": [to_numpy(b) for b in bank],
            "league": league,
            # Paths only. The anchor weights live in their own files and are
            # never trained, so copying them into every checkpoint would add
            # their full size for nothing -- and would hide the fact that an
            # anchor is meant to be a file you can point at and diff.
            "anchors": args.anchors,
            "key": np.asarray(jrandom.key_data(key)),
            "channels": args.channels,
            "blocks": args.blocks,
            "groups": args.groups,
            "stage": args.stage,
            "planes": PLANE_NAMES,
            "globals": GLOBAL_NAMES,
        }, fh)
    os.replace(tmp, path)


def check_features(blob, path):
    """Old checkpoints have a different observation, and flax only reports it as
    a cryptic kernel-shape error. Say what actually happened."""
    saved = blob.get("planes")
    if saved is not None and saved != PLANE_NAMES:
        added = [p for p in PLANE_NAMES if p not in saved]
        removed = [p for p in saved if p not in PLANE_NAMES]
        raise SystemExit(
            f"{path} was trained with {len(saved)} observation planes, the code now builds "
            f"{len(PLANE_NAMES)}.\n"
            f"  added since:   {added or 'none'}\n"
            f"  removed since: {removed or 'none'}\n"
            "Observation changes are not resumable. Start a fresh run, or `git stash` back to "
            "the code this checkpoint was trained with to use it."
        )


# Opponent pool. Index 0-2 are scripted, index 3 is a network (current policy
# or a banked checkpoint).
#
# RandomAgent is NOT here. Over 4828 episodes it produced 0.0% wins, 8.1% losses
# and 91.9% draws: it never dies, because neither side can find a general, and
# the 8% of losses are it stumbling onto ours. An opponent that cannot lose
# teaches nothing -- the terminal reward is the only episode-level signal, so a
# schedule full of timeouts is a schedule with no signal. The remaining three
# finish 39-82% of their games.
SCRIPTED = ("expander", "hunter", "harvester")
NETWORK = len(SCRIPTED)


def build_scripted():
    return [ExpanderAgent(), HunterAgent(), HarvesterAgent()]


class Opponent(NamedTuple):
    kind: int          # 0-2 scripted, 3 network
    label: str
    bank_idx: int = -1      # index into bank, or -1
    anchor_idx: int = -1    # index into anchors, or -1

    def one_hot(self):
        """Identity for the critic: each bot gets a slot; bank, anchors and
        self-play all share one. Which specific one does not matter -- what the
        critic needs is 'this game is worth ~+9.8' vs '~-2.6', and self-play sits
        in the same peer-level regime as a bank snapshot (this policy, or a very
        recent one, on both sides). An anchor is deliberately NOT distinguished
        from bank either: both are frozen/near-frozen opponents of comparable
        strength to what the bank already encodes."""
        return jnp.eye(NUM_OPPONENT_KINDS, dtype=jnp.float32)[self.kind]


LEAGUE_DECAY = 0.98      # per-update EMA on stats: ~50-update memory, so a
                         # regression against a "beaten" opponent resurfaces it
MIN_GAMES = 20           # below this an opponent counts as unbeaten (score 0)


def init_league(bank, anchors=()):
    return {
        "bots": {name: [0.0, 0.0, 0.0] for name in SCRIPTED},   # wins, draws, games
        "bank": [[0.0, 0.0, 0.0] for _ in bank],
        # Keyed by name, not position: adding or dropping an anchor must never
        # shift another anchor's history the way a list index would.
        "anchors": {a.name: [0.0, 0.0, 0.0] for a in anchors},
    }


def sync_anchors(league, anchors):
    """Add slots for new anchors, keep the history of ones already measured,
    drop stats for anchors no longer in the pool so the table cannot show a
    yardstick that is not being played."""
    previous = league.get("anchors") or {}
    league["anchors"] = {a.name: previous.get(a.name, [0.0, 0.0, 0.0]) for a in anchors}
    return league


def league_score(stats):
    wins, draws, games = stats
    return (wins + 0.5 * draws) / games if games >= MIN_GAMES else 0.0


def pfsp_weights(scores, power):
    """AlphaStar's f_hard: sample each opponent proportional to (1 - score)^p.
    Draws count half, so an opponent we only ever draw against stays "hard"."""
    w = [max(1.0 - s, 1e-3) ** power for s in scores]
    total = sum(w)
    return [x / total for x in w]


# Cold start: a fresh policy beats nothing, so PFSP -- which prioritises the
# HARDEST opponent -- points the wrong way, and self-play between two random
# policies is pure draws with no gradient. A run spent 60% of its first 300
# updates on self and had zero wins after 1755 updates.
#
# "Easiest" is scored by league_score, which counts draws as half a win. With
# RandomAgent in the pool that ranked it top (0.46, from a 92% draw rate) and
# poured the cold start into the one opponent that never dies. Dropping it makes
# the ranking mean what it says: every remaining bot can actually be beaten.
#
# AlphaStar avoids this phase entirely by bootstrapping from behaviour cloning
# before PFSP takes over; lacking that, bootstrap on the easiest opponent.
BOOTSTRAP_SCORE = 0.5    # once anything is beaten this well, switch to PFSP


def bootstrapping(league):
    scores = [league_score(st) for st in league["bots"].values()]
    return max(scores, default=0.0) < BOOTSTRAP_SCORE


def pfsp_pick(key, scores, power):
    weights = pfsp_weights(scores, power)
    return int(jrandom.choice(key, len(scores), p=jnp.asarray(weights)))


def choose_opponent(args, bank, anchors, league, key):
    """Easy-first until the agent can win, then PFSP hardest-first, split across
    up to four EXCLUSIVE slices of the roll: anchors, self-play, scripted bots,
    bank. Each floor means what it says, independent of how many entries the
    other pools happen to have.

    Self-play (`--self-floor`) plays train_state.params directly, which
    choose_opponent used to refuse outright: "the opponent is non-stationary at
    exactly the learning frequency; ML-Agents never does that -- it swaps between
    saved SNAPSHOTS every `swap_steps`, so the opponent is frozen while you learn
    against it. A 7900-update run played live-self or bank for 98%% of its last
    600 updates and drifted from 63%% to 45%% against its own bank." That risk is
    real and un-fixed here -- --self-floor is 0 by default, and turning it on is
    a deliberate choice to trade a frozen reference for however self-play alone
    performs against the bank/anchors, which still work exactly as before.

    Bots keep a floor for the same reason as always: PFSP's (1-score)^p stops
    wasting rollouts on opponents already beaten 100%, but that also means an
    UNFLOORED yardstick can vanish from the log entirely. --bot-floor 0 removes
    the scripted bots from the pool -- and with them, the bootstrap phase, which
    has nothing to measure if bots are never played (their score would sit at
    0.0 forever, permanently misreading "cold start").

    Bootstrap mode (nothing beaten yet, only reachable with --bot-floor > 0):
    the weighting flips to prefer the opponent we are doing BEST against -- the
    only one that can produce a first win to learn from.
    """
    k_floor, k_pick = jrandom.split(key)
    bot_scores = [league_score(league["bots"][n]) for n in SCRIPTED]

    if args.bot_floor > 0 and bootstrapping(league):
        weights = [(s + 0.05) ** args.pfsp_power for s in bot_scores]
        total = sum(weights)
        weights = [w / total for w in weights]
        i = int(jrandom.choice(k_pick, len(SCRIPTED), p=jnp.asarray(weights)))
        return Opponent(i, SCRIPTED[i])

    anchor_share = args.anchor_floor if anchors else 0.0
    self_share = args.self_floor
    bot_share = args.bot_floor
    bank_share = max(0.0, 1.0 - anchor_share - self_share - bot_share)
    if not bank:
        # Nothing to PFSP over -- redirect that slice to self-play rather than
        # crashing on an empty choice().
        self_share += bank_share
        bank_share = 0.0

    roll = float(jrandom.uniform(k_floor))
    if roll < anchor_share:
        i = pfsp_pick(k_pick, [league_score(league["anchors"][a.name]) for a in anchors],
                      args.pfsp_power)
        return Opponent(NETWORK, f"@{anchors[i].name}", anchor_idx=i)
    roll -= anchor_share

    if roll < self_share:
        return Opponent(NETWORK, "self")
    roll -= self_share

    if roll < bot_share:
        i = pfsp_pick(k_pick, bot_scores, args.pfsp_power)
        return Opponent(i, SCRIPTED[i])

    i = pfsp_pick(k_pick, [league_score(st) for st in league["bank"]], args.pfsp_power)
    return Opponent(NETWORK, f"bank[{i}]", bank_idx=i)


def league_stats(league, anchors, opponent):
    """The stats row this opponent's result belongs in, or None for live self --
    there is no fixed reference to accumulate self-play's outcomes against."""
    if opponent.anchor_idx >= 0:
        return league["anchors"][anchors[opponent.anchor_idx].name]
    if opponent.bank_idx >= 0:
        return league["bank"][opponent.bank_idx]
    if opponent.kind != NETWORK:
        return league["bots"][opponent.label]
    return None


def league_table(league):
    cells = ["BOOTSTRAP" if bootstrapping(league) else "pfsp"]
    cells += [f"{n} {league_score(st):.2f}" for n, st in league["bots"].items()]
    cells += [f"@{n} {league_score(st):.2f}" for n, st in league.get("anchors", {}).items()]
    cells += [f"b{i} {league_score(st):.2f}" for i, st in enumerate(league["bank"])]
    return "  ".join(cells)


def make_rollout(env, network, pool, scripted, config):
    def encode_side(obs, belief, player):
        belief = update_belief(belief, obs)
        planes, globals_ = extract(obs, belief, player)
        return belief, planes, globals_, action_mask(obs)

    def rollout(params, opp_params, state, learner_belief, opp_belief, ret, seat, opp_kind, opp_hot,
                alpha, w_explore, key):
        weights = config["POTENTIAL_WEIGHTS"]
        def env_step(carry, _):
            state, learner_belief, opp_belief, ret, key = carry
            obs0 = jax.vmap(get_observation, in_axes=(0, None))(state, 0)
            obs1 = jax.vmap(get_observation, in_axes=(0, None))(state, 1)
            learner_obs = jax.vmap(pick)(seat, obs0, obs1)
            opp_obs = jax.vmap(pick)(seat, obs1, obs0)
            learner_player = seat.astype(jnp.int32)

            learner_belief, planes, globals_, mask = jax.vmap(encode_side)(
                learner_obs, learner_belief, learner_player
            )
            opp_belief, o_planes, o_globals, o_mask = jax.vmap(encode_side)(
                opp_obs, opp_belief, 1 - learner_player
            )

            # width from the caller, not the constant: eval tools load checkpoints
            # trained with a different NUM_OPPONENT_KINDS and the value head's
            # input has to match whatever those weights expect.
            hot = jnp.broadcast_to(opp_hot, (seat.shape[0], opp_hot.shape[-1]))
            logits, value = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                params, planes, globals_, mask, hot)

            key, k_act, k_opp = jrandom.split(key, 3)
            n = seat.shape[0]
            action = jax.vmap(sample)(logits, jrandom.split(k_act, n))
            log_prob = jnp.take_along_axis(jax.vmap(log_probs)(logits), action[:, None], axis=1)[:, 0]

            height, width = state.armies.shape[1:]
            learner_move = jax.vmap(decode, in_axes=(0, None, None))(action, height, width)

            # lax.switch runs exactly one branch, so a scripted opponent costs no
            # network forward and vice versa. All branches are traced at compile.
            def net_branch(_):
                o_logits, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
                    opp_params, o_planes, o_globals, o_mask, hot)   # opponent's own value unused
                idx = jax.vmap(sample)(o_logits, jrandom.split(k_opp, n))
                return jax.vmap(decode, in_axes=(0, None, None))(idx, height, width)

            def bot_branch(agent):
                return lambda _: jax.vmap(agent.act)(opp_obs, jrandom.split(k_opp, n))

            opp_move = jax.lax.switch(
                opp_kind, [bot_branch(a) for a in scripted] + [net_branch], 0)

            # stamp each side's move into its own trail, so next turn sees it
            learner_belief = jax.vmap(record_move)(learner_belief, learner_move)
            opp_belief = jax.vmap(record_move)(opp_belief, opp_move)

            moves = jnp.stack([
                jnp.where(seat[:, None], opp_move, learner_move),
                jnp.where(seat[:, None], learner_move, opp_move),
            ], axis=1)

            timestep, new_state = jax.vmap(env.step, in_axes=(0, 0, None))(state, moves, pool)
            done = timestep.terminated | timestep.truncated
            raw = jnp.take_along_axis(timestep.reward, learner_player[:, None], axis=1)[:, 0]
            outcome = jax.vmap(terminal_reward, in_axes=(0, 0, 0, 0, 0, None))(
                raw, timestep.terminated, timestep.truncated,
                timestep.last_state, learner_player, config["OUTCOME"],
            )
            shaped = jax.vmap(shaping, in_axes=(0, 0, 0, 0, None, None, None))(
                state, timestep.last_state, done, learner_player, config["GAMMA"], w_explore, weights
            )

            transition = Transition(
                done=done,
                action=action,
                value=value,
                reward=outcome + alpha * shaped,
                log_prob=log_prob,
                planes=quantize(planes),
                globals_=globals_,
                mask=mask,
                opponent=hot,
            )
            learner_belief = jax.vmap(reset_belief_where)(learner_belief, done)
            opp_belief = jax.vmap(reset_belief_where)(opp_belief, done)

            # True cumulative reward per episode: accumulate every step, read it
            # out when the game ends, then reset. Carried across rollouts, since
            # episodes (up to 1200 turns) far outlive a 32-step rollout.
            ret = ret + transition.reward
            episode_return = jnp.where(done, ret, 0.0)
            ret = jnp.where(done, 0.0, ret)
            return ((new_state, learner_belief, opp_belief, ret, key),
                    (transition, raw, outcome, episode_return, done))

        carry, (traj, raw, terminal, ep_return, done) = jax.lax.scan(
            env_step, (state, learner_belief, opp_belief, ret, key), None, config["NUM_STEPS"]
        )
        state, learner_belief, opp_belief, ret, key = carry

        obs0 = jax.vmap(get_observation, in_axes=(0, None))(state, 0)
        obs1 = jax.vmap(get_observation, in_axes=(0, None))(state, 1)
        learner_obs = jax.vmap(pick)(seat, obs0, obs1)
        _, planes, globals_, mask = jax.vmap(encode_side)(learner_obs, learner_belief, seat.astype(jnp.int32))
        hot = jnp.broadcast_to(opp_hot, (seat.shape[0], opp_hot.shape[-1]))
        _, last_value = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0))(
            params, planes, globals_, mask, hot)

        return carry, traj, last_value, raw, terminal, ep_return, done

    return rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="small", choices=STAGES.keys())
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=2048)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9999)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--win", type=float, default=WIN)
    parser.add_argument("--loss", type=float, default=LOSS)
    parser.add_argument("--draw", type=float, default=DRAW)
    parser.add_argument("--draw-tiebreak", type=float, default=DRAW_TIEBREAK)
    parser.add_argument("--w-econ", type=float, default=1.0)
    parser.add_argument("--w-material", type=float, default=0.4)
    parser.add_argument("--w-territory", type=float, default=0.3)
    parser.add_argument("--w-safety", type=float, default=0.5)
    parser.add_argument("--w-threat", type=float, default=0.4)
    parser.add_argument("--w-explore", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anneal-frac", type=float, default=0.6)
    parser.add_argument("--opp-swap-every", type=int, default=10)
    parser.add_argument("--pfsp-power", type=float, default=2.0)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--bot-floor", type=float, default=0.15,
                        help="minimum share of updates played against the scripted bots, "
                             "so an absolute yardstick never leaves the log. 0 removes the "
                             "bots from the pool entirely (and skips the bootstrap phase, "
                             "which would otherwise stay on forever with nothing to measure "
                             "bot win rate from)")
    parser.add_argument("--self-floor", type=float, default=0.0,
                        help="share of updates played against the LIVE policy instead of a "
                             "frozen bank snapshot. choose_opponent's docstring warns against "
                             "this: a 7900-update run that played live-self/bank 98% of its "
                             "last 600 updates drifted 63%%->45%% against its own bank, because "
                             "the opponent is non-stationary at exactly the learning frequency. "
                             "0 (default) never does this")
    parser.add_argument("--anchors", default=None,
                        help="comma-separated frozen opponents, each 'path' or 'name=path'. "
                             "Accepts a .pkl checkpoint, an export .npz or a submission .zip. "
                             "Never trained, never banked, never evicted -- the only score in "
                             "the log that can rise (or fall) across the whole run, since the "
                             "bank's own eviction rule pins its win rate near 0.50")
    parser.add_argument("--anchor-floor", type=float, default=0.10,
                        help="share of updates played against the anchors (ignored with none)")
    parser.add_argument("--promote-threshold", type=float, default=0.55,
                        help="league_score needed against an anchor before the live policy's "
                             "weights replace it. 0.55 is AlphaGo Zero's own challenger-vs-best "
                             "promotion threshold -- clearly ahead, not just tied")
    parser.add_argument("--promote-min-games", type=int, default=100,
                        help="minimum EMA-tracked games against an anchor before its score is "
                             "trusted for promotion")
    parser.add_argument("--promote-min-updates", type=int, default=200,
                        help="minimum updates since an anchor's last promotion (or the run "
                             "start) before it can be promoted again. --promote-min-games is "
                             "NOT enough smoothing on its own: right after a reset, league_score "
                             "is an EMA that can cross any games threshold in a SINGLE update if "
                             "that update happens to finish many episodes, giving zero real "
                             "smoothing exactly when it matters most. This is the guard that "
                             "actually enforces a cooldown, independent of how episodes happen "
                             "to land in a batch")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--bank-size", type=int, default=10)
    parser.add_argument("--no-stagger", action="store_true")
    parser.add_argument("--out", default="ppo_params.pkl")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--reseed", action="store_true")
    args = parser.parse_args()

    config = dict(
        DEFAULT_CONFIG,
        NUM_ENVS=args.envs,
        NUM_STEPS=args.steps,
        NUM_MINIBATCHES=args.minibatches,
        UPDATE_EPOCHS=args.epochs,
        LR=args.lr,
        GAMMA=args.gamma,
        ENT_COEF=args.ent_coef,
        TARGET_KL=args.target_kl,
        OUTCOME=Outcome(win=args.win, loss=args.loss, draw=args.draw, tiebreak=args.draw_tiebreak),
        POTENTIAL_WEIGHTS=dict(
            econ=args.w_econ, material=args.w_material, territory=args.w_territory,
            safety=args.w_safety, threat=args.w_threat, explore=args.w_explore,
        ),
    )

    key = jrandom.PRNGKey(args.seed)
    key, k_pool, k_net = jrandom.split(key, 3)

    env = make_env(args.stage, args.pool_size)
    pool, _ = env.reset(k_pool)
    height, width = pool.armies.shape[1:]
    print(f"stage={args.stage} board={height}x{width} pool={env.pool_size} actions={num_actions(height, width)}")

    minibatch = args.envs * args.steps // args.minibatches
    buffer_gb = args.envs * args.steps * (NUM_PLANES * height * width + num_actions(height, width)) / 1e9
    activations_gb = minibatch * height * width * args.channels * 4 * (2 * args.blocks + 2) / 1e9
    print(
        f"batch={args.envs * args.steps} minibatch={minibatch} "
        f"| est VRAM: buffer {buffer_gb:.1f} GB + activations {activations_gb:.1f} GB"
    )

    states = jax.tree.map(lambda x: x[: args.envs], pool)
    states = states._replace(pool_idx=jnp.arange(args.envs, dtype=jnp.int32))
    seat = (jnp.arange(args.envs) % 2).astype(bool)
    # Each side needs its OWN memory. Feeding one belief to both, then keeping
    # only the learner's, hands the opponent its enemy's map knowledge and move
    # trail every rollout -- it plays blind and the learner "wins" 98% of games
    # against a crippled copy of itself.
    beliefs = jax.vmap(lambda _: init_belief((height, width)))(jnp.arange(args.envs))
    opp_beliefs = jax.vmap(lambda _: init_belief((height, width)))(jnp.arange(args.envs))
    returns_so_far = jnp.zeros((args.envs,), jnp.float32)

    # Stagger the env clocks. Every env starts at turn 0, so without this they
    # all reach the 1200-turn truncation on the SAME update and reset together:
    # the terminal signal arrives in one lump every ~37 updates and `eps` reads
    # 0 in between, leaving the agent trained almost entirely on shaping.
    # Offsetting the first episode spreads the endings out permanently. Capped
    # below the turn-800 deathtouch so the opening episode is not decided
    # instantly on a board that has not developed.
    if not args.no_stagger:
        key, k_stagger = jrandom.split(key)
        span = min(env.truncation, 700)
        states = states._replace(
            time=jrandom.randint(k_stagger, (args.envs,), 0, span).astype(states.time.dtype)
        )
        print(f"staggered env clocks across turns 0-{span - 1}")

    network = ActorCritic(channels=args.channels, blocks=args.blocks, groups=args.groups)
    params = network.init(
        k_net,
        jnp.zeros((height, width, NUM_PLANES), jnp.float32),
        jnp.zeros((NUM_GLOBALS,), jnp.float32),
        jnp.ones((num_actions(height, width),), bool),
        jnp.zeros((NUM_OPPONENT_KINDS,), jnp.float32),
    )
    # Keep the freshly built tree as the shape reference for anchors. `params`
    # below is overwritten by --resume, and a resumed checkpoint can itself carry
    # an old value-head width -- then it would be conforming anchors to the wrong
    # target. The init is always what the current code expects.
    reference = jax.tree.map(np.asarray, params)
    blob = None
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, "rb") as fh:
            blob = pickle.load(fh)
        check_features(blob, args.resume)
        params = to_jax(blob["params"])
    convs = 1 + 2 * args.blocks
    field = 1 + 2 * convs
    warn = "  << SMALLER THAN THE BOARD: raise --blocks" if field < max(height, width) else ""
    print(f"parameters: {sum(x.size for x in jax.tree.leaves(params)):,} "
          f"| receptive field {field} cells vs board {max(height, width)}{warn}")

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    frozen = anchors.load(args.anchors, reference)
    for a in frozen:
        print(f"anchor @{a.name}: {a.path}")
    anchor_params = [to_jax(a.params) for a in frozen]
    if not frozen and args.anchors:
        print("no anchors loaded -- nothing but the bank measures this run")

    start_update, bank, league = 0, [], None
    if blob is not None:
        start_update = blob.get("update", 0)
        bank = [to_jax(b) for b in blob.get("bank", [])]
        league = blob.get("league")
        if "opt_state" in blob:
            train_state = train_state.replace(
                opt_state=to_jax(blob["opt_state"]), step=blob.get("step", 0)
            )
        # The saved key makes a resume replay the identical stream -- same maps,
        # same opponent draws, same action samples. Two runs walked into the same
        # bad batch at update ~4080 that way. --reseed breaks the replay.
        if "key" in blob and not args.reseed:
            key = jrandom.wrap_key_data(jnp.asarray(blob["key"]))
        print(
            f"resumed {args.resume}: update {start_update}/{args.updates}, "
            f"bank {len(bank)}, optimizer {'restored' if 'opt_state' in blob else 'RESET'}"
        )
    if (not league or sorted(league.get("bots", {})) != sorted(SCRIPTED)
            or len(league.get("bank", [])) != len(bank)):
        league = init_league(bank, frozen)   # older checkpoint: everyone starts "unbeaten"
    else:
        # Adding anchors to a run in flight must NOT reset the bot and bank
        # history -- that history is the only thing that stops PFSP from
        # re-bootstrapping on opponents it already beat.
        sync_anchors(league, frozen)

    rollout = jax.jit(make_rollout(env, network, pool, build_scripted(), config))
    update_epoch = make_update(network.apply, config)

    @jax.jit
    def learn(train_state, traj, last_value, key):
        advantages, targets = compute_gae(traj, last_value, config["GAMMA"], config["GAE_LAMBDA"])
        carry, info = jax.lax.scan(
            update_epoch,
            (train_state, jnp.bool_(False), traj, advantages, targets, key),
            None, config["UPDATE_EPOCHS"],
        )
        return carry[0], info

    held = None
    # Grace period before an anchor can be promoted again -- see
    # --promote-min-updates. Starts at start_update (not 0) so a resumed run
    # does not treat every anchor as instantly eligible.
    promoted_at = {a.name: start_update for a in frozen}
    for update in range(start_update, args.updates):
        t0 = time.time()
        progress = update / max(args.updates * args.anneal_frac, 1)
        alpha = float(max(0.0, 1.0 - progress))
        w_explore = float(max(0.0, 0.2 * (1.0 - 3.0 * progress)))

        key, k_roll, k_learn, k_opp = jrandom.split(key, 4)
        # Hold one opponent for --opp-swap-every updates. Resampling every
        # update (what we did before) never gives a clean gradient toward
        # beating anyone; ML-Agents holds one fixed for ~10k steps for exactly
        # this reason. `held is None` covers the first update after a resume.
        if held is None or update % max(args.opp_swap_every, 1) == 0:
            held = choose_opponent(args, bank, frozen, league, k_opp)
        if held.anchor_idx >= 0:
            opp_params = anchor_params[held.anchor_idx]
        elif held.bank_idx >= 0:
            opp_params = bank[held.bank_idx]
        else:
            opp_params = train_state.params  # self-play, or nothing else matched

        carry, traj, last_value, outcome, terminal, ep_return, done = rollout(
            train_state.params, opp_params, states, beliefs, opp_beliefs, returns_so_far, seat,
            jnp.int32(held.kind), held.one_hot(), alpha, w_explore, k_roll,
        )
        states, beliefs, opp_beliefs, returns_so_far, _ = carry
        train_state, info = learn(train_state, traj, last_value, k_learn)

        episodes = int(done.sum())
        wins = int(((outcome > 0) & done).sum())
        losses = int(((outcome < 0) & done).sum())
        draws = episodes - wins - losses

        # Split the same counters by which SIDE the learner sat on. `seat` is
        # fixed for the whole run (env i -> player i%2, 1024 each way), so a game
        # between two identical policies -- which is exactly what `vs self` is,
        # since opp_params IS train_state.params -- must score 0.500 overall.
        #
        # It does not. Tallying a real run's `vs self` lines gave 7180W/8324L/424D
        # over 15928 games = 0.464, a 3.6-point deficit far outside noise. That
        # means the learner and the opponent are not interchangeable somewhere,
        # and since self-play is most of the training signal, the policy spends
        # most of its updates being told it is behind in a level matchup.
        #
        # This split says which bug it is:
        #   p0 and p1 straddle 0.500 (0.5-d / 0.5+d)  -> a seat/move-order
        #     asymmetry that is failing to cancel. _determine_move_order breaks
        #     exact army ties with `army_1 < army_0` (strict), so player 0 always
        #     moves first on a tie, and moving second is the advantage.
        #   BOTH halves below 0.500                   -> the opponent is not
        #     actually playing the same policy; the bug is in the rollout.
        p0 = ~seat                                   # learner is player 0
        w0 = int(((outcome > 0) & done & p0).sum())
        l0 = int(((outcome < 0) & done & p0).sum())
        n0 = int((done & p0).sum())
        w1, l1 = wins - w0, losses - l0
        n1 = episodes - n0
        s0 = (w0 + 0.5 * (n0 - w0 - l0)) / n0 if n0 else float("nan")
        s1 = (w1 + 0.5 * (n1 - w1 - l1)) / n1 if n1 else float("nan")

        elapsed = time.time() - t0
        total_loss, value_loss, actor_loss, ent, kl, applied = [float(jnp.mean(x)) for x in info]
        # rew: mean reward per env over this rollout (terminal + alpha*shaping).
        # ep:  mean reward per finished episode, 0 when none finished.
        rew = float(traj.reward.sum(axis=0).mean())
        # Mean cumulative reward of the episodes that finished (win +10 / loss
        # -10 / draw -5). Dividing total rollout reward by episode count
        # is meaningless when few episodes finish -- it attributes 65k
        # transitions of shaping across 2048 envs to a handful of endings.
        ep_rew = float((ep_return * done).sum()) / episodes if episodes else 0.0
        ep_term = float((terminal * done).sum()) / episodes if episodes else 0.0

        print(
            f"upd {update:5d} | loss {total_loss:+.3f} v {value_loss:.3f} pi {actor_loss:+.3f} "
            f"H {ent:.2f} kl {kl:.3f}/{applied:.0%} | rew {rew:+.3f} ep {ep_rew:+.2f} (T{ep_term:+.1f}) "
            f"| eps {episodes:4d} W/L/D {wins:3d}/{losses:3d}/{draws:3d} "
            f"| p0 {s0:.3f}({n0:4d}) p1 {s1:.3f}({n1:4d}) "
            f"| vs {held.label:12s} "
            f"| alpha {alpha:.2f} | sps {args.envs * args.steps / elapsed:7.0f} | {elapsed:5.1f}s"
        )

        if episodes:
            stats = league_stats(league, frozen, held)
            if stats is not None:
                for j, v in enumerate((wins, draws, episodes)):
                    stats[j] = LEAGUE_DECAY * stats[j] + v

        # Promote: replace an anchor's weights with the live policy once it has
        # CLEARLY passed it, not merely tied it -- AlphaGo Zero's own
        # challenger-vs-best protocol. Two independent guards, not one:
        # --promote-min-games AND --promote-min-updates. games alone is not
        # enough smoothing -- right after a reset, league_score's EMA can cross
        # any games threshold in a SINGLE update if that update happens to
        # finish many episodes, which is exactly zero smoothing at the moment
        # it matters most. --promote-min-updates enforces a real cooldown
        # regardless of how episodes happen to land in a batch.
        #
        # A promotion updates anchor_params IN MEMORY ONLY unless it is also
        # written to a.path. Every resume calls anchors.load(args.anchors, ...)
        # fresh from that file -- nothing about a past promotion is stored in
        # the periodic --out checkpoint's "anchors" field, which saves just the
        # CLI path string, not weights. Without this write, resuming silently
        # reverts the anchor to whatever it was BEFORE any promotion in the
        # previous session, and the live policy -- which resumed at its true,
        # fully-trained strength -- looks like it dominates from update one.
        # save() writes a "params" field in exactly the shape anchors.py's
        # read_params() expects for a .pkl; everything else in the blob (bank,
        # opt_state, ...) is irrelevant here and simply ignored on load.
        for i, a in enumerate(frozen):
            astats = league["anchors"][a.name]
            score = league_score(astats)
            if (astats[2] >= args.promote_min_games
                    and update - promoted_at[a.name] >= args.promote_min_updates
                    and score >= args.promote_threshold):
                games = astats[2]
                anchor_params[i] = jax.tree.map(jnp.copy, train_state.params)
                league["anchors"][a.name] = [0.0, 0.0, 0.0]
                promoted_at[a.name] = update
                if a.path.endswith(".pkl"):
                    save(a.path, train_state, update, bank, league, key, args)
                else:
                    print(f"    WARNING: anchor @{a.name}'s source {a.path} is not a .pkl -- "
                          f"cannot persist the promotion there, it WILL be lost on resume")
                promo_path = f"{args.out}.promoted_{a.name}_upd{update}.pkl"
                save(promo_path, train_state, update, bank, league, key, args)
                print(f"    promoted @{a.name} at update {update} "
                      f"(score {score:.3f} over {games:.0f} games) -> {promo_path}")

        if (update + 1) % args.checkpoint_every == 0:
            # Snapshot on a timer -- ML-Agents `save_steps`. The old gate ("bank
            # only while beating the pool") measured strength against a pool made
            # of recent selves, so the bar sank along with the policy and it kept
            # banking at 45% right through the decline.
            bank.append(jax.tree.map(jnp.copy, train_state.params))
            league["bank"].append([0.0, 0.0, 0.0])
            # `while`, not `if`: a resumed checkpoint can carry more entries than
            # THIS run's --bank-size (cwnh_0805.pkl has 10; this run asks for 5).
            # `if` only pops one per checkpoint -- append 1, evict 1, net zero --
            # so the bank would sit at its old size forever instead of shrinking.
            while len(bank) > args.bank_size:
                # Evict the entry we beat MOST, never the oldest. FIFO dropped the
                # peak policy out of the pool and the reference frame then tracked
                # the decline instead of anchoring against it. A fresh entry has
                # games < MIN_GAMES so it scores 0.0 and is safe until measured.
                weakest = max(range(len(bank)), key=lambda j: league_score(league["bank"][j]))
                bank.pop(weakest)
                league["bank"].pop(weakest)
            held = None                                        # re-pick vs the new pool
            print(f"    league: {league_table(league)}")
            save(args.out, train_state, update + 1, bank, league, key, args)

    save(args.out, train_state, args.updates, bank, league, key, args)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
