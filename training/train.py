import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from flax.training.train_state import TrainState

from generals.agents import ExpanderAgent
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

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
from .network import ActorCritic, log_probs, sample
from .potential import Outcome, shaping, terminal_reward
from .ppo_core import Transition, compute_gae, make_update, quantize

STAGES = {
    "tiny": dict(min_grid_size=8, max_grid_size=9, pad_to=10, truncation=1200),
    "small": dict(min_grid_size=10, max_grid_size=12, pad_to=12, truncation=1200),
    "medium": dict(min_grid_size=14, max_grid_size=16, pad_to=16, truncation=1200),
    "full": dict(mode="competition"),
}

DEFAULT_CONFIG = dict(
    LR=3e-4,
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


def save(path, train_state, update, bank, key, args):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({
            "params": to_numpy(train_state.params),
            "opt_state": to_numpy(train_state.opt_state),
            "step": int(train_state.step),
            "update": update,
            "bank": [to_numpy(b) for b in bank],
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


def make_rollout(env, network, pool, expander, config):
    def encode_side(obs, belief, player):
        belief = update_belief(belief, obs)
        planes, globals_ = extract(obs, belief, player)
        return belief, planes, globals_, action_mask(obs)

    def rollout(params, opp_params, state, learner_belief, opp_belief, seat, use_expander, alpha, w_explore, key):
        weights = config["POTENTIAL_WEIGHTS"]
        def env_step(carry, _):
            state, learner_belief, opp_belief, key = carry
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

            logits, value = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(params, planes, globals_, mask)
            o_logits, _ = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(opp_params, o_planes, o_globals, o_mask)

            key, k_act, k_opp, k_exp = jrandom.split(key, 4)
            n = seat.shape[0]
            action = jax.vmap(sample)(logits, jrandom.split(k_act, n))
            log_prob = jnp.take_along_axis(jax.vmap(log_probs)(logits), action[:, None], axis=1)[:, 0]
            o_action = jax.vmap(sample)(o_logits, jrandom.split(k_opp, n))

            height, width = state.armies.shape[1:]
            learner_move = jax.vmap(decode, in_axes=(0, None, None))(action, height, width)
            opp_move = jax.vmap(decode, in_axes=(0, None, None))(o_action, height, width)
            expander_move = jax.vmap(expander.act)(opp_obs, jrandom.split(k_exp, n))
            opp_move = jnp.where(use_expander, expander_move, opp_move)

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
            )
            learner_belief = jax.vmap(reset_belief_where)(learner_belief, done)
            opp_belief = jax.vmap(reset_belief_where)(opp_belief, done)
            return (new_state, learner_belief, opp_belief, key), (transition, raw, done)

        carry, (traj, outcome, done) = jax.lax.scan(
            env_step, (state, learner_belief, opp_belief, key), None, config["NUM_STEPS"]
        )
        state, learner_belief, opp_belief, key = carry

        obs0 = jax.vmap(get_observation, in_axes=(0, None))(state, 0)
        obs1 = jax.vmap(get_observation, in_axes=(0, None))(state, 1)
        learner_obs = jax.vmap(pick)(seat, obs0, obs1)
        _, planes, globals_, mask = jax.vmap(encode_side)(learner_obs, learner_belief, seat.astype(jnp.int32))
        _, last_value = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(params, planes, globals_, mask)

        return carry, traj, last_value, outcome, done

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
    parser.add_argument("--win", type=float, default=10.0)
    parser.add_argument("--loss", type=float, default=-6.0)
    parser.add_argument("--draw", type=float, default=-2.0)
    parser.add_argument("--draw-tiebreak", type=float, default=1.0)
    parser.add_argument("--w-econ", type=float, default=1.0)
    parser.add_argument("--w-material", type=float, default=0.4)
    parser.add_argument("--w-territory", type=float, default=0.3)
    parser.add_argument("--w-safety", type=float, default=0.5)
    parser.add_argument("--w-threat", type=float, default=0.4)
    parser.add_argument("--w-explore", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anneal-frac", type=float, default=0.6)
    parser.add_argument("--expander-prob", type=float, default=0.2)
    parser.add_argument("--self-prob", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--bank-size", type=int, default=10)
    parser.add_argument("--bank-threshold", type=float, default=0.55)
    parser.add_argument("--out", default="ppo_params.pkl")
    parser.add_argument("--resume", default=None)
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
    beliefs = jax.vmap(lambda _: init_belief((height, width)))(jnp.arange(args.envs))

    network = ActorCritic(channels=args.channels, blocks=args.blocks, groups=args.groups)
    params = network.init(
        k_net,
        jnp.zeros((height, width, NUM_PLANES), jnp.float32),
        jnp.zeros((NUM_GLOBALS,), jnp.float32),
        jnp.ones((num_actions(height, width),), bool),
    )
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

    start_update, bank = 0, []
    if blob is not None:
        start_update = blob.get("update", 0)
        bank = [to_jax(b) for b in blob.get("bank", [])]
        if "opt_state" in blob:
            train_state = train_state.replace(
                opt_state=to_jax(blob["opt_state"]), step=blob.get("step", 0)
            )
        if "key" in blob:
            key = jrandom.wrap_key_data(jnp.asarray(blob["key"]))
        print(
            f"resumed {args.resume}: update {start_update}/{args.updates}, "
            f"bank {len(bank)}, optimizer {'restored' if 'opt_state' in blob else 'RESET'}"
        )

    rollout = jax.jit(make_rollout(env, network, pool, ExpanderAgent(), config))
    update_epoch = make_update(network.apply, config)

    @jax.jit
    def learn(train_state, traj, last_value, key):
        advantages, targets = compute_gae(traj, last_value, config["GAMMA"], config["GAE_LAMBDA"])
        carry, info = jax.lax.scan(
            update_epoch, (train_state, traj, advantages, targets, key), None, config["UPDATE_EPOCHS"]
        )
        return carry[0], info

    recent_winrate = []
    for update in range(start_update, args.updates):
        t0 = time.time()
        progress = update / max(args.updates * args.anneal_frac, 1)
        alpha = float(max(0.0, 1.0 - progress))
        w_explore = float(max(0.0, 0.2 * (1.0 - 3.0 * progress)))

        key, k_roll, k_learn, k_opp = jrandom.split(key, 4)
        use_expander = jrandom.uniform(k_opp) < args.expander_prob
        if bank and float(jrandom.uniform(k_opp)) > args.self_prob:
            opp_params = bank[int(jrandom.randint(k_opp, (), 0, len(bank)))]
        else:
            opp_params = train_state.params

        carry, traj, last_value, outcome, done = rollout(
            train_state.params, opp_params, states, beliefs, beliefs, seat,
            use_expander, alpha, w_explore, k_roll,
        )
        states, learner_belief, opp_belief, _ = carry
        beliefs = learner_belief
        train_state, info = learn(train_state, traj, last_value, k_learn)

        episodes = int(done.sum())
        wins = int(((outcome > 0) & done).sum())
        losses = int(((outcome < 0) & done).sum())
        draws = episodes - wins - losses
        elapsed = time.time() - t0
        total_loss, value_loss, actor_loss, ent = [float(jnp.mean(x)) for x in info]

        print(
            f"upd {update:5d} | loss {total_loss:+.3f} v {value_loss:.3f} pi {actor_loss:+.3f} "
            f"H {ent:.2f} | eps {episodes:4d} W/L/D {wins:3d}/{losses:3d}/{draws:3d} "
            f"| alpha {alpha:.2f} | sps {args.envs * args.steps / elapsed:7.0f} | {elapsed:5.1f}s"
        )

        if episodes:
            recent_winrate.append(wins / episodes)
            recent_winrate[:] = recent_winrate[-20:]

        if (update + 1) % args.checkpoint_every == 0:
            # Only bank a checkpoint once it is actually beating the pool.
            # Appending on a fixed timer fills the league with weak versions,
            # which inflates the training win-rate and stops teaching anything.
            strong = (len(recent_winrate) >= 10
                      and sum(recent_winrate) / len(recent_winrate) >= args.bank_threshold)
            if strong:
                bank.append(jax.tree.map(jnp.copy, train_state.params))
                bank[:] = bank[-args.bank_size:]
                recent_winrate.clear()
            save(args.out, train_state, update + 1, bank, key, args)

    save(args.out, train_state, args.updates, bank, key, args)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
