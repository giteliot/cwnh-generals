import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jrandom

from training.features import NUM_GLOBALS, init_belief
from training.network import ActorCritic
from training.potential import Outcome
from training.train import SCRIPTED, STAGES, build_scripted, make_env, make_rollout, to_jax

# Shaping is switched off: the gauntlet reads the engine's raw +1/-1/0 only, so
# reward tuning cannot move the ranking.
EVAL_CONFIG = dict(
    GAMMA=0.9999,
    OUTCOME=Outcome(),
    POTENTIAL_WEIGHTS=dict(econ=0.0, material=0.0, territory=0.0, safety=0.0, threat=0.0, explore=0.0),
)


def entries(paths, with_bank):
    out = []
    for path in paths:
        blob = pickle.load(open(path, "rb"))
        stem = Path(path).stem
        out.append((stem, to_jax(blob["params"])))
        if with_bank:
            out += [(f"{stem}:b{i}", to_jax(b)) for i, b in enumerate(blob["bank"])]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="rank checkpoints by win rate against the fixed scripted bots -- the absolute "
                    "yardstick that self-play win% cannot provide")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--bank", action="store_true", help="also rank every banked policy inside each checkpoint")
    ap.add_argument("--stage", default="full", choices=STAGES.keys())
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--turns", type=int, default=1600)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = pickle.load(open(args.checkpoints[0], "rb"))
    env = make_env(args.stage, max(args.envs, 64))
    pool, _ = env.reset(jrandom.PRNGKey(args.seed))
    height, width = pool.armies.shape[1:]

    # The critic's opponent one-hot has changed width across runs (0 before critic
    # conditioning, 6 with RandomAgent in the pool, 5 without). Read it off the value
    # head instead of the current constant, so old checkpoints still load.
    kinds = spec["params"]["params"]["Dense_2"]["kernel"].shape[0] - (2 * spec["channels"] + NUM_GLOBALS)
    print(f"{Path(args.checkpoints[0]).name}: {spec['channels']}ch x {spec['blocks']} blocks, "
          f"critic sees {kinds} opponent slots")

    network = ActorCritic(channels=spec["channels"], blocks=spec["blocks"], groups=spec["groups"])
    rollout = jax.jit(make_rollout(env, network, pool, build_scripted(), dict(EVAL_CONFIG, NUM_STEPS=args.chunk)))

    base = jax.tree.map(lambda x: x[: args.envs], pool)._replace(
        pool_idx=jnp.arange(args.envs, dtype=jnp.int32))
    seat = (jnp.arange(args.envs) % 2).astype(bool)
    span = min(env.truncation, 700)

    def fresh():
        return jax.vmap(lambda _: init_belief((height, width)))(jnp.arange(args.envs))

    rows = []
    for name, params in entries(args.checkpoints, args.bank):
        record = {}
        for kind, bot in enumerate(SCRIPTED):
            key = jrandom.fold_in(jrandom.PRNGKey(args.seed), kind)
            state = base._replace(
                time=jrandom.randint(key, (args.envs,), 0, span).astype(base.time.dtype))
            belief, opp_belief, ret = fresh(), fresh(), jnp.zeros((args.envs,), jnp.float32)
            hot = jnp.zeros((kinds,), jnp.float32)
            wins = losses = draws = 0
            for _ in range(args.turns // args.chunk):
                carry, _, _, raw, _, _, done = rollout(
                    params, params, state, belief, opp_belief, ret, seat, kind, hot, 0.0, 0.0, key)
                state, belief, opp_belief, ret, key = carry
                wins += int(((raw > 0) & done).sum())
                losses += int(((raw < 0) & done).sum())
                draws += int(((raw == 0) & done).sum())
            record[bot] = (wins, losses, draws)
        played = sum(sum(v) for v in record.values())
        mean = sum(v[0] for v in record.values()) / played if played else 0.0
        rows.append((mean, name, record))
        print(f"  {name:<28} mean win {mean:6.1%}", flush=True)

    width_ = 30 + 14 * len(SCRIPTED) + 11
    print(f"\n{args.stage} board, {args.envs} envs x {args.turns} turns, sampled actions")
    print(f"{'checkpoint':<30}" + "".join(f"{b:>14}" for b in SCRIPTED) + f"{'mean':>11}")
    print("-" * width_)
    for mean, name, record in sorted(rows, reverse=True):
        cells = ""
        for bot in SCRIPTED:
            w, l, d = record[bot]
            n = w + l + d
            cells += f"{(w / n if n else 0):>13.1%} "
        print(f"{name:<30}{cells}{mean:>10.1%}")


if __name__ == "__main__":
    main()
