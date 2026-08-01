import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generals.agents import ExpanderAgent, HunterAgent, RandomAgent  # noqa: E402
from generals.agents.harvester_agent import HarvesterAgent  # noqa: E402
from generals.core.game import get_info, get_observation  # noqa: E402
from generals.gui import ReplayGUI  # noqa: E402
from generals.gui.properties import GuiMode  # noqa: E402
from training.play_agent import PPOAgent  # noqa: E402
from training.train import STAGES, make_env  # noqa: E402

AGENTS = {
    "random": RandomAgent,
    "expander": ExpanderAgent,
    "hunter": HunterAgent,
    "harvester": HarvesterAgent,
}


def build(spec, player_idx, temperature):
    if spec.startswith("ppo:"):
        path = spec[4:]
        return PPOAgent(path, id=Path(path).stem, player_idx=player_idx, temperature=temperature)
    return AGENTS[spec](id=spec)


def play_game(env, pool, agents, key, max_turns):
    state = jax.tree.map(lambda x: x[jrandom.randint(key, (), 0, pool.armies.shape[0])], pool)
    for agent in agents:
        agent.reset()

    states, infos = [state], [get_info(state)]
    for _ in range(max_turns):
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([
            agents[0].act(get_observation(state, 0), k0),
            agents[1].act(get_observation(state, 1), k1),
        ])
        timestep, next_state = env.step(state, actions, pool)
        states.append(timestep.last_state)
        infos.append(timestep.info)
        state = next_state
        if bool(timestep.terminated | timestep.truncated):
            return states, infos, int(timestep.info.winner)
    return states, infos, -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent0")
    parser.add_argument("agent1")
    parser.add_argument("--stage", default="full", choices=STAGES.keys())
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-turns", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    env = make_env(args.stage, 256)
    pool, _ = env.reset(jrandom.PRNGKey(args.seed))
    agents = [build(args.agent0, 0, args.temperature), build(args.agent1, 1, args.temperature)]
    names = [agents[0].id, agents[1].id]
    print(f"stage={args.stage} board={pool.armies.shape[1]}x{pool.armies.shape[2]}  {names[0]} vs {names[1]}")

    tally = {0: 0, 1: 0, -1: 0}
    last = None
    for game in range(args.games):
        key = jrandom.PRNGKey(args.seed + game)
        states, infos, winner = play_game(env, pool, agents, key, args.max_turns)
        tally[winner] += 1
        final = infos[-1]
        print(
            f"game {game + 1}/{args.games}  turns {len(states) - 1:4d}  "
            f"winner {names[winner] if winner >= 0 else 'draw':10s} "
            f"| land {int(final.land[0]):3d}/{int(final.land[1]):3d} "
            f"army {int(final.army[0]):4d}/{int(final.army[1]):4d}"
        )
        last = (states, infos)

    print(f"\n{names[0]} {tally[0]}  |  {names[1]} {tally[1]}  |  draws {tally[-1]}")

    if not args.no_gui and last is not None:
        gui = ReplayGUI(last[0][0], agent_ids=names, fps=args.fps, mode=GuiMode.REPLAY, start_paused=True)
        gui.play(last[0], last[1])


if __name__ == "__main__":
    main()
