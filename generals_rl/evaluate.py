"""Evaluation (Phase 13): play games against one opponent, report win/loss/draw.

Separate from training: no gradient updates, no exploration (greedy actions).
"""
from typing import NamedTuple

import jax.random as jrandom

from generals.agents.agent import Agent
from generals.core.action import compute_valid_move_mask_obs
from generals.core.env import GeneralsEnv
from generals_rl.env_adapter import RLEnvAdapter
from generals_rl.network import ActorCritic, select_greedy_action


class EvaluationResult(NamedTuple):
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_reward: float
    avg_episode_length: float

    def __str__(self):
        return (
            "Evaluation\n"
            "----------\n"
            f"Games: {self.games}\n"
            f"Wins:  {self.wins}\n"
            f"Losses: {self.losses}\n"
            f"Draws:  {self.draws}\n"
            f"Win rate: {self.win_rate * 100:.1f}%\n"
            f"Avg reward: {self.avg_reward:.4f}\n"
            f"Avg episode length: {self.avg_episode_length:.1f}"
        )


def evaluate(
    env: GeneralsEnv,
    model: ActorCritic,
    params,
    opponent: Agent,
    num_games: int,
    key,
    reward_kwargs: dict | None = None,
) -> EvaluationResult:
    reward_kwargs = reward_kwargs or {}
    adapter = RLEnvAdapter(env, **reward_kwargs)

    wins = losses = draws = 0
    total_reward = 0.0
    total_length = 0

    for _ in range(num_games):
        key, reset_key = jrandom.split(key)
        observation, adapter_state, pool = adapter.reset(reset_key)

        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            key, opponent_key = jrandom.split(key)
            own_obs = adapter._get_obs(adapter_state.state, 0)
            action = select_greedy_action(model, params, observation, compute_valid_move_mask_obs(own_obs))
            opponent_obs = adapter._get_obs(adapter_state.state, 1)
            opponent_action = opponent.act(opponent_obs, opponent_key)
            observation, info, done, adapter_state = adapter.step(adapter_state, action, opponent_action, pool)
            episode_reward += float(info["reward"])
            episode_length += 1

        winner = int(info["game_info"].winner)
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        total_reward += episode_reward
        total_length += episode_length

    return EvaluationResult(
        games=num_games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / num_games,
        avg_reward=total_reward / num_games,
        avg_episode_length=total_length / num_games,
    )


def _build_opponent(opponent_type: str) -> Agent:
    from generals.agents.expander_agent import ExpanderAgent
    from generals.agents.harvester_agent import HarvesterAgent
    from generals.agents.hunter_agent import HunterAgent
    from generals.agents.random_agent import RandomAgent

    opponents = {
        "random": RandomAgent,
        "expander": ExpanderAgent,
        "hunter": HunterAgent,
        "harvester": HarvesterAgent,
    }
    return opponents[opponent_type]()


def _main():
    import argparse

    from generals_rl import checkpoint as checkpoint_lib
    from generals_rl.ppo import create_train_state
    from generals_rl.train import _ppo_config_from

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint against an opponent.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--step", type=int, default=None, help="Defaults to the latest checkpoint.")
    parser.add_argument("--opponent", choices=["random", "expander", "hunter", "harvester"], default=None)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--truncation", type=int, default=500)
    args = parser.parse_args()

    step = args.step if args.step is not None else checkpoint_lib.latest_step(args.checkpoint_dir)
    if step is None:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")

    config = checkpoint_lib.load_config(args.checkpoint_dir, step)

    env = GeneralsEnv(grid_dims=(args.grid_size, args.grid_size), truncation=args.truncation)
    model = ActorCritic(num_channels=config.network.num_channels)

    adapter = RLEnvAdapter(env)
    dummy_obs, _, _ = adapter.reset(jrandom.PRNGKey(0))
    params = model.init(jrandom.PRNGKey(0), dummy_obs)
    ppo_config = _ppo_config_from(config)
    train_state_template = create_train_state(model, params, ppo_config)

    train_state, _ = checkpoint_lib.restore(args.checkpoint_dir, step, train_state_template, jrandom.PRNGKey(0))

    opponent = _build_opponent(args.opponent or config.opponent.opponent_type)
    result = evaluate(env, model, train_state.params, opponent, args.num_games, jrandom.PRNGKey(args.seed))
    print(f"Checkpoint: {args.checkpoint_dir} (step {step})")
    print(f"Opponent: {args.opponent or config.opponent.opponent_type}")
    print(result)


if __name__ == "__main__":
    _main()
