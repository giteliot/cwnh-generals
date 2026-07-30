import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generals import get_observation  # noqa: E402
from generals.agents import ExpanderAgent, HunterAgent, RandomAgent  # noqa: E402
from generals.agents.harvester_agent import HarvesterAgent  # noqa: E402
from generals.core.game import get_info  # noqa: E402
from generals.core.rendering import JaxGameAdapter  # noqa: E402
from generals.gui.gui import GUI  # noqa: E402
from generals.gui.properties import GuiMode  # noqa: E402
from training.features import build_cost  # noqa: E402
from training.play_agent import PPOAgent  # noqa: E402
from training.train import STAGES, make_env  # noqa: E402

AGENTS = {
    "random": RandomAgent,
    "expander": ExpanderAgent,
    "hunter": HunterAgent,
    "harvester": HarvesterAgent,
}

DIRECTION_KEYS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
}
DIRECTION_DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

CURSOR_COLOR = (255, 220, 0)
SOURCE_COLOR = (255, 90, 0)
SQUARE_SIZE = 50


def draw_status(state, player, cursor):
    obs = get_observation(state, player)
    own_structures = (obs.castles | obs.generals) & obs.owned_cells
    cost = int(build_cost(own_structures)[cursor[0], cursor[1]])
    army = int(obs.armies[cursor[0], cursor[1]])
    mine = bool(obs.owned_cells[cursor[0], cursor[1]])
    plain = mine and not bool(obs.castles[cursor[0], cursor[1]] or obs.generals[cursor[0], cursor[1]])
    castles = int(jnp.sum(own_structures)) - 1

    head = f"castles {castles}  |  "
    if not plain:
        text, colour = head + "(B) needs your own plain land", (170, 170, 170)
    elif army >= cost:
        text, colour = head + f"(B) BUILD: {cost}   army {army} -> {army - cost}", (110, 230, 130)
    else:
        text, colour = head + f"(B) build: {cost}   army {army}, need {cost - army} more", (230, 170, 90)

    screen = pygame.display.get_surface()
    font = pygame.font.Font(None, 26)
    label = font.render(text, True, colour)
    box = label.get_rect()
    box.topleft = (6, 4)
    pygame.draw.rect(screen, (0, 0, 0), box.inflate(10, 6))
    screen.blit(label, box)


def draw_cursor(cursor, source):
    screen = pygame.display.get_surface()
    if source is not None:
        pygame.draw.rect(
            screen, SOURCE_COLOR,
            (source[1] * SQUARE_SIZE, source[0] * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3,
        )
    pygame.draw.rect(
        screen, CURSOR_COLOR,
        (cursor[1] * SQUARE_SIZE, cursor[0] * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3,
    )
    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=[*AGENTS, "ppo"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--player", type=int, default=0, choices=[0, 1])
    parser.add_argument("--stage", default="full", choices=STAGES.keys())
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.stage, 256)
    key = jrandom.PRNGKey(args.seed)
    pool, state = env.reset(key)

    human, opp = args.player, 1 - args.player
    if args.agent == "ppo":
        bot = PPOAgent(args.checkpoint, id="ppo", player_idx=opp, temperature=args.temperature)
    else:
        bot = AGENTS[args.agent](id=args.agent)

    names = [None, None]
    names[human], names[opp] = "You", "Bot"
    adapter = JaxGameAdapter(state, names, get_info(state))
    agent_data = {names[human]: {"color": (70, 160, 90)}, names[opp]: {"color": (200, 70, 70)}}
    gui = GUI(adapter, agent_data, mode=GuiMode.TRAIN)
    gui.properties.agent_fov[names[opp]] = False
    pygame.display.toggle_fullscreen()

    cursor = [int(state.general_positions[human][0]), int(state.general_positions[human][1])]
    source = None

    while True:
        gui.render()
        draw_status(state, human, cursor)
        draw_cursor(cursor, source)

        event = pygame.event.wait()
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            break
        if event.type != pygame.KEYDOWN:
            continue

        human_action = None

        if event.key in DIRECTION_KEYS and source is None:
            dr, dc = DIRECTION_DELTA[DIRECTION_KEYS[event.key]]
            cursor[0] += dr
            cursor[1] += dc
        elif event.key == pygame.K_RETURN and source is None:
            source = tuple(cursor)
        elif event.key == pygame.K_ESCAPE:
            source = None
        elif event.key == pygame.K_SPACE:
            human_action = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)
        elif event.key == pygame.K_f:
            pygame.display.toggle_fullscreen()
        elif event.key == pygame.K_b:
            target = source if source is not None else tuple(cursor)
            human_action = jnp.array([2, target[0], target[1], 0, 0], dtype=jnp.int32)
            source = None
        elif event.key in DIRECTION_KEYS and source is not None:
            split = int(bool(pygame.key.get_mods() & pygame.KMOD_SHIFT))
            direction = DIRECTION_KEYS[event.key]
            human_action = jnp.array([0, source[0], source[1], direction, split], dtype=jnp.int32)
            source = None

        if human_action is None:
            continue

        key, k_opp = jrandom.split(key)
        opp_action = bot.act(get_observation(state, opp), k_opp)

        actions = [None, None]
        actions[human], actions[opp] = human_action, opp_action
        timestep, state = env.step(state, jnp.stack(actions), pool)
        adapter.update_from_state(state, timestep.info)

        if timestep.terminated or timestep.truncated:
            gui.render()
            winner = int(timestep.info.winner)
            print(f"Game over at turn {int(state.time)}. Winner: {names[winner] if winner != -1 else 'draw'}")
            pygame.time.wait(3000)
            break

    gui.close()


if __name__ == "__main__":
    main()
