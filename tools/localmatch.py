"""Competition-faithful local match driver.

Why this exists instead of `competition/matchup.py`: that runner drives
`game.step` directly, which bypasses `env.step` — the only place the
build-castles and deathtouch modifiers are applied. Under `--mode competition`
it therefore plays a castle-less, deathtouch-less game, and a build action
`[2, r, c, 0, 0]` is silently executed as a move upward. It also enforces no
per-turn time limit. See ARCHITECTURE.md ("Traps").

This driver speaks the identical wire protocol to the identical `run.sh`
subprocesses, but steps through `env.step` with `mode="competition"` so the
modifiers apply, and it times every reply against the published 150 ms budget.
Nothing under `competition/` is modified — `protocol.py` is imported read-only
so the frame format can never drift from the real one.

    python tools/localmatch.py                             # two expanders, 1 game
    python tools/localmatch.py A/run.sh B/run.sh --games 10 --swap
    python tools/localmatch.py A/run.sh B/run.sh --gui     # replay the last game

Caveats, so results aren't over-trusted:
  - Fault semantics are inferred. The docs say a bot forfeits at 50 faults in
    one game and that a crash forfeits immediately, but don't enumerate what
    counts as a fault. Here: a reply over budget, or an unparseable reply.
  - An over-budget reply is counted but still played. The real grader has to do
    something at the deadline (most likely substitute a pass), so a bot that
    routinely overruns will do *worse* there than here. Treat the latency
    report, not the win rate, as the signal for a slow bot.
  - Per-game board dimensions are sampled per seed and the general-distance
    floor is scaled to 0.8*min(h, w), following the eval-driver behaviour
    described in the preset comments in generals/core/env.py.
"""
import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import jax.random as jrandom

REPO_ROOT = Path(__file__).resolve().parent.parent
# protocol.py lives in competition/ and is not part of the installed `generals`
# package (setuptools ships only `generals*`), so it has to be imported by path.
sys.path.insert(0, str(REPO_ROOT / "competition"))

from generals import GeneralsEnv  # noqa: E402
from generals.core import game  # noqa: E402
from protocol import decode_action, encode_handshake, encode_observation  # noqa: E402

DEFAULT_AGENT = REPO_ROOT / "competition" / "agents" / "expander_python" / "run.sh"

# Published limits (https://www.generals.bot/docs). Not enforced by the engine.
TURN_BUDGET_MS = 150.0
FIRST_MOVE_BUDGET_MS = 10_000.0   # a model gets time to load on move one
MAX_FAULTS = 50                   # this many in one game forfeits it

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


class Forfeit(Exception):
    """A bot lost the game outright (crash, or too many faults)."""

    def __init__(self, player: int, reason: str):
        super().__init__(f"player {player} forfeits: {reason}")
        self.player = player
        self.reason = reason


class Bot:
    """One subprocess agent, with per-turn timing and fault accounting."""

    def __init__(self, run_sh: Path, player_id: int, H: int, W: int):
        self.path = run_sh
        self.name = run_sh.parent.name
        self.player_id = player_id
        self.faults = 0
        self.latencies_ms: list[float] = []

        self.proc = subprocess.Popen(
            ["bash", str(run_sh)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            bufsize=1, text=True, cwd=str(run_sh.parent),
        )
        self.proc.stdin.write(encode_handshake(player_id, H, W))
        self.proc.stdin.flush()

    def ask(self, obs, turn: int) -> jnp.ndarray:
        """Send one observation frame, time the reply, return the action."""
        budget = FIRST_MOVE_BUDGET_MS if turn == 0 else TURN_BUDGET_MS
        try:
            self.proc.stdin.write(encode_observation(obs))
            self.proc.stdin.flush()
            t0 = time.perf_counter()
            line = self.proc.stdout.readline()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        except (BrokenPipeError, ValueError) as e:
            raise Forfeit(self.player_id, f"pipe closed ({type(e).__name__})") from e

        if not line:
            raise Forfeit(self.player_id, "closed stdout (crashed?)")

        self.latencies_ms.append(elapsed_ms)
        if elapsed_ms > budget:
            self._fault(f"turn {turn}: {elapsed_ms:.0f} ms > {budget:.0f} ms budget")

        try:
            action = decode_action(line)
            if action.shape != (5,):
                raise ValueError(f"expected 5 ints, got {action.shape[0]}")
        except (ValueError, IndexError) as e:
            # Malformed output is a fault; the engine treats it as a pass.
            self._fault(f"turn {turn}: unparseable reply {line.strip()!r} ({e})")
            return PASS
        return action

    def _fault(self, detail: str) -> None:
        self.faults += 1
        if self.faults <= 3 or self.faults % 10 == 0:
            print(f"  [fault {self.faults}] {self.name}: {detail}", file=sys.stderr)
        if self.faults >= MAX_FAULTS:
            raise Forfeit(self.player_id, f"{self.faults} faults")

    def close(self) -> None:
        # The agent treats EOF on stdin as "game over, exit cleanly".
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def stats(self) -> str:
        if not self.latencies_ms:
            return "no replies"
        # Skip move one: it gets a 10 s allowance and would swamp the tail.
        tail = self.latencies_ms[1:] or self.latencies_ms
        p95 = statistics.quantiles(tail, n=20)[-1] if len(tail) >= 20 else max(tail)
        over = sum(1 for ms in tail if ms > TURN_BUDGET_MS)
        return (f"p50 {statistics.median(tail):6.1f} ms | p95 {p95:6.1f} ms | "
                f"max {max(tail):7.1f} ms | over budget {over}/{len(tail)}")


def build_agent(run_sh: Path) -> None:
    """Run `build.sh` next to `run.sh` if one exists. Aborts on failure."""
    build = run_sh.parent / "build.sh"
    if not build.exists():
        return
    print(f"[localmatch] building {build.parent.name}", file=sys.stderr)
    if subprocess.run(["bash", str(build)], cwd=str(build.parent)).returncode != 0:
        sys.exit(f"[localmatch] build failed: {build}")


def make_initial_state(env: GeneralsEnv, key: jnp.ndarray) -> tuple[object, tuple[int, int]]:
    """One competition board: sides drawn independently in [18, 21], padded to 22.

    env.init_state() would always produce max_grid_size squared, so it is not
    used here. The general-distance floor is scaled per board the way the eval
    driver is documented to (0.8*min(h, w)) rather than left at the pool-wide 14.
    """
    k_dims, k_grid = jrandom.split(key)
    h, w = (int(x) for x in jrandom.randint(
        k_dims, (2,), env.min_grid_size, env.max_grid_size + 1))
    env.min_generals_distance = round(0.8 * min(h, w))
    return env._make_single_state_fixed(k_grid, h, w), (h, w)


def play(env, pool, paths: tuple[Path, Path], seed: int, record: bool):
    """Run one game. Returns (winner, turns, dims, bots, states_log, infos_log)."""
    state, dims = make_initial_state(env, jrandom.PRNGKey(seed))
    H = W = env.pad_to
    get_obs = game.get_full_observation if env.perfect_info else game.get_observation

    states_log = [state] if record else None
    infos_log = [game.get_info(state)] if record else None

    bots = [Bot(paths[0], 0, H, W), Bot(paths[1], 1, H, W)]
    winner, turn = -1, 0
    try:
        while True:
            try:
                actions = jnp.stack([bots[i].ask(get_obs(state, i), turn) for i in (0, 1)])
            except Forfeit as f:
                print(f"  {f}", file=sys.stderr)
                return 1 - f.player, turn, dims, bots, states_log, infos_log

            # env.step is the whole point: it applies build-castles and
            # deathtouch, then auto-resets from the pool once the game ends —
            # so the post-step `state` is already the *next* game's board.
            # timestep.last_state is this game's true final state.
            timestep, state = env.step(state, actions, pool)
            turn += 1

            if record:
                states_log.append(timestep.last_state)
                infos_log.append(timestep.info)

            if bool(timestep.terminated) or bool(timestep.truncated):
                winner = int(timestep.info.winner)
                break
    finally:
        for bot in bots:
            bot.close()

    return winner, turn, dims, bots, states_log, infos_log


def replay(states_log, infos_log, names, fps):
    from generals.gui import ReplayGUI
    from generals.gui.properties import GuiMode

    gui = ReplayGUI(states_log[0], agent_ids=names, fps=fps,
                    mode=GuiMode.REPLAY, start_paused=True)
    print("[localmatch] replay open — controls are shown in the window")
    gui.play(states_log, infos_log)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent0", nargs="?", default=str(DEFAULT_AGENT), help="path to a run.sh")
    ap.add_argument("agent1", nargs="?", default=str(DEFAULT_AGENT), help="path to a run.sh")
    ap.add_argument("--games", type=int, default=1, help="games to play (default: 1)")
    ap.add_argument("--seed", type=int, default=0, help="base seed; game N uses seed+N")
    ap.add_argument("--swap", action="store_true",
                    help="alternate which bot is player 0 (halves seat bias)")
    ap.add_argument("--gui", action="store_true", help="replay the final game")
    ap.add_argument("--fps", type=int, default=8, help="replay frame rate (default: 8)")
    args = ap.parse_args()

    paths = [Path(args.agent0).resolve(), Path(args.agent1).resolve()]
    for p in paths:
        if not p.exists():
            sys.exit(f"agent script not found: {p}")
    build_agent(paths[0])
    if paths[1] != paths[0]:
        build_agent(paths[1])

    # A small pool: env.step needs one to auto-reset into, but we stop at the
    # first terminal step, so its contents are never actually played.
    env = GeneralsEnv(mode="competition", pool_size=32)
    print(f"[localmatch] mode=competition  truncation={env.truncation}  "
          f"deathtouch={env.deathtouch_turn}  build_castles={env.build_castles}  "
          f"fog={not env.perfect_info}")
    pool, _ = env.reset(jrandom.PRNGKey(args.seed))

    names = [p.parent.name for p in paths]
    if names[0] == names[1]:  # self-play: keep the summary rows distinguishable
        names = [f"{names[0]}#0", f"{names[1]}#1"]
    wins = [0, 0]
    draws = 0
    latencies: list[list[float]] = [[], []]
    faults = [0, 0]
    last = None

    for g in range(args.games):
        flip = args.swap and g % 2 == 1
        seats = [paths[1], paths[0]] if flip else [paths[0], paths[1]]
        record = args.gui and g == args.games - 1

        winner, turns, dims, bots, states_log, infos_log = play(
            env, pool, seats, args.seed + g, record)

        # Map seat index back to bot index.
        for seat, bot in enumerate(bots):
            bot_idx = (1 - seat) if flip else seat
            latencies[bot_idx].extend(bot.latencies_ms[1:])
            faults[bot_idx] += bot.faults

        if winner < 0:
            draws += 1
            outcome = "draw"
        else:
            bot_idx = (1 - winner) if flip else winner
            wins[bot_idx] += 1
            outcome = f"{names[bot_idx]} wins"

        seat_note = " (swapped)" if flip else ""
        print(f"game {g + 1:>3}/{args.games}  seed {args.seed + g:<5} "
              f"{dims[0]}x{dims[1]}{seat_note:<10}  turn {turns:<5} {outcome}")
        for bot in bots:
            print(f"    {bot.name:<20} {bot.stats()}")
        if record:
            last = (states_log, infos_log, [b.name for b in bots])

    if args.games > 1:
        print(f"\n=== {args.games} games ===")
        for i, name in enumerate(names):
            pct = 100.0 * wins[i] / args.games
            tail = latencies[i]
            lat = (f"p50 {statistics.median(tail):.1f} ms, max {max(tail):.1f} ms"
                   if tail else "no data")
            print(f"  {name:<20} {wins[i]:>3} wins ({pct:5.1f}%)  "
                  f"faults {faults[i]:<4} {lat}")
        print(f"  {'draws':<20} {draws:>3}")

    if last:
        replay(*last, args.fps)


if __name__ == "__main__":
    main()
