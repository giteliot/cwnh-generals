"""Checkpointing (Phase 12): actor/critic params, optimizer state, step, RNG key, config.

Params/opt_state/key are saved via orbax (array pytree). Config is saved as a
JSON sidecar file since it's plain Python scalars, not an array pytree.
"""
import dataclasses
import json
from pathlib import Path

import orbax.checkpoint as ocp

from generals_rl.config import Config, NetworkConfig, OpponentConfig, PPOHyperparams, RewardConfig, TrainingConfig
from generals_rl.ppo import TrainState


def _manager(checkpoint_dir: str | Path) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(Path(checkpoint_dir).resolve() / "state")


def save(checkpoint_dir: str | Path, step: int, train_state: TrainState, key, config: Config) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    manager = _manager(checkpoint_dir)
    state = {"params": train_state.params, "opt_state": train_state.opt_state, "key": key}
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()

    config_path = checkpoint_dir / f"config_{step}.json"
    config_path.write_text(json.dumps(dataclasses.asdict(config), indent=2))


def restore(checkpoint_dir: str | Path, step: int, train_state_template: TrainState, key_template) -> tuple[TrainState, object]:
    """Restore params/opt_state/key. `train_state_template`/`key_template` give the
    expected pytree structure and shapes (e.g. from a freshly-initialized model)."""
    manager = _manager(checkpoint_dir)
    target = {
        "params": train_state_template.params,
        "opt_state": train_state_template.opt_state,
        "key": key_template,
    }
    restored = manager.restore(step, args=ocp.args.StandardRestore(target))
    train_state = TrainState(params=restored["params"], opt_state=restored["opt_state"])
    return train_state, restored["key"]


def load_config(checkpoint_dir: str | Path, step: int) -> Config:
    config_path = Path(checkpoint_dir) / f"config_{step}.json"
    data = json.loads(config_path.read_text())
    return Config(
        ppo=PPOHyperparams(**data["ppo"]),
        reward=RewardConfig(**data["reward"]),
        opponent=OpponentConfig(**data["opponent"]),
        network=NetworkConfig(**data["network"]),
        training=TrainingConfig(**data["training"]),
    )


def latest_step(checkpoint_dir: str | Path) -> int | None:
    return _manager(checkpoint_dir).latest_step()
