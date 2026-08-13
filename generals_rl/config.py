"""Central training configuration (Phase 11)."""
from dataclasses import dataclass, field


@dataclass
class PPOHyperparams:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    num_epochs: int = 4
    minibatch_size: int = 64
    rollout_length: int = 256
    num_envs: int = 1
    max_grad_norm: float = 0.5


@dataclass
class RewardConfig:
    terminal_win_reward: float = 1.0
    terminal_loss_reward: float = -1.0
    terminal_draw_reward: float = 0.0
    shaping_alpha: float = 0.0
    shaping_max_ratio: float = 10.0


def _default_scripted_weights() -> dict[str, float]:
    return {"expander": 0.10, "hunter": 0.10, "random": 0.05, "harvester": 0.05}


@dataclass
class OpponentConfig:
    # Used only by the single-opponent adapter path (e.g. evaluate.py's CLI).
    # One of: "random", "expander", "hunter", "harvester".
    opponent_type: str = "expander"

    # Self-play opponent pool (Phase 14), used by train().
    pool_size: int = 3
    pool_checkpoint_weight: float = 0.7
    scripted_bot_weights: dict[str, float] = field(default_factory=_default_scripted_weights)
    pool_update_frequency: int = 100  # iterations between promotion checks
    pool_win_rate_window: int = 50  # rolling window of recent pool-game outcomes, from normal rollouts
    pool_win_rate_threshold: float = 0.45


@dataclass
class NetworkConfig:
    num_channels: int = 32


@dataclass
class TrainingConfig:
    seed: int = 0
    total_timesteps: int = 1_000_000
    checkpoint_frequency: int = 100  # iterations; checkpointing not yet implemented (Phase 12)
    evaluation_frequency: int = 100  # iterations; evaluation not yet implemented (Phase 13)
    log_frequency: int = 1  # iterations


@dataclass
class Config:
    ppo: PPOHyperparams = None
    reward: RewardConfig = None
    opponent: OpponentConfig = None
    network: NetworkConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        self.ppo = self.ppo or PPOHyperparams()
        self.reward = self.reward or RewardConfig()
        self.opponent = self.opponent or OpponentConfig()
        self.network = self.network or NetworkConfig()
        self.training = self.training or TrainingConfig()
