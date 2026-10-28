"""One serializable configuration for training, validation and deployment."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

from harmony_baselines.five_scenarios.common import OUT, SCENARIOS


VARIANTS = {
    "E1": ("PPO-Raw", False, False, False),
    "E2": ("PPO-Aug", True, False, False),
    "E3": ("PPO-LSTM", False, True, False),
    "E4": ("PPO-LSTM-Aug", True, True, False),
    "E5": ("Residual-PPO", False, False, True),
    "E6": ("Residual-PPO-LSTM", False, True, True),
}


@dataclass
class TrainConfig:
    variant: str = "E1"
    seed: int = 20261028
    episodes: int = 96
    batch_episodes: int = 8
    workers: int = 4
    max_seconds: float = 1800.
    hidden: int = 64
    learning_rate: float = 3e-4
    lstm_learning_rate: float = 1e-3
    lstm_hidden: int = 64
    lstm_epochs: int = 2
    replay_capacity: int = 256
    validate_every: int = 6
    early_stopping_patience: int = 10
    train_days: tuple = ("period-7", "period-8")
    validation_days: tuple = ("period-21",)
    test_days: tuple = ("period-24",)
    validation_roles: tuple = ("0", "8", "16", "24", "32", "40")
    scenarios: tuple = SCENARIOS
    market_probabilities: tuple = (.6, .3, .1)
    budget_mode: str = "original"
    budget_range: tuple = (.8, 1.2)
    opponent_randomization: float = .1
    opponents_path: str = str(OUT / "opponents_frozen.json")
    residual_spec: dict | None = None
    initial_alpha: float = 178.97980036190066
    smoke: bool = False
    ppo: dict = field(default_factory=lambda: {
        "gamma": 1., "gae_lambda": .95, "clip_epsilon": .2,
        "epochs": 4, "minibatch_size": 256, "value_coef": .5,
        "entropy_coef": .005, "max_grad_norm": .5, "target_kl": .03,
        "reward_scale": 1.,
    })

    @property
    def algorithm(self):
        return VARIANTS[self.variant][0]

    @property
    def augmentation(self):
        return VARIANTS[self.variant][1]

    @property
    def redistribution(self):
        return VARIANTS[self.variant][2]

    @property
    def residual(self):
        return VARIANTS[self.variant][3]

    def validate(self):
        if self.variant not in VARIANTS:
            raise ValueError("Unknown experiment variant")
        for key in ("episodes", "batch_episodes", "workers", "hidden", "validate_every", "replay_capacity"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("A finite positive runtime limit is mandatory")
        splits = list(map(set, (self.train_days, self.validation_days, self.test_days)))
        if not all(splits) or any(splits[a] & splits[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Train, validation and test complete periods must be disjoint")
        if not set(self.scenarios).issubset(SCENARIOS) or not self.scenarios:
            raise ValueError("Unknown/empty scenario set")
        if self.ppo.get("gamma") != 1.:
            raise ValueError("This experiment conserves undiscounted daily value; gamma must be 1")
        if len(self.market_probabilities) != 3 or min(self.market_probabilities) < 0 or abs(sum(self.market_probabilities)-1) > 1e-8:
            raise ValueError("Historical/bootstrap/stress probabilities must sum to one")
        if not 0 <= self.opponent_randomization <= .25:
            raise ValueError("Opponent randomization must be within the declared local range [0,.25]")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data).validate()
