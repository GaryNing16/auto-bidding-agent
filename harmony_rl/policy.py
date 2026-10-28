"""Frozen deployment adapter. Only own configuration and own history enter it."""
from pathlib import Path
import torch

from harmony_bidding.agent_api import Agent
from harmony_bidding.agents import make_agent
from .models import ActorCritic, StateEncoder, sample_action


def load_payload(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "harmony-ppo-lstm-v1":
        raise ValueError("Unsupported PPO checkpoint")
    return payload


def model_from_payload(payload):
    model = ActorCritic(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    encoder = StateEncoder.from_dict(payload["encoder"])
    return model, encoder


class PPOAgent(Agent):
    def __init__(self, checkpoint):
        super().__init__()
        self.payload = load_payload(checkpoint)
        self.model, self.encoder = model_from_payload(self.payload)
        self.base = None

    def reset(self, own_config, public_config):
        self.own_config, self.public_config = own_config, public_config
        if self.payload["model_config"]["residual"]:
            self.base = make_agent(self.payload["config"]["residual_spec"])
            self.base.reset(own_config, public_config)

    def bidding(self, observation):
        base_alpha = self.base.bidding(observation) if self.base is not None else 1.
        with torch.no_grad():
            return sample_action(self.model, self.encoder.encode(observation),
                                 base_alpha=base_alpha, deterministic=True)["alpha"]

    def update(self, own_feedback):
        if self.base is not None:
            self.base.update(own_feedback)
