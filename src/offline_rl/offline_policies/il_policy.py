from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tianshou.policy import BasePolicy, ImitationPolicy
from torch import nn

from offline_rl.offline_policies.policy_model import PolicyModel
from offline_rl.utils import extract_dimension

DEFAULT_POLICY_CONFIG = {
    "lr": 0.001,
}


class DQNVector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: int,
        device: str | int | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.device = device
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),  # Adjust input_dim and hidden layers as needed
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Linear(64, action_shape),
            nn.Softmax(),
        )

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        if info is None:
            info = {}
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state


class ILPolicyModel(PolicyModel):
    def default_config(self) -> dict[str, Any]:
        return DEFAULT_POLICY_CONFIG

    def create_policy_from_dict(
        self,
        policy_config: dict[str, Any],
        action_space: gym.core.ActType,
        observation_space: gym.core.ObsType,
    ) -> BasePolicy:
        observation_shape = extract_dimension(observation_space)
        action_shape = extract_dimension(action_space)

        device = "cpu"

        net = DQNVector(observation_shape, action_shape, device=device).to(device)

        optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])
        return ImitationPolicy(actor=net, optim=optim, action_space=action_space)
