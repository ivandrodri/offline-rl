from typing import Any

import gymnasium as gym
import tianshou
import torch
from tianshou.policy import BasePolicy
from tianshou.utils.net.common import Net

from offline_rl.offline_policies.policy_model import PolicyModel
from offline_rl.utils import extract_dimension

DEFAULT_POLICY_CONFIG = {
    "lr": 0.01,
    "gamma": 0.99,
    "device": "cpu",
    "hidden_sizes": [256, 256],
    "n_steps": 5,
    "target_freq": 300,
    "epsilon": 1.0,  # exploration noise
}


class DQNPolicyModel(PolicyModel):
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

        device = policy_config["device"]

        net = Net(
            state_shape=observation_shape,
            action_shape=action_shape,
            hidden_sizes=policy_config["hidden_sizes"],
            device=device,
        )
        optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])

        policy = tianshou.policy.DQNPolicy(
            model=net,
            optim=optim,
            action_space=action_space,
            discount_factor=policy_config["gamma"],
            estimation_step=policy_config["n_steps"],
            target_update_freq=policy_config["target_freq"],
        )
        policy.set_eps(policy_config["epsilon"])

        return policy
