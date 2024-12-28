from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
from tianshou.policy import BasePolicy


class PolicyModel(ABC):
    @abstractmethod
    def default_config(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def create_policy_from_dict(
        self,
        policy_config: dict[str, Any],
        action_space: gym.core.ActType,
        observation_space: gym.core.ObsType,
    ) -> BasePolicy:
        pass
