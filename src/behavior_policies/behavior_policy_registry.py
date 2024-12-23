from enum import Enum
from typing import Callable

from behavior_policies.custom_torcs_policy import get_torcs_expert_policy, get_torcs_expert_policy_with_noise, \
    get_torcs_drunk_driver_policy


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class BehaviorPolicyFactory(Enum):
    random = "random"
    torcs_expert_policy = "torcs_expert_policy"
    torcs_drunk_driver_policy = "torcs_drunk_driver_policy"
    torcs_expert_policy_with_noise = "torcs_expert_policy_with_noise"

    def create_policy(self) -> Callable:
        match self:
            case self.random:
                return lambda action_space: action_space.sample()
            case self.torcs_expert_policy:
                return get_torcs_expert_policy
