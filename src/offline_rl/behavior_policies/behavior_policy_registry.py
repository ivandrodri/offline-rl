from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np

from offline_rl.behavior_policies.custom_2d_grid_policy import (
    behavior_policy_8x8_grid_deterministic_0_0_to_4_7,
    behavior_policy_8x8_suboptimal_initial_0_0_final_0_7,
    move_left_with_noise,
    move_right,
    move_up_from_bottom_5_steps,
)


class BehaviorPolicy2dGridFactory(StrEnum):
    suboptimal_8x8 = "suboptimal_8x8"
    deterministic_8x8 = "deterministic_8x8"
    move_up = "move_up"
    random = "random"
    move_left = "move_left"
    move_right = "move_right"

    def __call__(self, state: np.ndarray, env: gym.Env) -> Any:
        policy_map = {
            BehaviorPolicy2dGridFactory.suboptimal_8x8: behavior_policy_8x8_suboptimal_initial_0_0_final_0_7,
            BehaviorPolicy2dGridFactory.deterministic_8x8: behavior_policy_8x8_grid_deterministic_0_0_to_4_7,
            BehaviorPolicy2dGridFactory.move_up: move_up_from_bottom_5_steps,
            BehaviorPolicy2dGridFactory.random: lambda state, env: env.action_space.sample(),
            BehaviorPolicy2dGridFactory.move_left: move_left_with_noise,
            BehaviorPolicy2dGridFactory.move_right: move_right,
        }
        return policy_map[self](state, env)
