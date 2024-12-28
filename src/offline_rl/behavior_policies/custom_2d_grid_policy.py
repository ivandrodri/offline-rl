import random

import numpy as np

from offline_rl.custom_envs.custom_2d_grid_env.simple_grid import Custom2DGridEnv
from offline_rl.utils import one_hot_to_integer

# MOVES:
#   0: (-1, 0),  # UP
#   1: (1, 0),  # DOWN
#   2: (0, -1),  # LEFT
#   3: (0, 1)  # RIGHT


def behavior_policy_8x8_suboptimal_initial_0_0_final_0_7(
    state: np.ndarray,
    env: Custom2DGridEnv,
) -> int:
    """:param state:
    :param env:
    :return:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    if state_xy[1] == 6:
        return 0
    if state_xy[0] < 4:
        return 1

    return 3


def behavior_policy_8x8_grid_deterministic_0_0_to_4_7(
    state: np.ndarray,
    env: Custom2DGridEnv,
) -> int:
    """Deterministic suboptimal policy to move agent from (4,0) towards (7,7).

    :param state: Agent state
    :param env:
    :return: The action
    :rtype:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)

    # if (4 <= state_xy[1] <= 6) and state_xy[0] == 2:
    #    return 2
    if state_xy[1] == 4:
        action = 0
    elif state_xy[0] == 7:
        action = 3
    else:
        action = 1

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]

    return action


def move_up_from_bottom_5_steps(state: np.ndarray, env: Custom2DGridEnv) -> int:
    """:param state:
    :param env:
    :return:
    """
    state_index = one_hot_to_integer(state)
    state_xy = env.to_xy(state_index)
    if state_xy[0] > 2:
        return 0
    return 1


def move_left_with_noise(state: np.ndarray, env: Custom2DGridEnv) -> int:
    """:param state:
    :param env:
    :return:
    """
    possible_directions = [0, 1, 2, 3]
    weights = [1, 1, 4, 1]

    action = random.choices(possible_directions, weights=weights)[0]

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]

    return action


def move_right(state: np.ndarray, env: Custom2DGridEnv) -> int:
    """:param state:
    :param env:
    :return:
    """
    action = 3

    if not env.discrete_action:
        return np.eye(env.action_space.shape[0])[action]

    return action
