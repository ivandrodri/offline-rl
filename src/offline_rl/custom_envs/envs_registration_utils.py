from dataclasses import dataclass
from enum import StrEnum

from gymnasium import register as gymnasium_register

from offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import (
    ObstacleTypes,
)

ENTRY_POINT_2D_GRID = "offline_rl.custom_envs.custom_2d_grid_env.simple_grid:Custom2DGridEnv"


class RenderMode(StrEnum):
    RGB_ARRAY_LIST = "rgb_array_list"
    RGB_ARRAY = "rgb_array"
    HUMAN = "human"


@dataclass
class GridEnvConfig:
    env_name: str
    obstacles: list[str] | None = None
    render_mode: RenderMode = None
    discrete_action: bool = True
    max_episode_steps: int = 50

    def __post_init__(self):
        if self.obstacles:
            # Compute GRID_DIMS from OBSTACLES if obstacles are provided
            num_rows = len(self.obstacles)
            num_cols = len(self.obstacles[0]) if num_rows > 0 else 0
            if not (num_rows > 0 and num_cols > 0):
                raise ValueError("To use obstacle maps The grid must be two dimensional!")


def _register_custom_grid_envs(
    env_name: str,
    obstacles: ObstacleTypes,
    discrete_action: bool,
    render_mode: RenderMode,
):
    """2D grid world environment registration in order to use it as you will do with any other gymnasium environments.

    :param env_name:
    :param obstacles:
    :param discrete_action:
    :param render_mode:
    :return:
    """

    def register_custom_grid_env(grid_env_config: GridEnvConfig):
        gymnasium_register(
            id=grid_env_config.env_name,
            entry_point=ENTRY_POINT_2D_GRID,
            max_episode_steps=grid_env_config.max_episode_steps,
            kwargs={
                "obstacle_map": grid_env_config.obstacles,
                "discrete_action": grid_env_config.discrete_action,
            },
        )

    config = {
        "env_name": env_name,
        "obstacles": obstacles,
        "discrete_action": discrete_action,
        "render_mode": render_mode,
    }
    config = GridEnvConfig(**config)
    register_custom_grid_env(config)


def _register_half_cheetah_v5_env(max_episode_steps=50, env_name: str = "HalfCheetah-v5"):
    """HalfCheetah environment with a custom max_episode_steps. Take it as an example for other
    gymnasium environments.

    :param max_episode_steps:
    :param env_name:
    :return:
    """
    entry_point = "gymnasium.envs.mujoco:HalfCheetahEnv"
    reward_threshold = 4800.0

    return gymnasium_register(
        id=env_name,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        reward_threshold=reward_threshold,
    )
