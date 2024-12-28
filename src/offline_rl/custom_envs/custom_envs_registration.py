from enum import StrEnum

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RenderCollection

from offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import (
    ObstacleTypes,
)
from offline_rl.custom_envs.env_wrappers import (
    Grid2DInitialConfig,
    InitialConfigCustom2DGridEnvWrapper,
)
from offline_rl.custom_envs.envs_registration_utils import (
    RenderMode,
    _register_custom_grid_envs,
    _register_half_cheetah_v5_env,
)

# ToDo or not ToDo: This Factory could be more simple to register a new environment but it could be also fine as it is
#   user friendly.

# To add a new environment:
# 1 - Create a new entry in CustomEnv,
# 2 - Register the environment in _register_grid_envs. For a 2D grid world just add a new config to grid_configs
# 3 - add the environment to EnvFactory

MAX_EPISODE_STEPS_HALF_CHEETAH = 5


class CustomEnv(StrEnum):
    HalfCheetah_v5 = "HalfCheetah-v5"
    Grid_2D_4x4_discrete = "Grid_2D_4x4_discrete"
    Grid_2D_4x4_continuous = "Grid_2D_4x4_continuous"
    Grid_2D_6x6_discrete = "Grid_2D_6x6_discrete"
    Grid_2D_6x6_continuous = "Grid_2D_6x6_continuous"
    Grid_2D_8x8_continuous = "Grid_2D_8x8_continuous"
    Grid_2D_8x8_discrete = "Grid_2D_8x8_discrete"


def _register_grid_envs():
    """Register custom environments."""
    grid_configs = [
        (CustomEnv.Grid_2D_4x4_discrete, ObstacleTypes.obst_free_4x4.value, True),
        (CustomEnv.Grid_2D_4x4_continuous, ObstacleTypes.obst_free_4x4.value, False),
        (CustomEnv.Grid_2D_6x6_discrete, ObstacleTypes.obst_free_6x6.value, True),
        (CustomEnv.Grid_2D_6x6_continuous, ObstacleTypes.obst_free_6x6.value, False),
        (CustomEnv.Grid_2D_8x8_discrete, ObstacleTypes.obst_free_8x8.value, True),
        (CustomEnv.Grid_2D_8x8_continuous, ObstacleTypes.obst_free_8x8.value, False),
    ]
    for env_name, obstacles, discrete_action in grid_configs:
        _register_custom_grid_envs(
            env_name=env_name,
            obstacles=obstacles,
            discrete_action=discrete_action,
            render_mode=RenderMode.RGB_ARRAY,
        )

    _register_half_cheetah_v5_env(
        max_episode_steps=MAX_EPISODE_STEPS_HALF_CHEETAH,
    )


class EnvFactory(StrEnum):
    HalfCheetah_v5 = CustomEnv.HalfCheetah_v5
    Grid_2D_4x4_discrete = CustomEnv.Grid_2D_4x4_discrete
    Grid_2D_4x4_continuous = CustomEnv.Grid_2D_4x4_continuous
    Grid_2D_6x6_discrete = CustomEnv.Grid_2D_6x6_discrete
    Grid_2D_6x6_continuous = CustomEnv.Grid_2D_6x6_continuous
    Grid_2D_8x8_continuous = CustomEnv.Grid_2D_8x8_continuous
    Grid_2D_8x8_discrete = CustomEnv.Grid_2D_8x8_discrete

    def get_env(
        self,
        render_mode: RenderMode | None = None,
        grid_config: Grid2DInitialConfig = None,
    ) -> Env:
        """Get a gym environment based on the factory configuration.

        :param render_mode:
        :param grid_config: if a 2D grid world environment is created then you can provide a custom initial
            configuration with your own obstacles and/or custom initial states and target position.
        :return: Gymnasium Environment
        """
        _register_grid_envs()
        match self:
            case self.Grid_2D_4x4_discrete | self.Grid_2D_4x4_continuous | self.Grid_2D_6x6_discrete | self.Grid_2D_6x6_continuous | self.Grid_2D_8x8_discrete | self.Grid_2D_8x8_continuous:
                env = (
                    gym.make(self)
                    if render_mode is None
                    else RenderCollection(gym.make(self, render_mode=render_mode.value))
                )
                return InitialConfigCustom2DGridEnvWrapper(env, env_config=grid_config)

            case self.HalfCheetah_v5:
                return gym.make(self)
