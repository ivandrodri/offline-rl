import pytest

from offline_rl.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import (
    ObstacleTypes,
)
from offline_rl.custom_envs.env_wrappers import Grid2DInitialConfig
from offline_rl.generate_custom_minari_datasets.generate_minari_dataset import (
    create_minari_datasets,
)


class TestMinari:
    @pytest.mark.parametrize(
        "grid_obstacles",
        [
            ObstacleTypes.obstacle_8x8_top_right,
            ObstacleTypes.obst_free_8x8,
        ],
    )
    def test_create_minari_dataset_and_check_metadata(
        self,
        grid_obstacles,
        get_grid_8x8_env,
        random_grid_policy,
        set_test_env_variables,
    ):
        env = get_grid_8x8_env
        test_data_identifier = "_collected_data_test"
        num_traj_steps = 3

        grid_env_config = Grid2DInitialConfig(
            obstacles=grid_obstacles,
        )

        data_set_config = create_minari_datasets(
            env_name=env.unwrapped.spec.id,
            dataset_identifier=test_data_identifier,
            num_collected_points=num_traj_steps,
            behavior_policy=random_grid_policy,
            env_2d_grid_initial_config=grid_env_config,
        )

        assert data_set_config.env_name == env.spec.id
        assert data_set_config.num_steps == num_traj_steps
        assert data_set_config.behavior_policy == random_grid_policy

    def test_load_minari(self):
        pass

    def test_combine_minari(self):
        pass
