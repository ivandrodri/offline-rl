import os

import gymnasium as gym
import pytest

from offline_rl.behavior_policies.behavior_policy_registry import (
    BehaviorPolicy2dGridFactory,
)
from offline_rl.custom_envs.custom_envs_registration import EnvFactory
from offline_rl.set_env_variables import set_env_variables


@pytest.fixture()
def get_grid_8x8_env() -> gym.Env:
    return EnvFactory.Grid_2D_8x8_discrete.get_env()


@pytest.fixture()
def random_grid_policy(get_grid_8x8_env) -> BehaviorPolicy2dGridFactory:
    return BehaviorPolicy2dGridFactory.random


@pytest.fixture()
def set_test_env_variables():
    current_directory = os.path.dirname(__file__)
    set_env_variables(current_directory)
