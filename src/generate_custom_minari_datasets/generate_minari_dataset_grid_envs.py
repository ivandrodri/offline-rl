from load_env_variables import load_env_variables
load_env_variables()

from generate_custom_minari_datasets.utils import generate_compatible_minari_dataset_name
from torcs_gymnasium.src.torcs_lidar_environment.torcs_lidar_env import TorcsLidarEnv
from utils import ignore_keyboard_interrupt, delete_minari_data_if_exists
import json
import os
import minari
import numpy as np
import gymnasium as gym
from behavior_policies.behavior_policy_registry import BehaviorPolicyFactory
from custom_envs.custom_envs_registration import EnvFactory
from dataclasses import asdict, dataclass
from typing import Dict, Tuple, List, Sequence
from minari import DataCollector, combine_datasets
from minari.data_collector.callbacks import StepDataCallback
from minari.storage import get_dataset_path

OVERRIDE_DATA_SET = True


@dataclass
class MinariDatasetConfig:
    env_name: str
    data_set_name: str
    num_steps: int
    behavior_policy: BehaviorPolicyFactory = None
    children_dataset_names: List[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(**config_dict)

    def save_to_file(self):
        data_set_path = get_dataset_path(self.data_set_name)
        file_name = "config.json"

        obj_to_saved = asdict(self)
        obj_to_saved["behavior_policy"] = self.behavior_policy.value

        with open(os.path.join(data_set_path, file_name), "w") as file:
            json.dump(obj_to_saved, file, indent=4)

    @classmethod
    def load_from_file(cls, dataset_id):
        filename = get_dataset_path(dataset_id)
        with open(os.path.join(filename, "config.json"), "r") as file:
            config_dict = json.load(file)

        config_dict["behavior_policy"] = BehaviorPolicyFactory[config_dict["behavior_policy"]]
        return cls(**config_dict)


def create_minari_collector_env_wrapper(
    env_name: str,
):
    """
    Creates a wrapper 'DataCollector' around the environment in order to collect data for minari dataset

    :param env_name: One of the environments defined in Gymnasium or in EnvFactory
    :return:
    """

    if env_name in EnvFactory.__members__.values():
        env = EnvFactory[env_name].get_env(
        )
    else:
        try:
            env = gym.make(env_name)
        except gym.error.UnregisteredEnv as e:
            print(f"Error: The environment is neither from Gymnasium or registered in EnvFactory {e}.")

    class CustomSubsetStepDataCallback(StepDataCallback):
        def __call__(self, env, **kwargs):
            step_data = super().__call__(env, **kwargs)
            # del step_data["observations"]["achieved_goal"]
            return step_data

    env = DataCollector(
        env,
        step_data_callback=CustomSubsetStepDataCallback,
        record_infos=False,
    )
    return env


def create_minari_config(
    env_name: str,
    dataset_name: str,
    dataset_identifier: str,
    version_dataset: str,
    num_steps: int,
    behavior_policy_name: BehaviorPolicyFactory,
) -> MinariDatasetConfig:
    name_expert_data = generate_compatible_minari_dataset_name(
        env_name, dataset_name, version_dataset
    )

    dataset_name += dataset_identifier

    dataset_config = {"env_name": env_name, "data_set_name": name_expert_data, "num_steps": num_steps,
                      "behavior_policy": behavior_policy_name}

    return MinariDatasetConfig.from_dict(dataset_config)


@ignore_keyboard_interrupt
def create_minari_datasets(
    env_name: str,
    dataset_name: str = "data",
    dataset_identifier: str = "",
    version_dataset: str = "v0",
    num_colected_points: int = 1000,
    behavior_policy_name: BehaviorPolicyFactory = BehaviorPolicyFactory.random,
) -> MinariDatasetConfig:
    """
    Creates a custom Minari dataset and save a MinariDatasetConfig metadata to file (see /data/offline_data).

    :param env_name: One of the environments defined in Gymnasium or in EnvFactory
    :param dataset_name:
    :param dataset_identifier:
    :param version_dataset:
    :param num_colected_points:
    :param behavior_policy_name: One of our registered behavioral policies (see behavior_policy_registry.py).
    :return:
    :rtype:
    """
    dataset_config = create_minari_config(
        env_name=env_name,
        dataset_name=dataset_name,
        dataset_identifier=dataset_identifier,
        version_dataset=version_dataset,
        num_steps=num_colected_points,
        behavior_policy_name=behavior_policy_name,
    )

    delete_minari_data_if_exists(dataset_config.data_set_name, override_dataset=OVERRIDE_DATA_SET)
    env = create_minari_collector_env_wrapper(
        dataset_config.env_name,
    )
    state, _ = env.reset()

    num_steps = 0
    for _ in range(dataset_config.num_steps):
        behavior_policy = behavior_policy_name.create_policy()
        if dataset_config.behavior_policy == BehaviorPolicyFactory.random:
            action = env.action_space.sample()
        elif isinstance(env.unwrapped, TorcsLidarEnv):
            raw_observation = env.raw_observation
            action = np.array(behavior_policy(raw_observation, env), dtype=np.float32)
        else:
            action = behavior_policy(state, env)

        next_state, reward, done, time_out, info = env.step(action)

        num_steps += 1

        if done or time_out:
            state, _ = env.reset()
            num_steps = 0
        else:
            state = next_state

    dataset = env.create_dataset(
        dataset_id=dataset_config.data_set_name,
        eval_env=env
    )

    dataset_config.save_to_file()

    if isinstance(env.unwrapped, TorcsLidarEnv):
        env.end()

    return dataset_config


# ToDo: Add a flag to keep or not the single datasets.
def create_combined_minari_dataset(
    env_name: str,
    dataset_names: Tuple[str, str] = ("data_I", "data_II"),
    dataset_identifiers: Tuple[str, str] = ("", ""),
    num_collected_points: Tuple[int, int] = (1000, 1000),
    behavior_policy_names: Tuple[BehaviorPolicyFactory, BehaviorPolicyFactory] = (
        BehaviorPolicyFactory.random,
        BehaviorPolicyFactory.random,
    ),
    combined_dataset_identifier: str = "combined_dataset",
    version_dataset: str = "v0",
) -> MinariDatasetConfig:
    """
    Combine two minari datsets into a single one and save metadata with useful information.
    """
    collected_dataset_names = []

    for dataset_name, dataset_identifier, num_points, behavior_policy in zip(
        dataset_names, dataset_identifiers, num_collected_points, behavior_policy_names
    ):
        dataset_config = create_minari_datasets(
            env_name=env_name,
            dataset_name=dataset_name,
            dataset_identifier=dataset_identifier,
            num_colected_points=num_points,
            behavior_policy_name=behavior_policy,
        )

        collected_dataset_names.append(dataset_config.data_set_name)

    name_combined_dataset = generate_compatible_minari_dataset_name(
        env_name=env_name, data_set_name=combined_dataset_identifier, version=version_dataset
    )

    delete_minari_data_if_exists(name_combined_dataset)

    minari_datasets = [minari.load_dataset(dataset_id) for dataset_id in collected_dataset_names]

    combined_dataset = combine_datasets(minari_datasets, new_dataset_id=name_combined_dataset)

    print(
        f"Number of episodes in dataset I:{len(minari_datasets[0])}, in dataset II:{len(minari_datasets[1])} and  "
        f"in the combined dataset: {len(combined_dataset)}"
    )

    total_num_steps = int(np.sum(num_collected_points))

    # Create metadata for the combined dataset (we can reuse the metadata of set 0 for simplicity)
    minari_combined_dataset_config = MinariDatasetConfig.load_from_file(collected_dataset_names[0])
    minari_combined_dataset_config.num_steps = total_num_steps
    minari_combined_dataset_config.data_set_name = name_combined_dataset
    minari_combined_dataset_config.children_dataset_names = collected_dataset_names
    minari_combined_dataset_config.save_to_file()

    return minari_combined_dataset_config


# ToDo:
# 1 - Add info to Minari no possible yet: bug in Minari - issue open: https://github.com/Farama-Foundation/Minari/issues/125
