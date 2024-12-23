import functools
import os
from pathlib import Path
from typing import Union, Dict
import gymnasium as gym
import minari
import numpy as np
from minari import EpisodeData
from minari.storage import get_dataset_path
from tianshou.data import ReplayBuffer


def extract_dimension(obs_or_action_type: Union[gym.core.ObsType, gym.core.ActType]) -> int:
    if isinstance(obs_or_action_type, gym.spaces.Discrete):
        n = obs_or_action_type.n
    elif isinstance(obs_or_action_type, gym.spaces.Box):
        n = obs_or_action_type.shape[0]
    else:
        raise ValueError(
            "Only observations or actions that are discrete or one-dim Gymnasium Box are allowed"
        )
    return n


def get_trained_policy_path(dataset_id) -> Path:
    datasets_path = os.path.join("data", "trained_models_data") if os.environ.get("TRAINED_POLICY_PATH") is None \
        else os.environ["TRAINED_POLICY_PATH"]

    file_path = os.path.join(datasets_path, dataset_id)
    os.makedirs(datasets_path, exist_ok=True)
    return Path(file_path)


def ignore_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass  # Ignore KeyboardInterrupt

    return wrapper


def delete_minari_data_if_exists(file_name: str, override_dataset=True):
    local_datasets = minari.list_local_datasets()
    data_set_minari_paths = get_dataset_path("")
    custom_local_datasets = os.listdir(data_set_minari_paths)
    data_set_expert_task_path = os.path.join(data_set_minari_paths, file_name)

    if override_dataset:
        if (data_set_expert_task_path in local_datasets) or (file_name in custom_local_datasets):
            minari.delete_dataset(file_name)
        # os.makedirs(data_set_expert_task_path)
    else:
        raise FileExistsError(
            f"A dataset with that name already exists in {data_set_expert_task_path}. "
            f"Please delete it or turn 'OVERRIDE_DATA_SET' to True."
        )


def _episode_data_lengths(episode: EpisodeData):
    obs = (
        episode.observations["observation"]
        if isinstance(episode.observations, Dict)
        else episode.observations
    )
    lens_data = [
        obs[1:].shape[0],
        episode.actions.shape[0],
        episode.rewards.shape[0],
        episode.truncations.shape[0],
        episode.terminations.shape[0],
    ]

    return min(lens_data)


def load_buffer_minari(expert_data_task: str) -> ReplayBuffer:
    data_set_minari_paths = get_dataset_path("")
    data_set_minari_paths = os.path.join(data_set_minari_paths, "")
    #    data_set_minari_paths = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
    data_set_expert_task_path = os.path.join(data_set_minari_paths, expert_data_task)

    if not os.path.exists(data_set_expert_task_path):
        minari.download_dataset(expert_data_task)

    dataset = minari.load_dataset(expert_data_task)

    print(f"Dataset {data_set_expert_task_path} downloaded. number of episodes: {len(dataset)}")

    observations_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []
    truncations_list = []
    next_observations_list = []

    for i, episode in enumerate(dataset):
        # For some data the len of the episode len data (observations, actions, etc.) is not the same
        common_len = _episode_data_lengths(episode)

        obs = (
            episode.observations["observation"]
            if isinstance(episode.observations, Dict)
            else episode.observations
        )

        next_observations_list.append(obs[1:][0:common_len])
        observations_list.append(obs[:-1][0:common_len])
        terminals_list.append(episode.terminations[0:common_len])
        truncations_list.append(episode.truncations[0:common_len])
        rewards_list.append(episode.rewards[0:common_len])
        actions_list.append(episode.actions[0:common_len])

    observations = np.concatenate(observations_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    terminals = np.concatenate(terminals_list, axis=0)
    next_observations = np.concatenate(next_observations_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    # truncations = np.concatenate(truncations_list, axis=0)

    replay_buffer = ReplayBuffer.from_data(
        obs=observations,
        act=actions,
        rew=rewards,
        done=terminals,
        obs_next=next_observations,
        terminated=terminals,
        truncated=np.zeros(len(terminals)),
    )
    return replay_buffer
