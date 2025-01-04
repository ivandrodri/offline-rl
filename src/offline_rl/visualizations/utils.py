import functools

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import ImitationPolicy
from torch import nn
from tqdm import tqdm

from offline_rl.utils import (
    compare_state_action_histograms,
    extract_dimension,
    one_hot_to_integer,
    state_action_histogram,
)


def ignore_keyboard_interrupt(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            pass  # Ignore KeyboardInterrupt

    return wrapper


def get_state_action_data_and_policy_grid_distributions(
    data: ReplayBuffer,
    env: gym.Env,
    policy: nn.Module | str | None = None,
    num_episodes: int = 1,
    logits_sampling: bool = False,
    plot: bool = True,
    normalized: bool = True,
) -> tuple[dict, dict]:
    """:param data: Tianshou ReplyBuffer dataset
    :param env:
    :param policy: a pytorch policy
    :param num_episodes: the number of episodes used to generate the policy state-action distribution.
    :param logits_sampling: if False the action will be provided (usually arg_max [Q(s,a)] ) otherwise the
        q-values will be sampled. Useful for imitation learning to compare the data and policy distributions.
    :param normalized: if True the histograms are normalized.
    :return:
    """
    state_shape = extract_dimension(env.observation_space)
    action_shape = extract_dimension(env.action_space)

    state_action_count_data = {
        (int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)
    }

    for episode_elem in data:
        observation = episode_elem.obs
        action = episode_elem.act

        action_value = (
            int(action) if len(action.shape) == 0 or action.shape[0] <= 1 else np.argmax(action)
        )
        state_action_count_data[(one_hot_to_integer(observation), action_value)] += 1

    if policy is not None:
        state_action_count_policy = {
            (int1, int2): 0 for int1 in range(state_shape + 1) for int2 in range(action_shape)
        }

        for _i in tqdm(range(num_episodes), desc="Processing", ncols=100):
            done = False
            truncated = False
            state, _ = env.reset()
            while not (done or truncated):
                if policy != "random":
                    tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
                    policy_output = policy(tensor_state)

                    if logits_sampling is False:
                        action = (
                            policy_output.act[0]
                            if (
                                isinstance(policy_output.act[0], np.ndarray)
                                or isinstance(policy_output.act, np.ndarray)
                            )
                            else policy_output.act[0].detach().numpy()
                        )
                    else:
                        if isinstance(policy, ImitationPolicy):
                            q_values = policy_output.logits
                            categorical = torch.distributions.Categorical(logits=q_values[0])
                            action = np.array(categorical.sample())

                else:
                    action = env.action_space.sample()

                action_value = (
                    int(action)
                    if len(action.shape) == 0 or action.shape[0] <= 1
                    else np.argmax(action)
                )
                state_action_count_policy[(one_hot_to_integer(state), action_value)] += 1
                next_state, reward, done, truncated, info = env.step(action_value)
                state = next_state

    else:
        state_action_count_policy = None

    if plot:
        new_keys = [
            (env.to_xy(state_action[0]), state_action[1])
            for state_action in list(state_action_count_data.keys())
        ]

        state_action_histogram(
            state_action_count_data,
            title="State-Action data distribution",
            new_keys_for_state_action_count_plot=new_keys,
            normalized=normalized,
        )
        if state_action_count_policy is not None:
            state_action_histogram(
                state_action_count_policy,
                title="State-Action policy distribution",
                new_keys_for_state_action_count_plot=new_keys,
                normalized=normalized,
            )
            compare_state_action_histograms(state_action_count_data, state_action_count_policy)

    return state_action_count_data, state_action_count_policy


def snapshot_env(env: gym.Env):
    env.reset()
    env.step(0)
    rendered_data = env.render()  # Capture the frame as a NumPy array
    rendered_data = rendered_data[0].reshape(256, 256, 3)
    plt.imshow(rendered_data)  # Display the frame using matplotlib
    plt.show()  # Show the frame in a separate window