import logging
import sys
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import mediapy
import numpy as np
import pygame
import torch
from matplotlib import pyplot as plt
from tianshou.data import Batch
from tianshou.policy import BasePolicy, ImitationPolicy
from torch import nn

from offline_rl.behavior_policies.behavior_policy_registry import (
    BehaviorPolicy2dGridFactory,
)
from offline_rl.custom_envs.custom_2d_grid_env.simple_grid import Custom2DGridEnv
from offline_rl.custom_envs.custom_envs_registration import EnvFactory
from offline_rl.custom_envs.env_wrappers import Grid2DInitialConfig
from offline_rl.custom_envs.envs_registration_utils import RenderMode
from offline_rl.utils import extract_dimension
from offline_rl.visualizations.utils import ignore_keyboard_interrupt

logging.basicConfig(level=logging.WARNING)


def snapshot_env(env: gym.Env):
    env.reset()
    env.step(0)
    rendered_data = env.render()
    rendered_data = rendered_data[0].reshape(256, 256, 3)
    plt.imshow(rendered_data)
    plt.show()


def render_rgb_frames_pygame(env: gym.Env, screen, time_frame=20):
    clock = pygame.time.Clock()
    desired_fps = time_frame
    clock.tick(desired_fps)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    rendered_data = env.render()

    frames = np.copy(rendered_data[0])

    if isinstance(env.unwrapped, Custom2DGridEnv):
        frames = np.transpose(frames, (1, 0, 2))

    pygame_surface = pygame.surfarray.make_surface(frames)
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()


def initialize_pygame(title="RL agent animation"):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode([256, 256])
    pygame.display.set_caption(title)
    return screen


def render_mediapy(
    list_of_frames: list[np.ndarray],
    fps: float = 1,
    title="2d-GridWorld",
    **kwargs: Any,
):
    if len(list_of_frames) == 0:
        return
    mediapy.show_video(list_of_frames, fps=fps, title=title, **kwargs)


@ignore_keyboard_interrupt
def offpolicy_rendering(
    env_or_env_name: gym.Env | str,
    render_mode: RenderMode | None = RenderMode.RGB_ARRAY,
    env_2d_grid_initial_config: Grid2DInitialConfig = None,
    behavior_policy: BehaviorPolicy2dGridFactory = None,
    policy_model: BasePolicy | Callable | nn.Module = None,
    num_frames: int = 100,
    imitation_policy_sampling: bool = False,
    inline: bool = True,
    fps: float = 8.0,
):
    r""":param env_or_env_name: A gym environment or an env name.
    :param render_mode:
    :param env_2d_grid_initial_config: Initial config, namely obstacles and initial and target positions. Only used
        for Custom2DGridEnv configuration when env_or_env_name is the registered environment name.
    :param behavior_policy: name of behavior policy (only if behavior_policy is None -
        see behavior_policy_registry.py)
    :param policy_model: A Tianshou policy mode or a callable that accept a state and the env and returns an action
    :param num_frames: Number of frames
    :param imitation_policy_sampling: Only for imitation learning policy. If False we compute the eps greedy of \pi(a|s).
    :param inline: only to visualize the rendering inline in a jupyter notebook.
    :param fps: only useful for inline plots.
    :return:

    Usage:
    ```
    register_grid_envs()

    env_2D_grid_initial_config = Grid2DInitialConfig(
        obstacles=ObstacleTypes.obst_middle_8x8,
        initial_state=(0,0),
        target_state=(7,7),
    )

    behavior_policy_rendering(
        env_name=CustomEnv.Grid_2D_8x8_discrete,
        render_mode=RenderMode.RGB_ARRAY_LIST,
        behavior_policy_name=BehaviorPolicyType.behavior_suboptimal_8x8_grid_discrete,
        env_2d_grid_initial_config=env_2D_grid_initial_config,
        num_frames=1000,
    )
    ```

    """
    if behavior_policy is None and policy_model is None:
        raise ValueError("Either behavior_policy_name or behavior_policy must be provided.")
    if behavior_policy is not None and policy_model is not None:
        raise ValueError(
            "Both behavior_policy_name and behavior_policy cannot be provided simultaneously.",
        )

    if isinstance(env_or_env_name, str):
        env = EnvFactory[env_or_env_name].get_env(
            render_mode=render_mode,
            grid_config=env_2d_grid_initial_config,
        )
    else:
        env = env_or_env_name

    state, _ = env.reset()

    state_shape = extract_dimension(env.observation_space)

    if not inline and render_mode == RenderMode.RGB_ARRAY_LIST:
        screen = initialize_pygame()

    list_of_frames = []
    for _ in range(num_frames):
        if behavior_policy is not None:
            action = behavior_policy(state, env)
        else:
            if isinstance(policy_model, BasePolicy):
                tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
                policy_output = policy_model(tensor_state)

                if imitation_policy_sampling and isinstance(policy_model, ImitationPolicy):
                    policy_output = policy_output.logits
                    categorical = torch.distributions.Categorical(logits=policy_output[0])
                    action = np.array(categorical.sample())
                else:
                    action = (
                        policy_output.act[0]
                        if (
                            isinstance(policy_output.act[0], np.ndarray)
                            or isinstance(policy_output.act, np.ndarray)
                        )
                        else policy_output.act[0].detach().cpu().numpy()
                    )
            elif isinstance(policy_model, nn.Module):
                action = policy_model(torch.Tensor(state)).detach().numpy()
            elif isinstance(policy_model, Callable):
                action = policy_model(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_frames += 1

        if render_mode == RenderMode.RGB_ARRAY:
            if inline:
                rendered_data = env.render()
                frames = rendered_data[0]

                if isinstance(env.unwrapped, Custom2DGridEnv):
                    frames = np.transpose(rendered_data[0], (0, 1, 2))
                list_of_frames.append(frames)
            else:
                render_rgb_frames_pygame(env, screen)
        elif render_mode is None:
            pass
        else:
            env.render()

        if done or time_out:
            state, _ = env.reset()
            num_frames = 0
        else:
            state = next_state

    if not inline:
        pygame.quit()

    if inline:
        render_mediapy(list_of_frames, fps=fps)
