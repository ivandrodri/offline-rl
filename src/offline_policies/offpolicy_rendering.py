import logging
import sys
import gymnasium as gym
import mediapy
import numpy as np
import pygame
import torch
from typing import List, Any, Union, Callable
from matplotlib import pyplot as plt
from tianshou.data import Batch
from tianshou.policy import BasePolicy, ImitationPolicy
from torch import nn
from behavior_policies.behavior_policy_registry import BehaviorPolicyFactory
from custom_envs.custom_envs_registration import RenderMode, EnvFactory
from torcs_gymnasium.src.torcs_lidar_environment.torcs_lidar_env import TorcsLidarEnv
from utils import ignore_keyboard_interrupt, extract_dimension

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

    pygame_surface = pygame.surfarray.make_surface(frames)
    screen.blit(pygame_surface, (0, 0))
    pygame.display.flip()


def initialize_pygame(title="RL agent animation"):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode([256, 256])
    pygame.display.set_caption(title)
    return screen


def render_mediapy(list_of_frames: List[np.ndarray], fps: float = 1, title="2d-GridWorld", **kwargs: Any):
    if len(list_of_frames) == 0:
        return
    mediapy.show_video(list_of_frames, fps=fps, title=title, **kwargs)


@ignore_keyboard_interrupt
def offpolicy_rendering(
    env_or_env_name: Union[gym.Env, str],
    render_mode: RenderMode | None = RenderMode.RGB_ARRAY_LIST,
    behavior_policy_name: BehaviorPolicyFactory | None = None,
    policy_model: Union[BasePolicy, Callable, nn.Module] = None,
    num_frames: int = 100,
    imitation_policy_sampling: bool = False,
    inline: bool = True,
    fps: float = 8.0,
):
    """
    :param env_or_env_name: A gym environment or an env name.
    :param render_mode:
    :param behavior_policy_name: name of behavior policy (only if behavior_policy is None -
        see behavior_policy_registry.py)
    :param policy_model: A Tianshou policy mode or a callable that accept an state and the env and returns an action
    :param num_frames: Number of frames
    :param imitation_policy_sampling: Only for imitation learning policy. If False we compute the eps greedy of \pi(a|s).
    :param inline: only to visualize the rendering inline in a jupyter notebook.
    :param fps: only useful for inline plots.
    :return:

    """
    if behavior_policy_name is None and policy_model is None:
        raise ValueError("Either behavior_policy_name or behavior_policy must be provided.")
    if behavior_policy_name is not None and policy_model is not None:
        raise ValueError(
            "Both behavior_policy_name and behavior_policy cannot be provided simultaneously."
        )

    if isinstance(env_or_env_name, str):
        env = EnvFactory[env_or_env_name].get_env(render_mode=render_mode)
    else:
        env = env_or_env_name

    # TORCS render mode must be configured from the TORCS api.
    if isinstance(env.unwrapped, TorcsLidarEnv):
        render_mode = None

    state, _ = env.reset()

    state_shape = extract_dimension(env.observation_space)

    screen = None
    if not inline:
        if render_mode == RenderMode.RGB_ARRAY_LIST:
            screen = initialize_pygame()

    list_of_frames = []
    action = None
    for _ in range(num_frames):
        if behavior_policy_name is not None:
            behavior_policy = behavior_policy_name.create_policy()

            if behavior_policy_name == BehaviorPolicyFactory.random:
                action = env.action_space.sample()
            elif isinstance(env.unwrapped, TorcsLidarEnv):
                raw_observation = env.raw_observation
                action = np.array(behavior_policy(raw_observation, env), dtype=np.float32)
            else:
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
                action=policy_model(torch.Tensor(state)).detach().numpy()
            elif isinstance(policy_model, Callable):
                action = policy_model(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_frames += 1

        if render_mode == RenderMode.RGB_ARRAY_LIST:
            if inline:
                rendered_data = env.render()
                frames = rendered_data[0]
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
    if isinstance(env.unwrapped, TorcsLidarEnv):
        env.end()

