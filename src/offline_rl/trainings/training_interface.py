from abc import ABC, abstractmethod

import gymnasium as gym
from pydantic import BaseModel
from tianshou.env import SubprocVectorEnv
from tianshou.policy import BasePolicy

from offline_rl.custom_envs.custom_envs_registration import EnvFactory
from offline_rl.offline_policies.policy_registry import RLPolicyFactory
from offline_rl.trainings.policy_config_data_class import TrainedPolicyConfig


class OfflineTrainingHyperparams(BaseModel):
    """A model to store hyperparameters for offline training.

    Attributes
    ----------
        offline_policy_config: Configuration for the trained policy.
        step_per_epoch: Number of steps per epoch.
        step_per_collect: Steps per data collection phase.
        num_epochs: Number of training epochs.
        batch_size: Size of the training batch.
        update_per_step: Number of updates per step.
        number_test_envs: Number of environments for testing.
        exploration_noise: Whether to use exploration noise.
        restore_training: Whether to restore training.
        seed: Random seed.
        policy_name_of_trained_policy: Name of the trained policy.
        test_in_train: Whether to test during training.
        episode_per_test: Number of episodes per test.
    """

    offline_policy_config: TrainedPolicyConfig
    step_per_epoch: int
    step_per_collect: int = 1
    num_epochs: int = 1
    batch_size: int = 64
    update_per_step: int = 1
    number_test_envs: int | None = 1
    exploration_noise: bool = True
    restore_training: bool = False
    seed: int | None = None
    policy_name_of_trained_policy: str | None = None
    test_in_train: bool = True
    episode_per_test: int = 1


class OnlineTrainingHyperparams(BaseModel):
    """Configuration for training a policy model.

    Attributes
    ----------
        trained_policy_config (TrainedPolicyConfig): Configuration for the trained policy.
        policy (RLPolicyFactory): one of the policies defined in RLPolicyFactory.
        num_epochs (int): Number of training epochs. Default is 1.
        batch_size (int): Number of samples per batch. Default is 64.
        buffer_size (int): Maximum size of the replay buffer. Default is 100,000.
        step_per_epoch (int): Number of steps per training epoch. Default is 100,000.
        step_per_collect (int): Number of steps to collect per training iteration. Default is 10.
        repeat_per_collect (int): Number of updates per data collection step. Default is 10.
        number_test_envs (int): Number of environments for testing. Default is 5.
        number_train_envs (int): Number of environments for training. Default is 10.
        exploration_noise (bool): Whether to add noise for exploration during training. Default is True.
        episode_per_test (int): Number of episodes to use for testing. Default is 10.
        frames_stack (int): Number of stacked frames as input to the model. Default is 1.
        seed (Optional[int]): Random seed for reproducibility. Default is None.
        restore_training (bool): Whether to restore training from a checkpoint. Default is False.
        policy_model_name (str): Name of the file to save the trained policy model. Default is 'policy.pth'.
    """

    trained_policy_config: TrainedPolicyConfig
    policy: RLPolicyFactory
    num_epochs: int = 1
    batch_size: int = 64
    buffer_size: int = 100_000
    step_per_epoch: int = 100_000
    step_per_collect: int = 10
    repeat_per_collect: int = 10
    number_test_envs: int = 5
    number_train_envs: int = 10
    exploration_noise: bool = True
    episode_per_test: int = 10
    frames_stack: int = 1
    seed: int | None = None
    restore_training: bool = False
    policy_name_of_trained_policy: str = "policy.pth"


class TrainingInterface(ABC):
    @staticmethod
    def _get_environments(
        policy_config: TrainedPolicyConfig,
        number_test_envs: int | None = None,
        number_train_envs: int | None = None,
    ) -> (gym.Env, SubprocVectorEnv):
        env_name = policy_config.minari_dataset_config.env_name
        render_mode = policy_config.render_mode
        env_config = policy_config.minari_dataset_config.initial_config_2d_grid_env

        if env_name in EnvFactory.__members__:
            env = EnvFactory[env_name].get_env(render_mode=render_mode, grid_config=env_config)
        else:
            env = gym.make(env_name, render_mode=render_mode)

        test_envs = train_envs = None

        if number_test_envs is not None:
            test_envs = SubprocVectorEnv(
                [
                    lambda: EnvFactory[env_name].get_env(grid_config=env_config)
                    if env_name in EnvFactory.__members__
                    else gym.make(env_name)
                    for _ in range(number_test_envs)
                ],
            )

        if number_train_envs is not None:
            train_envs = SubprocVectorEnv(
                [
                    lambda: EnvFactory[env_name].get_env(grid_config=env_config)
                    if env_name in EnvFactory.__members__
                    else gym.make(env_name)
                    for _ in range(number_train_envs)
                ],
            )

        return env, test_envs, train_envs

    @staticmethod
    def _create_policy(policy_config: TrainedPolicyConfig, env: gym.Env) -> BasePolicy:
        policy_name = policy_config.rl_policy_model
        policy_config = policy_config.policy_config
        policy = RLPolicyFactory[policy_name].get_policy()
        return policy.create_policy_from_dict(
            policy_config=policy_config,
            action_space=env.action_space,
            observation_space=env.observation_space,
        )

    @staticmethod
    @abstractmethod
    def training(training_config: OfflineTrainingHyperparams | OnlineTrainingHyperparams):
        """Training loop for offline or online RL trainings.

        :param training_config:
        :return:
        """
