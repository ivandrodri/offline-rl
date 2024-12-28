import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger

from offline_rl.custom_envs.custom_envs_registration import EnvFactory
from offline_rl.offline_policies.policy_registry import RLPolicyFactory
from offline_rl.trainings.custom_tensorboard_callbacks import CustomSummaryWriter
from offline_rl.trainings.policy_config_data_class import (
    TrainedPolicyConfig,
    get_trained_policy_path,
)
from offline_rl.trainings.training_interface import (
    OfflineTrainingHyperparams,
    TrainingInterface,
)
from offline_rl.utils import load_buffer_minari

POLICY_NAME_BEST_REWARD = "policy_best_reward.pth"
POLICY_NAME = "policy.pth"


class OfflineRLTraining(TrainingInterface):
    @staticmethod
    def _stop_fn(mean_rewards) -> bool:
        Warning("Stop function not implemented")
        return False

    @staticmethod
    def training(training_config: OfflineTrainingHyperparams):
        """Offline policy training with a Minari dataset. The policy could be one of the ones you can find in
        /offline_policies/policy_registry.py .

        :param training_config:
        :return:
        """
        if training_config.seed is not None:
            np.random.seed(training_config.seed)
            torch.manual_seed(training_config.seed)

        env, test_envs, _ = OfflineRLTraining._get_environments(
            training_config.offline_policy_config,
            training_config.number_test_envs,
        )
        name_expert_data = training_config.offline_policy_config.name_expert_data
        data_buffer = load_buffer_minari(name_expert_data)

        # Path to save models/config
        offline_policy_name = training_config.offline_policy_config.rl_policy_model
        log_name = Path(name_expert_data) / Path(offline_policy_name)
        log_path = get_trained_policy_path() / log_name

        # Policy creation/restoration
        policy = OfflineRLTraining._create_policy(training_config.offline_policy_config, env)

        if training_config.restore_training:
            policy_name = (
                training_config.policy_name_of_trained_policy
                if training_config.policy_name_of_trained_policy is not None
                else POLICY_NAME
            )
            policy_path = os.path.join(log_path, policy_name)
            policy.load_state_dict(
                torch.load(policy_path, map_location=training_config.offline_policy_config.device),
            )
            print("Loaded policy from: ", policy_path)

        # Create collector for testing
        test_collector = None
        if test_envs is not None:
            test_collector = Collector(
                policy,
                test_envs,
                exploration_noise=training_config.exploration_noise,
            )

        def save_best_fn(policy):
            torch.save(policy.state_dict(), str(log_path / POLICY_NAME_BEST_REWARD))

        # Tensorboard writer
        custom_writer = CustomSummaryWriter(log_path, env)
        custom_writer.log_custom_info()
        logger = TensorboardLogger(custom_writer)

        # Training
        _ = OfflineTrainer(
            policy=policy,
            buffer=data_buffer,
            test_collector=test_collector,
            max_epoch=training_config.num_epochs,
            step_per_epoch=training_config.step_per_epoch,
            step_per_collect=training_config.step_per_collect,
            update_per_step=training_config.update_per_step,
            episode_per_test=training_config.episode_per_test
            if training_config.number_test_envs is not None
            else None,
            batch_size=training_config.batch_size,
            stop_fn=OfflineRLTraining._stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=training_config.test_in_train,
        ).run()

        # Save final policy
        policy_name = (
            POLICY_NAME
            if training_config.policy_name_of_trained_policy is None
            else (training_config.policy_name_of_trained_policy)
        )
        torch.save(policy.state_dict(), str(log_path / policy_name))

        # Save config
        training_config.offline_policy_config.save_to_file()

    @staticmethod
    def restore_policy(
        offline_policy_config: TrainedPolicyConfig,
        best_policy: bool = False,
    ) -> BasePolicy:
        env_name = offline_policy_config.minari_dataset_config.env_name
        render_mode = offline_policy_config.render_mode
        env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

        if env_name in EnvFactory.__members__:
            env = EnvFactory[env_name].get_env(render_mode=render_mode, grid_config=env_config)
        else:
            env = gym.make(env_name, render_mode=render_mode)

        # Policy restoration
        policy_model = offline_policy_config.rl_policy_model
        policy_config = offline_policy_config.policy_config

        policy = (
            RLPolicyFactory[policy_model]
            .get_policy()
            .create_policy_from_dict(
                policy_config=policy_config,
                action_space=env.action_space,
                observation_space=env.observation_space,
            )
        )

        log_name = Path(offline_policy_config.name_expert_data) / Path(
            offline_policy_config.rl_policy_model,
        )

        policy_name = POLICY_NAME_BEST_REWARD if best_policy else POLICY_NAME
        log_path = get_trained_policy_path() / log_name / policy_name
        policy.load_state_dict(torch.load(str(log_path), map_location=offline_policy_config.device))
        return policy
