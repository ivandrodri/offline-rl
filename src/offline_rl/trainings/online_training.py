import os

import numpy as np
import torch
from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils import TensorboardLogger

from offline_rl.offline_policies.policy_registry import PolicyType
from offline_rl.trainings.custom_tensorboard_callbacks import CustomSummaryWriter
from offline_rl.trainings.policy_config_data_class import get_trained_policy_path
from offline_rl.trainings.training_interface import (
    OnlineTrainingHyperparams,
    TrainingInterface,
)

POLICY_NAME_BEST_REWARD = "policy_best_reward_online.pth"
POLICY_NAME = "policy_online.pth"


class OnlineRLTraining(TrainingInterface):
    @staticmethod
    def training(training_config: OnlineTrainingHyperparams):
        if training_config.seed is not None:
            np.random.seed(training_config.seed)
            torch.manual_seed(training_config.seed)

        # Create environments

        env, test_envs, train_envs = OnlineRLTraining._get_environments(
            policy_config=training_config.trained_policy_config,
            number_test_envs=training_config.number_test_envs,
            number_train_envs=training_config.number_train_envs,
        )

        # Path to save models/config
        policy_name = training_config.trained_policy_config.rl_policy_model
        name_expert_data = training_config.trained_policy_config.name_expert_data
        log_name = os.path.join(name_expert_data, policy_name)
        log_path = get_trained_policy_path(log_name)

        # Policy creation/restoration
        policy = OnlineRLTraining._create_policy(
            policy_config=training_config.trained_policy_config,
            env=env,
        )
        if training_config.restore_training:
            policy_path = os.path.join(log_path, "policy.pth")
            policy.load_state_dict(
                torch.load(policy_path, map_location=training_config.trained_policy_config.device),
            )
            print("Loaded policy from: ", policy_path)

        # Create collector for testing

        if training_config.number_train_envs > 1:
            # buffer = VectorReplayBuffer(buffer_size, len(train_envs))
            buffer = VectorReplayBuffer(
                total_size=training_config.buffer_size,
                buffer_num=len(train_envs),
                stack_num=training_config.frames_stack,
                # ignore_obs_next=True,
                # save_only_last_obs=True,
            )
        else:
            buffer = ReplayBuffer(
                buffer_size=training_config.buffer_size,
                buffer_num=len(train_envs),
                stack_num=training_config.frames_stack,
                # ignore_obs_next=True,
                # save_only_last_obs=True,
            )

        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        test_collector = Collector(policy, test_envs)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, POLICY_NAME_BEST_REWARD))

        def stop_fn(mean_rewards):
            return False

        # Tensorboard writer
        custom_writer = CustomSummaryWriter(log_path, env)
        custom_writer.log_custom_info()
        logger = TensorboardLogger(custom_writer)

        trainer = None
        if training_config.policy_type == PolicyType.offpolicy:
            trainer = OffpolicyTrainer
        elif training_config.policy_type == PolicyType.onpolicy:
            trainer = OnpolicyTrainer

        # Training
        _ = trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=training_config.num_epochs,
            step_per_epoch=training_config.step_per_epoch,
            step_per_collect=training_config.step_per_collect,
            repeat_per_collect=training_config.repeat_per_collect,
            episode_per_test=training_config.episode_per_test,
            batch_size=training_config.batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        ).run()

        # Save final policy
        policy_model_name = (
            POLICY_NAME
            if training_config.policy_name_of_trained_policy is None
            else training_config.policy_name_of_trained_policy
        )
        torch.save(policy.state_dict(), os.path.join(log_path, policy_model_name))

        # Save config
        training_config.trained_policy_config.save_to_file()
