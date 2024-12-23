import os
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Literal, Dict

from bcq_continuous_policy import bcq_continuous_default_config, create_bcq_continuous_policy_from_dict
from cql_continuous_policy import cql_continuous_default_config, create_cql_continuous_policy_from_dict
from custom_envs.custom_envs_registration import RenderMode
from dagger_torcs_policy import dagger_torcs_default_config, create_dagger_torcs_policy_from_dict
from generate_custom_minari_datasets.generate_minari_dataset_grid_envs import MinariDatasetConfig
from il_torcs_policy import il_torcs_default_config, create_il_torcs_policy_from_dict
from utils import get_trained_policy_path


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class PolicyType(str, Enum):
    offline = "offline"


class PolicyName(str, Enum):
    cql_continuous = "cql_continuous"
    imitation_learning_torcs = "imitation_learning_torcs"
    dagger_torcs = "dagger_torcs"
    bcq_continuous = "bcq_continuous"


class DefaultPolicyConfigFactoryRegistry(CallableEnum):
    cql_continuous = cql_continuous_default_config
    bcq_continuous = bcq_continuous_default_config
    imitation_learning_torcs = il_torcs_default_config
    dagger_torcs = dagger_torcs_default_config


class PolicyFactoryRegistry(CallableEnum):
    cql_continuous = create_cql_continuous_policy_from_dict
    bcq_continuous = create_bcq_continuous_policy_from_dict
    imitation_learning_torcs = create_il_torcs_policy_from_dict
    dagger_torcs = create_dagger_torcs_policy_from_dict


@dataclass
class TrainedPolicyConfig:
    """
    This class is used to store metadata associated to the offline policy (policy_name, policy_config) and the collected
    MINARI dataset used to trained it (name_expert_data, minari_dataset_config).

    See the notebooks to see how to use it.
    """

    policy_name: PolicyName
    name_expert_data: str | None = None
    render_mode: RenderMode = None
    minari_dataset_config: MinariDatasetConfig = None
    policy_config: DefaultPolicyConfigFactoryRegistry = None
    device: Literal["cpu", "cuda"] = "cpu"

    def __post_init__(self):
        if self.policy_config is None:
            self.policy_config = DefaultPolicyConfigFactoryRegistry.__dict__[self.policy_name]()
            self.policy_config["device"]=self.device
        if self.name_expert_data:
            self.minari_dataset_config = MinariDatasetConfig.load_from_file(self.name_expert_data)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TrainedPolicyConfig":
        return cls(**config_dict)

    def save_to_file(self):
        data_set_path = get_trained_policy_path(self.minari_dataset_config.data_set_name)
        file_name = "config.json"
        obj_to_save = asdict(self)

        if self.minari_dataset_config.initial_config_2d_grid_env is not None:
            obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ] = obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ].value
        with open(os.path.join(data_set_path, self.policy_name, file_name), "w") as file:
            json.dump(obj_to_save, file, indent=4)

    @classmethod
    def load_from_file(cls, dataset_id) -> "TrainedPolicyConfig":
        filename = get_trained_policy_path(dataset_id)
        with open(os.path.join(filename, "config.json"), "r") as file:
            config_dict = json.load(file)

        config_dict["minari_dataset_config"] = MinariDatasetConfig.from_dict(
            config_dict["minari_dataset_config"]
        )
        return cls(**config_dict)
















