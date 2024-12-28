import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from offline_rl.config import get_trained_policy_path
from offline_rl.custom_envs.envs_registration_utils import RenderMode
from offline_rl.generate_custom_minari_datasets.generate_minari_dataset import (
    MinariDatasetConfig,
)
from offline_rl.offline_policies.policy_registry import RLPolicyFactory

logging.basicConfig()


@dataclass
class TrainedPolicyConfig:
    """Metadata for the trained policy. For offlineRL it includes Minari metadata, too."""

    rl_policy_model: RLPolicyFactory
    device: Literal["cpu", "cuda"]
    name_expert_data: str | None = None
    render_mode: RenderMode = None
    minari_dataset_config: MinariDatasetConfig = None
    policy_config: dict["str", Any] = None

    def __post_init__(self):
        if self.policy_config is None:
            self.policy_config = self.rl_policy_model.get_policy().default_config()
            self.policy_config["device"] = self.device
        if self.name_expert_data:
            self.minari_dataset_config = MinariDatasetConfig.load_from_file(self.name_expert_data)

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)

    def save_to_file(self):
        data_set_path = get_trained_policy_path() / self.minari_dataset_config.data_set_name
        file_name = "config.json"
        obj_to_save = asdict(self)

        if self.minari_dataset_config.initial_config_2d_grid_env is not None:
            obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ] = obj_to_save["minari_dataset_config"]["initial_config_2d_grid_env"][
                "obstacles"
            ].value

        path_to_save = data_set_path.name / Path(self.rl_policy_model) / Path(file_name)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)

        with open(str(path_to_save), "w") as file:
            json.dump(obj_to_save, file, indent=4)

        logging.info(f"TrainedPolicyConfig metadata saved to {path_to_save}")

    @classmethod
    def load_from_file(cls, dataset_name: str | Path, policy_name: str | Path):
        """Load config from a json file.

        :param dataset_name: Just the name of the dataset. You don't need to specify the full path
        :param policy_name: the name of the offline policy
        :return:
        """
        filename = get_trained_policy_path() / Path(dataset_name) / Path(policy_name)

        file_to_config = filename / Path("config.json")

        with open(str(file_to_config)) as file:
            config_dict = json.load(file)

        config_dict["minari_dataset_config"] = MinariDatasetConfig.from_dict(
            config_dict["minari_dataset_config"],
        )
        new_config = cls(**config_dict)

        logging.info(f"New TrainedPolicyConfig for dataset: {dataset_name} created")

        return new_config
