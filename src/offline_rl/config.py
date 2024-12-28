import os
from pathlib import Path


def get_offline_rl_abs_path():
    """:return:"""
    path = Path(os.path.dirname(os.path.realpath(__file__)))
    os.makedirs(path, exist_ok=True)
    return path


def get_custom_envs_abs_path() -> Path:
    """:return:"""
    path = get_offline_rl_abs_path() / Path("custom_envs")
    os.makedirs(path, exist_ok=True)
    return path


def get_trained_policy_path():
    """:return:"""
    path = get_offline_rl_abs_path() / Path("data") / Path("trained_models_data")
    os.makedirs(path, exist_ok=True)
    return path
