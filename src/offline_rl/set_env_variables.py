import os


def set_env_variables(current_directory: str | None = None):
    current_directory = (
        os.path.dirname(__file__) if current_directory is None else current_directory
    )
    mujoco_directory = os.path.expanduser("~/.mujoco")

    return os.environ.update(
        {
            "LD_LIBRARY_PATH": os.path.join(mujoco_directory, "mujoco210", "bin")
            + ":"
            + "/usr/lib/nvidia",
            "MINARI_DATASETS_PATH": os.path.join(current_directory, "data", "offline_data"),
        },
    )
