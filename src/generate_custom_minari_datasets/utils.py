import re


def is_v_plus_number(input_string):
    pattern = r"^v\d+$"  # This pattern matches 'v' followed by one or more digits
    return bool(re.match(pattern, input_string))


def generate_compatible_minari_dataset_name(env_name: str, data_set_name: str, version: str):
    full_data_set_name = env_name + "-" + data_set_name + "-" + version
    if not is_v_plus_number(version):
        raise ValueError(
            f"Your minari file is call {full_data_set_name} but the version should be a lower 'v' "
            f"followed by a number, e.g 'v0', in order to be compatible with Minari library."
        )
    return full_data_set_name

