from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
import yaml
import torch


def _read_config(config_file: Optional[Union[str, Path]] = None) -> Dict:
    """
    Read the config file (only called internally)

    Args:
        config_file: Optionally provide a filename for another config

    Returns:
        Dict: Modified content of the config file
    """

    project_root = (Path(__file__).parents[2]).resolve()

    if config_file is None:
        config_file = project_root / "config.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config["PROJECT_ROOT"] = project_root
    config["dtype"] = torch.float32

    return config


config = _read_config()


def get_path(config_keys: List[str], __c=config, __testing=False, **kwargs):
    """
    Get an absolute path from the config file, providing the hierarchy of keys leading to it. E.g. ["paths", "final_model"]

    :param config_keys: Hierarchical list of keys
    :param __c: Config dict to use (only used for testing)
    :param __testing: Whether to use the testing config dir
    :param kwargs: Additional kwargs to pass to the format function, replacing placeholders in the config paths
    :return:

    Return an absolute path for provided config_keys
    """

    path = __c["PROJECT_ROOT"]
    if __testing:
        path /= config["test_dir"]

    dict_level = __c
    for key in config_keys:
        dict_level = dict_level[key]

    path /= dict_level.format(**kwargs)

    return path


def model_path_from_name(model_name: str):
    """
    Translate model names (e.g. biogpt) to their respective model weigth paths (e.g. microsoft/biogpt)
    """

    try:
        path_name = config["model_name_path_map"][model_name]
    except KeyError:
        return model_name
    else:
        local = Path(config["PROJECT_ROOT"] / path_name)

        if local.exists():
            return local.as_posix()
        else:
            if "resources" in path_name or "results" in path_name:
                logging.warning(
                    "Model path not found, despite 'results' or 'resources' in path. Maybe you need to download it manually"
                )

            return path_name
