from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
import yaml
import torch
import os


def _read_config(config_file: Optional[Union[str, Path]] = None) -> Dict:
    """
    Read the config file. First try to read from the current directory, then from the project root, then fail

    Args:
        config_file: Optionally provide a filename for another config

    Returns:
        Dict: Modified content of the config file
    """

    if config_file is not None:
        config_file = Path(config_file).resolve()
    elif Path("config.yaml").exists():
        config_file = Path("config.yaml").resolve()
    else:
        config_file = (Path(__file__).parents[2] / "config.yaml").resolve()

    assert config_file.exists(), f"Config file {config_file} not found"
    logging.info(f"Reading config from {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config["PROJECT_ROOT"] = config_file.parent
    config["dtype"] = torch.float32

    return config


config = _read_config()


def get_cache_dir():
    return os.getenv("CELLWHISPERER_CACHE", get_path(["paths", "cache"]))


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
