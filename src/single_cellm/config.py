from typing import List, Dict, Optional, Union
from pathlib import Path
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
    Params:
        config_keys: list of keys
    Return an absolute path for provided config_keys
    """

    if len(config_keys) == 1:
        if __testing:
            return (
                config["PROJECT_ROOT"]
                / config["test_dir"]
                / __c[config_keys[0]].format_map(kwargs)
            )
        else:
            return config["PROJECT_ROOT"] / __c[config_keys[0]].format_map(kwargs)
    else:
        return get_path(
            config_keys[1:], __c[config_keys[0]], __testing=__testing, **kwargs
        )
