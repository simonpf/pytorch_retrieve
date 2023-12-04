"""
pytorch_retrieve.config
=======================

The 'pytorch_retrieve.config' module implements functionality for
 reading configuration files.
"""
from pathlib import Path

import yaml
import toml


def read_config_file(path: Path) -> dict:
    """
    Read a configuration file and returns its contents as a 'dict'.

    Args:
        path: Path of the configuration file to read.

    Return:
        A dictionary containing the parsed configuration file.

    Raises:
        RuntimeError if the given path points to a non-existing file or
        if the file suffix is neither '.toml' nor '.yaml'.
    """
    if not path.exists():
        raise RuntimeError(
            "The given path '%s' does not point to an existing file.",
            path
        )

    suffix = path.suffix
    if not suffix in [".yaml", ".toml"]:
        raise RuntimeError(
            "Config files must be in '.yaml' or '.toml' format and the file "
            "suffix should be either '.yaml' or '.toml'."

        )

    if suffix == ".yaml":
        config = yaml.safe_load(open(path).read())
        return config

    config = toml.loads(open(path).read())
    return config
