import threading
import typing as t
from pathlib import Path

import ruamel.yaml

import dbt_osmosis.core.logger as logger

__all__ = [
    "_read_yaml",
    "_YAML_BUFFER_CACHE",
]

_YAML_BUFFER_CACHE: dict[Path, t.Any] = {}
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits."""


def _read_yaml(
    yaml_handler: ruamel.yaml.YAML, yaml_handler_lock: threading.Lock, path: Path
) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent."""
    with yaml_handler_lock:
        if path not in _YAML_BUFFER_CACHE:
            if not path.is_file():
                logger.debug(":warning: Path => %s is not a file. Returning empty doc.", path)
                return _YAML_BUFFER_CACHE.setdefault(path, {})
            logger.debug(":open_file_folder: Reading YAML doc => %s", path)
            _YAML_BUFFER_CACHE[path] = t.cast(dict[str, t.Any], yaml_handler.load(path))
    return _YAML_BUFFER_CACHE[path]
