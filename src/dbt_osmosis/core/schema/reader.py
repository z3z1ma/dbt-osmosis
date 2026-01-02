import threading
import typing as t
from pathlib import Path

import ruamel.yaml
from ruamel.yaml import YAMLError

import dbt_osmosis.core.logger as logger

__all__ = [
    "_read_yaml",
    "_YAML_BUFFER_CACHE",
]

_YAML_BUFFER_CACHE: dict[Path, t.Any] = {}
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits."""

_YAML_BUFFER_CACHE_LOCK = threading.Lock()
"""Lock to protect _YAML_BUFFER_CACHE from concurrent access."""


def _read_yaml(
    yaml_handler: ruamel.yaml.YAML, yaml_handler_lock: threading.Lock, path: Path
) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent."""
    with yaml_handler_lock:
        with _YAML_BUFFER_CACHE_LOCK:
            if path not in _YAML_BUFFER_CACHE:
                if not path.is_file():
                    logger.debug(":warning: Path => %s is not a file. Returning empty doc.", path)
                    return _YAML_BUFFER_CACHE.setdefault(path, {})
                logger.debug(":open_file_folder: Reading YAML doc => %s", path)
                try:
                    # Add null check - yaml_handler.load() can return None for empty files
                    content = yaml_handler.load(path)
                    _YAML_BUFFER_CACHE[path] = t.cast(
                        dict[str, t.Any], content if content is not None else {}
                    )
                except YAMLError as e:
                    logger.error(
                        ":boom: Failed to parse YAML file => %s: %s. "
                        "Please check the file for syntax errors.",
                        path,
                        e,
                    )
                    raise
                except (OSError, IOError) as e:
                    logger.error(":boom: Failed to read YAML file => %s: %s", path, e)
                    raise
    with _YAML_BUFFER_CACHE_LOCK:
        return t.cast(dict[str, t.Any], _YAML_BUFFER_CACHE[path])
