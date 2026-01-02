"""YAML file reading and caching for dbt-osmosis.

Thread-safety:
    - _YAML_BUFFER_CACHE is protected by _YAML_BUFFER_CACHE_LOCK
    - All cache reads and writes must be synchronized using this lock
    - _read_yaml() acquires both yaml_handler_lock and _YAML_BUFFER_CACHE_LOCK
    - The cache is unbounded and may grow indefinitely (known issue: dbt-osmosis-5n7)
"""

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
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits.

Thread-safety: Protected by _YAML_BUFFER_CACHE_LOCK. All reads and writes
must be guarded by this lock. The cache is unbounded and may grow indefinitely.
"""

_YAML_BUFFER_CACHE_LOCK = threading.Lock()
"""Lock to protect _YAML_BUFFER_CACHE from concurrent access.

Critical sections: _read_yaml() and _write_yaml() perform cache operations
under this lock. All access to _YAML_BUFFER_CACHE must be synchronized.
"""


def _read_yaml(
    yaml_handler: ruamel.yaml.YAML, yaml_handler_lock: threading.Lock, path: Path
) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent.

    Thread-safety: This function is thread-safe. It acquires both yaml_handler_lock
    and _YAML_BUFFER_CACHE_LOCK to ensure synchronized access to the shared cache.
    Multiple threads can safely call this function concurrently.

    Returns:
        Parsed YAML content as a dictionary, or empty dict if file doesn't exist.
    """
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
