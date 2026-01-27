"""YAML file reading and caching for dbt-osmosis.

Thread-safety:
    - _YAML_BUFFER_CACHE is protected by _YAML_BUFFER_CACHE_LOCK
    - All cache reads and writes must be synchronized using this lock
    - _read_yaml() acquires both yaml_handler_lock and _YAML_BUFFER_CACHE_LOCK
    - The cache now uses LRU eviction policy with a size limit of 256 entries
"""

import threading
import typing as t
from collections import OrderedDict
from pathlib import Path

import ruamel.yaml
from ruamel.yaml import YAMLError

from dbt_osmosis.core import logger

__all__ = [
    "_YAML_BUFFER_CACHE",
    "_YAML_ORIGINAL_CACHE",
    "_read_yaml",
]


class LRUCache:
    """Thread-safe LRU cache implementation with dictionary-like interface.

    This cache automatically evicts the least recently used items when it reaches
    its maximum size limit. All operations are thread-safe.

    Args:
        maxsize: Maximum number of items to store before eviction

    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._cache: OrderedDict[Path, t.Any] = OrderedDict()
        self._lock = threading.Lock()

    def __getitem__(self, key: Path) -> t.Any:
        with self._lock:
            if key in self._cache:
                # Move to end to mark as recently used
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            raise KeyError(key)

    def __setitem__(self, key: Path, value: t.Any) -> None:
        with self._lock:
            if key in self._cache:
                # Move to end if already exists
                self._cache.pop(key)
            elif len(self._cache) >= self.maxsize:
                # Remove oldest item if at capacity
                self._cache.popitem(last=False)
            self._cache[key] = value

    def __contains__(self, key: Path) -> bool:
        with self._lock:
            return key in self._cache

    def __delitem__(self, key: Path) -> None:
        with self._lock:
            del self._cache[key]

    def setdefault(self, key: Path, default: t.Any) -> t.Any:
        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = default
            return self._cache[key]

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()

    def keys(self) -> t.Any:
        """Return a view of the cache's keys.

        This method provides dict-like interface for iteration.
        Returns the keys view from the internal OrderedDict.
        """
        with self._lock:
            return self._cache.keys()

    def values(self) -> t.Any:
        """Return a view of the cache's values.

        This method provides dict-like interface for iteration.
        Returns the values view from the internal OrderedDict.
        """
        with self._lock:
            return self._cache.values()

    def items(self) -> t.Any:
        """Return a view of the cache's items.

        This method provides dict-like interface for iteration.
        Returns the items view from the internal OrderedDict.
        """
        with self._lock:
            return self._cache.items()

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        with self._lock:
            return len(self._cache)


# Use LRU cache with 256 entries to prevent unbounded growth
# The maxsize should be large enough for typical projects but small enough to limit memory
_YAML_BUFFER_CACHE: LRUCache = LRUCache(maxsize=256)
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits.

Thread-safety: Protected by _YAML_BUFFER_CACHE_LOCK. All reads and writes
must be guarded by this lock. Uses LRU eviction policy with max size of 256 entries.
"""

_YAML_ORIGINAL_CACHE: LRUCache = LRUCache(maxsize=256)
"""Cache for original unfiltered YAML content to preserve filtered sections.

This cache stores the complete original YAML content (including semantic_models, macros, etc.)
before filtering. When writing YAML back to disk, we merge the original filtered sections
with the processed content to ensure nothing is lost.

Thread-safety: Protected by _YAML_BUFFER_CACHE_LOCK. All reads and writes
must be guarded by this lock. Uses LRU eviction policy with max size of 256 entries.
"""

_YAML_BUFFER_CACHE_LOCK = threading.Lock()
"""Lock to protect _YAML_BUFFER_CACHE and _YAML_ORIGINAL_CACHE from concurrent access.

Critical sections: _read_yaml() and _write_yaml() perform cache operations
under this lock. All access to both caches must be synchronized.
"""


def _read_yaml(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    path: Path,
) -> dict[str, t.Any]:
    """Read a yaml file from disk. Adds an entry to the buffer cache so all operations on a path are consistent.

    Thread-safety: This function is thread-safe. It acquires both yaml_handler_lock
    and _YAML_BUFFER_CACHE_LOCK to ensure synchronized access to the shared cache.
    Multiple threads can safely call this function concurrently.

    Returns:
        Parsed YAML content as a dictionary, or empty dict if file doesn't exist.

    """
    with yaml_handler_lock, _YAML_BUFFER_CACHE_LOCK:
        if path not in _YAML_BUFFER_CACHE:
            if not path.is_file():
                logger.debug(":warning: Path => %s is not a file. Returning empty doc.", path)
                return _YAML_BUFFER_CACHE.setdefault(path, {})
            logger.debug(":open_file_folder: Reading YAML doc => %s", path)
            try:
                # Read the file using the filtered YAML handler (OsmosisYAML)
                # This filters out semantic_models, macros, etc.
                filtered_content = yaml_handler.load(path)

                # Also read the original unfiltered content to preserve filtered sections
                # We use a standard YAML parser without filtering
                unfiltered_handler = ruamel.yaml.YAML()
                unfiltered_handler.preserve_quotes = True
                original_content = unfiltered_handler.load(path)

                # Store both filtered content (for processing) and original (for preservation)
                _YAML_BUFFER_CACHE[path] = t.cast(
                    "dict[str, t.Any]",
                    filtered_content if filtered_content is not None else {},
                )
                _YAML_ORIGINAL_CACHE[path] = t.cast(
                    "dict[str, t.Any]",
                    original_content if original_content is not None else {},
                )
            except YAMLError as e:
                logger.error(
                    ":boom: Failed to parse YAML file => %s: %s. "
                    "Please check the file for syntax errors.",
                    path,
                    e,
                )
                raise
            except OSError as e:
                logger.error(":boom: Failed to read YAML file => %s: %s", path, e)
                raise
    with _YAML_BUFFER_CACHE_LOCK:
        return t.cast("dict[str, t.Any]", _YAML_BUFFER_CACHE[path])
