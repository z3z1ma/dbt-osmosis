"""YAML file reading and caching for dbt-osmosis.

Thread-safety:
    - _YAML_BUFFER_CACHE is protected by _YAML_BUFFER_CACHE_LOCK
    - All cache reads and writes must be synchronized using this lock
    - _read_yaml() acquires both yaml_handler_lock and _YAML_BUFFER_CACHE_LOCK
    - Clean cache entries use LRU eviction, but dirty entries stay pinned until commit
"""

import threading
import typing as t
from collections import OrderedDict
from pathlib import Path

import ruamel.yaml
from ruamel.yaml import YAMLError
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.scalarstring import (
    DoubleQuotedScalarString,
    PlainScalarString,
    SingleQuotedScalarString,
)

from dbt_osmosis.core import logger
from dbt_osmosis.core.schema.parser import _partition_yaml_top_level_sections

__all__ = [
    "_YAML_BUFFER_CACHE",
    "_YAML_ORIGINAL_CACHE",
    "_read_yaml",
]


class LRUCache:
    """Thread-safe LRU cache implementation with dictionary-like interface.

    This cache automatically evicts the least recently used clean items when it
    reaches its maximum size limit. Dirty entries can be pinned so buffered YAML
    mutations are not silently discarded before they are committed.

    Args:
        maxsize: Maximum number of items to store before eviction

    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._cache: OrderedDict[Path, t.Any] = OrderedDict()
        self._lock = threading.Lock()
        self._dirty_keys: set[Path] = set()

    def _evict_one_clean_locked(self) -> bool:
        for candidate in list(self._cache.keys()):
            if candidate in self._dirty_keys:
                continue
            del self._cache[candidate]
            return True
        return False

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
            elif len(self._cache) >= self.maxsize and not self._evict_one_clean_locked():
                logger.debug(
                    ":pin: Retaining dirty YAML buffers beyond cache max size => %s",
                    self.maxsize,
                )
            self._cache[key] = value

    def __contains__(self, key: Path) -> bool:
        with self._lock:
            return key in self._cache

    def __delitem__(self, key: Path) -> None:
        with self._lock:
            self._dirty_keys.discard(key)
            del self._cache[key]

    def setdefault(self, key: Path, default: t.Any) -> t.Any:
        with self._lock:
            if key not in self._cache:
                if len(self._cache) >= self.maxsize and not self._evict_one_clean_locked():
                    logger.debug(
                        ":pin: Retaining dirty YAML buffers beyond cache max size => %s",
                        self.maxsize,
                    )
                self._cache[key] = default
            return self._cache[key]

    def mark_dirty(self, key: Path) -> None:
        """Pin a cache entry so it cannot be evicted before commit."""
        with self._lock:
            if key in self._cache:
                self._dirty_keys.add(key)

    def mark_clean(self, key: Path) -> None:
        """Allow a cache entry to be evicted again."""
        with self._lock:
            self._dirty_keys.discard(key)

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._dirty_keys.clear()

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


# Use LRU cache with 256 clean entries to prevent unbounded growth.
# Dirty entries may temporarily exceed the limit so buffered edits are not lost.
_YAML_BUFFER_CACHE: LRUCache = LRUCache(maxsize=256)
"""Cache for yaml file buffers to avoid redundant disk reads/writes and simplify edits.

Thread-safety: Protected by _YAML_BUFFER_CACHE_LOCK. All reads and writes
must be guarded by this lock. Clean entries use LRU eviction with max size 256,
while dirty entries stay pinned until they are written or discarded.
"""

_YAML_ORIGINAL_CACHE: LRUCache = LRUCache(maxsize=256)
"""Cache for original unfiltered YAML content to preserve filtered sections.

This cache stores the complete original YAML content (including semantic_models, macros, etc.)
before filtering. When writing YAML back to disk, we merge the original filtered sections
with the processed content to ensure nothing is lost.

Thread-safety: Protected by _YAML_BUFFER_CACHE_LOCK. All reads and writes
must be guarded by this lock. Clean entries use LRU eviction with max size 256,
while dirty entries stay pinned until they are written or discarded.
"""

_YAML_BUFFER_CACHE_LOCK = threading.Lock()
"""Lock to protect _YAML_BUFFER_CACHE and _YAML_ORIGINAL_CACHE from concurrent access.

Critical sections: _read_yaml() and _write_yaml() perform cache operations
under this lock. All access to both caches must be synchronized.
"""

_QUOTED_SCALAR_STRING_TYPES = (DoubleQuotedScalarString, SingleQuotedScalarString)
_YAML_1_1_BOOLEAN_LIKE_STRINGS = frozenset({
    "y",
    "Y",
    "yes",
    "Yes",
    "YES",
    "n",
    "N",
    "no",
    "No",
    "NO",
    "on",
    "On",
    "ON",
    "off",
    "Off",
    "OFF",
})


def _normalize_quoted_scalar_style(
    value: DoubleQuotedScalarString | SingleQuotedScalarString,
    *,
    width: int,
    prefix_colon: str | None,
    description_indent: int = 0,
) -> str:
    scalar = str(value)
    if scalar in _YAML_1_1_BOOLEAN_LIKE_STRINGS:
        return scalar
    if len(scalar.splitlines()) > 1:
        return scalar
    description_threshold = width - description_indent - len(f"description{prefix_colon or ''}: ")
    if len(scalar) > description_threshold:
        return scalar
    return PlainScalarString(scalar)


def _nested_column_description_indent(path: tuple[str, ...], indent_mapping: int) -> int:
    if len(path) < 3 or path[-3:] != ("columns", "[]", "description"):
        return 0
    return max(len(path) - 1, 0) * indent_mapping


def _mark_yaml_caches_dirty(path: Path) -> None:
    """Pin buffered YAML state so later cache churn cannot discard pending edits."""
    for cache in (_YAML_BUFFER_CACHE, _YAML_ORIGINAL_CACHE):
        marker = getattr(cache, "mark_dirty", None)
        if callable(marker):
            marker(path)


def _discard_yaml_caches(path: Path) -> None:
    """Remove a YAML path from both caches after a truthful disk outcome."""
    for cache in (_YAML_BUFFER_CACHE, _YAML_ORIGINAL_CACHE):
        if path in cache:
            del cache[path]


def _has_yaml_anchor(value: t.Any) -> bool:
    anchor = getattr(value, "anchor", None)
    return bool(getattr(anchor, "value", None))


def _normalize_managed_quote_styles(
    value: t.Any,
    *,
    width: int,
    prefix_colon: str | None,
    indent_mapping: int,
    path: tuple[str, ...] = (),
) -> t.Any:
    """Convert unanchored managed quoted scalars to plain strings in place."""
    if _has_yaml_anchor(value):
        return value
    if isinstance(value, _QUOTED_SCALAR_STRING_TYPES):
        return _normalize_quoted_scalar_style(
            value,
            width=width,
            prefix_colon=prefix_colon,
            description_indent=_nested_column_description_indent(path, indent_mapping),
        )
    if isinstance(value, dict):
        for key, item in list(value.items()):
            normalized_key = _normalize_managed_quote_styles(
                key,
                width=width,
                prefix_colon=prefix_colon,
                indent_mapping=indent_mapping,
                path=path,
            )
            item_path = (*path, str(key)) if isinstance(key, str) else path
            normalized_item = _normalize_managed_quote_styles(
                item,
                width=width,
                prefix_colon=prefix_colon,
                indent_mapping=indent_mapping,
                path=item_path,
            )
            if normalized_key is not key or normalized_item is not item:
                if isinstance(value, CommentedMap):
                    index = list(value).index(key)
                    del value[key]
                    value.insert(index, normalized_key, normalized_item)
                else:
                    del value[key]
                    value[normalized_key] = normalized_item
    elif isinstance(value, CommentedSeq):
        for index, item in enumerate(list(value)):
            normalized = _normalize_managed_quote_styles(
                item,
                width=width,
                prefix_colon=prefix_colon,
                indent_mapping=indent_mapping,
                path=(*path, "[]"),
            )
            if normalized is not item:
                value[index] = normalized
    elif isinstance(value, list):
        for index, item in enumerate(value):
            normalized = _normalize_managed_quote_styles(
                item,
                width=width,
                prefix_colon=prefix_colon,
                indent_mapping=indent_mapping,
                path=(*path, "[]"),
            )
            if normalized is not item:
                value[index] = normalized
    return value


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
                # Read the file once without top-level filtering so managed and
                # preserved sections stay in the same ruamel object graph. This
                # keeps cross-section anchors and aliases valid when the writer
                # restores unmanaged sections.
                unfiltered_handler = ruamel.yaml.YAML()
                unfiltered_handler.preserve_quotes = True
                original_content = unfiltered_handler.load(path)
                if isinstance(original_content, dict):
                    filtered_content, _ = _partition_yaml_top_level_sections(original_content)
                    if not yaml_handler.preserve_quotes:
                        _normalize_managed_quote_styles(
                            filtered_content,
                            width=t.cast("int", yaml_handler.width),
                            prefix_colon=yaml_handler.prefix_colon,
                            indent_mapping=t.cast("int", yaml_handler.map_indent),
                        )
                else:
                    filtered_content = original_content

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
