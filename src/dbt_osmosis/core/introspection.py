# pyright: reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false
from __future__ import annotations

import re
import threading
import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path

# pyright: reportPrivateImportUsage=false
from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.exceptions.compilation import ApproximateMatchError
from dbt.contracts.graph.nodes import ResultNode  # pyright: ignore[reportPrivateImportUsage]
from dbt_common.contracts.metadata import (
    ColumnMetadata,  # pyright: ignore[reportPrivateImportUsage]
)

from dbt_osmosis.core import logger
from dbt_osmosis.core.catalog_operations import _generate_catalog, _load_catalog

__all__ = [
    "_COLUMN_LIST_CACHE",
    # New configuration sources for US1
    "ConfigMetaSource",
    "ConfigSourceName",
    # Foundational classes for unified config resolution
    "ConfigurationError",
    "ConfigurationSource",
    "ProjectVarsSource",
    # Unified property access for US2
    "PropertyAccessor",
    "PropertySource",
    "SettingsResolver",
    "SupplementaryFileSource",
    "UnrenderedConfigSource",
    "_find_first",
    "_generate_catalog",
    "_get_effective_column_meta",
    "_get_effective_column_tags",
    "_load_catalog",
    "_maybe_use_precise_dtype",
    "get_columns",
    "normalize_column_name",
    "resolve_setting",
]

T = t.TypeVar("T")
_MISSING = object()


@dataclass(frozen=True)
class _WarehouseColumnCacheKey:
    """Scope warehouse column cache entries to the active dbt connection context.

    We intentionally cache raw adapter columns instead of processed ColumnMetadata
    because the final metadata depends on per-call settings, ignore patterns, and
    node-level overrides.
    """

    rendered_relation: str
    project_root: str
    profile_name: str
    target_name: str
    database_type: str


_COLUMN_LIST_CACHE: dict[_WarehouseColumnCacheKey, tuple[BaseColumn, ...]] = {}
"""Cache raw warehouse columns to avoid redundant live introspection.

Thread-safety: Protected by _COLUMN_LIST_CACHE_LOCK. All reads and writes
must be guarded by this lock. The cache is unbounded and may grow indefinitely.
"""

_COLUMN_LIST_CACHE_LOCK = threading.Lock()
"""Lock to protect _COLUMN_LIST_CACHE from concurrent access.

Critical sections: get_columns() function performs cache reads and writes
under this lock. All access to _COLUMN_LIST_CACHE must be synchronized.
"""


def _build_column_cache_key(context: t.Any, rendered_relation: str) -> _WarehouseColumnCacheKey:
    """Build a warehouse cache key for a relation in the active dbt context."""
    runtime_cfg = context.project.runtime_cfg
    credentials = getattr(runtime_cfg, "credentials", None)
    return _WarehouseColumnCacheKey(
        rendered_relation=rendered_relation,
        project_root=str(getattr(runtime_cfg, "project_root", "") or ""),
        profile_name=str(getattr(runtime_cfg, "profile_name", "") or ""),
        target_name=str(getattr(runtime_cfg, "target_name", "") or ""),
        database_type=str(getattr(credentials, "type", "") or ""),
    )


# =============================================================================
# Foundational Classes for Unified Configuration Resolution System
# =============================================================================


class ConfigurationError(Exception):
    """Exception raised when configuration file is invalid or cannot be read.

    This exception is used throughout the unified configuration resolution system
    to indicate errors related to configuration file parsing, validation, or access.

    Attributes:
        message: The error message describing what went wrong.
        file_path: Optional path to the configuration file that caused the error.

    Example:
        >>> raise ConfigurationError("Invalid YAML syntax", "/path/to/config.yml")
        ConfigurationError: Invalid YAML syntax (file: /path/to/config.yml)

    """

    def __init__(self, message: str, file_path: str | None = None) -> None:
        self.file_path = file_path
        self.message = message
        if file_path:
            full_message = f"{message} (file: {file_path})"
        else:
            full_message = message
        super().__init__(full_message)


class ConfigSourceName(Enum):
    """Enumeration of configuration source names for logging and identification.

    Each source name corresponds to a specific location where configuration
    values can be retrieved. These names are used for logging which source
    provided a resolved value.

    Values:
        COLUMN_META: Column-level meta dictionary (highest priority)
        NODE_META: Node-level meta dictionary
        CONFIG_EXTRA: Node config.extra dictionary
        CONFIG_META: Node config.meta dictionary (dbt 1.10+)
        UNRENDERED_CONFIG: Node unrendered_config dictionary (dbt 1.10+)
        CONTEXT_SETTINGS: Explicit runtime context settings
        PROJECT_VARS: Project-level vars from dbt_project.yml
        SUPPLEMENTARY_FILE: Supplementary dbt-osmosis.yml file
        FALLBACK: Default fallback value (lowest priority)
    """

    COLUMN_META = "column_meta"
    NODE_META = "node_meta"
    CONFIG_EXTRA = "config_extra"
    CONFIG_META = "config_meta"
    UNRENDERED_CONFIG = "unrendered_config"
    CONTEXT_SETTINGS = "context_settings"
    PROJECT_VARS = "project_vars"
    SUPPLEMENTARY_FILE = "supplementary_file"
    FALLBACK = "fallback"


class PropertySource(Enum):
    """Enumeration of property sources for model and column metadata.

    This enum defines where model properties (like descriptions, tags, meta)
    can be retrieved from. It's used by the PropertyAccessor to specify
    which source to read from.

    Values:
        MANIFEST: Parsed manifest.json with rendered jinja values
        YAML: Raw YAML files with unrendered jinja templates
        DATABASE: Unsupported placeholder for future warehouse metadata introspection

    Example:
        >>> # Get unrendered description from YAML
        >>> accessor.get_description(node, source=PropertySource.YAML)

    """

    MANIFEST = "manifest"
    YAML = "yaml"
    DATABASE = "database"


def _get_mapping_value(source: t.Any, key: str) -> t.Any | None:
    """Read a value from either a mapping or an object attribute."""
    if isinstance(source, t.Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _get_options_value(
    options: t.Any,
    kebab_key: str,
    snake_key: str,
) -> t.Any:
    """Read a setting from an options object while preserving falsey values."""
    if not isinstance(options, t.Mapping):
        return _MISSING
    if kebab_key in options:
        return options[kebab_key]
    if snake_key in options:
        return options[snake_key]
    return _MISSING


def _setting_key_variants(setting_name: str) -> tuple[str, str]:
    """Return kebab-case and snake_case variants for a setting name."""
    return setting_name.replace("_", "-"), setting_name.replace("-", "_")


def _get_setting_from_mapping(
    source: t.Any,
    setting_name: str,
    *,
    direct_keys: bool,
) -> t.Any:
    """Read a setting from a mapping while preserving explicit falsey values."""
    if not isinstance(source, t.Mapping):
        return _MISSING

    kebab_key, snake_key = _setting_key_variants(setting_name)
    for prefixed_name in (f"dbt-osmosis-{kebab_key}", f"dbt_osmosis_{snake_key}"):
        if prefixed_name in source:
            return source[prefixed_name]

    if direct_keys:
        if kebab_key in source:
            return source[kebab_key]
        if snake_key in source:
            return source[snake_key]

    for options_name in ("dbt-osmosis-options", "dbt_osmosis_options"):
        value = _get_options_value(source.get(options_name, {}), kebab_key, snake_key)
        if value is not _MISSING:
            return value
    return _MISSING


def _setting_value_or_none(
    source: t.Any,
    setting_name: str,
    *,
    direct_keys: bool,
) -> t.Any | None:
    """Return a setting value for debug chains, or None when the source is missing."""
    value = _get_setting_from_mapping(source, setting_name, direct_keys=direct_keys)
    return None if value is _MISSING else value


def _same_setting_value(left: t.Any, right: t.Any) -> bool:
    """Compare settings values with Click tuple defaults matching dataclass lists."""
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return list(left) == list(right)
    return left == right


def _get_explicit_context_setting_value(context: t.Any, setting_name: str) -> t.Any:
    """Return a non-default runtime setting value, or _MISSING when not explicit."""
    if context is None or not hasattr(context, "settings"):
        return _MISSING

    attr_name = setting_name.replace("-", "_")
    settings_obj = context.settings
    if not hasattr(settings_obj, attr_name):
        return _MISSING

    current_value = getattr(settings_obj, attr_name)

    try:
        from dbt_osmosis.core.settings import YamlRefactorSettings

        default_value = getattr(YamlRefactorSettings(), attr_name)
    except Exception:  # noqa: BLE001
        return _MISSING

    if not _same_setting_value(current_value, default_value):
        return current_value
    return _MISSING


def _project_vars_dict(context: t.Any) -> dict[str, t.Any] | None:
    """Return project vars as a dictionary when available."""
    if not hasattr(context, "project"):
        return None
    if not hasattr(context.project, "runtime_cfg"):
        return None
    if not hasattr(context.project.runtime_cfg, "vars"):
        return None

    vars_dict = context.project.runtime_cfg.vars
    if hasattr(vars_dict, "to_dict"):
        vars_dict = vars_dict.to_dict()
    if isinstance(vars_dict, dict):
        return vars_dict
    return None


def _project_vars_sources(vars_dict: dict[str, t.Any]) -> tuple[t.Any, t.Any, t.Any]:
    """Return project vars lookup sections in precedence order."""
    return (
        vars_dict.get("dbt-osmosis", {}),
        vars_dict.get("dbt_osmosis", {}),
        vars_dict,
    )


def _merge_column_meta(
    legacy_meta: t.Mapping[str, t.Any],
    config_meta: t.Mapping[str, t.Any],
) -> dict[str, t.Any]:
    """Merge legacy column meta with dbt 1.10+ config.meta.

    config.meta is the newer dbt location and wins key conflicts; legacy top-level
    meta remains supported. Nested dbt-osmosis options objects are merged so one
    location does not discard unrelated options from the other.
    """
    merged = dict(legacy_meta)
    for key, value in config_meta.items():
        current = merged.get(key)
        if (
            key in {"dbt-osmosis-options", "dbt_osmosis_options"}
            and isinstance(
                current,
                dict,
            )
            and isinstance(value, dict)
        ):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _get_effective_column_meta(column: t.Any) -> dict[str, t.Any]:
    """Return column meta with legacy meta plus dbt 1.10+ config.meta."""
    legacy_meta = _get_mapping_value(column, "meta")
    config = _get_mapping_value(column, "config")
    config_meta = _get_mapping_value(config, "meta") if config is not None else None

    legacy_meta = legacy_meta if isinstance(legacy_meta, t.Mapping) else {}
    config_meta = config_meta if isinstance(config_meta, t.Mapping) else {}
    return _merge_column_meta(legacy_meta, config_meta)


def _get_effective_column_tags(column: t.Any) -> list[str]:
    """Return column tags with legacy tags followed by dbt 1.10+ config.tags."""
    legacy_tags = _get_mapping_value(column, "tags")
    config = _get_mapping_value(column, "config")
    config_tags = _get_mapping_value(config, "tags") if config is not None else None

    tags: list[str] = []
    for source in (legacy_tags, config_tags):
        if not isinstance(source, (list, tuple)):
            continue
        for tag in source:
            if isinstance(tag, str) and tag not in tags:
                tags.append(tag)
    return tags


class ConfigurationSource(ABC):
    """Abstract base class for configuration sources in the resolution chain.

    Each configuration source knows how to extract values from a specific
    location (column meta, node meta, config.extra, etc.). Sources are
    checked in precedence order, and the first non-None value is returned.

    Concrete implementations must implement the get() method to retrieve
    values from their specific location.

    Attributes:
        name: The ConfigSourceName enum value for this source (used for logging).

    Example:
        >>> class ColumnMetaSource(ConfigurationSource):
        ...     def __init__(self, node: ResultNode, column: str):
        ...         super().__init__(ConfigSourceName.COLUMN_META)
        ...         self.node = node
        ...         self.column = column
        ...
        ...     def get(self, key: str) -> Any | None:
        ...         if column := self.node.columns.get(self.column):
        ...             return column.meta.get(key)
        ...         return None

    """

    def __init__(self, name: ConfigSourceName) -> None:
        self._name = name

    @property
    def name(self) -> ConfigSourceName:
        """Return the ConfigSourceName enum value for this source."""
        return self._name

    @abstractmethod
    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from this source.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """


class ConfigMetaSource(ConfigurationSource):
    """Configuration source for node.config.meta.

    This source reads configuration from the config.meta dictionary,
    which is available in dbt 1.10 and later versions. Older source
    configs retain project-level meta in config.extra instead.

    Supported key variants:
    - dbt-osmosis-<key> (kebab-case with prefix)
    - dbt_osmosis_<key> (snake_case with prefix)
    - <key> (direct key without prefix)
    - dbt-osmosis-options.<key> (nested options object)

    Example:
        >>> source = ConfigMetaSource(node)
        >>> value = source.get("output-to-lower")

    """

    def __init__(self, node: ResultNode) -> None:
        super().__init__(ConfigSourceName.CONFIG_META)
        self._node = node

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from config.meta.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        config = getattr(self._node, "config", None)
        extra = getattr(config, "extra", {})
        config_meta = getattr(config, "meta", _get_mapping_value(extra, "meta"))
        if not isinstance(config_meta, dict):
            return None

        # Normalize key to both kebab and snake variants
        kebab_key = key.replace("_", "-")
        snake_key = key.replace("-", "_")

        # Check prefixed variants first (highest precedence within this source)
        prefixed_kebab = f"dbt-osmosis-{kebab_key}"
        prefixed_snake = f"dbt_osmosis_{snake_key}"

        if prefixed_kebab in config_meta:
            return config_meta[prefixed_kebab]
        if prefixed_snake in config_meta:
            return config_meta[prefixed_snake]

        # Check direct key variants
        if kebab_key in config_meta:
            return config_meta[kebab_key]
        if snake_key in config_meta:
            return config_meta[snake_key]

        # Check options objects
        options_kebab = config_meta.get("dbt-osmosis-options", {})
        options_snake = config_meta.get("dbt_osmosis_options", {})

        value = _get_options_value(options_kebab, kebab_key, snake_key)
        if value is not _MISSING:
            return value
        value = _get_options_value(options_snake, kebab_key, snake_key)
        if value is not _MISSING:
            return value

        return None


class UnrenderedConfigSource(ConfigurationSource):
    """Configuration source for node.unrendered_config (dbt 1.10+).

    This source reads configuration from the unrendered_config dictionary,
    which is available in dbt 1.10 and later versions. It gracefully
    handles versions where this field doesn't exist.

    Supported key variants:
    - dbt-osmosis-<key> (kebab-case with prefix)
    - dbt_osmosis_<key> (snake_case with prefix)
    - dbt-osmosis-options.<key> (nested options object)

    Note: This source only supports prefixed variants (not direct keys),
    as unrendered_config is typically used for config() blocks which
    require valid Python identifiers.

    Example:
        >>> source = UnrenderedConfigSource(node)
        >>> value = source.get("skip-add-columns")

    """

    def __init__(self, node: ResultNode) -> None:
        super().__init__(ConfigSourceName.UNRENDERED_CONFIG)
        self._node = node

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from unrendered_config.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        # Gracefully handle dbt versions < 1.10 where unrendered_config doesn't exist
        if not hasattr(self._node, "unrendered_config"):
            return None

        unrendered_config = self._node.unrendered_config
        if not isinstance(unrendered_config, dict):
            return None

        # Normalize key to both kebab and snake variants
        kebab_key = key.replace("_", "-")
        snake_key = key.replace("-", "_")

        # Check prefixed variants only (unrendered_config is for config() blocks)
        prefixed_kebab = f"dbt-osmosis-{kebab_key}"
        prefixed_snake = f"dbt_osmosis_{snake_key}"

        if prefixed_kebab in unrendered_config:
            return unrendered_config[prefixed_kebab]
        if prefixed_snake in unrendered_config:
            return unrendered_config[prefixed_snake]

        # Check options objects
        options_kebab = unrendered_config.get("dbt-osmosis-options", {})
        options_snake = unrendered_config.get("dbt_osmosis_options", {})

        value = _get_options_value(options_kebab, kebab_key, snake_key)
        if value is not _MISSING:
            return value
        value = _get_options_value(options_snake, kebab_key, snake_key)
        if value is not _MISSING:
            return value

        return None


class ProjectVarsSource(ConfigurationSource):
    """Configuration source for project-level vars in dbt_project.yml.

    This source reads configuration from the project's runtime_cfg.vars,
    which contains variables defined in dbt_project.yml under the vars: section.

    Supported key variants:
    - dbt-osmosis.<key> (under dbt-osmosis top-level key)
    - dbt_osmosis.<key> (under dbt_osmosis top-level key)
    - dbt-osmosis-options.<key> / dbt_osmosis_options.<key> nested in those sections
    - dbt-osmosis-<key> / dbt_osmosis_<key> and <key> direct top-level vars

    Example:
        >>> source = ProjectVarsSource(context)
        >>> value = source.get("skip-add-tags")

    """

    def __init__(self, context: t.Any) -> None:
        super().__init__(ConfigSourceName.PROJECT_VARS)
        self._context = context

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from project vars.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        vars_dict = _project_vars_dict(self._context)
        if vars_dict is None:
            return None

        for source in _project_vars_sources(vars_dict):
            value = _get_setting_from_mapping(source, key, direct_keys=True)
            if value is not _MISSING:
                return value

        return None


class SupplementaryFileSource(ConfigurationSource):
    """Configuration source for dbt-osmosis.yml supplementary file.

    This source reads configuration from a dbt-osmosis.yml file in the
    project root, allowing users to define configuration outside of
    dbt's hot path.

    Supported key variants:
    - dbt-osmosis-<key> (kebab-case with prefix)
    - dbt_osmosis_<key> (snake_case with prefix)
    - <key> (direct key without prefix)
    - dbt-osmosis-options.<key> (nested options object)

    The file is optional - if it doesn't exist, this source returns None
    for all keys without error.

    Example:
        >>> source = SupplementaryFileSource(context)
        >>> value = source.get("skip-add-tags")

    """

    _SHARED_CONFIG_CACHE: t.ClassVar[dict[tuple[Path, int, int], dict[str, t.Any]]] = {}
    _SHARED_CONFIG_CACHE_LOCK: t.ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, context: t.Any) -> None:
        super().__init__(ConfigSourceName.SUPPLEMENTARY_FILE)
        self._context = context
        self._config_cache: dict[str, t.Any] | None = None

    def _load_config(self) -> dict[str, t.Any]:
        """Load dbt-osmosis.yml from project root.

        Returns:
            Configuration dictionary, empty dict if file doesn't exist.

        Raises:
            ConfigurationError: If the file exists but contains invalid YAML syntax.

        """
        if self._config_cache is not None:
            return self._config_cache

        # Get project root
        if not hasattr(self._context, "project"):
            return {}
        if not hasattr(self._context.project, "runtime_cfg"):
            return {}
        if not hasattr(self._context.project.runtime_cfg, "project_root"):
            return {}

        try:
            project_root = Path(self._context.project.runtime_cfg.project_root)
        except TypeError:
            return {}
        config_file = project_root / "dbt-osmosis.yml"

        # Check if file exists first
        if not config_file.is_file():
            return {}

        try:
            stat_result = config_file.stat()
        except OSError as e:
            raise ConfigurationError(
                f"Error reading {config_file.name}: {e}",
                file_path=str(config_file),
            ) from e

        cache_key = (config_file, stat_result.st_mtime_ns, stat_result.st_size)
        with self._SHARED_CONFIG_CACHE_LOCK:
            cached_config = self._SHARED_CONFIG_CACHE.get(cache_key)
        if cached_config is not None:
            self._config_cache = cached_config
            return self._config_cache

        # Use standard ruamel.yaml (not OsmosisYAML) to avoid filtering
        # dbt-osmosis.yml contains arbitrary config keys, not dbt structures
        import ruamel.yaml

        yaml_handler = ruamel.yaml.YAML()
        yaml_handler.preserve_quotes = True

        try:
            with Path(config_file).open("r") as f:
                content = yaml_handler.load(f)
                # Empty file or None content is OK, treat as empty config
                if content is None:
                    self._config_cache = {}
                elif isinstance(content, dict):
                    self._config_cache = content
                else:
                    # File exists but content is not a dict (e.g., just a string)
                    # This is invalid configuration
                    raise ConfigurationError(
                        f"Invalid configuration in {config_file.name}: "
                        f"expected a dictionary, got {type(content).__name__}",
                        file_path=str(config_file),
                    )
        except ruamel.yaml.YAMLError as e:
            # Invalid YAML syntax - raise ConfigurationError with helpful message
            raise ConfigurationError(
                f"Invalid YAML syntax in {config_file.name}: {e}",
                file_path=str(config_file),
            ) from e
        except ConfigurationError:
            # Re-raise ConfigurationError as-is
            raise
        except Exception as e:
            # Other errors (file read permissions, etc.) - also raise
            raise ConfigurationError(
                f"Error reading {config_file.name}: {e}",
                file_path=str(config_file),
            ) from e

        with self._SHARED_CONFIG_CACHE_LOCK:
            self._SHARED_CONFIG_CACHE[cache_key] = self._config_cache

        return self._config_cache

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from dbt-osmosis.yml.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        config = self._load_config()
        if not isinstance(config, dict):
            return None

        # Normalize key
        kebab_key = key.replace("_", "-")
        snake_key = key.replace("-", "_")

        # Check prefixed variants first
        prefixed_kebab = f"dbt-osmosis-{kebab_key}"
        prefixed_snake = f"dbt_osmosis_{snake_key}"

        if prefixed_kebab in config:
            return config[prefixed_kebab]
        if prefixed_snake in config:
            return config[prefixed_snake]

        # Check direct key variants
        if kebab_key in config:
            return config[kebab_key]
        if snake_key in config:
            return config[snake_key]

        # Check options objects
        options_kebab = config.get("dbt-osmosis-options", {})
        options_snake = config.get("dbt_osmosis_options", {})

        value = _get_options_value(options_kebab, kebab_key, snake_key)
        if value is not _MISSING:
            return value
        value = _get_options_value(options_snake, kebab_key, snake_key)
        if value is not _MISSING:
            return value

        return None


def _configuration_source_value(source: ConfigurationSource, setting_name: str) -> t.Any:
    """Read a ConfigurationSource using _MISSING for absent values."""
    value = source.get(setting_name)
    return _MISSING if value is None else value


def _node_dict_setting_sources(
    node: ResultNode,
    column_name: str | None,
) -> list[tuple[str, t.Any, bool]]:
    """Return mapping-backed node setting sources in precedence order."""
    sources: list[tuple[str, t.Any, bool]] = []
    if column_name and (column := node.columns.get(column_name)):
        sources.append(("column_meta", _get_effective_column_meta(column), True))
    sources.extend([
        ("node_meta", getattr(node, "meta", {}), True),
        ("config_extra", getattr(getattr(node, "config", None), "extra", {}), False),
    ])
    return sources


def _node_configuration_sources(node: ResultNode) -> list[tuple[str, ConfigurationSource]]:
    """Return object-backed setting sources in precedence order."""
    sources: list[tuple[str, ConfigurationSource]] = []
    if hasattr(node, "config"):
        sources.append(("config.meta", ConfigMetaSource(node)))
    if hasattr(node, "unrendered_config"):
        sources.append(("unrendered_config (dbt 1.10+)", UnrenderedConfigSource(node)))
    return sources


def _log_resolved_setting(setting_name: str, source_name: str) -> None:
    logger.debug(":gear: Resolved setting '%s' from %s", setting_name, source_name)


def _resolve_node_setting(
    setting_name: str,
    node: ResultNode,
    column_name: str | None,
) -> t.Any:
    """Resolve a setting from node-backed sources, or _MISSING."""
    for source_name, source, direct_keys in _node_dict_setting_sources(node, column_name):
        value = _get_setting_from_mapping(source, setting_name, direct_keys=direct_keys)
        if value is not _MISSING:
            _log_resolved_setting(setting_name, source_name)
            return value

    for source_name, source in _node_configuration_sources(node):
        value = _configuration_source_value(source, setting_name)
        if value is not _MISSING:
            _log_resolved_setting(setting_name, source_name)
            return value
    return _MISSING


def _explicit_context_source_value(
    active_context: t.Any,
    setting_name: str,
    fallback: t.Any,
) -> t.Any:
    """Return explicit runtime settings when they match the existing fallback rule."""
    current_value = _get_explicit_context_setting_value(active_context, setting_name)
    if current_value is _MISSING:
        return _MISSING
    if not _same_setting_value(current_value, fallback):
        return _MISSING
    return current_value


def _context_configuration_sources(active_context: t.Any) -> tuple[ConfigurationSource, ...]:
    """Return context-backed project setting sources in precedence order."""
    return (
        SupplementaryFileSource(active_context),
        ProjectVarsSource(active_context),
    )


def _resolve_context_setting(
    setting_name: str,
    active_context: t.Any | None,
    fallback: t.Any,
) -> t.Any:
    """Resolve a setting from explicit context, supplementary file, or project vars."""
    explicit_value = _explicit_context_source_value(active_context, setting_name, fallback)
    if explicit_value is not _MISSING:
        _log_resolved_setting(setting_name, "explicit context settings")
        return explicit_value

    if active_context is None:
        return _MISSING

    for context_source in _context_configuration_sources(active_context):
        value = _configuration_source_value(context_source, setting_name)
        if value is not _MISSING:
            _log_resolved_setting(setting_name, context_source.name.value)
            return value
    return _MISSING


def _node_precedence_entries(
    setting_name: str,
    node: ResultNode,
    column_name: str | None,
) -> list[tuple[ConfigSourceName, t.Any | None]]:
    """Return node-backed debug precedence entries."""
    entries: list[tuple[ConfigSourceName, t.Any | None]] = []
    if column_name and (column := node.columns.get(column_name)):
        entries.append((
            ConfigSourceName.COLUMN_META,
            _setting_value_or_none(
                _get_effective_column_meta(column),
                setting_name,
                direct_keys=True,
            ),
        ))

    entries.extend([
        (
            ConfigSourceName.NODE_META,
            _setting_value_or_none(getattr(node, "meta", {}), setting_name, direct_keys=True),
        ),
        (
            ConfigSourceName.CONFIG_EXTRA,
            _setting_value_or_none(
                getattr(getattr(node, "config", None), "extra", {}),
                setting_name,
                direct_keys=False,
            ),
        ),
        (ConfigSourceName.CONFIG_META, _config_meta_precedence_value(setting_name, node)),
        (
            ConfigSourceName.UNRENDERED_CONFIG,
            _unrendered_config_precedence_value(setting_name, node),
        ),
    ])
    return entries


def _config_meta_precedence_value(setting_name: str, node: ResultNode) -> t.Any | None:
    return ConfigMetaSource(node).get(setting_name)


def _unrendered_config_precedence_value(setting_name: str, node: ResultNode) -> t.Any | None:
    if hasattr(node, "unrendered_config"):
        return UnrenderedConfigSource(node).get(setting_name)
    return None


def _context_precedence_entries(
    setting_name: str,
    active_context: t.Any,
) -> list[tuple[ConfigSourceName, t.Any | None]]:
    """Return context-backed debug precedence entries."""
    context_settings_value = _get_explicit_context_setting_value(active_context, setting_name)
    if context_settings_value is _MISSING:
        context_settings_value = None
    return [
        (ConfigSourceName.CONTEXT_SETTINGS, context_settings_value),
        (
            ConfigSourceName.SUPPLEMENTARY_FILE,
            SupplementaryFileSource(active_context).get(setting_name),
        ),
        (ConfigSourceName.PROJECT_VARS, ProjectVarsSource(active_context).get(setting_name)),
    ]


def _yaml_path_template_from_mapping(source: t.Any) -> str | None:
    """Return a dbt-osmosis path template from a mapping-like source."""
    if not isinstance(source, t.Mapping):
        return None
    for key in ("dbt-osmosis", "dbt_osmosis"):
        if key in source:
            return t.cast("str | None", source[key])
    return None


def _yaml_path_template_sources(node: ResultNode) -> list[tuple[str, t.Any]]:
    """Return YAML path template sources in precedence order."""
    sources: list[tuple[str, t.Any]] = []
    if hasattr(node, "config") and hasattr(node.config, "extra"):
        sources.append(("config.extra", node.config.extra))
    if hasattr(node, "config") and hasattr(node.config, "meta"):
        sources.append(("config.meta", getattr(node.config, "meta", None)))
    if hasattr(node, "meta"):
        sources.append(("node.meta", node.meta))
    if hasattr(node, "unrendered_config"):
        sources.append(("unrendered_config", node.unrendered_config))
    return sources


# =============================================================================
# Existing Settings Resolver (to be extended)
# =============================================================================


@dataclass
class SettingsResolver:
    """Resolves configuration settings for dbt nodes from multiple sources with clear precedence rules.

    This class encapsulates the complex settings resolution logic that was previously in
    _get_setting_for_node. It provides a clean, testable interface for retrieving
    configuration values from various sources with defined precedence.

    Settings Resolution Precedence (highest to lowest):
    1. Column level settings (if column specified)
       - Column meta: <key>
       - Column meta: dbt-osmosis-<key>
       - Column meta: dbt_osmosis_<key> (python identifier variant)
       - Column meta: dbt-osmosis-options.<key>
       - Column meta: dbt_osmosis_options.<key> (python identifier variant)

    2. Node level settings
       - Node meta: <key>
       - Node meta: dbt-osmosis-<key>
       - Node meta: dbt_osmosis_<key> (python identifier variant)
       - Node meta: dbt-osmosis-options.<key>
       - Node meta: dbt_osmosis_options.<key> (python identifier variant)
       - Node config extra: dbt-osmosis-<key>
       - Node config extra: dbt_osmosis_<key> (python identifier variant)
       - Node config extra: dbt-osmosis-options.<key>
       - Node config extra: dbt_osmosis_options.<key> (python identifier variant)
       - Node config extra: <key> (direct key)
       - Node config extra: <identifier> (python identifier variant)

    3. dbt 1.10+ node config sources
       - Node config.meta
       - Node unrendered_config

    4. Context-backed project sources (when context is provided)
       - Supplementary dbt-osmosis.yml
       - Project vars

    5. Fallback value
    """

    context: t.Any | None = None

    def resolve(
        self,
        setting_name: str,
        node: ResultNode | None = None,
        column_name: str | None = None,
        *,
        context: t.Any | None = None,
        fallback: t.Any | None = None,
    ) -> t.Any:
        """Resolve a setting value from the configured sources.

        Args:
            setting_name: The name of the setting to resolve (supports both kebab-case and snake_case)
            node: The dbt node to resolve settings for
            column_name: Optional column name to check column-level settings
            context: Optional dbt-osmosis context for supplementary file and project vars
            fallback: Default value if setting not found in any source

        Returns:
            The resolved setting value or fallback if not found

        """
        active_context = context if context is not None else self.context

        if node is not None:
            node_value = _resolve_node_setting(setting_name, node, column_name)
            if node_value is not _MISSING:
                return node_value

        context_value = _resolve_context_setting(setting_name, active_context, fallback)
        if context_value is not _MISSING:
            return context_value

        logger.debug(
            ":gear: Setting '%s' not found, using fallback: %s",
            setting_name,
            fallback,
        )
        return fallback

    def has(
        self,
        setting_name: str,
        node: ResultNode | None = None,
        column_name: str | None = None,
        *,
        context: t.Any | None = None,
    ) -> bool:
        """Check if a setting exists in any source.

        Args:
            setting_name: The name of the setting to check
            node: The dbt node to check for settings
            column_name: Optional column name to check column-level settings
            context: Optional dbt-osmosis context for supplementary file and project vars

        Returns:
            True if the setting exists in any source, False otherwise

        """
        active_context = context if context is not None else self.context
        if _get_explicit_context_setting_value(active_context, setting_name) is not _MISSING:
            return True

        # Use resolve with a sentinel value to check if setting exists
        sentinel = object()
        result = self.resolve(setting_name, node, column_name, context=context, fallback=sentinel)
        return result is not sentinel

    def get_precedence_chain(
        self,
        setting_name: str,
        node: ResultNode | None = None,
        column_name: str | None = None,
        *,
        context: t.Any | None = None,
    ) -> list[tuple[ConfigSourceName, t.Any | None]]:
        """Get the full precedence chain for a setting with values from each source.

        This is useful for debugging and understanding which source provided
        the final value.

        Args:
            setting_name: The name of the setting to check
            node: The dbt node to check for settings
            column_name: Optional column name to check column-level settings

        Returns:
            A list of tuples (source_name, value) for each source in precedence order.
            Values are None if the source doesn't have the setting.

        """
        active_context = context if context is not None else self.context
        chain: list[tuple[ConfigSourceName, t.Any | None]] = []

        if node is not None:
            chain.extend(_node_precedence_entries(setting_name, node, column_name))

        if active_context is not None:
            chain.extend(_context_precedence_entries(setting_name, active_context))

        chain.append((ConfigSourceName.FALLBACK, None))

        return chain

    def get_yaml_path_template(
        self,
        node: ResultNode,
    ) -> str | None:
        """Get the YAML path template for a node.

        The path template is a special configuration value that specifies where
        the node's YAML file should be located. It uses the bare `dbt-osmosis` or
        `dbt_osmosis` key (without a setting suffix) in config sources.

        Precedence (highest to lowest):
            1. node.config.extra["dbt-osmosis"] or ["dbt_osmosis"]
            2. node.config.meta["dbt-osmosis"] or ["dbt_osmosis"] (dbt 1.10+)
            3. node.meta["dbt-osmosis"] or ["dbt_osmosis"]
            4. node.unrendered_config["dbt-osmosis"] or ["dbt_osmosis"] (dbt 1.10+)

        Args:
            node: The dbt node to get the path template for.

        Returns:
            The path template string, or None if not found.

        """
        if node is None:
            return None

        for source_name, source in _yaml_path_template_sources(node):
            result = _yaml_path_template_from_mapping(source)
            if result:
                logger.debug(":gear: Found YAML path template in %s: %s", source_name, result)
                return result

        logger.debug(":gear: No YAML path template found in node config")
        return None


_SETTINGS_RESOLVER = SettingsResolver()


@t.overload
def _find_first(coll: t.Iterable[T], predicate: t.Callable[[T], bool], default: T) -> T:
    pass


@t.overload
def _find_first(
    coll: t.Iterable[T],
    predicate: t.Callable[[T], bool],
    default: None = ...,
) -> T | None:
    pass


def _find_first(
    coll: t.Iterable[T],
    predicate: t.Callable[[T], bool],
    default: T | None = None,
) -> T | None:
    """Find the first item in a container that satisfies a predicate."""
    for item in coll:
        if predicate(item):
            return item
    return default


def normalize_column_name(column: str, credentials_type: str) -> str:
    """Apply case normalization to a column name based on the credentials type."""
    if credentials_type == "snowflake" and column.startswith('"') and column.endswith('"'):
        logger.debug(":snowflake: Column name found with double-quotes => %s", column)
    elif credentials_type == "snowflake":
        return column.upper()
    return column.strip('"').strip("`").strip("[]")


def _maybe_use_precise_dtype(
    col: BaseColumn | ColumnMetadata,
    settings: t.Any,
    node: ResultNode | None = None,
    *,
    context: t.Any | None = None,
) -> str:
    """Use precise data type if enabled in settings."""
    use_num_prec = _SETTINGS_RESOLVER.resolve(
        "numeric-precision-and-scale",
        node,
        column_name=col.name,
        context=context,
        fallback=settings.numeric_precision_and_scale,
    )
    use_chr_prec = _SETTINGS_RESOLVER.resolve(
        "string-length",
        node,
        column_name=col.name,
        context=context,
        fallback=settings.string_length,
    )
    # Handle BaseColumn from introspection (has is_numeric/is_string methods)
    # vs ColumnMetadata from catalog (no such methods, type already set)
    if isinstance(col, BaseColumn):
        if (col.is_numeric() and use_num_prec) or (col.is_string() and use_chr_prec):
            logger.debug(":ruler: Using precise data type => %s", col.data_type)
            return col.data_type
        if hasattr(col, "mode"):
            return col.data_type
        return col.dtype
    # ColumnMetadata from catalog - type is already set correctly
    return col.type


def _get_setting_for_node(
    opt: str,
    /,
    node: ResultNode | None = None,
    col: str | None = None,
    *,
    fallback: t.Any | None = None,
) -> t.Any:
    """Get a configuration value for a dbt node from the node's meta and config.

    models: # dbt_project
      project:
        staging:
          +dbt-osmosis: path/spec.yml
          +dbt-osmosis-options:
            string-length: true
            numeric-precision-and-scale: true
            skip-add-columns: true
          +dbt-osmosis-skip-add-tags: true

    models: # schema
      - name: foo
        meta:
          string-length: false
          prefix: user_ # we strip this prefix to inherit from columns upstream, useful in staging models that prefix everything
        columns:
          - bar:
            meta:
              dbt-osmosis-skip-meta-merge: true # per-column options
              dbt-osmosis-options:
                output-to-lower: true

    {{ config(..., dbt_osmosis_options={"prefix": "account_"}) }} -- sql

    We check for
    From node column meta
    - <key>
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    From node meta
    - <key>
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    From node config
    - dbt-osmosis-<key>
    - dbt-osmosis-options.<key>
    - dbt_osmosis_<key> # allows use in {{ config(...) }} by being a valid python identifier
    - dbt_osmosis_options.<key> # allows use in {{ config(...) }} by being a valid python identifier
    """
    return _SETTINGS_RESOLVER.resolve(
        opt,
        node,
        column_name=col,
        fallback=fallback,
    )


def resolve_setting(
    context: t.Any,
    setting_name: str,
    /,
    node: ResultNode | None = None,
    col: str | None = None,
    *,
    fallback: t.Any | None = None,
) -> t.Any:
    """Resolve a dbt-osmosis setting using node sources plus context-backed project sources."""
    return _SETTINGS_RESOLVER.resolve(
        setting_name,
        node,
        column_name=col,
        context=context,
        fallback=fallback,
    )


def _relation_for_column_collection(
    context: t.Any,
    relation: BaseRelation | ResultNode,
) -> tuple[BaseRelation, ResultNode | None]:
    """Return a dbt relation plus the source node when one was provided."""
    if isinstance(relation, BaseRelation):
        return relation, None

    return (
        context.project.adapter.Relation.create_from(
            context.project.adapter.config,  # pyright: ignore[reportUnknownArgumentType]
            relation,  # pyright: ignore[reportArgumentType]
        ),
        relation,
    )


def _rendered_relation_name(relation: BaseRelation) -> str:
    """Render a relation if possible, otherwise stringify it."""
    if not relation:
        return ""
    renderer = getattr(t.cast(t.Any, relation), "render", None)
    return t.cast("str", renderer()) if callable(renderer) else str(relation)


def _iter_flattened_columns(column: BaseColumn | ColumnMetadata) -> t.Iterable[t.Any]:
    """Yield a column plus any adapter-provided flattened child columns."""
    yield column
    flattener = getattr(t.cast(t.Any, column), "flatten", None)
    if callable(flattener):
        yield from t.cast(t.Iterable[t.Any], flattener())


def _column_matches_ignore_pattern(context: t.Any, column_name: str) -> bool:
    """Return True when a column should be skipped by configured patterns."""
    return any(re.match(pattern, column_name) for pattern in context.ignore_patterns)


def _column_comment(column: BaseColumn) -> str | None:
    """Return adapter-specific column comments."""
    return getattr(column, "description", None) or getattr(column, "comment", None)


def _column_metadata_for_output(
    context: t.Any,
    result_node: ResultNode | None,
    column: BaseColumn | ColumnMetadata,
    normalized_name: str,
    index: int,
) -> ColumnMetadata:
    """Return manifest-compatible metadata for a warehouse or catalog column."""
    if isinstance(column, ColumnMetadata):
        return column

    return ColumnMetadata(
        name=normalized_name,
        type=_maybe_use_precise_dtype(column, context.settings, result_node, context=context),
        index=index,
        comment=_column_comment(column),
    )


def _add_processed_columns(
    context: t.Any,
    result_node: ResultNode | None,
    normalized_columns: OrderedDict[str, ColumnMetadata],
    raw_column: BaseColumn | ColumnMetadata,
    index: int,
) -> int:
    """Normalize and add one raw column plus flattened children."""
    credentials_type = context.project.runtime_cfg.credentials.type
    for column in _iter_flattened_columns(raw_column):
        if _column_matches_ignore_pattern(context, column.name):
            logger.debug(
                ":no_entry_sign: Skipping column => %s due to skip pattern match.",
                column.name,
            )
            continue
        normalized_name = normalize_column_name(column.name, credentials_type)
        normalized_columns[normalized_name] = _column_metadata_for_output(
            context,
            result_node,
            column,
            normalized_name,
            index,
        )
        index += 1
    return index


def _processed_columns(
    context: t.Any,
    result_node: ResultNode | None,
    columns: t.Iterable[BaseColumn | ColumnMetadata],
    *,
    catalog: bool = False,
) -> OrderedDict[str, ColumnMetadata]:
    """Convert raw warehouse or catalog columns into normalized metadata."""
    normalized_columns: OrderedDict[str, ColumnMetadata] = OrderedDict()
    index = 0
    for column in columns:
        if catalog:
            # Older Core catalog classes need conversion but already carry ordinals.
            index = t.cast("ColumnMetadata", column).index
        index = _add_processed_columns(context, result_node, normalized_columns, column, index)
    return normalized_columns


def _catalog_entry_matches(matcher: t.Any, entry: t.Any) -> bool:
    """Return True when a catalog entry matches the active relation."""
    if not callable(matcher):
        return False
    try:
        return bool(matcher(*entry.key()))
    except ApproximateMatchError:
        # For Snowflake and other case-insensitive databases, an approximate
        # match (case difference) IS the same relation, so treat as match.
        return True


def _postgres_quoted_catalog_components_match(relation: BaseRelation, entry: t.Any) -> bool:
    """Preserve PostgreSQL quoted case before catalog keys erase it."""
    metadata = entry.metadata
    return (
        (
            metadata.database is None
            or not relation.quote_policy.database
            or relation.database == metadata.database
        )
        and (not relation.quote_policy.schema or relation.schema == metadata.schema)
        and (not relation.quote_policy.identifier or relation.identifier == metadata.name)
    )


def _postgres_unquoted_catalog_component(
    value: str | None, *, quoted: bool, included: bool
) -> str | None:
    """Resolve only evidenced ASCII SQL components, not connection database names."""
    if value is not None and value.isascii() and included and not quoted:
        return value.lower()
    return None


def _catalog_columns_for_relation(
    context: t.Any,
    relation: BaseRelation,
    rendered_relation: str,
) -> tuple[ColumnMetadata, ...] | None:
    """Return catalog columns for a relation when a matching catalog entry exists."""
    catalog = context.read_catalog()
    if not catalog:
        return None

    logger.debug(":blue_book: Catalog found => Checking for ref => %s", rendered_relation)
    is_postgres = context.project.runtime_cfg.credentials.type == "postgres"
    unquoted_schema = unquoted_identifier = None
    if is_postgres:
        unquoted_schema = _postgres_unquoted_catalog_component(
            relation.schema,
            quoted=relation.quote_policy.schema,
            included=relation.include_policy.schema,
        )
        unquoted_identifier = _postgres_unquoted_catalog_component(
            relation.identifier,
            quoted=relation.quote_policy.identifier,
            included=relation.include_policy.identifier,
        )
    # Compare raw catalog identity with the requested SQL identity before the
    # folded fallback. PostgreSQL unquoted ASCII tokens name lowercase objects.
    relation_identity = (
        relation.database,
        relation.schema if unquoted_schema is None else unquoted_schema,
        relation.identifier if unquoted_identifier is None else unquoted_identifier,
    )
    catalog_entry = _find_first(
        chain(catalog.nodes.values(), catalog.sources.values()),
        lambda entry: (
            (
                entry.metadata.database,
                entry.metadata.schema,
                entry.metadata.name,
            )
            == relation_identity
        ),
    )
    if catalog_entry is None:
        matcher = getattr(t.cast(t.Any, relation), "matches", None)
        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()),
            lambda entry: (
                (not is_postgres or _postgres_quoted_catalog_components_match(relation, entry))
                and (unquoted_schema is None or entry.metadata.schema == unquoted_schema)
                and (unquoted_identifier is None or entry.metadata.name == unquoted_identifier)
                and _catalog_entry_matches(matcher, entry)
            ),
        )
    if not catalog_entry:
        return None

    logger.info(
        ":books: Found catalog entry for => %s. Using it to process columns.",
        rendered_relation,
    )
    return tuple(catalog_entry.columns.values())


def _cached_warehouse_columns(
    context: t.Any,
    relation: BaseRelation,
    rendered_relation: str,
) -> tuple[BaseColumn, ...]:
    """Return cached warehouse columns or introspect and cache them."""
    cache_key = _build_column_cache_key(context, rendered_relation)
    with _COLUMN_LIST_CACHE_LOCK:
        cached_columns = _COLUMN_LIST_CACHE.get(cache_key)

    if cached_columns is not None:
        logger.debug(":blue_book: Column list cache HIT => %s", rendered_relation)
        return cached_columns

    try:
        logger.info(":mag: Introspecting columns in warehouse for => %s", rendered_relation)
        warehouse_columns = tuple(
            t.cast(
                "t.Iterable[BaseColumn]",
                context.project.adapter.get_columns_in_relation(relation),
            ),
        )
    except Exception as ex:  # noqa: BLE001
        logger.warning(":warning: Could not introspect columns for %s: %s", rendered_relation, ex)
        return ()

    with _COLUMN_LIST_CACHE_LOCK:
        _COLUMN_LIST_CACHE[cache_key] = warehouse_columns
    return warehouse_columns


def get_columns(
    context: t.Any,
    relation: BaseRelation | ResultNode | None,
) -> dict[str, ColumnMetadata]:
    """Collect column metadata from database or catalog.

    Thread-safety: This function is thread-safe. It uses _COLUMN_LIST_CACHE_LOCK
    to synchronize access to the shared _COLUMN_LIST_CACHE. Multiple threads can
    safely call this function concurrently.

    Returns:
        OrderedDict mapping normalized column names to ColumnMetadata.

    """
    if relation is None:
        logger.debug(":blue_book: Relation is empty, skipping column collection.")
        return OrderedDict()

    relation, result_node = _relation_for_column_collection(context, relation)
    rendered_relation = _rendered_relation_name(relation)
    logger.info(":mag_right: Collecting columns for table => %s", rendered_relation)

    catalog_columns = _catalog_columns_for_relation(context, relation, rendered_relation)
    if catalog_columns is not None:
        return _processed_columns(context, result_node, catalog_columns, catalog=True)

    if context.project.config.disable_introspection:
        logger.warning(
            ":warning: Introspection is disabled, cannot introspect columns and no catalog entry.",
        )
        return OrderedDict()

    warehouse_columns = _cached_warehouse_columns(context, relation, rendered_relation)
    return _processed_columns(context, result_node, warehouse_columns)


def _manifest_column_property(column: t.Any, property_key: str) -> t.Any | None:
    """Return a column property from manifest metadata."""
    if property_key == "description":
        return getattr(column, "description", None)
    if property_key == "data_type":
        return getattr(column, "data_type", None)
    if property_key == "tags":
        return _get_effective_column_tags(column)
    if property_key == "meta":
        return _get_effective_column_meta(column)
    if property_key == "name":
        return getattr(column, "name", None)
    return getattr(column, property_key, None)


def _manifest_node_property(node: ResultNode, property_key: str) -> t.Any | None:
    """Return a node property from manifest metadata."""
    if property_key == "description":
        return getattr(node, "description", None)
    if property_key == "tags":
        return getattr(node, "tags", None)
    if property_key == "meta":
        return getattr(node, "meta", None)
    if property_key == "name":
        return getattr(node, "name", None)
    return getattr(node, property_key, None)


def _node_log_id(node: ResultNode) -> str:
    """Return a stable node identifier for property-access log messages."""
    return t.cast("str", getattr(node, "unique_id", "unknown"))


def _node_has_yaml_patch(node: ResultNode) -> bool:
    """Return True when a node has a patch path suitable for YAML access."""
    return hasattr(node, "patch_path") and node.patch_path is not None


def _yaml_column_property(column: t.Mapping[str, t.Any], property_key: str) -> t.Any | None:
    """Return a column property from raw YAML column content."""
    if property_key == "tags":
        config = column.get("config")
        if "tags" not in column and not (isinstance(config, t.Mapping) and "tags" in config):
            return None
        return _get_effective_column_tags(column)
    if property_key == "meta":
        config = column.get("config")
        if "meta" not in column and not (isinstance(config, t.Mapping) and "meta" in config):
            return None
        return _get_effective_column_meta(column)
    return column.get(property_key)


def _yaml_column_value(
    yaml_content: t.Mapping[str, t.Any],
    property_key: str,
    column_name: str,
) -> t.Any | None:
    """Return a property for a named YAML column."""
    columns = yaml_content.get("columns", [])
    for column in columns:
        if isinstance(column, t.Mapping) and column.get("name") == column_name:
            return _yaml_column_property(column, property_key)
    return None


# =============================================================================
# PropertyAccessor: Unified Model Property Access
# =============================================================================


class PropertyAccessor:
    """Unified interface for accessing model properties from multiple sources.

    The PropertyAccessor provides a single interface for accessing model properties
    (descriptions, tags, meta, data types) from either:
    - Manifest: Rendered jinja values (pre-compiled by dbt)
    - YAML: Unrendered jinja templates (raw {{ doc(...) }} syntax)
    - Auto: Automatically selects based on unrendered jinja detection

    ``PropertySource.DATABASE`` is reserved for future warehouse metadata
    introspection and raises ``NotImplementedError`` when requested.

    This enables the unrendered jinja feature (doc blocks) by allowing users to
    choose between rendered and unrendered property values.

    Example:
        >>> accessor = PropertyAccessor(context)
        >>> # Get rendered description from manifest
        >>> desc = accessor.get_description(node, source="manifest")
        >>> # Get unrendered description from YAML (preserves {{ doc(...) }})
        >>> desc = accessor.get_description(node, source="yaml")
        >>> # Auto-detect based on jinja presence
        >>> desc = accessor.get_description(node, source="auto")

    """

    def __init__(self, context: t.Any) -> None:
        self._context = context

    def _get_from_manifest(
        self,
        node: ResultNode,
        property_key: str,
        column_name: str | None = None,
    ) -> t.Any | None:
        """Get a property value from the manifest (rendered jinja).

        The manifest contains pre-rendered values where jinja templates
        like {{ doc('foo') }} have already been resolved.

        Args:
            node: The dbt node (model, source, seed, etc.)
            property_key: The property to retrieve (e.g., "description", "tags", "meta")
            column_name: Optional column name for column-level properties

        Returns:
            The property value from manifest, or None if not found

        """
        if column_name:
            column = node.columns.get(column_name)
            if column is None:
                return None
            return _manifest_column_property(column, property_key)

        return _manifest_node_property(node, property_key)

    def _get_from_yaml(
        self,
        node: ResultNode,
        property_key: str,
        column_name: str | None = None,
    ) -> t.Any | None:
        """Get a property value from YAML files (unrendered jinja).

        YAML files contain raw jinja templates like {{ doc('foo') }} that
        haven't been rendered yet. This is useful for preserving doc blocks.

        Args:
            node: The dbt node (model, source, seed, etc.)
            property_key: The property to retrieve (e.g., "description", "tags", "meta")
            column_name: Optional column name for column-level properties

        Returns:
            The property value from YAML, or None if not found

        """
        from dbt_osmosis.core.node_yaml import _get_node_yaml

        if not _node_has_yaml_patch(node):
            logger.debug(
                ":page_facing_up: Node %s has no patch_path, skipping YAML access",
                _node_log_id(node),
            )
            return None

        try:
            yaml_content = _get_node_yaml(self._context, node)
            if yaml_content is None:
                logger.debug(
                    ":page_facing_up: No YAML content found for node %s",
                    _node_log_id(node),
                )
                return None

            if column_name:
                return _yaml_column_value(yaml_content, property_key, column_name)

            return yaml_content.get(property_key)

        except FileNotFoundError:
            logger.warning(
                ":warning: YAML file not found for node %s, falling back to manifest",
                _node_log_id(node),
            )
            return None
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                ":warning: Error reading YAML for node %s: %s",
                _node_log_id(node),
                ex,
            )
            return None

    def _has_unrendered_jinja(self, value: t.Any) -> bool:
        """Check if a value contains unrendered jinja templates.

        Detects common jinja patterns used in dbt:
        - {{ doc('block_name') }} for doc blocks
        - {% docs block_name %}...{% enddocs %} for doc blocks
        - {{ var('variable_name') }} for variables
        - {{ env_var('ENV_VAR') }} for environment variables
        - {{ ... }} for generic jinja expressions
        - {% ... %} for generic jinja statements

        Handles nested structures (lists, dicts) by recursively checking values.

        Args:
            value: The value to check (string, list, dict, etc.)

        Returns:
            True if unrendered jinja is detected, False otherwise

        """
        # Handle lists (e.g., policy_tags)
        if isinstance(value, list):
            return any(self._has_unrendered_jinja(item) for item in value)

        # Handle dicts (e.g., meta fields)
        if isinstance(value, dict):
            return any(self._has_unrendered_jinja(v) for v in value.values())

        if not isinstance(value, str):
            return False

        # Check for common unrendered jinja patterns
        patterns = [
            "{{ doc(",  # Doc block function
            "{% docs ",  # Doc block start tag
            "{% enddocs %}",  # Doc block end tag
            "{{ var(",  # Variable substitution
            "{{ env_var(",  # Environment variable substitution
            "{{ ",  # Generic jinja expression start
            "{% ",  # Generic jinja statement start
        ]

        return any(pattern in value for pattern in patterns)

    def get(
        self,
        property_key: str,
        node: ResultNode,
        *,
        column_name: str | None = None,
        source: PropertySource | str = PropertySource.MANIFEST,
    ) -> t.Any | None:
        """Get a property value from the specified source.

        Args:
            property_key: The property to retrieve (e.g., "description", "tags", "meta")
            node: The dbt node (model, source, seed, etc.)
            column_name: Optional column name for column-level properties
            source: The source to read from ("manifest", "yaml", or "auto").
                "database" is reserved and currently unsupported.

        Returns:
            The property value, or None if not found

        Raises:
            ValueError: If an invalid source is specified
            NotImplementedError: If the unsupported database source is requested

        """
        # Handle "auto" as a special case before enum conversion
        if isinstance(source, str) and source == "auto":
            # Auto-detect: prefer YAML if it has unrendered jinja
            yaml_value = self._get_from_yaml(node, property_key, column_name)
            if yaml_value is not None and self._has_unrendered_jinja(yaml_value):
                logger.debug(
                    ":magic_wand: Detected unrendered jinja in YAML for %s, using YAML source",
                    getattr(node, "unique_id", "unknown"),
                )
                return yaml_value
            # Fall back to manifest
            return self._get_from_manifest(node, property_key, column_name)

        # Normalize source to enum
        if isinstance(source, str):
            try:
                source = PropertySource(source)
            except ValueError:
                raise ValueError(
                    f"Invalid source '{source}'. Must be one of: "
                    f"'auto', {', '.join([s.value for s in PropertySource])}",
                )

        if source == PropertySource.MANIFEST:
            return self._get_from_manifest(node, property_key, column_name)

        if source == PropertySource.YAML:
            yaml_value = self._get_from_yaml(node, property_key, column_name)
            # Fall back to manifest if YAML doesn't have the value
            if yaml_value is None:
                logger.debug(
                    ":page_facing_up: Property '%s' not in YAML for %s, falling back to manifest",
                    property_key,
                    getattr(node, "unique_id", "unknown"),
                )
                return self._get_from_manifest(node, property_key, column_name)
            return yaml_value

        if source == PropertySource.DATABASE:
            raise NotImplementedError(
                "database property source is not implemented for PropertyAccessor; "
                + "use source='manifest', source='yaml', or source='auto' instead",
            )

        # This shouldn't happen with enum validation, but just in case
        raise ValueError(
            f"Invalid source '{source}'. Must be one of: "
            f"'auto', {', '.join([s.value for s in PropertySource])}",
        )

    def get_description(
        self,
        node: ResultNode,
        *,
        column_name: str | None = None,
        source: PropertySource | str = PropertySource.MANIFEST,
    ) -> str | None:
        """Get the description for a node or column.

        Convenience method for getting descriptions.

        Args:
            node: The dbt node (model, source, seed, etc.)
            column_name: Optional column name for column-level descriptions
            source: The source to read from ("manifest", "yaml", or "auto").
                "database" is reserved and currently unsupported.

        Returns:
            The description string, or None if not found

        """
        return t.cast(
            "str | None",
            self.get("description", node, column_name=column_name, source=source),
        )

    def get_meta(
        self,
        node: ResultNode,
        *,
        column_name: str | None = None,
        source: PropertySource | str = PropertySource.MANIFEST,
        meta_key: str | None = None,
    ) -> t.Any | None:
        """Get the meta dictionary for a node or column.

        Convenience method for getting metadata.

        Args:
            node: The dbt node (model, source, seed, etc.)
            column_name: Optional column name for column-level meta
            source: The source to read from ("manifest", "yaml", or "auto")
            meta_key: Optional specific key within the meta dictionary

        Returns:
            The meta dictionary if meta_key is None, or the specific meta value
            if meta_key is specified. Returns None if not found.

        """
        meta = self.get("meta", node, column_name=column_name, source=source)
        if meta is None:
            return None
        if meta_key is not None:
            return meta.get(meta_key) if isinstance(meta, dict) else None
        return meta

    def has_property(
        self,
        property_key: str,
        node: ResultNode,
        *,
        column_name: str | None = None,
    ) -> bool:
        """Check if a property exists in either manifest or YAML.

        Args:
            property_key: The property to check for
            node: The dbt node (model, source, seed, etc.)
            column_name: Optional column name for column-level properties

        Returns:
            True if the property exists in manifest or YAML, False otherwise

        """
        manifest_value = self._get_from_manifest(node, property_key, column_name)
        if manifest_value is not None:
            return True

        yaml_value = self._get_from_yaml(node, property_key, column_name)
        return yaml_value is not None
