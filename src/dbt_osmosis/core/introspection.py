# pyright: reportPrivateImportUsage=false, reportOptionalMemberAccess=false, reportUnknownMemberType=false
from __future__ import annotations

import json
import re
import threading
import typing as t
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from pathlib import Path

# pyright: reportPrivateImportUsage=false
from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.relation import BaseRelation
from dbt.adapters.exceptions.compilation import ApproximateMatchError
from dbt.artifacts.schemas.catalog import CatalogArtifact, CatalogResults  # pyright: ignore[reportPrivateImportUsage]
from dbt.contracts.graph.nodes import ResultNode  # pyright: ignore[reportPrivateImportUsage]
from dbt_common.contracts.metadata import ColumnMetadata  # pyright: ignore[reportPrivateImportUsage]
from dbt.task.docs.generate import Catalog

from dbt_osmosis.core import logger

__all__ = [
    "_find_first",
    "normalize_column_name",
    "_maybe_use_precise_dtype",
    "get_columns",
    "SettingsResolver",
    "resolve_setting",
    "_load_catalog",
    "_generate_catalog",
    "_COLUMN_LIST_CACHE",
    # Foundational classes for unified config resolution
    "ConfigurationError",
    "ConfigSourceName",
    "PropertySource",
    "ConfigurationSource",
    # Unified property access for US2
    "PropertyAccessor",
    # New configuration sources for US1
    "ConfigMetaSource",
    "UnrenderedConfigSource",
    "ProjectVarsSource",
    "SupplementaryFileSource",
    "_get_effective_column_meta",
    "_get_effective_column_tags",
]

T = t.TypeVar("T")
_MISSING = object()


class _CatalogArtifactProtocol(t.Protocol):
    nodes: t.Mapping[str, object]
    sources: t.Mapping[str, object]

    def write(self, path: str) -> None: ...


class _CatalogArtifactFactoryProtocol(t.Protocol):
    @staticmethod
    def from_dict(data: object) -> _CatalogArtifactProtocol: ...

    @staticmethod
    def from_results(
        *,
        nodes: object,
        sources: object,
        generated_at: datetime,
        compile_results: object,
        errors: list[str] | None,
    ) -> _CatalogArtifactProtocol: ...


def _catalog_artifact_factory() -> _CatalogArtifactFactoryProtocol:
    """Return a typed view over dbt's versioned catalog artifact factory.

    dbt's shipped type surface varies across supported core lines even though the
    runtime object exposes the methods dbt-osmosis needs. Centralize the cast so
    compatibility shims stay local to catalog loading/generation.
    """
    return t.cast("_CatalogArtifactFactoryProtocol", t.cast("object", CatalogArtifact))


def _as_catalog_results(artifact: _CatalogArtifactProtocol) -> CatalogResults:
    """Normalize a versioned catalog artifact to the concrete CatalogResults alias."""
    return t.cast("CatalogResults", t.cast("object", artifact))


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
        """Initialize a ConfigurationError.

        Args:
            message: The error message describing what went wrong.
            file_path: Optional path to the configuration file that caused the error.

        """
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
        DATABASE: Warehouse metadata via introspection (future use)

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
    except Exception:
        return _MISSING

    if not _same_setting_value(current_value, default_value):
        return current_value
    return _MISSING


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
        """Initialize a ConfigurationSource.

        Args:
            name: The ConfigSourceName enum value for this source.

        """
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
    """Configuration source for node.config.meta (dbt 1.10+).

    This source reads configuration from the config.meta dictionary,
    which is available in dbt 1.10 and later versions. It gracefully
    handles versions where this field doesn't exist.

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
        """Initialize ConfigMetaSource.

        Args:
            node: The dbt node to read config.meta from.

        """
        super().__init__(ConfigSourceName.CONFIG_META)
        self._node = node

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from config.meta.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        # Gracefully handle dbt versions < 1.10 where config.meta doesn't exist
        if not hasattr(self._node, "config") or not hasattr(self._node.config, "meta"):
            return None

        config_meta = getattr(self._node.config, "meta", None)
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
        """Initialize UnrenderedConfigSource.

        Args:
            node: The dbt node to read unrendered_config from.

        """
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
        """Initialize ProjectVarsSource.

        Args:
            context: The dbt context with project.runtime_cfg.vars.

        """
        super().__init__(ConfigSourceName.PROJECT_VARS)
        self._context = context

    def get(self, key: str) -> t.Any | None:
        """Get a configuration value from project vars.

        Args:
            key: The configuration key to look up.

        Returns:
            The configuration value if found, None otherwise.

        """
        # Safely access runtime_cfg.vars
        if not hasattr(self._context, "project"):
            return None
        if not hasattr(self._context.project, "runtime_cfg"):
            return None
        if not hasattr(self._context.project.runtime_cfg, "vars"):
            return None

        # Get vars as dictionary
        vars_dict = self._context.project.runtime_cfg.vars
        if hasattr(vars_dict, "to_dict"):
            vars_dict = vars_dict.to_dict()
        if not isinstance(vars_dict, dict):
            return None

        # Normalize key
        kebab_key = key.replace("_", "-")
        snake_key = key.replace("-", "_")

        def mapping_value(source: t.Any) -> t.Any:
            if not isinstance(source, t.Mapping):
                return _MISSING

            for prefixed_name in (f"dbt-osmosis-{kebab_key}", f"dbt_osmosis_{snake_key}"):
                if prefixed_name in source:
                    return source[prefixed_name]

            if kebab_key in source:
                return source[kebab_key]
            if snake_key in source:
                return source[snake_key]

            for options_name in ("dbt-osmosis-options", "dbt_osmosis_options"):
                value = _get_options_value(source.get(options_name, {}), kebab_key, snake_key)
                if value is not _MISSING:
                    return value

            return _MISSING

        # Check dbt-osmosis top-level key
        dbt_osmosis_vars = vars_dict.get("dbt-osmosis", {})
        value = mapping_value(dbt_osmosis_vars)
        if value is not _MISSING:
            return value

        # Check dbt_osmosis top-level key (snake_case variant)
        dbt_osmosis_vars_snake = vars_dict.get("dbt_osmosis", {})
        value = mapping_value(dbt_osmosis_vars_snake)
        if value is not _MISSING:
            return value

        value = mapping_value(vars_dict)
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
        """Initialize SupplementaryFileSource.

        Args:
            context: The dbt context with project root path.

        """
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

        def dict_value(source: t.Any, *, direct_keys: bool) -> t.Any:
            if not isinstance(source, t.Mapping):
                return _MISSING

            kebab_name = setting_name.replace("_", "-")
            snake_name = setting_name.replace("-", "_")

            for prefixed_name in (f"dbt-osmosis-{kebab_name}", f"dbt_osmosis_{snake_name}"):
                if prefixed_name in source:
                    return source[prefixed_name]

            if direct_keys:
                if kebab_name in source:
                    return source[kebab_name]
                if snake_name in source:
                    return source[snake_name]

            options_kebab = source.get("dbt-osmosis-options", {})
            options_snake = source.get("dbt_osmosis_options", {})
            value = _get_options_value(options_kebab, kebab_name, snake_name)
            if value is not _MISSING:
                return value
            value = _get_options_value(options_snake, kebab_name, snake_name)
            if value is not _MISSING:
                return value
            return _MISSING

        def source_value(source: ConfigurationSource) -> t.Any:
            value = source.get(setting_name)
            if value is None:
                return _MISSING
            return value

        def explicit_context_setting_value() -> t.Any:
            current_value = _get_explicit_context_setting_value(active_context, setting_name)
            if current_value is _MISSING:
                return _MISSING
            if not _same_setting_value(current_value, fallback):
                return _MISSING
            return current_value

        if node is not None:
            node_sources: list[tuple[str, t.Any, bool]] = []

            # Column-level sources (if column specified) - HIGHEST precedence
            if column_name and (column := node.columns.get(column_name)):
                column_meta = _get_effective_column_meta(column)
                node_sources.append(("column_meta", column_meta, True))

            node_sources.extend([
                ("node_meta", getattr(node, "meta", {}), True),
                ("config_extra", getattr(getattr(node, "config", None), "extra", {}), False),
            ])

            # Check each source for the setting (in order - highest precedence first)
            for source_name, source, direct_keys in node_sources:
                value = dict_value(source, direct_keys=direct_keys)
                if value is not _MISSING:
                    logger.debug(
                        ":gear: Resolved setting '%s' from %s",
                        setting_name,
                        source_name,
                    )
                    return value

            # Check dbt 1.10+ sources AFTER existing sources (lower precedence)
            # Check config.meta (dbt 1.10+)
            if hasattr(node, "config") and hasattr(node.config, "meta"):
                config_meta_source = ConfigMetaSource(node)
                value = source_value(config_meta_source)
                if value is not _MISSING:
                    logger.debug(
                        ":gear: Resolved setting '%s' from config.meta (dbt 1.10+)",
                        setting_name,
                    )
                    return value

            # Check unrendered_config (dbt 1.10+)
            if hasattr(node, "unrendered_config"):
                unrendered_source = UnrenderedConfigSource(node)
                value = source_value(unrendered_source)
                if value is not _MISSING:
                    logger.debug(
                        ":gear: Resolved setting '%s' from unrendered_config (dbt 1.10+)",
                        setting_name,
                    )
                    return value

        explicit_context_value = explicit_context_setting_value()
        if explicit_context_value is not _MISSING:
            logger.debug(
                ":gear: Resolved setting '%s' from explicit context settings",
                setting_name,
            )
            return explicit_context_value

        if active_context is not None:
            for context_source in (
                SupplementaryFileSource(active_context),
                ProjectVarsSource(active_context),
            ):
                value = source_value(context_source)
                if value is not _MISSING:
                    logger.debug(
                        ":gear: Resolved setting '%s' from %s",
                        setting_name,
                        context_source.name.value,
                    )
                    return value

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
        chain = []

        kebab_name = setting_name.replace("_", "-")
        snake_name = setting_name.replace("-", "_")

        # Helper to extract value from a dict source
        def extract_value(source: t.Any, *, direct_keys: bool) -> t.Any | None:
            if not isinstance(source, t.Mapping):
                return None
            # Check prefixed variants
            for prefixed_name in (f"dbt-osmosis-{kebab_name}", f"dbt_osmosis_{snake_name}"):
                if prefixed_name in source:
                    return source[prefixed_name]
            # Check direct variants
            if direct_keys:
                if kebab_name in source:
                    return source[kebab_name]
                if snake_name in source:
                    return source[snake_name]
            for options_name in ("dbt-osmosis-options", "dbt_osmosis_options"):
                value = _get_options_value(source.get(options_name, {}), kebab_name, snake_name)
                if value is not _MISSING:
                    return value
            return None

        if node is not None:
            # Column-level sources
            if column_name and (column := node.columns.get(column_name)):
                value = extract_value(_get_effective_column_meta(column), direct_keys=True)
                chain.append((ConfigSourceName.COLUMN_META, value))

            # Node meta sources
            value = extract_value(node.meta, direct_keys=True)
            chain.append((ConfigSourceName.NODE_META, value))

            # Node config.extra
            value = extract_value(node.config.extra, direct_keys=False)
            chain.append((ConfigSourceName.CONFIG_EXTRA, value))

            # Config.meta (dbt 1.10+)
            if hasattr(node, "config") and hasattr(node.config, "meta"):
                config_meta_source = ConfigMetaSource(node)
                value = config_meta_source.get(setting_name)
                chain.append((ConfigSourceName.CONFIG_META, value))
            else:
                chain.append((ConfigSourceName.CONFIG_META, None))

            # Unrendered config (dbt 1.10+)
            if hasattr(node, "unrendered_config"):
                unrendered_source = UnrenderedConfigSource(node)
                value = unrendered_source.get(setting_name)
                chain.append((ConfigSourceName.UNRENDERED_CONFIG, value))
            else:
                chain.append((ConfigSourceName.UNRENDERED_CONFIG, None))

        if active_context is not None:
            context_settings_value = _get_explicit_context_setting_value(
                active_context,
                setting_name,
            )
            if context_settings_value is _MISSING:
                context_settings_value = None
            chain.append((ConfigSourceName.CONTEXT_SETTINGS, context_settings_value))

            supplementary_source = SupplementaryFileSource(active_context)
            supplementary_value = supplementary_source.get(setting_name)
            chain.append((ConfigSourceName.SUPPLEMENTARY_FILE, supplementary_value))
            project_vars_source = ProjectVarsSource(active_context)
            project_vars_value = project_vars_source.get(setting_name)
            chain.append((ConfigSourceName.PROJECT_VARS, project_vars_value))

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

        # Keys to check (both kebab and snake variants)
        keys = ("dbt-osmosis", "dbt_osmosis")

        # Helper to check a dict for the path template
        def check_dict(source: dict[str, t.Any]) -> str | None:
            for key in keys:
                if key in source:
                    return source[key]
            return None

        # Check config.extra first (highest priority)
        if hasattr(node, "config") and hasattr(node.config, "extra"):
            result = check_dict(node.config.extra)
            if result:
                logger.debug(
                    ":gear: Found YAML path template in config.extra: %s",
                    result,
                )
                return result

        # Check config.meta (dbt 1.10+)
        if hasattr(node, "config") and hasattr(node.config, "meta"):
            config_meta = getattr(node.config, "meta", None)
            if isinstance(config_meta, dict):
                result = check_dict(config_meta)
                if result:
                    logger.debug(
                        ":gear: Found YAML path template in config.meta: %s",
                        result,
                    )
                    return result

        # Check node.meta (for YAML-level meta definitions)
        if hasattr(node, "meta"):
            node_meta = node.meta
            if isinstance(node_meta, dict):
                result = check_dict(node_meta)
                if result:
                    logger.debug(
                        ":gear: Found YAML path template in node.meta: %s",
                        result,
                    )
                    return result

        # Check unrendered_config (dbt 1.10+)
        if hasattr(node, "unrendered_config"):
            unrendered = node.unrendered_config
            if isinstance(unrendered, dict):
                result = check_dict(unrendered)
                if result:
                    logger.debug(
                        ":gear: Found YAML path template in unrendered_config: %s",
                        result,
                    )
                    return result

        logger.debug(":gear: No YAML path template found in node config")
        return None


_SETTINGS_RESOLVER = SettingsResolver()


@t.overload
def _find_first(coll: t.Iterable[T], predicate: t.Callable[[T], bool], default: T) -> T: ...


@t.overload
def _find_first(
    coll: t.Iterable[T],
    predicate: t.Callable[[T], bool],
    default: None = ...,
) -> T | None: ...


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
    normalized_columns: OrderedDict[str, ColumnMetadata] = OrderedDict()

    if relation is None:
        logger.debug(":blue_book: Relation is empty, skipping column collection.")
        return normalized_columns

    result_node: ResultNode | None = None
    if not isinstance(relation, BaseRelation):
        # NOTE: Technically, we should use `isinstance(relation, ResultNode)` to verify it's a ResultNode,
        #       but since ResultNode is defined as a Union[...], Python 3.9 raises
        #       > TypeError: Subscripted generics cannot be used with class and instance checks
        #       To avoid that, we're skipping the isinstance check.
        result_node = relation  # may be a ResultNode
        relation = context.project.adapter.Relation.create_from(
            context.project.adapter.config,  # pyright: ignore[reportUnknownArgumentType]
            relation,  # pyright: ignore[reportArgumentType]
        )

    relation_any = t.cast(t.Any, relation)
    if relation:
        renderer = getattr(relation_any, "render", None)
        rendered_relation = t.cast("str", renderer()) if callable(renderer) else str(relation)
    else:
        rendered_relation = ""

    logger.info(":mag_right: Collecting columns for table => %s", rendered_relation)
    index = 0

    def process_column(c: BaseColumn | ColumnMetadata, /) -> None:
        nonlocal index

        columns = [c]
        flattener = getattr(t.cast(t.Any, c), "flatten", None)
        if callable(flattener):
            for flattened in t.cast(t.Iterable[t.Any], flattener()):
                columns.append(flattened)

        for column in columns:
            if any(re.match(b, column.name) for b in context.ignore_patterns):
                logger.debug(
                    ":no_entry_sign: Skipping column => %s due to skip pattern match.",
                    column.name,
                )
                continue
            normalized = normalize_column_name(
                column.name,
                context.project.runtime_cfg.credentials.type,
            )
            if not isinstance(column, ColumnMetadata):
                dtype = _maybe_use_precise_dtype(
                    column,
                    context.settings,
                    result_node,
                    context=context,
                )
                # BigQuery uses "description" attribute, other adapters use "comment"
                col_comment = getattr(column, "description", None) or getattr(
                    column,
                    "comment",
                    None,
                )
                column = ColumnMetadata(
                    name=normalized,
                    type=dtype,
                    index=index,
                    comment=col_comment,
                )
            normalized_columns[normalized] = column
            index += 1

    if catalog := context.read_catalog():
        logger.debug(":blue_book: Catalog found => Checking for ref => %s", rendered_relation)
        matcher = getattr(relation_any, "matches", None)

        def matches_relation(entry: t.Any) -> bool:
            if not callable(matcher):
                return False
            try:
                return bool(matcher(*entry.key()))
            except ApproximateMatchError:
                # For Snowflake and other case-insensitive databases, an approximate
                # match (case difference) IS the same relation, so treat as match
                return True

        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()),
            matches_relation,
        )
        if catalog_entry:
            logger.info(
                ":books: Found catalog entry for => %s. Using it to process columns.",
                rendered_relation,
            )
            for column in catalog_entry.columns.values():
                process_column(column)
            return normalized_columns

    if context.project.config.disable_introspection:
        logger.warning(
            ":warning: Introspection is disabled, cannot introspect columns and no catalog entry.",
        )
        return normalized_columns

    cache_key = _build_column_cache_key(context, rendered_relation)
    with _COLUMN_LIST_CACHE_LOCK:
        cached_columns = _COLUMN_LIST_CACHE.get(cache_key)

    if cached_columns is not None:
        logger.debug(":blue_book: Column list cache HIT => %s", rendered_relation)
        for column in cached_columns:
            process_column(column)
        return normalized_columns

    try:
        logger.info(":mag: Introspecting columns in warehouse for => %s", rendered_relation)
        warehouse_columns = tuple(
            t.cast(
                "t.Iterable[BaseColumn]",
                context.project.adapter.get_columns_in_relation(relation),
            ),
        )
    except Exception as ex:
        logger.warning(":warning: Could not introspect columns for %s: %s", rendered_relation, ex)
        return normalized_columns

    with _COLUMN_LIST_CACHE_LOCK:
        _COLUMN_LIST_CACHE[cache_key] = warehouse_columns

    for column in warehouse_columns:
        process_column(column)

    return normalized_columns


def _load_catalog(settings: t.Any) -> CatalogResults | None:
    """Load the catalog file if it exists and return a CatalogResults instance."""
    logger.debug(":mag: Attempting to load catalog from => %s", settings.catalog_path)
    if not settings.catalog_path:
        return None
    fp = Path(settings.catalog_path)
    if not fp.exists():
        logger.warning(":warning: Catalog path => %s does not exist.", fp)
        return None
    logger.info(":books: Loading existing catalog => %s", fp)
    return _as_catalog_results(_catalog_artifact_factory().from_dict(json.loads(fp.read_text())))


# NOTE: this is mostly adapted from dbt-core with some cruft removed, strict pyright is not a fan of dbt's shenanigans
def _generate_catalog(context: t.Any) -> CatalogResults | None:
    """Generate dbt catalog file for the project."""
    import dbt.utils as dbt_utils  # pyright: ignore[reportPrivateImportUsage]

    if context.config.disable_introspection:
        logger.warning(":warning: Introspection is disabled, cannot generate catalog.")
        return None
    logger.info(
        ":books: Generating a new catalog for the project => %s",
        context.runtime_cfg.project_name,
    )
    catalogable_nodes = chain(
        [
            t.cast("t.Any", node)  # pyright: ignore[reportInvalidCast]
            for node in context.manifest.nodes.values()
            if node.is_relational and not node.is_ephemeral_model
        ],
        [t.cast("t.Any", node) for node in context.manifest.sources.values()],  # pyright: ignore[reportInvalidCast]
    )
    table, exceptions = context.adapter.get_filtered_catalog(
        catalogable_nodes,
        context.manifest.get_used_schemas(),  # pyright: ignore[reportArgumentType]
    )

    logger.debug(":mag_right: Building catalog from returned table => %s", table)
    catalog = Catalog(
        [dict(zip(table.column_names, map(dbt_utils._coerce_decimal, row))) for row in table],  # pyright: ignore[reportUnknownArgumentType,reportPrivateUsage]
    )

    errors: list[str] | None = None
    if exceptions:
        errors = [str(e) for e in exceptions]
        logger.warning(":warning: Exceptions encountered in get_filtered_catalog => %s", errors)

    nodes, sources = catalog.make_unique_id_map(context.manifest)
    artifact = _catalog_artifact_factory().from_results(
        nodes=nodes,
        sources=sources,
        generated_at=datetime.now(timezone.utc),
        compile_results=None,
        errors=errors,
    )
    artifact_path = Path(context.runtime_cfg.project_target_path, "catalog.json")
    logger.info(":bookmark_tabs: Writing fresh catalog => %s", artifact_path)
    artifact.write(str(artifact_path.resolve()))  # Cache it, same as dbt
    return _as_catalog_results(artifact)


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
        """Initialize the PropertyAccessor.

        Args:
            context: YamlRefactorContext containing project, manifest, yaml_handler, etc.

        """
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
        # Handle column-level properties
        if column_name:
            column = node.columns.get(column_name)
            if column is None:
                return None
            # Map property keys to column attributes
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
            # Try generic attribute access
            return getattr(column, property_key, None)

        # Handle node-level properties
        if property_key == "description":
            return getattr(node, "description", None)
        if property_key == "tags":
            return getattr(node, "tags", None)
        if property_key == "meta":
            return getattr(node, "meta", None)
        if property_key == "name":
            return getattr(node, "name", None)
        # Try generic attribute access
        return getattr(node, property_key, None)

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
        from dbt_osmosis.core.inheritance import _get_node_yaml

        # Check if node has a YAML file (ephemeral models may not)
        if not hasattr(node, "patch_path") or node.patch_path is None:
            logger.debug(
                ":page_facing_up: Node %s has no patch_path, skipping YAML access",
                getattr(node, "unique_id", "unknown"),
            )
            return None

        try:
            # Get the YAML content for this node
            yaml_content = _get_node_yaml(self._context, node)
            if yaml_content is None:
                logger.debug(
                    ":page_facing_up: No YAML content found for node %s",
                    getattr(node, "unique_id", "unknown"),
                )
                return None

            # Handle column-level properties
            if column_name:
                columns = yaml_content.get("columns", [])
                for column in columns:
                    if not isinstance(column, t.Mapping):
                        continue
                    if column.get("name") == column_name:
                        if property_key == "tags":
                            config = column.get("config")
                            if "tags" not in column and not (
                                isinstance(config, t.Mapping) and "tags" in config
                            ):
                                return None
                            return _get_effective_column_tags(column)
                        if property_key == "meta":
                            config = column.get("config")
                            if "meta" not in column and not (
                                isinstance(config, t.Mapping) and "meta" in config
                            ):
                                return None
                            return _get_effective_column_meta(column)
                        return column.get(property_key)
                return None

            # Handle node-level properties
            return yaml_content.get(property_key)

        except FileNotFoundError:
            logger.warning(
                ":warning: YAML file not found for node %s, falling back to manifest",
                getattr(node, "unique_id", "unknown"),
            )
            return None
        except Exception as ex:
            logger.warning(
                ":warning: Error reading YAML for node %s: %s",
                getattr(node, "unique_id", "unknown"),
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
            source: The source to read from ("manifest", "yaml", or "auto")

        Returns:
            The property value, or None if not found

        Raises:
            ValueError: If an invalid source is specified

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
            # Database introspection not yet implemented for PropertyAccessor
            logger.debug(
                ":mag: Database source not yet implemented for PropertyAccessor, "
                "falling back to manifest",
            )
            return self._get_from_manifest(node, property_key, column_name)

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
            source: The source to read from ("manifest", "yaml", or "auto")

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
