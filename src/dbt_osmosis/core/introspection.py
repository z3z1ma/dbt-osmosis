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

from dbt.adapters.base.column import Column as BaseColumn
from dbt.adapters.base.relation import BaseRelation
from dbt.contracts.graph.nodes import ResultNode
from dbt.contracts.results import CatalogArtifact, CatalogResults, ColumnMetadata
from dbt.task.docs.generate import Catalog

from dbt_osmosis.core import logger

__all__ = [
    "_find_first",
    "normalize_column_name",
    "_maybe_use_precise_dtype",
    "get_columns",
    "SettingsResolver",
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
]

T = t.TypeVar("T")

_COLUMN_LIST_CACHE: dict[str, OrderedDict[str, ColumnMetadata]] = {}


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
        PROJECT_VARS: Project-level vars from dbt_project.yml
        SUPPLEMENTARY_FILE: Supplementary dbt-osmosis.yml file
        FALLBACK: Default fallback value (lowest priority)
    """

    COLUMN_META = "column_meta"
    NODE_META = "node_meta"
    CONFIG_EXTRA = "config_extra"
    CONFIG_META = "config_meta"
    UNRENDERED_CONFIG = "unrendered_config"
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

        config_meta = self._node.config.meta
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

        if isinstance(options_kebab, dict) and kebab_key in options_kebab:
            return options_kebab[kebab_key]
        if isinstance(options_snake, dict) and kebab_key in options_snake:
            return options_snake[kebab_key]

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

        if isinstance(options_kebab, dict) and kebab_key in options_kebab:
            return options_kebab[kebab_key]
        if isinstance(options_snake, dict) and kebab_key in options_snake:
            return options_snake[kebab_key]

        return None


class ProjectVarsSource(ConfigurationSource):
    """Configuration source for project-level vars in dbt_project.yml.

    This source reads configuration from the project's runtime_cfg.vars,
    which contains variables defined in dbt_project.yml under the vars: section.

    Supported key variants:
    - dbt-osmosis.<key> (under dbt-osmosis top-level key)
    - dbt_osmosis.<key> (under dbt_osmosis top-level key)
    - <key> (direct key, if at top level of vars)

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

        # Check dbt-osmosis top-level key
        dbt_osmosis_vars = vars_dict.get("dbt-osmosis", {})
        if isinstance(dbt_osmosis_vars, dict):
            # Check both kebab and snake variants
            if kebab_key in dbt_osmosis_vars:
                return dbt_osmosis_vars[kebab_key]
            if snake_key in dbt_osmosis_vars:
                return dbt_osmosis_vars[snake_key]

        # Check dbt_osmosis top-level key (snake_case variant)
        dbt_osmosis_vars_snake = vars_dict.get("dbt_osmosis", {})
        if isinstance(dbt_osmosis_vars_snake, dict):
            if kebab_key in dbt_osmosis_vars_snake:
                return dbt_osmosis_vars_snake[kebab_key]
            if snake_key in dbt_osmosis_vars_snake:
                return dbt_osmosis_vars_snake[snake_key]

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

        project_root = Path(self._context.project.runtime_cfg.project_root)
        config_file = project_root / "dbt-osmosis.yml"

        # Check if file exists first
        if not config_file.is_file():
            return {}

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

        if isinstance(options_kebab, dict) and kebab_key in options_kebab:
            return options_kebab[kebab_key]
        if isinstance(options_snake, dict) and kebab_key in options_snake:
            return options_snake[kebab_key]

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

    3. Fallback value
    """

    def resolve(
        self,
        setting_name: str,
        node: ResultNode | None = None,
        column_name: str | None = None,
        *,
        fallback: t.Any | None = None,
    ) -> t.Any:
        """Resolve a setting value from the configured sources.

        Args:
            setting_name: The name of the setting to resolve (supports both kebab-case and snake_case)
            node: The dbt node to resolve settings for
            column_name: Optional column name to check column-level settings
            fallback: Default value if setting not found in any source

        Returns:
            The resolved setting value or fallback if not found

        """
        if node is None:
            return fallback

        # Convert between kebab-case and snake_case for different naming conventions
        kebab_name = setting_name.replace("_", "-")
        snake_name = setting_name.replace("-", "_")

        # Build list of sources in precedence order
        sources = []

        # Column-level sources (if column specified) - HIGHEST precedence
        if column_name and (column := node.columns.get(column_name)):
            sources = [
                column.meta,
                column.meta.get("dbt-osmosis-options", {}),
                column.meta.get("dbt_osmosis_options", {}),
            ]
            # Add node-level sources after column sources
            sources.extend([
                node.meta,
                node.meta.get("dbt-osmosis-options", {}),
                node.meta.get("dbt_osmosis_options", {}),
                node.config.extra,
                node.config.extra.get("dbt-osmosis-options", {}),
                node.config.extra.get("dbt_osmosis_options", {}),
            ])
        else:
            # Only node-level sources
            sources = [
                node.meta,
                node.meta.get("dbt-osmosis-options", {}),
                node.meta.get("dbt_osmosis_options", {}),
                node.config.extra,
                node.config.extra.get("dbt-osmosis-options", {}),
                node.config.extra.get("dbt_osmosis_options", {}),
            ]

        # Check each source for the setting (in order - highest precedence first)
        for source in sources:
            # Check prefixed variants first
            for prefixed_name in (f"dbt-osmosis-{kebab_name}", f"dbt_osmosis_{snake_name}"):
                if prefixed_name in source:
                    logger.debug(
                        ":gear: Resolved setting '%s' from source (prefixed variant)",
                        setting_name,
                    )
                    return source[prefixed_name]

            # For non-config.extra sources, check direct key variants
            if source is not node.config.extra:
                if kebab_name in source:
                    logger.debug(
                        ":gear: Resolved setting '%s' from source (direct kebab variant)",
                        setting_name,
                    )
                    return source[kebab_name]
                if snake_name in source:
                    logger.debug(
                        ":gear: Resolved setting '%s' from source (direct snake variant)",
                        setting_name,
                    )
                    return source[snake_name]
            # config.extra only checks prefixed variants, not direct keys

        # Check dbt 1.10+ sources AFTER existing sources (lower precedence)
        # Check config.meta (dbt 1.10+)
        if hasattr(node, "config") and hasattr(node.config, "meta"):
            config_meta_source = ConfigMetaSource(node)
            value = config_meta_source.get(setting_name)
            if value is not None:
                logger.debug(
                    ":gear: Resolved setting '%s' from config.meta (dbt 1.10+)",
                    setting_name,
                )
                return value

        # Check unrendered_config (dbt 1.10+)
        if hasattr(node, "unrendered_config"):
            unrendered_source = UnrenderedConfigSource(node)
            value = unrendered_source.get(setting_name)
            if value is not None:
                logger.debug(
                    ":gear: Resolved setting '%s' from unrendered_config (dbt 1.10+)",
                    setting_name,
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
    ) -> bool:
        """Check if a setting exists in any source.

        Args:
            setting_name: The name of the setting to check
            node: The dbt node to check for settings
            column_name: Optional column name to check column-level settings

        Returns:
            True if the setting exists in any source, False otherwise

        """
        if node is None:
            return False

        # Use resolve with a sentinel value to check if setting exists
        sentinel = object()
        result = self.resolve(setting_name, node, column_name, fallback=sentinel)
        return result is not sentinel

    def get_precedence_chain(
        self,
        setting_name: str,
        node: ResultNode | None = None,
        column_name: str | None = None,
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
        chain = []

        if node is None:
            chain.append((ConfigSourceName.FALLBACK, None))
            return chain

        # Convert between kebab-case and snake_case
        kebab_name = setting_name.replace("_", "-")
        snake_name = setting_name.replace("-", "_")

        # Helper to extract value from a dict source
        def extract_value(source: dict[str, t.Any]) -> t.Any | None:
            # Check prefixed variants
            for prefixed_name in (f"dbt-osmosis-{kebab_name}", f"dbt_osmosis_{snake_name}"):
                if prefixed_name in source:
                    return source[prefixed_name]
            # Check direct variants
            if kebab_name in source:
                return source[kebab_name]
            if snake_name in source:
                return source[snake_name]
            return None

        # Column-level sources
        if column_name and (column := node.columns.get(column_name)):
            value = extract_value(column.meta)
            chain.append((ConfigSourceName.COLUMN_META, value))

        # Node meta sources
        value = extract_value(node.meta)
        chain.append((ConfigSourceName.NODE_META, value))

        # Node config.extra
        value = extract_value(node.config.extra)
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

        # Note: Project vars and supplementary file require context,
        # which isn't available in this stateless resolver

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
            3. node.unrendered_config["dbt-osmosis"] or ["dbt_osmosis"] (dbt 1.10+)

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
            config_meta = node.config.meta
            if isinstance(config_meta, dict):
                result = check_dict(config_meta)
                if result:
                    logger.debug(
                        ":gear: Found YAML path template in config.meta: %s",
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


# Global resolver instance for backward compatibility
_resolver = SettingsResolver()


def _get_setting_for_node(
    opt: str,
    /,
    node: ResultNode | None = None,
    col: str | None = None,
    *,
    fallback: t.Any | None = None,
) -> t.Any:
    """Get a configuration value for a dbt node from the node's meta and config.

    DEPRECATED: Use SettingsResolver directly instead. This function is kept for
    backward compatibility and will be removed in a future version.

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
    # For backward compatibility, use the resolver directly
    return _resolver.resolve(opt, node, col, fallback=fallback)


_COLUMN_LIST_CACHE: dict[str, OrderedDict[str, ColumnMetadata]] = {}
"""Cache for column lists to avoid redundant introspection.

Thread-safety: Protected by _COLUMN_LIST_CACHE_LOCK. All reads and writes
must be guarded by this lock. The cache is unbounded and may grow indefinitely.
"""

_COLUMN_LIST_CACHE_LOCK = threading.Lock()
"""Lock to protect _COLUMN_LIST_CACHE from concurrent access.

Critical sections: get_columns() function performs cache reads and writes
under this lock. All access to _COLUMN_LIST_CACHE must be synchronized.
"""


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
    col: BaseColumn,
    settings: t.Any,
    node: ResultNode | None = None,
) -> str:
    """Use the precise data type if enabled in the settings."""
    use_num_prec = _get_setting_for_node(
        "numeric-precision-and-scale",
        node,
        col.name,
        fallback=settings.numeric_precision_and_scale,
    )
    use_chr_prec = _get_setting_for_node(
        "string-length",
        node,
        col.name,
        fallback=settings.string_length,
    )
    if (col.is_numeric() and use_num_prec) or (col.is_string() and use_chr_prec):
        logger.debug(":ruler: Using precise data type => %s", col.data_type)
        return col.data_type
    if hasattr(col, "mode"):
        return col.data_type
    return col.dtype


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
    if node is None:
        return fallback
    k, identifier = opt.replace("_", "-"), opt.replace("-", "_")
    sources = [
        node.meta,
        node.meta.get("dbt-osmosis-options", {}),
        node.meta.get("dbt_osmosis_options", {}),
        node.config.extra,
        node.config.extra.get("dbt-osmosis-options", {}),
        node.config.extra.get("dbt_osmosis_options", {}),
    ]
    if col and (column := node.columns.get(col)):
        sources = [
            column.meta,
            column.meta.get("dbt-osmosis-options", {}),
            column.meta.get("dbt_osmosis_options", {}),
            *sources,
        ]
    for source in sources:
        for variation in (f"dbt-osmosis-{k}", f"dbt_osmosis_{identifier}"):
            if variation in source:
                return source[variation]
        if source is not node.config.extra:
            if k in source:
                return source[k]
            if identifier in source:
                return source[identifier]
    return fallback


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

    rendered_relation = relation.render()
    with _COLUMN_LIST_CACHE_LOCK:
        if rendered_relation in _COLUMN_LIST_CACHE:
            logger.debug(":blue_book: Column list cache HIT => %s", rendered_relation)
            return _COLUMN_LIST_CACHE[rendered_relation]

    logger.info(":mag_right: Collecting columns for table => %s", rendered_relation)
    index = 0

    def process_column(c: BaseColumn | ColumnMetadata, /) -> None:
        nonlocal index

        columns = [c]
        if hasattr(c, "flatten"):
            columns.extend(c.flatten())  # pyright: ignore[reportUnknownMemberType]

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
                dtype = _maybe_use_precise_dtype(column, context.settings, result_node)
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
        catalog_entry = _find_first(
            chain(catalog.nodes.values(), catalog.sources.values()),
            lambda c: relation.matches(*c.key()),
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

    try:
        logger.info(":mag: Introspecting columns in warehouse for => %s", rendered_relation)
        for column in t.cast(
            "t.Iterable[BaseColumn]",
            context.project.adapter.get_columns_in_relation(relation),
        ):
            process_column(column)
    except Exception as ex:
        logger.warning(":warning: Could not introspect columns for %s: %s", rendered_relation, ex)

    with _COLUMN_LIST_CACHE_LOCK:
        _COLUMN_LIST_CACHE[rendered_relation] = normalized_columns
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
    return t.cast("CatalogResults", CatalogArtifact.from_dict(json.loads(fp.read_text())))


# NOTE: this is mostly adapted from dbt-core with some cruft removed, strict pyright is not a fan of dbt's shenanigans
def _generate_catalog(context: t.Any) -> CatalogResults | None:
    """Generate the dbt catalog file for the project."""
    import dbt.utils as dbt_utils

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
    artifact = CatalogArtifact.from_results(
        nodes=nodes,
        sources=sources,
        generated_at=datetime.now(timezone.utc),
        compile_results=None,
        errors=errors,
    )
    artifact_path = Path(context.runtime_cfg.project_target_path, "catalog.json")
    logger.info(":bookmark_tabs: Writing fresh catalog => %s", artifact_path)
    artifact.write(str(artifact_path.resolve()))  # Cache it, same as dbt
    return t.cast("CatalogResults", artifact)


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
                return getattr(column, "tags", None)
            if property_key == "meta":
                return getattr(column, "meta", None)
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
                    if column.get("name") == column_name:
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
