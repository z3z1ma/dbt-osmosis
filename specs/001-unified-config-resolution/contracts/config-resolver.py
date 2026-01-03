"""
ConfigResolver API Contract

This contract defines the interface for the unified configuration resolution system.
Implementations must follow this contract to ensure compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

# Type aliases for clarity
NodeRef = Any  # dbt.contracts.graph.nodes.ResultNode
ColumnName = str | None
FallbackValue = Any
ConfigValue = Any


class ConfigResolver(Protocol):
    """
    Protocol for configuration resolution from multiple sources.

    The resolver checks sources in precedence order and returns the first
    non-None value found. If no source has the key, the fallback is returned.

    Precedence (highest to lowest):
        1. Column-level meta (when column specified)
        2. Node-level meta
        3. Node-level config.extra
        4. Node-level config.meta (dbt 1.10+)
        5. Node-level unrendered_config (dbt 1.10+)
        6. Project-level vars (dbt_project.yml)
        7. Supplementary dbt-osmosis.yml
        8. Fallback value
    """

    def resolve(
        self,
        setting_name: str,
        node: NodeRef | None,
        column_name: ColumnName = None,
        *,
        fallback: FallbackValue = None,
    ) -> ConfigValue:
        """
        Resolve a configuration setting from the highest-priority source.

        Args:
            setting_name: The setting to resolve. Supports both kebab-case
                (e.g., "skip-add-tags") and snake_case (e.g., "skip_add_tags").
                Prefixes ("dbt-osmosis-", "dbt_osmosis_") are optional.
            node: The dbt node (model, source, seed, etc.) to resolve for.
                If None, returns fallback immediately.
            column_name: Optional column name for column-level overrides.
                If specified, column-level meta is checked before node-level.
            fallback: Default value if setting not found in any source.

        Returns:
            The resolved configuration value from the highest-priority source,
            or the fallback value if not found.

        Example:
            >>> resolver = ConfigResolver()
            >>> # Node-level setting
            >>> skip_tags = resolver.resolve(
            ...     "skip-add-tags",
            ...     node=my_model,
            ...     fallback=False
            ... )
            >>> # Column-level override
            >>> use_lower = resolver.resolve(
            ...     "output-to-lower",
            ...     node=my_model,
            ...     column_name="user_id",
            ...     fallback=False
            ... )
        """
        ...

    def has(
        self,
        setting_name: str,
        node: NodeRef | None,
        column_name: ColumnName = None,
    ) -> bool:
        """
        Check if a configuration setting exists in any source.

        Unlike resolve(), this method does not provide a fallback and
        returns a boolean indicating presence.

        Args:
            setting_name: The setting to check for.
            node: The dbt node to check on.
            column_name: Optional column name for column-level checks.

        Returns:
            True if the setting exists in any source, False otherwise.

        Example:
            >>> resolver = ConfigResolver()
            >>> if resolver.has("custom-setting", node):
            ...     value = resolver.resolve("custom-setting", node)
        """
        ...

    def get_precedence_chain(
        self,
        node: NodeRef | None,
        setting_name: str,
        column_name: ColumnName = None,
    ) -> list[str]:
        """
        Get the list of source names in precedence order for debugging.

        Args:
            node: The dbt node.
            column_name: Optional column name.

        Returns:
            List of source names that will be checked, in order.

        Example:
            >>> resolver = ConfigResolver()
            >>> chain = resolver.get_precedence_chain(node, "user_id")
            >>> print(chain)
            ['column_meta', 'node_meta', 'config_extra', ...]
        """
        ...


class PropertySource:
    """Enum for property source selection."""

    MANIFEST = "manifest"  # Parsed manifest.json (rendered jinja)
    YAML = "yaml"  # Raw YAML files (unrendered jinja)
    AUTO = "auto"  # Prefer YAML if has unrendered jinja, else manifest


class PropertyAccessor(Protocol):
    """
    Protocol for unified model property access from manifest or YAML.

    Provides a single interface for accessing properties like descriptions,
    tags, and meta from either the parsed manifest (rendered values) or
    raw YAML files (unrendered jinja templates).
    """

    def get(
        self,
        property_key: str,
        node: NodeRef,
        column_name: ColumnName = None,
        source: Literal["manifest", "yaml", "auto"] = "auto",
    ) -> Any:
        """
        Get a model or column property from the specified source.

        Args:
            property_key: The property to retrieve (e.g., "description",
                "tags", "meta", "data_type", "name").
            node: The dbt node.
            column_name: Optional column name. If specified, retrieves
                the column property instead of node property.
            source: Which source to read from:
                - "manifest": Read from parsed manifest (rendered values)
                - "yaml": Read from raw YAML files (unrendered jinja)
                - "auto": Prefer YAML if contains unrendered jinja,
                  otherwise use manifest

        Returns:
            The property value, or None if not found.

        Raises:
            ValueError: If source is not one of "manifest", "yaml", "auto"
            ConfigurationError: If YAML file is malformed

        Example:
            >>> accessor = PropertyAccessor(context)
            >>> # Get unrendered description
            >>> desc = accessor.get(
            ...     "description",
            ...     node=my_model,
            ...     column_name="user_id",
            ...     source="yaml"
            ... )
            >>> # desc may contain "{{ doc('my_doc') }}"
        """
        ...

    def get_description(
        self,
        node: NodeRef,
        column_name: ColumnName = None,
        source: Literal["manifest", "yaml", "auto"] = "auto",
    ) -> str | None:
        """
        Convenience method for retrieving descriptions.

        Args:
            node: The dbt node.
            column_name: Optional column name.
            source: Which source to read from.

        Returns:
            The description string, or None if not found.

        Example:
            >>> accessor = PropertyAccessor(context)
            >>> desc = accessor.get_description(my_model, "user_id")
        """
        ...

    def get_meta(
        self,
        node: NodeRef,
        key: str,
        column_name: ColumnName = None,
        source: Literal["manifest", "yaml", "auto"] = "auto",
    ) -> Any:
        """
        Convenience method for retrieving metadata values.

        Args:
            node: The dbt node.
            key: The meta key to retrieve.
            column_name: Optional column name.
            source: Which source to read from.

        Returns:
            The meta value, or None if not found.

        Example:
            >>> accessor = PropertyAccessor(context)
            >>> pii = accessor.get_meta(my_model, "pii", "email")
        """
        ...

    def has_property(
        self,
        property_key: str,
        node: NodeRef,
        column_name: ColumnName = None,
    ) -> bool:
        """
        Check if a property exists in any source.

        Args:
            property_key: The property to check for.
            node: The dbt node.
            column_name: Optional column name.

        Returns:
            True if the property exists in manifest or YAML, False otherwise.

        Example:
            >>> accessor = PropertyAccessor(context)
            >>> if accessor.has_property("description", my_model):
            ...     desc = accessor.get("description", my_model)
        """
        ...


# Validation rules as type annotations
class SettingName:
    """
    Valid configuration setting name.

    Rules:
    - Non-empty string
    - Kebab-case and snake_case are equivalent
    - Prefixes ("dbt-osmosis-", "dbt_osmosis_") are optional
    - Nested keys use dot notation: "dbt-osmosis-options.key"
    """

    def __init__(self, value: str):
        if not value or not isinstance(value, str):
            raise ValueError("Setting name must be a non-empty string")
        self.value = value

    @property
    def base_name(self) -> str:
        """Setting name without prefix."""
        # Remove "dbt-osmosis-" or "dbt_osmosis_" prefix
        for prefix in ("dbt-osmosis-", "dbt_osmosis_"):
            if self.value.startswith(prefix):
                return self.value[len(prefix) :]
        return self.value

    @property
    def kebab_variant(self) -> str:
        """Kebab-case version of base name."""
        return self.base_name.replace("_", "-")

    @property
    def snake_variant(self) -> str:
        """Snake_case version of base name."""
        return self.base_name.replace("-", "_")

    def __str__(self) -> str:
        return self.value


class ConfigurationError(Exception):
    """Raised when configuration file is invalid or cannot be read."""

    def __init__(self, message: str, file_path: str | None = None):
        self.file_path = file_path
        super().__init__(message)


# Backward compatibility layer


def _get_setting_for_node(
    setting_name: str,
    node: NodeRef | None = None,
    column: ColumnName = None,
    *,
    fallback: FallbackValue = None,
) -> ConfigValue:
    """
    Legacy function for backward compatibility.

    DEPRECATED: Use ConfigResolver.resolve() instead.

    This function is maintained for compatibility with existing code.
    New code should use ConfigResolver directly.

    Args:
        setting_name: The setting to resolve.
        node: The dbt node.
        column: Optional column name.
        fallback: Default value if not found.

    Returns:
        The resolved configuration value.

    Example:
        >>> # Old way (still works)
        >>> value = _get_setting_for_node("skip-add-tags", node, fallback=False)
        >>> # New way (preferred)
        >>> resolver = ConfigResolver()
        >>> value = resolver.resolve("skip-add-tags", node, fallback=False)
    """
    # Implementation delegates to ConfigResolver
    ...


# Export list for __init__.py
__all__ = [
    "ConfigResolver",
    "PropertyAccessor",
    "PropertySource",
    "SettingName",
    "ConfigurationError",
    "_get_setting_for_node",  # Exported for backward compatibility
]
