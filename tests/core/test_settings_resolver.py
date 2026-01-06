"""Tests for SettingsResolver class."""

from __future__ import annotations

import pytest

from dbt_osmosis.core.introspection import SettingsResolver


class MockColumn:
    """Mock column for testing."""

    def __init__(self, name: str, meta: dict | None = None) -> None:
        self.name = name
        self.meta = meta or {}


class MockConfig:
    """Mock config for testing."""

    def __init__(self, extra: dict | None = None) -> None:
        self.extra = extra or {}


class MockNode:
    """Mock node for testing."""

    def __init__(
        self,
        meta: dict | None = None,
        config_extra: dict | None = None,
        columns: dict[str, MockColumn] | None = None,
    ) -> None:
        self.meta = meta or {}
        self.config = MockConfig(config_extra)
        self.columns = columns or {}


@pytest.fixture
def resolver() -> SettingsResolver:
    """Create a SettingsResolver instance for testing."""
    return SettingsResolver()


@pytest.fixture
def sample_node() -> MockNode:
    """Create a sample node for testing."""
    return MockNode(
        meta={
            "string-length": True,
            "dbt-osmosis-string-length": False,
            "dbt-osmosis-options": {"numeric-precision-and-scale": True},
            "dbt_osmosis_options": {"prefix": "test_"},
        },
        config_extra={
            "skip-add-columns": False,
            "dbt-osmosis-skip-add-tags": True,
            "dbt_osmosis_skip_add_tags": False,
            "dbt-osmosis-options": {"output-to-lower": True},
        },
        columns={
            "col1": MockColumn(
                "col1",
                {"output-to-lower": True, "dbt-osmosis-output-to-lower": False},
            ),
            "col2": MockColumn("col2", {"dbt_osmosis_prefix": "col_"}),
        },
    )


def test_resolve_with_no_node(resolver: SettingsResolver) -> None:
    """Test that fallback is returned when node is None."""
    assert resolver.resolve("string-length", None, fallback=True) is True


def test_resolve_with_no_matching_setting(
    resolver: SettingsResolver,
    sample_node: MockNode,
) -> None:
    """Test that fallback is returned when no matching setting is found."""
    assert resolver.resolve("unknown-setting", sample_node, fallback="default") == "default"


def test_resolve_from_column_meta(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test resolving setting from column meta."""
    # Column-level setting should take precedence
    result = resolver.resolve("output-to-lower", sample_node, column_name="col1", fallback=False)
    # The column has output-to-lower: True, but also has dbt-osmosis-output-to-lower: False
    # The prefixed variant should take precedence over direct key
    assert result is False


def test_resolve_from_column_prefixed_setting(
    resolver: SettingsResolver,
    sample_node: MockNode,
) -> None:
    """Test resolving setting from column prefixed setting."""
    # Column col2 has dbt_osmosis_prefix: "col_" (not output-to-lower)
    # Should fall back to node-level settings and find output-to-lower in config.extra
    result = resolver.resolve("output-to-lower", sample_node, column_name="col2", fallback=False)
    assert result is True


def test_resolve_from_node_meta(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test resolving setting from node meta."""
    result = resolver.resolve("string-length", sample_node, fallback=False)
    # Node meta has string-length: True, but also has dbt-osmosis-string-length: False
    # Prefixed variant should take precedence
    assert result is False


def test_resolve_from_node_prefixed_setting(
    resolver: SettingsResolver,
    sample_node: MockNode,
) -> None:
    """Test resolving setting from node prefixed setting."""
    result = resolver.resolve("string-length", sample_node, fallback=None)
    # Node-level prefixed setting should override direct setting
    assert result is False


def test_resolve_from_node_config(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test resolving setting from node config extra."""
    # Note: config.extra only checks prefixed variants, not direct keys
    result = resolver.resolve("skip-add-columns", sample_node, fallback=True)
    assert result is True  # Should fall back to True since no prefixed variant exists


def test_resolve_from_node_config_prefixed(
    resolver: SettingsResolver,
    sample_node: MockNode,
) -> None:
    """Test resolving setting from node config prefixed setting."""
    result = resolver.resolve("skip-add-tags", sample_node, fallback=None)
    # Prefixed variant should take precedence
    assert result is True


def test_precedence_order(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test that column-level settings take precedence over node-level settings."""
    # Set a setting in both column and node meta
    sample_node.columns["col1"].meta["test-setting"] = "column-value"
    sample_node.meta["test-setting"] = "node-value"

    result = resolver.resolve("test-setting", sample_node, column_name="col1", fallback="fallback")
    assert result == "column-value"


def test_kebab_case_support(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test that both kebab-case and snake_case are supported."""
    # Add setting with snake_case variant
    sample_node.meta["snake_case_setting"] = "snake-value"

    # Should be able to resolve with kebab-case
    result = resolver.resolve("snake-case-setting", sample_node, fallback=None)
    assert result == "snake-value"

    # Should also be able to resolve with snake_case
    result = resolver.resolve("snake_case_setting", sample_node, fallback=None)
    assert result == "snake-value"


def test_options_object_support(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test that dbt-osmosis-options objects are supported."""
    result = resolver.resolve("numeric-precision-and-scale", sample_node, fallback=None)
    assert result is True


def test_fallback_value(resolver: SettingsResolver) -> None:
    """Test that fallback value is returned when no setting is found."""
    node = MockNode()
    result = resolver.resolve("unknown-setting", node, fallback="default-value")
    assert result == "default-value"


def test_column_not_found(resolver: SettingsResolver, sample_node: MockNode) -> None:
    """Test that unknown columns don't break resolution."""
    result = resolver.resolve(
        "string-length",
        sample_node,
        column_name="unknown-column",
        fallback=False,
    )
    # Should fall back to node-level settings (prefixed variant takes precedence)
    assert result is False
