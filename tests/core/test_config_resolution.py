"""Tests for ConfigResolver class.

This module tests the unified configuration resolution system that consolidates
configuration access from multiple sources including:
- Column-level meta
- Node-level meta
- Node-level config.extra
- Node-level config.meta (dbt 1.10+)
- Node-level unrendered_config (dbt 1.10+)
- Project-level vars in dbt_project.yml
- Supplementary dbt-osmosis.yml file
- Fallback values

The ConfigResolver provides a single interface for resolving configuration
settings with proper precedence handling across dbt versions 1.8-1.11+.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from dbt_osmosis.core.introspection import (
    ConfigMetaSource,
    ConfigSourceName,
    ProjectVarsSource,
    SettingsResolver,
    SupplementaryFileSource,
    UnrenderedConfigSource,
)
from dbt_osmosis.core.settings import YamlRefactorContext


class MockColumn:
    """Mock column for testing configuration resolution."""

    def __init__(self, name: str, meta: dict[str, Any] | None = None) -> None:
        """Initialize a mock column.

        Args:
            name: Column name
            meta: Optional metadata dictionary for the column

        """
        self.name = name
        self.meta = meta or {}


class MockConfig:
    """Mock config object for testing configuration resolution."""

    def __init__(
        self,
        extra: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a mock config object.

        Args:
            extra: Optional extra configuration dictionary
            meta: Optional meta configuration dictionary (dbt 1.10+)

        """
        self.extra = extra or {}
        self.meta = meta or {}


class MockNode:
    """Mock node for testing configuration resolution."""

    def __init__(
        self,
        meta: dict[str, Any] | None = None,
        config_extra: dict[str, Any] | None = None,
        config_meta: dict[str, Any] | None = None,
        columns: dict[str, MockColumn] | None = None,
        unique_id: str = "model.test.my_model",
    ) -> None:
        """Initialize a mock node.

        Args:
            meta: Optional node-level metadata
            config_extra: Optional config.extra dictionary
            config_meta: Optional config.meta dictionary (dbt 1.10+)
            columns: Optional dictionary of columns
            unique_id: Unique identifier for the node

        """
        self.meta = meta or {}
        self.config = MockConfig(config_extra, config_meta)
        self.columns = columns or {}
        self.unique_id = unique_id


@pytest.fixture
def sample_project_dir(tmp_path: Path) -> Path:
    """Create a sample project directory with dbt_project.yml.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to the temporary project directory

    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create a basic dbt_project.yml with dbt-osmosis vars
    dbt_project = project_dir / "dbt_project.yml"
    dbt_project.write_text(
        """
name: test_project
version: 1.0.0
config-version: 2

profile: test

vars:
  dbt-osmosis:
    skip-add-tags: true
    yaml_settings:
      map_indent: 2
""",
    )

    return project_dir


@pytest.fixture
def sample_node_with_config() -> MockNode:
    """Create a sample node with various configuration sources.

    Returns:
        MockNode with populated configuration across multiple sources

    """
    return MockNode(
        meta={
            # Node-level meta with various key formats
            "string-length": True,
            "dbt-osmosis-string-length": False,  # Prefixed should override
            "dbt-osmosis-options": {
                "output-to-lower": True,
                "numeric-precision-and-scale": False,
            },
        },
        config_extra={
            # Config.extra with prefixed keys
            "dbt-osmosis-skip-add-columns": False,
            "skip-add-tags": True,  # Non-prefixed (lower precedence)
        },
        config_meta={
            # Config.meta (dbt 1.10+) with prefixed keys
            "dbt-osmosis-output-to-lower": True,
        },
        columns={
            "col1": MockColumn(
                "col1",
                {
                    "output-to-lower": True,  # Direct key
                    "dbt-osmosis-output-to-lower": False,  # Prefixed override
                },
            ),
            "col2": MockColumn("col2", {"dbt_osmosis_prefix": "col_"}),
        },
    )


class TestConfigResolver:
    """Test suite for ConfigResolver class.

    Tests cover:
    - Configuration resolution from all supported sources
    - Precedence order (column > node config.meta > node meta > project vars)
    - Support for kebab-case and snake_case variants
    - Support for prefixed and non-prefixed keys
    - Nested options object support
    - Fallback value handling
    - Cross-dbt-version compatibility
    """

    def test_config_meta_source_dbt_110(self) -> None:
        """T020: Test ConfigMetaSource reads from node.config.meta (dbt 1.10+)."""
        # Create a mock node with config.meta
        mock_node = Mock()
        mock_node.config = Mock()
        mock_node.config.meta = {
            "dbt-osmosis-output-to-lower": True,
            "dbt_osmosis_skip_add_tags": False,
            "dbt-osmosis-options": {
                "numeric-precision-and-scale": True,
            },
        }

        source = ConfigMetaSource(mock_node)

        # Test prefixed kebab-case
        assert source.get("output-to-lower") is True
        # Test prefixed snake_case
        assert source.get("skip_add_tags") is False
        # Test options object
        assert source.get("numeric-precision-and-scale") is True
        # Test missing key returns None
        assert source.get("nonexistent") is None

    def test_config_meta_source_graceful_version_handling(self) -> None:
        """T020: Test ConfigMetaSource handles missing config.meta gracefully (dbt < 1.10)."""
        # Create a mock node without config.meta attribute (dbt 1.8)
        mock_node = Mock()
        mock_node.config = Mock()
        # Don't set meta attribute - simulates dbt 1.8
        del mock_node.config.meta

        source = ConfigMetaSource(mock_node)

        # Should return None for all keys without error
        assert source.get("output-to-lower") is None
        assert source.get("skip_add_tags") is None

    def test_unrendered_config_source_dbt_110(self) -> None:
        """T021: Test UnrenderedConfigSource reads from node.unrendered_config (dbt 1.10+)."""
        # Create a mock node with unrendered_config
        mock_node = Mock()
        mock_node.unrendered_config = {
            "dbt-osmosis-skip-add-columns": True,
            "dbt_osmosis_skip_add_tags": False,
        }

        source = UnrenderedConfigSource(mock_node)

        # Test prefixed variants
        assert source.get("skip-add-columns") is True
        assert source.get("skip_add_tags") is False
        # Test missing key
        assert source.get("nonexistent") is None

    def test_unrendered_config_source_graceful_version_handling(self) -> None:
        """T021: Test UnrenderedConfigSource handles missing unrendered_config gracefully."""
        # Create a mock node without unrendered_config attribute
        mock_node = Mock()
        # Don't set unrendered_config attribute
        del mock_node.unrendered_config

        source = UnrenderedConfigSource(mock_node)

        # Should return None for all keys without error
        assert source.get("skip-add-columns") is None
        assert source.get("skip_add_tags") is None

    def test_project_vars_source(self) -> None:
        """T022: Test ProjectVarsSource reads from runtime_cfg.vars."""
        # Create a mock context with project vars
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.vars = Mock()
        mock_context.project.runtime_cfg.vars.to_dict = Mock(
            return_value={
                "dbt-osmosis": {
                    "skip-add-tags": True,
                    "dbt-osmosis-skip-add-columns": False,
                    "dbt-osmosis-options": {
                        "string-length": True,
                    },
                    "yaml-settings": {"map_indent": 2},
                },
                "dbt_osmosis": {
                    "output-to-lower": False,
                    "dbt_osmosis_output_to_upper": True,
                    "dbt_osmosis_options": {
                        "numeric_precision_and_scale": False,
                    },
                },
                "dbt-osmosis-force-inherit-descriptions": True,
                "prefer-yaml-values": False,
            },
        )

        source = ProjectVarsSource(mock_context)

        # Test dbt-osmosis kebab-case prefix
        assert source.get("skip-add-tags") is True
        # Test prefixed and nested variants inside project vars namespaces
        assert source.get("skip-add-columns") is False
        assert source.get("string-length") is True
        # Test dbt_osmosis snake_case prefix
        assert source.get("output-to-lower") is False
        assert source.get("output-to-upper") is True
        assert source.get("numeric-precision-and-scale") is False
        # Test documented top-level project var variants
        assert source.get("force-inherit-descriptions") is True
        assert source.get("prefer-yaml-values") is False
        # Test nested yaml-settings
        assert source.get("yaml-settings") == {"map_indent": 2}
        # Test missing key
        assert source.get("nonexistent") is None

    def test_project_vars_source_key_normalization(self) -> None:
        """T022: Test ProjectVarsSource supports all key variants."""
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.vars = Mock()
        mock_context.project.runtime_cfg.vars.to_dict = Mock(
            return_value={
                "dbt-osmosis": {
                    "skip-add-tags": True,
                    "skip_add_tags": False,
                },
            },
        )

        source = ProjectVarsSource(mock_context)

        # Both kebab and snake variants should work
        assert source.get("skip-add-tags") is True
        assert source.get("skip_add_tags") is True  # Normalizes to kebab

    def test_supplementary_file_source(self, tmp_path: Path) -> None:
        """T023: Test SupplementaryFileSource reads dbt-osmosis.yml."""
        # Create a test dbt-osmosis.yml file
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
skip-add-tags: true
yaml-settings:
  map_indent: 2
dbt-osmosis-options:
  output-to-lower: false
""",
        )

        # Create a mock context with project_dir
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Test direct key
        assert source.get("skip-add-tags") is True
        # Test nested yaml-settings
        assert source.get("yaml-settings") == {"map_indent": 2}
        # Test options object
        assert source.get("output-to-lower") is False
        # Test missing key
        assert source.get("nonexistent") is None

    def test_nested_options_support_snake_case_inner_keys(self, tmp_path: Path) -> None:
        """Nested dbt_osmosis_options should accept Python-friendly inner keys."""
        mock_node = Mock()
        mock_node.meta = {"dbt_osmosis_options": {"string_length": True}}
        mock_node.config = Mock()
        mock_node.config.extra = {"dbt_osmosis_options": {"output_to_lower": False}}
        mock_node.config.meta = {"dbt_osmosis_options": {"skip_add_tags": True}}
        mock_node.unrendered_config = {
            "dbt_osmosis_options": {"numeric_precision_and_scale": True},
        }
        mock_node.columns = {}

        assert ConfigMetaSource(mock_node).get("skip-add-tags") is True
        assert UnrenderedConfigSource(mock_node).get("numeric-precision-and-scale") is True

        resolver = SettingsResolver()
        assert resolver.resolve("string-length", mock_node, fallback=False) is True
        assert resolver.resolve("output-to-lower", mock_node, fallback=True) is False

        (tmp_path / "dbt-osmosis.yml").write_text(
            "dbt_osmosis_options:\n  force_inherit_descriptions: true\n",
        )
        mock_context = SimpleNamespace(
            project=SimpleNamespace(runtime_cfg=SimpleNamespace(project_root=tmp_path)),
        )
        assert SupplementaryFileSource(mock_context).get("force-inherit-descriptions") is True

    def test_nested_options_support_kebab_case_inner_keys(self, tmp_path: Path) -> None:
        """Nested dbt-osmosis options keep accepting kebab-case inner keys."""
        mock_node = Mock()
        mock_node.config = Mock()
        mock_node.config.meta = {
            "dbt-osmosis-options": {"output-to-lower": True},
        }
        mock_node.unrendered_config = {
            "dbt-osmosis-options": {"skip-add-tags": False},
        }

        assert ConfigMetaSource(mock_node).get("output-to-lower") is True
        assert UnrenderedConfigSource(mock_node).get("skip-add-tags") is False

        (tmp_path / "dbt-osmosis.yml").write_text(
            "dbt-osmosis-options:\n  numeric-precision-and-scale: true\n",
        )
        mock_context = SimpleNamespace(
            project=SimpleNamespace(runtime_cfg=SimpleNamespace(project_root=tmp_path)),
        )
        assert SupplementaryFileSource(mock_context).get("numeric-precision-and-scale") is True

    def test_dbt110_sources_support_snake_inner_keys_in_snake_namespace(self) -> None:
        """dbt 1.10+ config containers accept snake_case inner option keys."""
        mock_node = Mock()
        mock_node.config = Mock()
        mock_node.config.meta = {
            "dbt_osmosis_options": {"output_to_lower": True},
        }
        mock_node.unrendered_config = {
            "dbt_osmosis_options": {"output_to_lower": True, "skip_add_tags": False},
        }

        assert ConfigMetaSource(mock_node).get("output-to-lower") is True
        assert UnrenderedConfigSource(mock_node).get("output-to-lower") is True
        assert UnrenderedConfigSource(mock_node).get("skip-add-tags") is False

    def test_supplementary_file_supports_snake_inner_keys_in_both_namespaces(
        self,
        tmp_path: Path,
    ) -> None:
        """Supplementary YAML accepts snake_case inner keys in both option namespaces."""
        (tmp_path / "dbt-osmosis.yml").write_text(
            """
dbt-osmosis-options:
  output_to_lower: false
dbt_osmosis_options:
  force_inherit_descriptions: true
""",
        )
        mock_context = SimpleNamespace(
            project=SimpleNamespace(runtime_cfg=SimpleNamespace(project_root=tmp_path)),
        )

        source = SupplementaryFileSource(mock_context)
        assert source.get("output-to-lower") is False
        assert source.get("force-inherit-descriptions") is True

    def test_nested_options_prefer_kebab_inner_key_over_snake_variant(
        self,
        tmp_path: Path,
    ) -> None:
        """When both inner key spellings exist, kebab-case wins."""
        mock_node = Mock()
        mock_node.config = Mock()
        mock_node.config.meta = {
            "dbt_osmosis_options": {
                "output-to-lower": False,
                "output_to_lower": True,
            },
        }
        mock_node.unrendered_config = {
            "dbt-osmosis-options": {
                "output-to-lower": "kebab",
                "output_to_lower": "snake",
            },
        }

        assert ConfigMetaSource(mock_node).get("output-to-lower") is False
        assert UnrenderedConfigSource(mock_node).get("output-to-lower") == "kebab"

        (tmp_path / "dbt-osmosis.yml").write_text(
            """
dbt-osmosis-options:
  output-to-lower: kebab
  output_to_lower: snake
""",
        )
        mock_context = SimpleNamespace(
            project=SimpleNamespace(runtime_cfg=SimpleNamespace(project_root=tmp_path)),
        )
        assert SupplementaryFileSource(mock_context).get("output-to-lower") == "kebab"

    def test_supplementary_file_source_reuses_shared_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated supplementary source instances should not reparse unchanged YAML."""
        import ruamel.yaml

        (tmp_path / "dbt-osmosis.yml").write_text("skip-add-tags: true\n")
        mock_context = SimpleNamespace(
            project=SimpleNamespace(runtime_cfg=SimpleNamespace(project_root=tmp_path)),
        )

        load_count = 0
        original_load = ruamel.yaml.YAML.load

        def counting_load(self, stream):
            nonlocal load_count
            load_count += 1
            return original_load(self, stream)

        monkeypatch.setattr(ruamel.yaml.YAML, "load", counting_load)

        assert SupplementaryFileSource(mock_context).get("skip-add-tags") is True
        assert SupplementaryFileSource(mock_context).get("skip-add-tags") is True
        assert load_count == 1

    def test_supplementary_file_source_missing_file(self, tmp_path: Path) -> None:
        """T023: Test SupplementaryFileSource handles missing file gracefully."""
        # Create a mock context without dbt-osmosis.yml
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Should return None for all keys without error
        assert source.get("skip-add-tags") is None
        assert source.get("yaml-settings") is None

    def test_full_precedence_chain(self) -> None:
        """T012: Test full precedence chain across all 7 sources."""
        # Create a resolver with all sources
        resolver = SettingsResolver()

        # Mock node with configuration in all sources
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-test-setting": "node-meta",
        }
        mock_node.config = Mock()
        mock_node.config.extra = {
            "dbt-osmosis-test-setting": "config-extra",
        }
        mock_node.config.meta = {
            "dbt-osmosis-test-setting": "config-meta",
        }
        mock_node.unrendered_config = {
            "dbt-osmosis-test-setting": "unrendered-config",
        }
        mock_node.columns = {}

        # Mock context
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.vars = Mock()
        mock_context.project.runtime_cfg.vars.to_dict = Mock(
            return_value={
                "dbt-osmosis": {"test-setting": "project-vars"},
            },
        )
        mock_context.project.runtime_cfg.project_root = Path()

        # Create resolver with context (we'll need to update SettingsResolver to accept context)
        # For now, test with existing sources
        result = resolver.resolve("test-setting", mock_node, fallback="fallback-value")

        # Should return node-meta (highest precedence among existing sources)
        assert result == "node-meta"

    def test_key_normalization_kebab_snake(self) -> None:
        """T013: Test kebab-case/snake_case key normalization."""
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-skip-add-tags": True,
            "dbt_osmosis_output_to_lower": False,
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Test kebab input finds kebab source
        assert resolver.resolve("skip-add-tags", mock_node) is True
        # Test snake input finds snake source
        assert resolver.resolve("output_to_lower", mock_node) is False
        # Test snake input finds kebab source (normalized)
        assert resolver.resolve("skip_add_tags", mock_node) is True
        # Test kebab input finds snake source (normalized)
        assert resolver.resolve("output-to-lower", mock_node) is False

    def test_prefix_handling(self) -> None:
        """T014: Test prefix handling (dbt-osmosis-, dbt_osmosis_)."""
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-skip-add-tags": True,
            "dbt_osmosis_output_to_lower": False,
            "skip-add-columns": "direct",
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Prefixed variants should be found
        assert resolver.resolve("skip-add-tags", mock_node) is True
        assert resolver.resolve("output-to-lower", mock_node) is False
        # Direct key should also be found
        assert resolver.resolve("skip-add-columns", mock_node) == "direct"

    def test_has_method(self) -> None:
        """T015: Test SettingsResolver.has() method."""
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-existing-key": True,
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Test existing key
        assert resolver.has("existing-key", mock_node) is True
        # Test missing key
        assert resolver.has("nonexistent-key", mock_node) is False
        # Test with column
        mock_column = Mock()
        mock_column.meta = {"dbt-osmosis-col-key": "value"}
        mock_node.columns = {"test_col": mock_column}
        assert resolver.has("col-key", mock_node, column_name="test_col") is True

    def test_get_precedence_chain(self) -> None:
        """T016: Test SettingsResolver.get_precedence_chain() method."""
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {"test": "value"}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Get precedence chain
        chain = resolver.get_precedence_chain("test-setting", mock_node)

        # Should return list of sources in precedence order
        assert isinstance(chain, list)
        assert len(chain) > 0
        # Each item should be a tuple of (source_name, value)
        for item in chain:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], ConfigSourceName)

    def test_dbt_version_compatibility(self) -> None:
        """T017: Test dbt version compatibility (1.8 vs 1.11)."""
        resolver = SettingsResolver()

        # Test dbt 1.8 (no config.meta or unrendered_config)
        mock_node_18 = Mock()
        mock_node_18.meta = {"dbt-osmosis-test": "v1.8"}
        mock_node_18.config = Mock()
        mock_node_18.config.extra = {}
        # No config.meta or unrendered_config attributes
        del mock_node_18.config.meta
        del mock_node_18.unrendered_config
        mock_node_18.columns = {}

        assert resolver.resolve("test", mock_node_18) == "v1.8"

        # Test dbt 1.11 (has config.meta and unrendered_config)
        mock_node_111 = Mock()
        mock_node_111.meta = {}
        mock_node_111.config = Mock()
        mock_node_111.config.extra = {}
        mock_node_111.config.meta = {"dbt-osmosis-test": "v1.11"}
        mock_node_111.unrendered_config = {}
        mock_node_111.columns = {}

        assert resolver.resolve("test", mock_node_111) == "v1.11"

    def test_missing_source_graceful_handling(self) -> None:
        """T018: Test missing source graceful handling."""
        resolver = SettingsResolver()

        # Test with None node
        assert resolver.resolve("test", None, fallback="default") == "default"
        assert resolver.has("test", None) is False

        # Test with node missing attributes
        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Should handle missing attributes gracefully
        assert resolver.resolve("test", mock_node, fallback="default") == "default"
        assert resolver.has("test", mock_node) is False

    def test_integration_precedence_order(self, sample_node_with_config: MockNode) -> None:
        """T019: Integration test with full precedence order."""
        resolver = SettingsResolver()

        # Test column-level has highest precedence
        result = resolver.resolve("output-to-lower", sample_node_with_config, column_name="col1")
        # Column meta has: "output-to-lower": True, "dbt-osmosis-output-to-lower": False
        # Prefixed should take precedence
        assert result is False

        # Test node meta
        result = resolver.resolve("string-length", sample_node_with_config)
        # Node meta has: "string-length": True, "dbt-osmosis-string-length": False
        # Prefixed should take precedence
        assert result is False


class TestColumnLevelConfigurationOverride:
    """Test suite for User Story 4: Column-Level Configuration Override.

    Tests cover:
    - Column meta taking precedence over node meta
    - Column-specific config inheritance
    - Column fallback to node-level when no column override
    - Multiple columns with different overrides
    - Integration with demo_duckdb project
    """

    def test_column_meta_precedence_over_node_meta(self) -> None:
        """T057: Test column meta taking precedence over node meta.

        When a setting is defined at both column-level (in column.meta) and
        node-level (in node.meta), the column-level setting should take precedence.
        """
        resolver = SettingsResolver()

        # Create a node with conflicting settings at column and node level
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-output-to-lower": False,  # Node-level says False
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Create column with conflicting setting
        mock_column = Mock()
        mock_column.meta = {
            "dbt-osmosis-output-to-lower": True,  # Column-level says True (should win)
        }
        mock_node.columns = {"test_column": mock_column}

        # Column-level should take precedence
        result = resolver.resolve("output-to-lower", mock_node, column_name="test_column")
        assert result is True, "Column meta should override node meta"

    def test_column_meta_precedence_over_node_config_meta(self) -> None:
        """T057: Test column meta taking precedence over node config.meta (dbt 1.10+)."""
        resolver = SettingsResolver()

        # Create a node with config.meta (dbt 1.10+)
        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {
            "dbt-osmosis-skip-add-tags": True,  # Config.meta says True
        }
        mock_node.unrendered_config = {}

        # Create column with conflicting setting
        mock_column = Mock()
        mock_column.meta = {
            "dbt-osmosis-skip-add-tags": False,  # Column-level says False (should win)
        }
        mock_node.columns = {"test_column": mock_column}

        # Column-level should take precedence over config.meta
        result = resolver.resolve("skip-add-tags", mock_node, column_name="test_column")
        assert result is False, "Column meta should override node config.meta"

    def test_column_specific_config_inheritance(self) -> None:
        """T058: Test column-specific config inheritance from node-level.

        When a column doesn't have a specific setting defined, it should
        inherit from node-level settings.
        """
        resolver = SettingsResolver()

        # Create a node with a setting defined
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-string-length": True,  # Node-level setting
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Create column WITHOUT the setting (should inherit from node)
        mock_column = Mock()
        mock_column.meta = {}  # Empty - no column-level override
        mock_node.columns = {"my_column": mock_column}

        # Should inherit from node-level
        result = resolver.resolve("string-length", mock_node, column_name="my_column")
        assert result is True, "Column should inherit setting from node meta when not overridden"

    def test_column_fallback_to_node_level_when_no_override(self) -> None:
        """T059: Test column fallback to node-level when no column override.

        This test verifies the full fallback chain:
        1. Check column meta (not found)
        2. Check node meta (found - use this)
        3. Check config.extra (skipped since node meta had it)
        """
        resolver = SettingsResolver()

        # Create a node with setting in multiple sources
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-numeric-precision": True,  # Node-level
        }
        mock_node.config = Mock()
        mock_node.config.extra = {
            "dbt-osmosis-numeric-precision": False,  # Config.extra (lower precedence)
        }

        # Create column without any override
        mock_column = Mock()
        mock_column.meta = {}  # No column-level setting
        mock_node.columns = {"amount": mock_column}

        # Should fall back to node meta (skip config.extra)
        result = resolver.resolve("numeric-precision", mock_node, column_name="amount")
        assert result is True, "Should fallback to node meta when column has no override"

    def test_multiple_columns_with_different_overrides(self) -> None:
        """T060: Test multiple columns with different overrides.

        Different columns on the same node can have different configurations.
        Each column should use its own override when present, otherwise inherit.
        """
        resolver = SettingsResolver()

        # Create a node with a default setting
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-output-to-lower": False,  # Default for columns without override
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Create multiple columns with different configurations
        col1 = Mock()
        col1.meta = {
            "dbt-osmosis-output-to-lower": True,  # Override to True
        }

        col2 = Mock()
        col2.meta = {
            "dbt-osmosis-output-to-lower": False,  # Explicit False (same as node)
        }

        col3 = Mock()
        col3.meta = {}  # No override - should inherit node-level (False)

        col4 = Mock()
        col4.meta = {
            "dbt-osmosis-output-to-lower": True,  # Another override to True
        }

        mock_node.columns = {
            "uppercase_col": col1,
            "lowercase_col": col2,
            "default_col": col3,
            "custom_col": col4,
        }

        # Each column should use its own configuration
        assert resolver.resolve("output-to-lower", mock_node, column_name="uppercase_col") is True
        assert resolver.resolve("output-to-lower", mock_node, column_name="lowercase_col") is False
        assert (
            resolver.resolve("output-to-lower", mock_node, column_name="default_col") is False
        )  # Inherits from node
        assert resolver.resolve("output-to-lower", mock_node, column_name="custom_col") is True

    def test_column_with_mixed_key_formats(self) -> None:
        """T060: Test column overrides work with all key format variants.

        Column-level overrides should support:
        - Direct keys: "output-to-lower"
        - Prefixed kebab: "dbt-osmosis-output-to-lower"
        - Prefixed snake: "dbt_osmosis_output_to_lower"
        - Options object: "dbt-osmosis-options.output-to-lower"
        """
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Test direct key format
        col1 = Mock()
        col1.meta = {"output-to-lower": True}
        mock_node.columns = {"col1": col1}
        assert resolver.resolve("output-to-lower", mock_node, column_name="col1") is True

        # Test prefixed kebab format
        col2 = Mock()
        col2.meta = {"dbt-osmosis-output-to-lower": True}
        mock_node.columns = {"col2": col2}
        assert resolver.resolve("output-to-lower", mock_node, column_name="col2") is True

        # Test prefixed snake format
        col3 = Mock()
        col3.meta = {"dbt_osmosis_output_to_lower": True}
        mock_node.columns = {"col3": col3}
        assert resolver.resolve("output-to-lower", mock_node, column_name="col3") is True

        # Test options object format
        col4 = Mock()
        col4.meta = {"dbt-osmosis-options": {"output-to-lower": True}}
        mock_node.columns = {"col4": col4}
        assert resolver.resolve("output-to-lower", mock_node, column_name="col4") is True

    def test_column_has_method(self) -> None:
        """T060: Test SettingsResolver.has() works correctly with column-level settings.

        The has() method should check if a setting exists at column level.
        """
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Column with setting
        col_with_setting = Mock()
        col_with_setting.meta = {"dbt-osmosis-skip-add-tags": True}
        mock_node.columns = {"my_col": col_with_setting}

        # Should return True for existing setting
        assert resolver.has("skip-add-tags", mock_node, column_name="my_col") is True

        # Should return False for non-existing setting
        assert resolver.has("nonexistent-setting", mock_node, column_name="my_col") is False

        # Column without setting should check node level
        col_without_setting = Mock()
        col_without_setting.meta = {}
        mock_node.columns = {"other_col": col_without_setting}
        mock_node.meta = {"dbt-osmosis-skip-add-tags": True}

        # Should find node-level setting when column doesn't have it
        assert resolver.has("skip-add-tags", mock_node, column_name="other_col") is True

    def test_column_precedence_chain(self) -> None:
        """T060: Test get_precedence_chain() includes column-level source.

        The precedence chain should show column meta as the first source
        when a column_name is provided.
        """
        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-test": "node_value"}
        mock_node.config = Mock()
        mock_node.config.extra = {}

        mock_column = Mock()
        mock_column.meta = {"dbt-osmosis-test": "column_value"}
        mock_node.columns = {"test_col": mock_column}

        # Get precedence chain for column
        chain = resolver.get_precedence_chain("test", mock_node, column_name="test_col")

        # First item should be COLUMN_META
        assert len(chain) > 0
        assert chain[0][0] == ConfigSourceName.COLUMN_META
        assert chain[0][1] == "column_value"

        # Second item should be NODE_META
        assert chain[1][0] == ConfigSourceName.NODE_META
        assert chain[1][1] == "node_value"

    def test_integration_with_demo_duckdb_column_configs(
        self,
        yaml_context: YamlRefactorContext,
    ) -> None:
        """T061: Integration test with demo_duckdb column configs.

        This test verifies that column-level configuration overrides work correctly
        with a real dbt project. It uses the orders model which has:
        - Node-level: dbt-osmosis-output-to-lower: false
        - order_id column: dbt-osmosis-output-to-lower: true (override)
        - customer_id column: no override (inherits from node)
        - amount column: dbt-osmosis-string-length: true (different setting)

        The test validates:
        1. Column-level overrides take precedence
        2. Columns inherit from node when no override
        3. Different columns can have different settings
        """
        resolver = SettingsResolver()

        # Get the orders model from the manifest
        orders_node = None
        for node_id, node in yaml_context.manifest.nodes.items():
            if hasattr(node, "name") and node.name == "orders":
                orders_node = node
                break

        assert orders_node is not None, "Orders model not found in manifest"

        # Test 1: order_id column has override to True
        order_id_output_lower = resolver.resolve(
            "output-to-lower",
            orders_node,
            column_name="order_id",
        )
        assert order_id_output_lower is True, (
            "order_id column should have output-to-lower=True from column meta"
        )

        # Test 2: customer_id column inherits node-level (False)
        customer_id_output_lower = resolver.resolve(
            "output-to-lower",
            orders_node,
            column_name="customer_id",
        )
        assert customer_id_output_lower is False, (
            "customer_id column should inherit output-to-lower=False from node meta"
        )

        # Test 3: amount column has a different setting (string-length)
        amount_string_length = resolver.resolve("string-length", orders_node, column_name="amount")
        assert amount_string_length is True, (
            "amount column should have string-length=True from column meta"
        )

        # Test 4: Verify has() method works correctly
        assert resolver.has("output-to-lower", orders_node, column_name="order_id") is True
        assert resolver.has("output-to-lower", orders_node, column_name="customer_id") is True
        assert resolver.has("string-length", orders_node, column_name="amount") is True


class TestSupplementaryFileValidation:
    """Test suite for SupplementaryFileSource validation.

    Tests cover:
    - Reading dbt-osmosis.yml with valid configuration
    - Precedence over project vars
    - Node-level config precedence over dbt-osmosis.yml
    - Missing file handling (no errors)
    - Invalid YAML syntax error handling
    - Integration with demo_duckdb project
    """

    def test_read_dbt_osmosis_yml(self, tmp_path: Path) -> None:
        """T047: Test reading dbt-osmosis.yml with valid configuration."""
        # Create a valid dbt-osmosis.yml file
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
# dbt-osmosis configuration
yaml_settings:
  map_indent: 2
  sequence_indent: 4

skip-add-tags: true
skip-add-columns: false

dbt-osmosis-options:
  output-to-lower: false
  numeric-precision-and-scale: true
""",
        )

        # Create a mock context with project_dir
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Test reading various key formats
        assert source.get("skip-add-tags") is True
        assert source.get("skip-add-columns") is False
        assert source.get("yaml-settings") == {"map_indent": 2, "sequence_indent": 4}
        assert source.get("output-to-lower") is False
        assert source.get("numeric-precision-and-scale") is True

        # Test missing key
        assert source.get("nonexistent-key") is None

    def test_supplementary_file_precedence_over_project_vars(self, tmp_path: Path) -> None:
        """T048: Test dbt-osmosis.yml precedence over project vars."""
        # Create dbt-osmosis.yml with higher precedence value
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
skip-add-tags: true
output-to-lower: false
""",
        )

        # Create project vars with lower precedence value
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path
        mock_context.project.runtime_cfg.vars = Mock()
        mock_context.project.runtime_cfg.vars.to_dict = Mock(
            return_value={
                "dbt-osmosis": {
                    "skip-add-tags": False,  # Lower precedence
                    "output-to-lower": True,  # Lower precedence
                },
            },
        )

        # Supplementary file should take precedence over project vars
        sup_source = SupplementaryFileSource(mock_context)
        vars_source = ProjectVarsSource(mock_context)

        # Supplementary file values should win
        assert sup_source.get("skip-add-tags") is True
        assert sup_source.get("output-to-lower") is False

        # Project vars have different values (lower precedence)
        assert vars_source.get("skip-add-tags") is False
        assert vars_source.get("output-to-lower") is True

    def test_node_level_config_precedence_over_supplementary_file(self, tmp_path: Path) -> None:
        """T049: Test node-level config precedence over dbt-osmosis.yml."""
        # Create dbt-osmosis.yml with default value
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
skip-add-tags: false
output-to-lower: true
""",
        )

        # Create a mock context
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        # Create a mock node with node-level config that overrides supplementary file
        mock_node = Mock()
        mock_node.meta = {
            "dbt-osmosis-skip-add-tags": True,  # Higher precedence than supplementary file
            "dbt-osmosis-output-to-lower": False,  # Higher precedence
        }
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Supplementary file source
        sup_source = SupplementaryFileSource(mock_context)
        assert sup_source.get("skip-add-tags") is False
        assert sup_source.get("output-to-lower") is True

        # Node meta should override supplementary file
        resolver = SettingsResolver()
        assert resolver.resolve("skip-add-tags", mock_node) is True
        assert resolver.resolve("output-to-lower", mock_node) is False

    def test_missing_dbt_osmosis_yml_handling(self, tmp_path: Path) -> None:
        """T050: Test missing dbt-osmosis.yml handling (no errors)."""
        # Create a mock context without dbt-osmosis.yml
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Should return None for all keys without error
        assert source.get("skip-add-tags") is None
        assert source.get("yaml-settings") is None
        assert source.get("output-to-lower") is None

        # Multiple calls should work fine
        assert source.get("nonexistent") is None
        assert source.get("another-key") is None

    def test_invalid_yaml_error_handling(self, tmp_path: Path) -> None:
        """T051: Test invalid YAML error handling with ConfigurationError."""
        from dbt_osmosis.core.introspection import ConfigurationError

        # Create an invalid YAML file
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
yaml_settings:
  map_indent: 2
    sequence_indent: 4  # Invalid indentation
skip-add-tags: true
""",
        )

        # Create a mock context
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        # Create a new source instance to trigger validation
        source = SupplementaryFileSource(mock_context)

        # Should raise ConfigurationError when trying to read
        with pytest.raises(ConfigurationError) as exc_info:
            source.get("skip-add-tags")

        # Error message should mention the file
        assert str(tmp_path / "dbt-osmosis.yml") in str(exc_info.value)
        assert "invalid" in str(exc_info.value).lower() or "yaml" in str(exc_info.value).lower()

    def test_invalid_yaml_syntax_error_handling(self, tmp_path: Path) -> None:
        """T051: Test invalid YAML syntax error handling."""
        from dbt_osmosis.core.introspection import ConfigurationError

        # Create a file with invalid YAML syntax
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
yaml_settings:
  map_indent: 2
  - invalid YAML sequence in mapping
skip-add-tags: [unclosed list
""",
        )

        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Should raise ConfigurationError
        with pytest.raises(ConfigurationError) as exc_info:
            source.get("skip-add-tags")

        # Verify error mentions the file
        assert "dbt-osmosis.yml" in str(exc_info.value)

    def test_integration_with_demo_duckdb(self) -> None:
        """T052: Integration test with demo_duckdb dbt-osmosis.yml."""
        # Get the demo_duckdb directory
        demo_dir = Path(__file__).parent.parent.parent / "demo_duckdb"
        assert demo_dir.exists(), "demo_duckdb directory should exist"

        config_file = demo_dir / "dbt-osmosis.yml"
        assert config_file.exists(), "demo_duckdb/dbt-osmosis.yml should exist"

        # Create a mock context pointing to demo_duckdb
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = demo_dir

        source = SupplementaryFileSource(mock_context)

        # Test reading actual configuration from demo_duckdb/dbt-osmosis.yml
        # These values are from the actual file
        yaml_settings = source.get("yaml-settings")
        assert yaml_settings is not None
        assert isinstance(yaml_settings, dict)
        assert "map_indent" in yaml_settings
        assert "sequence_indent" in yaml_settings

        # Test other documented settings
        # Note: These are the defaults documented in the file
        assert source.get("skip-add-tags") is not None  # Should be False
        assert source.get("skip-add-columns") is not None  # Should be False
        assert source.get("numeric-precision-and-scale") is not None  # Should be False
        assert source.get("string-length") is not None  # Should be False

    def test_empty_yaml_file_handling(self, tmp_path: Path) -> None:
        """Test handling of empty dbt-osmosis.yml file."""
        # Create an empty YAML file
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text("")

        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Empty file should return None for all keys
        assert source.get("skip-add-tags") is None
        assert source.get("yaml-settings") is None

    def test_yaml_file_with_comments_only(self, tmp_path: Path) -> None:
        """Test handling of dbt-osmosis.yml with only comments."""
        # Create a file with only comments
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
# This is a comment
# Another comment
""",
        )

        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path

        source = SupplementaryFileSource(mock_context)

        # Comments-only file should return None for all keys
        assert source.get("skip-add-tags") is None
        assert source.get("yaml-settings") is None

    def test_supplementary_file_with_project_context_precedence(self, tmp_path: Path) -> None:
        """Test full precedence chain including supplementary file."""
        # Create dbt-osmosis.yml
        config_file = tmp_path / "dbt-osmosis.yml"
        config_file.write_text(
            """
test-setting: "supplementary-file"
""",
        )

        # Create project vars
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path
        mock_context.project.runtime_cfg.vars = Mock()
        mock_context.project.runtime_cfg.vars.to_dict = Mock(
            return_value={"dbt-osmosis": {"test-setting": "project-vars"}},
        )

        # Verify sources individually
        sup_source = SupplementaryFileSource(mock_context)
        vars_source = ProjectVarsSource(mock_context)

        assert sup_source.get("test-setting") == "supplementary-file"
        assert vars_source.get("test-setting") == "project-vars"

        # Create a node with node-level config
        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-test-setting": "node-meta"}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Test precedence: node meta > supplementary file > project vars
        resolver = SettingsResolver()
        result = resolver.resolve("test-setting", mock_node)

        # Node meta should win (highest precedence among existing sources)
        assert result == "node-meta"

        # Context-backed resolver should consult supplementary file before vars
        # when no higher-precedence node setting exists.
        mock_node.meta = {}
        result = resolver.resolve("test-setting", mock_node, context=mock_context)
        assert result == "supplementary-file"

    def test_context_backed_resolver_uses_project_vars_before_fallback(self) -> None:
        """Project vars participate in SettingsResolver precedence with context."""
        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.vars = {"dbt-osmosis": {"skip-add-tags": True}}

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.columns = {}

        resolver = SettingsResolver()
        assert (
            resolver.resolve("skip-add-tags", mock_node, context=mock_context, fallback=False)
            is True
        )

    def test_context_backed_resolver_preserves_explicit_falsey_values(self, tmp_path: Path) -> None:
        """False, 0, and empty string settings must not fall through as missing."""
        (tmp_path / "dbt-osmosis.yml").write_text(
            """
empty-string: ""
zero-value: 0
dbt-osmosis-options:
  skip-add-tags: false
""",
        )

        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.project.runtime_cfg = Mock()
        mock_context.project.runtime_cfg.project_root = tmp_path
        mock_context.project.runtime_cfg.vars = {
            "dbt-osmosis": {
                "empty-string": "from-vars",
                "zero-value": 99,
                "skip-add-tags": True,
            },
        }

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.columns = {}

        resolver = SettingsResolver()

        assert (
            resolver.resolve("skip-add-tags", mock_node, context=mock_context, fallback=True)
            is False
        )
        assert resolver.resolve("zero-value", mock_node, context=mock_context, fallback=42) == 0
        assert (
            resolver.resolve("empty-string", mock_node, context=mock_context, fallback="fallback")
            == ""
        )

    def test_context_settings_tuple_default_does_not_shadow_project_config(self) -> None:
        """Click's empty tuple default should match the dataclass empty list default."""
        mock_context = SimpleNamespace(
            project=SimpleNamespace(
                runtime_cfg=SimpleNamespace(
                    vars={
                        "dbt-osmosis": {
                            "add-inheritance-for-specified-keys": ["policy_tags"],
                        },
                    },
                ),
            ),
            settings=SimpleNamespace(add_inheritance_for_specified_keys=()),
        )
        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.columns = {}

        resolver = SettingsResolver()

        assert resolver.resolve(
            "add-inheritance-for-specified-keys",
            mock_node,
            context=mock_context,
            fallback=(),
        ) == ["policy_tags"]

    def test_context_backed_has_and_precedence_chain_without_node(self, tmp_path: Path) -> None:
        """Debug helpers should expose context-backed project settings."""
        (tmp_path / "dbt-osmosis.yml").write_text("test-setting: supplementary\n")
        mock_context = SimpleNamespace(
            project=SimpleNamespace(
                runtime_cfg=SimpleNamespace(
                    project_root=tmp_path,
                    vars={"dbt-osmosis": {"vars-only": "project-vars"}},
                ),
            ),
        )

        resolver = SettingsResolver()

        assert resolver.has("vars-only", context=mock_context) is True
        chain = resolver.get_precedence_chain("test-setting", context=mock_context)
        assert chain == [
            (ConfigSourceName.CONTEXT_SETTINGS, None),
            (ConfigSourceName.SUPPLEMENTARY_FILE, "supplementary"),
            (ConfigSourceName.PROJECT_VARS, None),
            (ConfigSourceName.FALLBACK, None),
        ]

        settings_context = SimpleNamespace(
            project=mock_context.project,
            settings=SimpleNamespace(skip_add_tags=True),
        )
        assert resolver.has("skip-add-tags", context=settings_context) is True
        settings_chain = resolver.get_precedence_chain("skip-add-tags", context=settings_context)
        assert settings_chain[0] == (ConfigSourceName.CONTEXT_SETTINGS, True)


class TestBackwardCompatibility:
    """Test suite for backward compatibility with _get_setting_for_node.

    Tests cover:
    - _get_setting_for_node delegates to SettingsResolver
    - All existing call sites continue to work
    - Public API contract for SettingsResolver
    - Public API contract for PropertyAccessor
    """

    def test_get_setting_for_node_delegates_to_resolver(self) -> None:
        """T067: Test _get_setting_for_node delegates to SettingsResolver."""
        from dbt_osmosis.core.introspection import _get_setting_for_node

        mock_node = Mock()
        with patch.object(
            SettingsResolver,
            "resolve",
            return_value="resolved-value",
        ) as resolve:
            result = _get_setting_for_node(
                "test-setting",
                mock_node,
                col="test_col",
                fallback="default",
            )

        assert result == "resolved-value"
        resolve.assert_called_once_with(
            "test-setting",
            mock_node,
            column_name="test_col",
            fallback="default",
        )

    def test_get_setting_for_node_uses_resolver_sources(self) -> None:
        """T067: Test _get_setting_for_node reaches config.meta and unrendered_config."""
        from dbt_osmosis.core.introspection import _get_setting_for_node

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {"dbt-osmosis-output-to-lower": True}
        mock_node.unrendered_config = {"dbt_osmosis_prefix": "orders_"}
        mock_node.columns = {}

        assert _get_setting_for_node("output-to-lower", mock_node, fallback=False) is True
        assert _get_setting_for_node("prefix", mock_node, fallback=None) == "orders_"

    def test_get_setting_for_node_with_column(self) -> None:
        """T067: Test _get_setting_for_node works with column-level settings."""
        from dbt_osmosis.core.introspection import _get_setting_for_node

        # Create a mock node with column-level config
        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-output-to-lower": False}
        mock_node.config = Mock()
        mock_node.config.extra = {}

        # Create column with different setting
        mock_column = Mock()
        mock_column.meta = {"dbt-osmosis-output-to-lower": True}
        mock_node.columns = {"test_col": mock_column}

        # Column-level should take precedence
        result = _get_setting_for_node("output-to-lower", mock_node, col="test_col")
        assert result is True

        # Without column, should use node-level
        result = _get_setting_for_node("output-to-lower", mock_node)
        assert result is False

    def test_settings_resolver_public_api_contract(self) -> None:
        """T065: Test SettingsResolver public API contract.

        Verifies that SettingsResolver exposes the expected public methods:
        - resolve(setting_name, node, column_name, fallback)
        - has(setting_name, node, column_name)
        - get_precedence_chain(setting_name, node, column_name)
        """
        from dbt_osmosis.core.introspection import SettingsResolver

        resolver = SettingsResolver()

        # Test resolve method exists and works
        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-test": "value"}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        assert hasattr(resolver, "resolve")
        assert callable(resolver.resolve)
        result = resolver.resolve("test", mock_node)
        assert result == "value"

        # Test has method exists and works
        assert hasattr(resolver, "has")
        assert callable(resolver.has)
        assert resolver.has("test", mock_node) is True
        assert resolver.has("nonexistent", mock_node) is False

        # Test get_precedence_chain method exists and works
        assert hasattr(resolver, "get_precedence_chain")
        assert callable(resolver.get_precedence_chain)
        chain = resolver.get_precedence_chain("test", mock_node)
        assert isinstance(chain, list)

    def test_property_accessor_public_api_contract(self) -> None:
        """T066: Test PropertyAccessor public API contract.

        Verifies that PropertyAccessor exposes the expected public methods:
        - get(property_key, node, column_name, source)
        - get_description(node, column_name, source)
        - get_meta(node, column_name, source, meta_key)
        - has_property(property_key, node, column_name)
        """
        from dbt_osmosis.core.introspection import PropertyAccessor, PropertySource

        mock_context = Mock()
        mock_context.project = Mock()
        mock_context.yaml_handler = Mock()
        mock_context.yaml_handler_lock = Mock()

        accessor = PropertyAccessor(context=mock_context)

        # Test get method exists and works
        assert hasattr(accessor, "get")
        assert callable(accessor.get)

        mock_node = Mock()
        mock_node.description = "Test description"
        mock_node.tags = ["tag1", "tag2"]
        mock_node.meta = {}

        result = accessor.get("description", mock_node, source=PropertySource.MANIFEST)
        assert result == "Test description"

        # Test get_description convenience method
        assert hasattr(accessor, "get_description")
        assert callable(accessor.get_description)
        result = accessor.get_description(mock_node, source=PropertySource.MANIFEST)
        assert result == "Test description"

        # Test get_meta convenience method
        assert hasattr(accessor, "get_meta")
        assert callable(accessor.get_meta)
        result = accessor.get_meta(mock_node, source=PropertySource.MANIFEST)
        assert result == {}

        # Test has_property method
        assert hasattr(accessor, "has_property")
        assert callable(accessor.has_property)
        assert accessor.has_property("description", mock_node) is True
        # Note: has_property checks if a property exists in manifest OR YAML
        # Since Mock objects return truthy values for any attribute access,
        # we can't reliably test for nonexistent properties with a basic Mock
        # Instead, we'll verify the method is callable and works for known properties

    def test_settings_resolver_respects_precedence(self) -> None:
        """T065: Test SettingsResolver respects configuration precedence.

        Precedence order (highest to lowest):
        1. Column meta
        2. Node meta
        3. Node config.extra
        4. Node config.meta (dbt 1.10+)
        5. Node unrendered_config (dbt 1.10+)
        6. Project vars
        7. Supplementary file
        8. Fallback
        """
        from dbt_osmosis.core.introspection import SettingsResolver

        resolver = SettingsResolver()

        # Create node with multiple config sources
        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-test": "meta"}
        mock_node.config = Mock()
        mock_node.config.extra = {"dbt-osmosis-test": "extra"}
        mock_node.config.meta = {"dbt-osmosis-test": "config_meta"}
        mock_node.unrendered_config = {"dbt-osmosis-test": "unrendered"}
        mock_node.columns = {}

        # Node meta should have highest precedence among these
        result = resolver.resolve("test", mock_node)
        assert result == "meta"

    def test_settings_resolver_key_normalization(self) -> None:
        """T065: Test SettingsResolver handles kebab-case and snake_case keys."""
        from dbt_osmosis.core.introspection import SettingsResolver

        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {"dbt-osmosis-test-key": "value"}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # kebab-case input
        assert resolver.resolve("test-key", mock_node) == "value"
        # snake_case input (normalized)
        assert resolver.resolve("test_key", mock_node) == "value"

    def test_settings_resolver_fallback_value(self) -> None:
        """T065: Test SettingsResolver uses fallback when setting not found."""
        from dbt_osmosis.core.introspection import SettingsResolver

        resolver = SettingsResolver()

        mock_node = Mock()
        mock_node.meta = {}
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.columns = {}

        # Should return fallback when setting not found
        assert resolver.resolve("nonexistent", mock_node, fallback="default") == "default"
        assert resolver.resolve("nonexistent", mock_node) is None

    def test_property_accessor_source_parameter(self) -> None:
        """T066: Test PropertyAccessor accepts different source parameters."""
        from dbt_osmosis.core.introspection import PropertyAccessor, PropertySource

        mock_context = Mock()
        accessor = PropertyAccessor(context=mock_context)

        mock_node = Mock()
        mock_node.description = "Test"

        # Test with enum
        result = accessor.get_description(mock_node, source=PropertySource.MANIFEST)
        assert result == "Test"

        # Test with string
        result = accessor.get_description(mock_node, source="manifest")
        assert result == "Test"

        # Test with auto
        result = accessor.get_description(mock_node, source="auto")
        assert result == "Test"

    def test_exports_available_from_osmosis(self) -> None:
        """T069/T070: Test SettingsResolver and PropertyAccessor are exported from osmosis module."""
        # Test that we can import from osmosis
        from dbt_osmosis.core.osmosis import PropertyAccessor, SettingsResolver

        # Verify they're the correct classes
        assert SettingsResolver is not None
        assert PropertyAccessor is not None

        # Test basic functionality
        resolver = SettingsResolver()
        assert hasattr(resolver, "resolve")

    def test_exports_available_from_introspection(self) -> None:
        """T069: Test SettingsResolver and PropertyAccessor are exported from introspection module."""
        from dbt_osmosis.core.introspection import (
            ConfigSourceName,
            PropertyAccessor,
            PropertySource,
            SettingsResolver,
        )

        # Verify all expected exports are available
        assert SettingsResolver is not None
        assert PropertyAccessor is not None
        assert ConfigSourceName is not None
        assert PropertySource is not None
