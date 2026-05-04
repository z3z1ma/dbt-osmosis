"""Tests for PropertyAccessor class.

This module tests the unified property access system that provides a single
interface for accessing model properties from multiple sources:
- Manifest (rendered jinja, pre-compiled by dbt)
- YAML files (unrendered jinja, raw templates like {{ doc(...) }})
- Database introspection (runtime schema information)

The PropertyAccessor enables the unrendered jinja feature (doc blocks)
by allowing users to choose between rendered and unrendered property values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# Import will work once PropertyAccessor is implemented
from dbt_osmosis.core.introspection import PropertyAccessor


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


class MockColumnConfig:
    """Mock column config for dbt 1.10+ column properties."""

    def __init__(
        self,
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.meta = meta or {}
        self.tags = tags or []


class MockColumn:
    """Mock column for testing property access."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        meta: dict[str, Any] | None = None,
        data_type: str | None = None,
        tags: list[str] | None = None,
        config_meta: dict[str, Any] | None = None,
        config_tags: list[str] | None = None,
    ) -> None:
        """Initialize a mock column.

        Args:
            name: Column name
            description: Optional column description
            meta: Optional metadata dictionary
            data_type: Optional data type
            tags: Optional list of tags

        """
        self.name = name
        self.description = description
        self.meta = meta or {}
        self.data_type = data_type
        self.tags = tags or []
        self.config = MockColumnConfig(config_meta, config_tags)


class MockNode:
    """Mock node for testing property access."""

    def __init__(
        self,
        unique_id: str = "model.test.my_model",
        description: str | None = None,
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        columns: dict[str, MockColumn] | None = None,
        raw_code: str | None = None,
        patch_path: str | None = "models/my_model.yml",
        resource_type: str = "model",
    ) -> None:
        """Initialize a mock node.

        Args:
            unique_id: Unique identifier for the node
            description: Optional node description
            meta: Optional metadata dictionary
            tags: Optional list of tags
            columns: Optional dictionary of columns
            raw_code: Optional raw SQL code
            patch_path: Optional YAML file path
            resource_type: Optional resource type (model, source, seed, etc.)

        """
        self.unique_id = unique_id
        self.description = description
        self.meta = meta or {}
        self.tags = tags or []
        self.columns = columns or {}
        self.raw_code = raw_code
        self.patch_path = patch_path
        self.resource_type = resource_type


@pytest.fixture
def sample_yaml_file(tmp_path: Path) -> Path:
    """Create a sample YAML file with model definitions.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to the created YAML file

    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    yaml_file = models_dir / "my_model.yml"
    yaml_file.write_text(
        """
version: 2

models:
  - name: my_model
    description: |
      This model uses {{ doc('my_doc_block') }} for documentation.
    columns:
      - name: id
        description: Unique identifier using {{ doc('id_doc') }}
        meta:
          dbt-osmosis:
            output-to-lower: true
      - name: name
        description: Customer name
      - name: created_at
        description: Timestamp with {% docs timestamp_docs %}special formatting{% enddocs %}
""",
    )

    return yaml_file


@pytest.fixture
def sample_node_with_unrendered() -> MockNode:
    """Create a sample node with unrendered jinja in descriptions.

    Returns:
        MockNode with unrendered jinja templates in descriptions

    """
    return MockNode(
        unique_id="model.test.my_model",
        description="This model uses {{ doc('my_doc_block') }} for documentation.",
        tags=["nightly", "piña"],
        columns={
            "id": MockColumn(
                "id",
                description="Unique identifier using {{ doc('id_doc') }}",
                meta={"dbt-osmosis": {"output-to-lower": True}},
            ),
            "name": MockColumn("name", description="Customer name"),
            "created_at": MockColumn(
                "created_at",
                description="Timestamp with {% docs timestamp_docs %}special formatting{% enddocs %}",
            ),
        },
    )


@pytest.fixture
def sample_node_rendered() -> MockNode:
    """Create a sample node with rendered descriptions (from manifest).

    Returns:
        MockNode with pre-rendered descriptions as they appear in manifest

    """
    return MockNode(
        unique_id="model.test.my_model",
        description="This model uses comprehensive documentation for documentation.",
        tags=["nightly", "piña"],
        columns={
            "id": MockColumn(
                "id",
                description="Unique identifier using unique identifier documentation",
                meta={"dbt-osmosis": {"output-to-lower": True}},
            ),
            "name": MockColumn("name", description="Customer name"),
            "created_at": MockColumn(
                "created_at",
                description="Timestamp with ISO 8601 formatted timestamp",
            ),
        },
    )


@pytest.fixture
def mock_context() -> Mock:
    """Create a mock YamlRefactorContext.

    Returns:
        Mock context with necessary attributes

    """
    context = Mock()
    context.project = Mock()
    context.project.manifest = Mock()
    context.yaml_handler = Mock()
    context.yaml_handler_lock = Mock()
    return context


class TestPropertyAccessor:
    """Test suite for PropertyAccessor class.

    Tests cover:
    - Accessing properties from manifest (rendered)
    - Accessing properties from YAML (unrendered)
    - Source preference (prefer_unrendered flag)
    - Fallback behavior when sources are missing
    - Column-level property access
    - Node-level property access
    - Handling of missing properties
    """

    def test_get_property_from_manifest(
        self,
        sample_node_rendered: MockNode,
        mock_context: Mock,
    ) -> None:
        """Test getting a property from the manifest (rendered)."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("description", sample_node_rendered, source="manifest")
        assert "comprehensive documentation" in result

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_get_property_from_yaml(
        self,
        mock_get_yaml,
        sample_node_with_unrendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test getting a property from YAML (unrendered)."""
        # Mock the YAML reading to return unrendered content
        mock_get_yaml.return_value = {
            "name": "my_model",
            "description": "This model uses {{ doc('my_doc_block') }} for documentation.",
            "columns": [
                {
                    "name": "id",
                    "description": "Unique identifier using {{ doc('id_doc') }}",
                    "meta": {"dbt-osmosis": {"output-to-lower": True}},
                },
            ],
        }
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("description", sample_node_with_unrendered, source="yaml")
        assert "{{ doc('my_doc_block') }}" in result

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_prefer_unrendered_true(
        self,
        mock_get_yaml,
        sample_node_rendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test that source='auto' with unrendered jinja prefers YAML."""
        # Mock the YAML reading to return unrendered content
        mock_get_yaml.return_value = {
            "name": "my_model",
            "description": "This model uses {{ doc('my_doc_block') }} for documentation.",
        }
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("description", sample_node_rendered, source="auto")
        # Should prefer YAML when unrendered jinja is detected
        assert "{{ doc" in result

    def test_prefer_unrendered_false(
        self,
        sample_node_rendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test that source='manifest' always uses manifest."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("description", sample_node_rendered, source="manifest")
        assert "comprehensive documentation" in result
        assert "{{ doc" not in result

    def test_get_column_property(
        self,
        sample_node_rendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test getting a column-level property."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get(
            "description",
            sample_node_rendered,
            column_name="id",
            source="manifest",
        )
        assert "unique identifier documentation" in result

    def test_get_tags(self, sample_node_rendered: MockNode, mock_context: Mock) -> None:
        """Test getting tags from a node."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("tags", sample_node_rendered, source="manifest")
        assert "nightly" in result
        assert "piña" in result

    def test_get_meta(self, sample_node_rendered: MockNode, mock_context: Mock) -> None:
        """Test getting meta from a node."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get_meta(sample_node_rendered, column_name="id", source="manifest")
        assert result is not None
        assert "output-to-lower" in str(result).lower() or "dbt-osmosis" in str(result).lower()

    def test_get_column_meta_merges_legacy_and_config_meta(
        self,
        mock_context: Mock,
    ) -> None:
        """Column config.meta is effective meta and wins conflicts over legacy meta."""
        node = MockNode(
            columns={
                "id": MockColumn(
                    "id",
                    meta={"legacy": True, "shared": "legacy"},
                    config_meta={"classification": "restricted", "shared": "config"},
                ),
            },
        )
        accessor = PropertyAccessor(context=mock_context)

        result = accessor.get_meta(node, column_name="id", source="manifest")

        assert result == {
            "legacy": True,
            "classification": "restricted",
            "shared": "config",
        }

    def test_get_column_tags_merges_legacy_and_config_tags(
        self,
        mock_context: Mock,
    ) -> None:
        """Column config.tags are effective tags after legacy tags with duplicates removed."""
        node = MockNode(
            columns={
                "id": MockColumn(
                    "id",
                    tags=["legacy", "shared"],
                    config_tags=["config", "shared"],
                ),
            },
        )
        accessor = PropertyAccessor(context=mock_context)

        result = accessor.get("tags", node, column_name="id", source="manifest")

        assert result == ["legacy", "shared", "config"]

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_get_yaml_column_config_meta_and_tags_are_effective(
        self,
        mock_get_yaml,
        mock_context: Mock,
    ) -> None:
        """YAML column config.meta and config.tags participate in effective values."""
        mock_get_yaml.return_value = {
            "name": "my_model",
            "columns": [
                {
                    "name": "id",
                    "meta": {
                        "legacy": True,
                        "shared": "legacy",
                        "dbt-osmosis-options": {
                            "legacy_only": True,
                            "shared_option": "legacy",
                        },
                    },
                    "tags": ["legacy", "shared"],
                    "config": {
                        "meta": {
                            "classification": "restricted",
                            "shared": "config",
                            "dbt-osmosis-options": {
                                "config_only": True,
                                "shared_option": "config",
                            },
                        },
                        "tags": ["config", "shared"],
                    },
                },
            ],
        }
        node = MockNode(
            columns={
                "id": MockColumn(
                    "id",
                    meta={"manifest": True},
                    tags=["manifest"],
                ),
            },
        )
        accessor = PropertyAccessor(context=mock_context)

        meta_result = accessor.get_meta(node, column_name="id", source="yaml")
        tags_result = accessor.get("tags", node, column_name="id", source="yaml")

        assert meta_result == {
            "legacy": True,
            "classification": "restricted",
            "shared": "config",
            "dbt-osmosis-options": {
                "legacy_only": True,
                "config_only": True,
                "shared_option": "config",
            },
        }
        assert tags_result == ["legacy", "shared", "config"]

    def test_yaml_source_reads_version_level_column_properties(
        self,
        yaml_context,
        fresh_caches,
    ) -> None:
        """YAML source should read selected versions[].columns and top-level fallbacks."""
        from dbt_osmosis.core.schema.reader import _read_yaml

        manifest = yaml_context.project.manifest
        node = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
        project_dir = Path(yaml_context.project.runtime_cfg.project_root)
        path = project_dir.joinpath(node.patch_path.split("://")[-1])
        yaml_doc = _read_yaml(yaml_context.yaml_handler, yaml_context.yaml_handler_lock, path)
        model = next(model for model in yaml_doc["models"] if model["name"] == node.name)
        model["description"] = "Top-level stg_customers fallback description"
        model["tags"] = ["top_level_tag"]
        model["meta"] = {"owner": "analytics"}
        version = next(version for version in model["versions"] if version["v"] == 2)
        version["columns"].insert(0, {"include": "*"})
        id_column = next(column for column in version["columns"] if column.get("name") == "id")
        id_column["description"] = "Versioned id via {{ doc('versioned_customer_id') }}"
        id_column["meta"] = {"classification": "versioned"}
        id_column["tags"] = ["versioned_id"]
        id_column["tests"] = ["unique", "not_null"]

        accessor = PropertyAccessor(context=yaml_context)

        assert accessor.get_description(node, source="yaml") == (
            "Top-level stg_customers fallback description"
        )
        assert accessor.get("tags", node, source="yaml") == ["top_level_tag"]
        assert accessor.get_meta(node, source="yaml") == {"owner": "analytics"}
        assert accessor.get_description(node, column_name="id", source="yaml") == (
            "Versioned id via {{ doc('versioned_customer_id') }}"
        )
        assert accessor.get_meta(node, column_name="id", source="yaml") == {
            "classification": "versioned"
        }
        assert accessor.get("tags", node, column_name="id", source="yaml") == ["versioned_id"]
        assert accessor.get("tests", node, column_name="id", source="yaml") == [
            "unique",
            "not_null",
        ]

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_yaml_column_without_meta_or_tags_falls_back_to_manifest(
        self,
        mock_get_yaml,
        mock_context: Mock,
    ) -> None:
        """YAML source falls back when a YAML column lacks meta/tags fields."""
        mock_get_yaml.return_value = {
            "name": "my_model",
            "columns": [
                {"name": "id", "description": "YAML description without governance fields"},
            ],
        }
        node = MockNode(
            columns={
                "id": MockColumn(
                    "id",
                    meta={"legacy": True},
                    tags=["legacy"],
                    config_meta={"classification": "restricted"},
                    config_tags=["config"],
                ),
            },
        )
        accessor = PropertyAccessor(context=mock_context)

        meta_result = accessor.get_meta(node, column_name="id", source="yaml")
        tags_result = accessor.get("tags", node, column_name="id", source="yaml")

        assert meta_result == {"legacy": True, "classification": "restricted"}
        assert tags_result == ["legacy", "config"]

    def test_get_data_type(self, mock_context: Mock) -> None:
        """Test getting data type for a column."""
        node = MockNode(
            columns={
                "id": MockColumn("id", data_type="integer"),
                "name": MockColumn("name", data_type="varchar"),
            },
        )
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("data_type", node, column_name="id", source="manifest")
        assert result == "integer"

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_missing_yaml_file(
        self,
        mock_get_yaml,
        sample_node_rendered: MockNode,
        mock_context: Mock,
    ) -> None:
        """Test behavior when YAML file doesn't exist (fallback to manifest)."""
        # Mock missing YAML file
        mock_get_yaml.return_value = None
        accessor = PropertyAccessor(context=mock_context)
        # Should fall back to manifest
        result = accessor.get("description", sample_node_rendered, source="yaml")
        # Should still get manifest value as fallback
        assert result is not None

    def test_missing_property_returns_none(self, mock_context: Mock) -> None:
        """Test that missing properties return None."""
        node = MockNode(description="Existing description")
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("nonexistent_property", node, source="manifest")
        assert result is None

    def test_ephemeral_model_handling(self, mock_context: Mock) -> None:
        """Test handling of ephemeral models that may not have YAML files."""
        ephemeral_node = MockNode(
            unique_id="model.test.ephemeral_model",
            description="Ephemeral model",
            patch_path=None,  # No YAML file for ephemeral models
        )
        accessor = PropertyAccessor(context=mock_context)
        # Should handle gracefully
        result = accessor.get("description", ephemeral_node, source="yaml")
        # Should fall back to manifest or return None
        assert result is None or result == "Ephemeral model"

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_column_not_in_yaml(
        self,
        mock_get_yaml,
        sample_node_rendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test accessing a column that exists in manifest but not in YAML."""
        # Add a column that doesn't exist in YAML
        sample_node_rendered.columns["extra_column"] = MockColumn(
            "extra_column",
            description="Extra",
        )
        # Mock YAML that doesn't have the extra_column
        mock_get_yaml.return_value = {
            "name": "my_model",
            "columns": [
                {"name": "id", "description": "ID column"},
            ],
        }
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get(
            "description",
            sample_node_rendered,
            column_name="extra_column",
            source="yaml",
        )
        # Should fall back to manifest if column not in YAML
        assert result is not None

    @patch("dbt_osmosis.core.inheritance._get_node_yaml")
    def test_unrendered_jinja_preservation(
        self,
        mock_get_yaml,
        sample_node_with_unrendered: MockNode,
        sample_yaml_file: Path,
        mock_context: Mock,
    ) -> None:
        """Test that unrendered jinja templates are preserved."""
        mock_get_yaml.return_value = {
            "name": "my_model",
            "columns": [
                {
                    "name": "id",
                    "description": "Unique identifier using {{ doc('id_doc') }}",
                },
            ],
        }
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get(
            "description",
            sample_node_with_unrendered,
            column_name="id",
            source="yaml",
        )
        assert "{{ doc('id_doc') }}" in result

    def test_multiple_doc_blocks(self, mock_context: Mock) -> None:
        """Test handling of multiple doc blocks in a single description."""
        node = MockNode(description="Start {{ doc('first') }} middle {{ doc('second') }} end")
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get("description", node, source="manifest")
        # Manifest should have rendered version
        assert "{{ doc" not in result or result == node.description

    def test_get_description_convenience(
        self,
        sample_node_rendered: MockNode,
        mock_context: Mock,
    ) -> None:
        """Test get_description() convenience method."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.get_description(sample_node_rendered, source="manifest")
        assert "comprehensive documentation" in result

    def test_has_property_true(self, sample_node_rendered: MockNode, mock_context: Mock) -> None:
        """Test has_property() returns True for existing properties."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.has_property("description", sample_node_rendered)
        assert result is True

    def test_has_property_false(self, sample_node_rendered: MockNode, mock_context: Mock) -> None:
        """Test has_property() returns False for missing properties."""
        accessor = PropertyAccessor(context=mock_context)
        result = accessor.has_property("nonexistent", sample_node_rendered)
        assert result is False

    def test_invalid_source_raises_error(
        self,
        sample_node_rendered: MockNode,
        mock_context: Mock,
    ) -> None:
        """Test that invalid source raises ValueError."""
        accessor = PropertyAccessor(context=mock_context)
        with pytest.raises(ValueError, match="Invalid source"):
            accessor.get("description", sample_node_rendered, source="invalid_source")


class TestPropertyAccessorIntegration:
    """Integration tests for PropertyAccessor with real dbt structures.

    These tests use the demo_duckdb project to verify end-to-end functionality.
    """

    @pytest.mark.skip(reason="Demo project fixture not yet set up")
    def test_access_with_demo_project(self, demo_project: Path) -> None:
        """Test property accessor with the demo_duckdb project."""

    @pytest.mark.skip(reason="Demo project fixture not yet set up")
    def test_source_definitions(self, demo_project: Path) -> None:
        """Test accessing properties from source definitions."""

    @pytest.mark.skip(reason="Demo project fixture not yet set up")
    def test_seed_definitions(self, demo_project: Path) -> None:
        """Test accessing properties from seed definitions."""
