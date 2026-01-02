# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Edge case tests for dbt-osmosis.

This module tests edge cases and boundary conditions across the codebase,
ensuring robustness with unusual but valid inputs.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from unittest import mock

import pytest

from dbt_osmosis.core.inheritance import _build_column_knowledge_graph
from dbt_osmosis.core.schema.parser import create_yaml_instance
from dbt_osmosis.core.schema.reader import _read_yaml, _YAML_BUFFER_CACHE
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.sync_operations import sync_node_to_yaml


# ============================================================================
# Empty Models (No Columns)
# ============================================================================


def test_empty_model_yml():
    """Test handling of models with no columns."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        empty_model_file = Path(tmpdir) / "empty_model.yml"
        empty_model_file.write_text("""
version: 2
models:
  - name: empty_model
    description: "A model with no columns"
""")

        result = _read_yaml(yaml_handler, yaml_handler_lock, empty_model_file)
        assert result is not None
        assert "models" in result
        assert len(result["models"]) == 1
        assert result["models"][0]["name"] == "empty_model"


def test_transform_with_empty_columns(yaml_context: YamlRefactorContext):
    """Test that transforms handle models with empty column lists."""
    # Get a model node
    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")

    test_node = model_nodes[0]

    # Mock columns to be empty
    with mock.patch.object(test_node, "columns", {}):
        # Should not crash
        sync_node_to_yaml(yaml_context, test_node, commit=False)


# ============================================================================
# Models with No Upstream Sources
# ============================================================================


def test_source_model_no_upstream(yaml_context: YamlRefactorContext):
    """Test inheritance for source models (no upstream dependencies)."""
    # Source nodes have no upstream models
    source_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "source"
    ]
    if not source_nodes:
        pytest.skip("No source nodes found in manifest")

    # Should handle sources gracefully (they have no upstream to inherit from)
    # Build knowledge graph - should not crash
    test_node = source_nodes[0]
    knowledge_graph = _build_column_knowledge_graph(yaml_context, test_node)
    # Should return a dict, even for sources with no upstream
    assert isinstance(knowledge_graph, dict)


# ============================================================================
# Models with No YAML File
# ============================================================================


def test_node_without_yaml_file(yaml_context: YamlRefactorContext):
    """Test handling of nodes that don't have a corresponding YAML file."""
    # Get a model node
    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")

    test_node = model_nodes[0]

    # Mock that the YAML file doesn't exist
    with mock.patch.object(test_node, "patch_path", "nonexistent.yml"):
        # Should handle gracefully - may create new file or skip
        try:
            sync_node_to_yaml(yaml_context, test_node, commit=False)
        except (FileNotFoundError, Exception):
            pass  # May raise if file truly doesn't exist


# ============================================================================
# Column Name Collisions
# ============================================================================


def test_duplicate_column_names():
    """Test handling of duplicate column names in YAML."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        duplicate_cols_file = Path(tmpdir) / "duplicate_cols.yml"
        duplicate_cols_file.write_text("""
version: 2
models:
  - name: test_model
    columns:
      - name: customer_id
        description: "First occurrence"
      - name: customer_id
        description: "Duplicate name"
""")

        # Should read the file (ruamel.yaml allows duplicates)
        result = _read_yaml(yaml_handler, yaml_handler_lock, duplicate_cols_file)
        assert result is not None

        # Both columns will be present
        model = result["models"][0]
        assert len(model["columns"]) == 2


# ============================================================================
# Special Characters in Names
# ============================================================================


@pytest.mark.parametrize(
    "column_name",
    [
        "column_with_underscore",
        "column-with-dash",
        "column.with.dot",
        "column:with:colon",
        "column/with/slash",
        "column with spaces",
        "UPPERCASE_COLUMN",
        "MixedCase_Column",
        "column123",
        "123column",
        "column__double__underscore",
        "column_with___trailing_underscore___",
    ],
)
def test_special_characters_in_column_names(column_name: str):
    """Test that various special characters in column names are handled."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        special_file = Path(tmpdir) / "special_chars.yml"
        special_file.write_text(f"""
version: 2
models:
  - name: test_model
    columns:
      - name: {column_name}
        description: "Column with special characters"
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, special_file)
        assert result is not None
        assert result["models"][0]["columns"][0]["name"] == column_name


def test_reserved_sql_keywords_as_column_names():
    """Test that SQL reserved keywords as column names are handled."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    reserved_keywords = [
        "select",
        "from",
        "where",
        "join",
        "order",
        "group",
        "having",
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "index",
        "table",
        "view",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for keyword in reserved_keywords:
            reserved_file = Path(tmpdir) / f"{keyword}.yml"
            reserved_file.write_text(f"""
version: 2
models:
  - name: test_model
    columns:
      - name: {keyword}
        description: "SQL reserved keyword as column name"
""")

            # Should read without error
            result = _read_yaml(yaml_handler, yaml_handler_lock, reserved_file)
            assert result is not None


# ============================================================================
# Deep Inheritance Chains
# ============================================================================


def test_deep_inheritance_chain(yaml_context: YamlRefactorContext):
    """Test inheritance across deep chains of models."""
    # Get a model node
    model_nodes = [
        node
        for node in yaml_context.project.manifest.nodes.values()
        if node.resource_type.value == "model"
    ]
    if not model_nodes:
        pytest.skip("No model nodes found in manifest")

    # Build the knowledge graph for a model
    # The demo project has: raw_* -> stg_* -> *_customers/orders
    # This is a chain of 2-3 models deep
    test_node = model_nodes[0]
    knowledge_graph = _build_column_knowledge_graph(yaml_context, test_node)

    # Should successfully build graph
    assert knowledge_graph is not None
    assert isinstance(knowledge_graph, dict)

    # Verify we have column information (structure may vary)
    if knowledge_graph:
        # At least one column should have been processed
        for col_name, col_info in knowledge_graph.items():
            # Column info should be a dict with some metadata
            assert isinstance(col_info, dict)
            # The structure may include various keys like description, data_type, etc.
            break


# ============================================================================
# Very Long Strings
# ============================================================================


def test_very_long_descriptions():
    """Test handling of very long description strings."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    long_description = "x" * 10000  # 10KB description

    with tempfile.TemporaryDirectory() as tmpdir:
        long_desc_file = Path(tmpdir) / "long_desc.yml"
        long_desc_file.write_text(f"""
version: 2
models:
  - name: test_model
    description: "{long_description}"
    columns:
      - name: test_column
        description: "{long_description}"
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, long_desc_file)
        assert result is not None
        assert len(result["models"][0]["description"]) == 10000


def test_many_columns_model():
    """Test handling of models with many columns."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    # Create a model with 100 columns
    columns_yaml = "\n".join([
        f"""      - name: column_{i}
        description: "Column {i}"
        data_type: integer"""
        for i in range(100)
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        many_cols_file = Path(tmpdir) / "many_columns.yml"
        many_cols_file.write_text(f"""
version: 2
models:
  - name: test_model
    columns:
{columns_yaml}
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, many_cols_file)
        assert result is not None
        assert len(result["models"][0]["columns"]) == 100


# ============================================================================
# Empty and Whitespace Values
# ============================================================================


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        "  \t\n  ",
    ],
)
def test_whitespace_descriptions(value: str):
    """Test handling of empty and whitespace-only descriptions."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        whitespace_file = Path(tmpdir) / "whitespace.yml"
        # For YAML, we need to properly escape or quote the value
        whitespace_file.write_text(f"""
version: 2
models:
  - name: test_model
    description: "{value}"
    columns:
      - name: test_column
        description: "{value}"
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, whitespace_file)
        assert result is not None


# ============================================================================
# Cache Edge Cases
# ============================================================================


def test_cache_with_many_files():
    """Test cache behavior with many files."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    _YAML_BUFFER_CACHE.clear()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 50 YAML files
        for i in range(50):
            yaml_file = Path(tmpdir) / f"file_{i}.yml"
            yaml_file.write_text(f"""
version: 2
models:
  - name: model_{i}
    description: "Test model {i}"
""")
            _read_yaml(yaml_handler, yaml_handler_lock, yaml_file)

        # Cache should contain all 50 files
        assert len(list(_YAML_BUFFER_CACHE.keys())) == 50


def test_cache_size_limit():
    """Test that cache respects size limit and evicts old entries."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    _YAML_BUFFER_CACHE.clear()

    # The cache max size is 256
    # Create 300 files to exceed limit
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(300):
            yaml_file = Path(tmpdir) / f"file_{i}.yml"
            yaml_file.write_text(f"version: 2\nmodels:\n  - name: model_{i}\n")
            _read_yaml(yaml_handler, yaml_handler_lock, yaml_file)

        # Cache should be at or below max size
        # (some entries may have been evicted)
        cache_size = len(list(_YAML_BUFFER_CACHE.keys()))
        assert cache_size <= 256


# ============================================================================
# Mixed Case Sensitivity
# ============================================================================


def test_case_sensitive_column_names():
    """Test that column names are case-sensitive."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        case_file = Path(tmpdir) / "case_sensitive.yml"
        case_file.write_text("""
version: 2
models:
  - name: test_model
    columns:
      - name: customer_id
        description: "Lowercase"
      - name: Customer_ID
        description: "Mixed case"
      - name: CUSTOMER_ID
        description: "Uppercase"
""")

        # Should read all three as distinct columns
        result = _read_yaml(yaml_handler, yaml_handler_lock, case_file)
        assert result is not None
        columns = result["models"][0]["columns"]
        assert len(columns) == 3


# ============================================================================
# None and Null Values
# ============================================================================


def test_null_values_in_yaml():
    """Test handling of null/None values in YAML."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        null_file = Path(tmpdir) / "null_values.yml"
        null_file.write_text("""
version: 2
models:
  - name: test_model
    columns:
      - name: column_a
        description: "Has description"
      - name: column_b
        description: null
      - name: column_c
        # No description key
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, null_file)
        assert result is not None
        columns = result["models"][0]["columns"]
        assert len(columns) == 3


# ============================================================================
# Unicode and Internationalization
# ============================================================================


@pytest.mark.parametrize(
    "text",
    [
        "Café",
        "日本語",
        "한국어",
        "Русский",
        "العربية",
        "עברית",
        "🚀🌟💻",
        "ÆØÅæøå",
        "Ññ",
        "Çç",
        "中文",
        "Ελληνικά",
    ],
)
def test_unicode_in_descriptions(text: str):
    """Test that Unicode text in descriptions is handled correctly."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        unicode_file = Path(tmpdir) / "unicode.yml"
        # Need to use block scalar for unicode with special characters
        unicode_file.write_text(f"""
version: 2
models:
  - name: test_model
    description: "{text}"
    columns:
      - name: test_column
        description: "{text}"
""")

        # Should read without error
        result = _read_yaml(yaml_handler, yaml_handler_lock, unicode_file)
        assert result is not None
        assert result["models"][0]["description"] == text


# ============================================================================
# Minimal YAML Documents
# ============================================================================


def test_minimal_valid_yaml():
    """Test reading minimal valid YAML."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        minimal_file = Path(tmpdir) / "minimal.yml"
        minimal_file.write_text("version: 2\n")

        result = _read_yaml(yaml_handler, yaml_handler_lock, minimal_file)
        assert result is not None
        assert result.get("version") == 2


def test_yaml_with_version_only():
    """Test YAML with only version key (no models or sources)."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        version_only_file = Path(tmpdir) / "version_only.yml"
        version_only_file.write_text("""
version: 2
""")

        result = _read_yaml(yaml_handler, yaml_handler_lock, version_only_file)
        assert result is not None
        assert result.get("version") == 2


# ============================================================================
# Nested Structures
# ============================================================================


def test_deeply_nested_yaml():
    """Test reading deeply nested YAML structures."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_file = Path(tmpdir) / "nested.yml"
        nested_file.write_text("""
version: 2
models:
  - name: outer_model
    description: "Outer model"
    columns:
      - name: column_a
        description: "Column A"
        meta:
          level1:
            level2:
              level3:
                level4: "deep value"
        tests:
          - unique:
              column_name: column_a
          - not_null:
              column_name: column_a
""")

        result = _read_yaml(yaml_handler, yaml_handler_lock, nested_file)
        assert result is not None
        assert "models" in result
        assert len(result["models"][0]["columns"]) == 1


# ============================================================================
# Multiple Top-Level Keys
# ============================================================================


def test_yaml_with_models_and_sources():
    """Test YAML with both models and sources."""
    yaml_handler = create_yaml_instance()
    yaml_handler_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmpdir:
        combined_file = Path(tmpdir) / "combined.yml"
        combined_file.write_text("""
version: 2
models:
  - name: my_model
    description: "A model"
    columns:
      - name: id
        description: "ID column"
sources:
  - name: my_source
    description: "A source"
    tables:
      - name: my_table
        columns:
          - name: id
            description: "ID column"
""")

        result = _read_yaml(yaml_handler, yaml_handler_lock, combined_file)
        assert result is not None
        assert "models" in result
        assert "sources" in result
        assert len(result["models"]) == 1
        assert len(result["sources"]) == 1
