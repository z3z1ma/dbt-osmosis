# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.diff import (
    ChangeCategory,
    ChangeSeverity,
    ColumnAdded,
    ColumnRemoved,
    ColumnRenamed,
    ColumnTypeChanged,
    SchemaDiff,
    SchemaDiffResult,
)
from dbt_osmosis.core.settings import YamlRefactorContext


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_schema_diff_initialization(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that SchemaDiff can be initialized with a context."""
    differ = SchemaDiff(yaml_context)
    assert differ._context == yaml_context
    assert differ._fuzzy_match_threshold == 85.0
    assert differ._detect_column_renames is True


def test_schema_diff_custom_threshold(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that SchemaDiff accepts custom fuzzy match threshold."""
    differ = SchemaDiff(yaml_context, fuzzy_match_threshold=90.0)
    assert differ._fuzzy_match_threshold == 90.0


def test_schema_diff_disable_renames(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that SchemaDiff can disable rename detection."""
    differ = SchemaDiff(yaml_context, detect_column_renames=False)
    assert differ._detect_column_renames is False


def test_schema_diff_compare_node(yaml_context: YamlRefactorContext, fresh_caches):
    """Test comparing a single node's schema."""
    import datetime

    differ = SchemaDiff(yaml_context)

    # Get a model node from the manifest
    from dbt.artifacts.resources.types import NodeType

    for node_id, node in yaml_context.manifest.nodes.items():
        if node.resource_type == NodeType.Model:
            result = differ.compare_node(node)
            assert isinstance(result, SchemaDiffResult)
            assert result.node == node
            assert isinstance(result.yaml_columns, dict)
            assert isinstance(result.database_columns, dict)
            assert isinstance(result.changes, list)
            assert isinstance(result.timestamp, datetime.datetime)
            break
    else:
        pytest.skip("No model nodes found in manifest")


def test_schema_diff_compare_all(yaml_context: YamlRefactorContext, fresh_caches):
    """Test comparing all nodes in the manifest."""
    differ = SchemaDiff(yaml_context)
    results = differ.compare_all()

    assert isinstance(results, dict)
    # Results should only contain nodes with changes
    for node_id, result in results.items():
        assert isinstance(result, SchemaDiffResult)
        assert result.has_changes is True


def test_schema_diff_result_properties(yaml_context: YamlRefactorContext, fresh_caches):
    """Test SchemaDiffResult properties."""
    differ = SchemaDiff(yaml_context)

    # Find a node and get its diff result
    from dbt.artifacts.resources.types import NodeType

    for node_id, node in yaml_context.manifest.nodes.items():
        if node.resource_type == NodeType.Model:
            result = differ.compare_node(node)

            # Test summary property
            summary = result.summary
            assert isinstance(summary, dict)

            # Test has_changes property
            assert isinstance(result.has_changes, bool)

            # Test breaking_changes property
            breaking = result.breaking_changes
            assert isinstance(breaking, list)

            # Test safe_changes property
            safe = result.safe_changes
            assert isinstance(safe, list)

            # Verify all changes in safe/breaking are in changes
            all_change_ids = {id(c) for c in result.changes}
            safe_ids = {id(c) for c in safe}
            breaking_ids = {id(c) for c in breaking}
            assert safe_ids.issubset(all_change_ids)
            assert breaking_ids.issubset(all_change_ids)

            break


def test_change_category_enum():
    """Test ChangeCategory enum values."""
    assert ChangeCategory.COLUMN_ADDED.value == "column_added"
    assert ChangeCategory.COLUMN_REMOVED.value == "column_removed"
    assert ChangeCategory.COLUMN_RENAMED.value == "column_renamed"
    assert ChangeCategory.TYPE_CHANGED.value == "type_changed"


def test_change_severity_enum():
    """Test ChangeSeverity enum values."""
    assert ChangeSeverity.SAFE.value == "safe"
    assert ChangeSeverity.MODERATE.value == "moderate"
    assert ChangeSeverity.BREAKING.value == "breaking"


def test_column_added_creation(yaml_context: YamlRefactorContext):
    """Test ColumnAdded change creation."""
    # Use a real node from the context
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnAdded(
        category=ChangeCategory.COLUMN_ADDED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        column_name="new_col",
        data_type="VARCHAR(255)",
        comment="A new column",
    )

    assert change.column_name == "new_col"
    assert change.data_type == "VARCHAR(255)"
    assert change.comment == "A new column"
    assert change.severity == ChangeSeverity.SAFE
    assert "new_col" in change.description


def test_column_removed_creation(yaml_context: YamlRefactorContext):
    """Test ColumnRemoved change creation."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnRemoved(
        category=ChangeCategory.COLUMN_REMOVED,
        severity=ChangeSeverity.MODERATE,
        node=node,
        description="",
        column_name="old_col",
        data_type="INTEGER",
    )

    assert change.column_name == "old_col"
    assert change.data_type == "INTEGER"
    assert change.severity == ChangeSeverity.MODERATE
    assert "old_col" in change.description


def test_column_renamed_creation(yaml_context: YamlRefactorContext):
    """Test ColumnRenamed change creation."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnRenamed(
        category=ChangeCategory.COLUMN_RENAMED,
        severity=ChangeSeverity.SAFE,
        node=node,
        description="",
        old_name="old_col",
        new_name="new_col",
        similarity_score=90.5,
        data_type="VARCHAR(100)",
    )

    assert change.old_name == "old_col"
    assert change.new_name == "new_col"
    assert change.similarity_score == 90.5
    assert change.data_type == "VARCHAR(100)"
    assert change.severity == ChangeSeverity.SAFE
    assert "old_col" in change.description
    assert "new_col" in change.description


def test_column_type_changed_creation(yaml_context: YamlRefactorContext):
    """Test ColumnTypeChanged change creation."""
    from dbt.artifacts.resources.types import NodeType

    node = next(
        (n for n in yaml_context.manifest.nodes.values() if n.resource_type == NodeType.Model),
        None,
    )
    if not node:
        pytest.skip("No model nodes found in manifest")

    change = ColumnTypeChanged(
        category=ChangeCategory.TYPE_CHANGED,
        severity=ChangeSeverity.MODERATE,
        node=node,
        description="",
        column_name="my_col",
        old_type="VARCHAR(50)",
        new_type="VARCHAR(100)",
    )

    assert change.column_name == "my_col"
    assert change.old_type == "VARCHAR(50)"
    assert change.new_type == "VARCHAR(100)"
    assert change.severity == ChangeSeverity.MODERATE
    assert "my_col" in change.description
    assert "VARCHAR(50)" in change.description
    assert "VARCHAR(100)" in change.description


def test_schema_diff_type_narrowing_detection(yaml_context: YamlRefactorContext):
    """Test type narrowing detection in SchemaDiff."""
    differ = SchemaDiff(yaml_context)

    # Test precision narrowing (VARCHAR(100) -> VARCHAR(50))
    is_narrowing = differ._is_type_narrowing("VARCHAR(100)", "VARCHAR(50)")
    assert is_narrowing is True

    # Test precision widening (VARCHAR(50) -> VARCHAR(100))
    is_narrowing = differ._is_type_narrowing("VARCHAR(50)", "VARCHAR(100)")
    assert is_narrowing is False

    # Test same type (no change)
    is_narrowing = differ._is_type_narrowing("VARCHAR(100)", "VARCHAR(100)")
    assert is_narrowing is False


def test_schema_diff_type_change_classification(yaml_context: YamlRefactorContext):
    """Test type change severity classification."""
    differ = SchemaDiff(yaml_context)

    # Same type = safe
    severity = differ._classify_type_change("VARCHAR(100)", "VARCHAR(100)")
    assert severity == ChangeSeverity.SAFE

    # Same family = safe (VARCHAR widening)
    severity = differ._classify_type_change("VARCHAR(50)", "VARCHAR(100)")
    assert severity == ChangeSeverity.SAFE

    # Integer narrowing = moderate
    severity = differ._classify_type_change("BIGINT", "INT")
    assert severity == ChangeSeverity.MODERATE

    # Different families = breaking (INT -> TEXT)
    severity = differ._classify_type_change("INTEGER", "TEXT")
    assert severity == ChangeSeverity.BREAKING
