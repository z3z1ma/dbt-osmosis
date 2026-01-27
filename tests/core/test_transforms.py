# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.transforms import (
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_alphabetically,
    sort_columns_as_configured,
    sort_columns_as_in_database,
    synchronize_data_types,
)


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext, fresh_caches):
    """Minimal test that runs the inheritance logic on all matched nodes in the real project."""
    inherit_upstream_column_knowledge(yaml_context)


def test_inject_missing_columns(yaml_context: YamlRefactorContext, fresh_caches):
    """If the DB has columns the YAML/manifest doesn't, we inject them.
    We run on all matched nodes to ensure no errors.
    """
    inject_missing_columns(yaml_context)


def test_remove_columns_not_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """If the manifest has columns the DB does not, we remove them.
    Typically, your real project might not have any extra columns, so this is a sanity test.
    """
    remove_columns_not_in_database(yaml_context)


def test_sort_columns_as_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """Sort columns in the order the DB sees them.
    With duckdb, this is minimal but we can still ensure no errors.
    """
    sort_columns_as_in_database(yaml_context)


def test_sort_columns_alphabetically(yaml_context: YamlRefactorContext, fresh_caches):
    """Check that sort_columns_alphabetically doesn't blow up in real project usage."""
    sort_columns_alphabetically(yaml_context)


def test_sort_columns_as_configured(yaml_context: YamlRefactorContext, fresh_caches):
    """By default, the sort_by is 'database', but let's confirm it doesn't blow up."""
    sort_columns_as_configured(yaml_context)


def test_synchronize_data_types(yaml_context: YamlRefactorContext, fresh_caches):
    """Synchronizes data types with the DB."""
    synchronize_data_types(yaml_context)


def test_sort_columns_alphabetically_with_output_to_lower(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting respects output-to-lower setting.

    When output-to-lower is enabled, columns should be sorted based on their
    lowercase form, not their original case. This ensures that after case
    conversion, the columns remain in alphabetical order.

    For example, with columns ["ZEBRA", "apple", "Banana"]:
    - Without fix: Sorted as ["ZEBRA", "Banana", "apple"] (ASCII order)
    - After lower: ["zebra", "banana", "apple"] (WRONG - not alphabetical)
    - With fix: Sorted as ["apple", "Banana", "ZEBRA"] (lowercase order)
    - After lower: ["apple", "banana", "zebra"] (CORRECT - alphabetical)
    """
    # Create a mock node with mixed-case column names that would sort
    # incorrectly in ASCII order
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "ZEBRA": mock.MagicMock(name="ZEBRA"),
        "apple": mock.MagicMock(name="apple"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context with output_to_lower enabled
    context_with_lower = mock.MagicMock()
    context_with_lower.settings.output_to_lower = True
    context_with_lower.settings.output_to_upper = False

    # Sort with output_to_lower enabled
    sort_columns_alphabetically(context_with_lower, mock_node)

    # Verify columns are sorted by lowercase name
    column_names = list(mock_node.columns.keys())
    assert column_names == ["apple", "Banana", "ZEBRA"], (
        f"Columns should be sorted by lowercase name, got {column_names}"
    )


def test_sort_columns_alphabetically_with_output_to_upper(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting respects output-to-upper setting.

    When output-to-upper is enabled, columns should be sorted based on their
    uppercase form, not their original case.
    """
    # Create a mock node with mixed-case column names
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "zebra": mock.MagicMock(name="zebra"),
        "APPLE": mock.MagicMock(name="APPLE"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context with output_to_upper enabled
    context_with_upper = mock.MagicMock()
    context_with_upper.settings.output_to_lower = False
    context_with_upper.settings.output_to_upper = True

    # Sort with output_to_upper enabled
    sort_columns_alphabetically(context_with_upper, mock_node)

    # Verify columns are sorted by uppercase name
    column_names = list(mock_node.columns.keys())
    assert column_names == ["APPLE", "Banana", "zebra"], (
        f"Columns should be sorted by uppercase name, got {column_names}"
    )


def test_sort_columns_alphabetically_without_case_conversion(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting works correctly when no case conversion is enabled.

    When neither output-to-lower nor output-to-upper is set, columns should be
    sorted using their original case (standard lexicographic order).
    """
    # Create a mock node with mixed-case column names
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "ZEBRA": mock.MagicMock(name="ZEBRA"),
        "apple": mock.MagicMock(name="apple"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context without case conversion
    context_no_conversion = mock.MagicMock()
    context_no_conversion.settings.output_to_lower = False
    context_no_conversion.settings.output_to_upper = False

    # Sort without case conversion
    sort_columns_alphabetically(context_no_conversion, mock_node)

    # Verify columns are sorted by original name (ASCII order)
    column_names = list(mock_node.columns.keys())
    # In ASCII: uppercase letters come before lowercase
    assert column_names == ["Banana", "ZEBRA", "apple"], (
        f"Columns should be sorted by original name (ASCII order), got {column_names}"
    )
