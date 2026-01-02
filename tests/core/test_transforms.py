# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.transforms import (
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_alphabetically,
    sort_columns_as_configured,
    sort_columns_as_in_database,
    synchronize_data_types,
)


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    """
    Creates a YamlRefactorContext for the real 'demo_duckdb' project.
    """
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    cfg.vars = {"dbt-osmosis": {}}

    project_context = create_dbt_project_context(cfg)
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,
            use_unrendered_descriptions=True,
        ),
    )
    return context


@pytest.fixture(scope="function")
def fresh_caches():
    """
    Patches the internal caches so each test starts with a fresh state.
    """
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that upstream column knowledge inheritance works correctly.
    Verifies that documentation and tests are inherited from upstream models.
    """
    # Get initial state

    # Run the inheritance transform
    result = inherit_upstream_column_knowledge(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for inheritance"

    # Find a specific node to verify inheritance (customers model should inherit from stg_customers)
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found in processed nodes"

    # Verify that the customers model has columns with inherited documentation
    columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert columns is not None, "No columns found for customers model"
    assert len(columns) > 0, "No columns found for customers model"

    # Verify specific inheritance - customer_id should have inherited tests from stg_customers
    customer_id_column = None
    for col in columns:
        if col.name == "customer_id":
            customer_id_column = col
            break

    assert customer_id_column is not None, "customer_id column not found"
    # The column should maintain its original description
    assert "unique identifier" in customer_id_column.description.lower(), (
        f"Expected 'unique identifier' in customer_id description, got: {customer_id_column.description}"
    )


def test_inject_missing_columns(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that missing columns from the database are properly injected into YAML.
    Verifies that columns present in DB but missing from YAML are added.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before injection
    initial_column_count = 0
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)

    # Run the injection transform
    result = inject_missing_columns(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for column injection"

    # Count total columns after injection
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # The transform should not decrease column count
    assert final_column_count >= initial_column_count, (
        f"Column count decreased from {initial_column_count} to {final_column_count}"
    )

    # Verify specific models have expected columns
    # Check customers model has basic columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify essential columns exist
    column_names = [col.name for col in customers_columns]
    expected_columns = ["customer_id", "first_name", "last_name"]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model. Columns: {column_names}"
        )

    # Verify that data types are preserved
    for col in customers_columns:
        if col.name == "customer_id":
            assert col.data_type == "INTEGER", (
                f"Expected customer_id data_type to be INTEGER, got {col.data_type}"
            )
        elif col.name == "first_name":
            assert col.data_type == "VARCHAR", (
                f"Expected first_name data_type to be VARCHAR, got {col.data_type}"
            )


def test_remove_columns_not_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that columns not present in the database are removed from YAML.
    Verifies stale columns are cleaned up while valid columns are preserved.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before removal
    initial_column_count = 0
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)

    # Run the removal transform
    result = remove_columns_not_in_database(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for column removal"

    # Count total columns after removal
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # The transform should not increase column count
    assert final_column_count <= initial_column_count, (
        f"Column count increased from {initial_column_count} to {final_column_count}"
    )

    # Verify specific models still have valid columns
    # Check customers model still has essential columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify essential columns still exist (they should be in the database)
    column_names = [col.name for col in customers_columns]
    expected_columns = ["customer_id", "first_name", "last_name"]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model after cleanup. Columns: {column_names}"
        )

    # Verify that all columns have valid data types
    for col in customers_columns:
        assert col.data_type is not None, f"Column {col.name} has no data_type"
        assert col.data_type != "", f"Column {col.name} has empty data_type"

    # Verify that valid columns are preserved
    # customer_id should still be INTEGER
    customer_id_column = next((col for col in customers_columns if col.name == "customer_id"), None)
    assert customer_id_column is not None, "customer_id column was removed"
    assert customer_id_column.data_type == "INTEGER", (
        f"Expected customer_id to remain INTEGER, got {customer_id_column.data_type}"
    )


def test_sort_columns_as_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that columns are sorted according to database order.
    Verifies the transform runs without errors and preserves column integrity.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before sorting
    initial_column_count = 0
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)

    # Run the sorting transform
    result = sort_columns_as_in_database(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for database sorting"

    # Count total columns after sorting
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # Column count should remain the same
    assert final_column_count == initial_column_count, (
        f"Column count changed from {initial_column_count} to {final_column_count} during database sorting"
    )

    # Verify specific models still have all their columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify all expected columns are still present
    column_names = [col.name for col in customers_columns]
    expected_columns = [
        "customer_id",
        "first_name",
        "last_name",
        "first_order",
        "most_recent_order",
        "number_of_orders",
        "customer_lifetime_value",
        "customer_average_value",
    ]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model after database sorting. Columns: {column_names}"
        )

    # Verify that column metadata is preserved during sorting
    for col in customers_columns:
        assert col.name is not None, "Column has no name after sorting"
        assert col.data_type is not None, f"Column {col.name} has no data_type after sorting"
        assert col.data_type != "", f"Column {col.name} has empty data_type after sorting"


def test_sort_columns_alphabetically(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that columns are sorted alphabetically by name.
    Verifies the transform runs without errors and preserves column integrity.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before sorting
    initial_column_count = 0
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)

    # Run the alphabetical sorting transform
    result = sort_columns_alphabetically(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for alphabetical sorting"

    # Count total columns after sorting
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # Column count should remain the same
    assert final_column_count == initial_column_count, (
        f"Column count changed from {initial_column_count} to {final_column_count} during alphabetical sorting"
    )

    # Verify specific models still have all their columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify all expected columns are still present
    column_names = [col.name for col in customers_columns]
    expected_columns = [
        "customer_id",
        "first_name",
        "last_name",
        "first_order",
        "most_recent_order",
        "number_of_orders",
        "customer_lifetime_value",
        "customer_average_value",
    ]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model after alphabetical sorting. Columns: {column_names}"
        )

    # Verify columns are sorted alphabetically (for customers model)
    column_names_sorted = column_names.copy()
    column_names_sorted.sort()
    assert column_names == column_names_sorted, (
        f"Columns in customers model are not sorted alphabetically. Got: {column_names}, Expected: {column_names_sorted}"
    )

    # Verify that column metadata is preserved during sorting
    for col in customers_columns:
        assert col.name is not None, "Column has no name after alphabetical sorting"
        assert col.data_type is not None, (
            f"Column {col.name} has no data_type after alphabetical sorting"
        )
        assert col.data_type != "", (
            f"Column {col.name} has empty data_type after alphabetical sorting"
        )


def test_sort_columns_as_configured(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that columns are sorted according to configuration settings.
    Verifies the transform runs without errors and preserves column integrity.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before sorting
    initial_column_count = 0
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)

    # Run the configuration-based sorting transform
    result = sort_columns_as_configured(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for configuration-based sorting"

    # Count total columns after sorting
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # Column count should remain the same
    assert final_column_count == initial_column_count, (
        f"Column count changed from {initial_column_count} to {final_column_count} during configuration-based sorting"
    )

    # Verify specific models still have all their columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify all expected columns are still present
    column_names = [col.name for col in customers_columns]
    expected_columns = [
        "customer_id",
        "first_name",
        "last_name",
        "first_order",
        "most_recent_order",
        "number_of_orders",
        "customer_lifetime_value",
        "customer_average_value",
    ]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model after configuration-based sorting. Columns: {column_names}"
        )

    # Verify that column metadata is preserved during sorting
    for col in customers_columns:
        assert col.name is not None, "Column has no name after configuration-based sorting"
        assert col.data_type is not None, (
            f"Column {col.name} has no data_type after configuration-based sorting"
        )
        assert col.data_type != "", (
            f"Column {col.name} has empty data_type after configuration-based sorting"
        )


def test_synchronize_data_types(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that data types are synchronized with the database.
    Verifies that data types are updated correctly while preserving other column properties.
    """
    # Get initial state
    initial_nodes = yaml_context.project_context.get_matched_nodes()

    # Count total columns before synchronization
    initial_column_count = 0
    initial_data_types = {}
    for node in initial_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            initial_column_count += len(columns)
            for col in columns:
                initial_data_types[f"{node.name}.{col.name}"] = col.data_type

    # Run the data type synchronization transform
    result = synchronize_data_types(yaml_context)

    # Verify the transform completed successfully
    assert result is not None

    # Check that some nodes were processed
    processed_nodes = yaml_context.project_context.get_matched_nodes()
    assert len(processed_nodes) > 0, "No nodes were processed for data type synchronization"

    # Count total columns after synchronization
    final_column_count = 0

    for node in processed_nodes:
        columns = yaml_context.schema_reader.get_model_columns(node)
        if columns:
            final_column_count += len(columns)

    # Column count should remain the same
    assert final_column_count == initial_column_count, (
        f"Column count changed from {initial_column_count} to {final_column_count} during data type synchronization"
    )

    # Verify specific models still have all their columns
    customers_node = None
    for node in processed_nodes:
        if node.name == "customers":
            customers_node = node
            break

    assert customers_node is not None, "customers node not found"

    customers_columns = yaml_context.schema_reader.get_model_columns(customers_node)
    assert customers_columns is not None, "No columns found for customers model"

    # Verify all expected columns are still present
    column_names = [col.name for col in customers_columns]
    expected_columns = [
        "customer_id",
        "first_name",
        "last_name",
        "first_order",
        "most_recent_order",
        "number_of_orders",
        "customer_lifetime_value",
        "customer_average_value",
    ]

    for expected_col in expected_columns:
        assert expected_col in column_names, (
            f"Expected column '{expected_col}' not found in customers model after data type synchronization. Columns: {column_names}"
        )

    # Verify that data types are valid and non-empty
    for col in customers_columns:
        assert col.data_type is not None, (
            f"Column {col.name} has no data_type after synchronization"
        )
        assert col.data_type != "", f"Column {col.name} has empty data_type after synchronization"
        # Data type should be a valid SQL type (basic check)
        assert isinstance(col.data_type, str), f"Column {col.name} data_type is not a string"

    # Verify specific data type expectations for known columns
    data_type_checks = {
        "customer_id": "INTEGER",
        "first_name": "VARCHAR",
        "last_name": "VARCHAR",
        "first_order": "DATE",
        "most_recent_order": "DATE",
        "number_of_orders": "BIGINT",
        "customer_lifetime_value": "DOUBLE",
        "customer_average_value": "DECIMAL(18,3)",
    }

    for col_name, expected_type in data_type_checks.items():
        col = next((c for c in customers_columns if c.name == col_name), None)
        assert col is not None, f"Column {col_name} not found after synchronization"
        # The data type should match the expected type (assuming it matches the DB)
        assert col.data_type == expected_type, (
            f"Expected {col_name} data_type to be {expected_type}, got {col.data_type}"
        )

    # Verify that column metadata (other than data_type) is preserved
    for col in customers_columns:
        if col.name == "customer_id":
            # Check if description is preserved
            assert col.description is not None, (
                f"Column {col.name} has no description after synchronization"
            )
            assert "unique identifier" in col.description.lower(), (
                f"Expected 'unique identifier' in customer_id description after synchronization, got: {col.description}"
            )
        elif col.name == "first_name":
            assert col.description is not None, (
                f"Column {col.name} has no description after synchronization"
            )
            assert "first name" in col.description.lower(), (
                f"Expected 'first name' in first_name description after synchronization, got: {col.description}"
            )
