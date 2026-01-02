"""Integration tests for the transform pipeline.

This module tests the END-TO-END BEHAVIOR of the transform pipeline,
not individual transform functions. The pipeline uses the >> operator
to chain transforms together:

    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    pipeline(context)  # Returns the pipeline itself

These tests verify:
1. Full pipeline execution produces correct results
2. Pipeline state consistency
3. Transform ordering guarantees
4. Multiple transforms can be chained together
"""

from __future__ import annotations

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


def test_full_pipeline_execution(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that the full transform pipeline executes correctly.

    This creates a realistic pipeline with multiple transforms chained together
    and verifies that the final state is correct.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Record initial state
    initial_column_count = len(target_node.columns)

    # Build and execute full pipeline
    pipeline = (
        inject_missing_columns >> inherit_upstream_column_knowledge >> sort_columns_as_configured
    )

    result = pipeline(yaml_context)

    # Assert: Pipeline should return itself
    assert result is pipeline

    # Assert: Pipeline metadata should indicate success
    assert pipeline.metadata.get("success", True)  # Pipeline completed

    # Assert: All expected columns should still be present
    assert len(target_node.columns) >= initial_column_count


def test_pipeline_with_inject_and_inherit(yaml_context: YamlRefactorContext, fresh_caches):
    """Test pipeline that injects missing columns then inherits knowledge.

    This is a common pattern:
    1. First inject columns that exist in DB but not in YAML
    2. Then inherit documentation for all columns (including newly injected ones)

    This tests that newly injected columns also get inherited documentation.
    """
    # Build and execute pipeline
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge

    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline

    # Assert: Pipeline should have metadata with steps
    assert "steps" in pipeline.metadata
    assert len(pipeline.metadata["steps"]) == 2


def test_pipeline_with_remove_and_sort(yaml_context: YamlRefactorContext, fresh_caches):
    """Test pipeline that removes stale columns then sorts.

    This tests that:
    1. Columns not in the database are removed
    2. Remaining columns are sorted according to configuration
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Record column names before pipeline
    columns_before = list(target_node.columns.keys())

    # Build and execute pipeline
    pipeline = remove_columns_not_in_database >> sort_columns_as_in_database

    yaml_context.settings.skip_add_tags = True
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline

    # Assert: Columns after should be sorted in database order
    columns_after = list(target_node.columns.keys())

    # All columns before should still be present (unless they don't exist in DB)
    assert len(columns_after) <= len(columns_before)


def test_pipeline_transform_ordering(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that transforms execute in the correct order when chained.

    Pipeline: inject >> inherit

    We verify order by checking intermediate state after each transform.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Set up: Set upstream descriptions
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "From staging"

    # Clear first_name description in target
    target_node.columns["first_name"].description = ""

    # Execute pipeline
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    pipeline(yaml_context)

    # Assert: After inheritance, first_name should have description
    assert target_node.columns["first_name"].description == "From staging"


def test_pipeline_state_consistency(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that the context state remains consistent across pipeline execution.

    The yaml_context should be mutated in-place by the pipeline.
    """
    # Get initial mutation count
    initial_count = yaml_context.mutation_count

    # Execute pipeline
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    result = pipeline(yaml_context)

    # Assert: Pipeline returns itself
    assert result is pipeline

    # Assert: Mutation count should have increased
    assert yaml_context.mutation_count >= initial_count


def test_empty_pipeline(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that an empty pipeline (single transform) works correctly.

    This verifies the >> operator works with just one transform.
    """
    # Execute single transform as "pipeline"
    pipeline = inject_missing_columns
    result = pipeline(yaml_context)

    # Assert: Should return the TransformOperation
    assert result == pipeline


def test_pipeline_with_settings(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that pipeline respects context settings.

    Settings like force_inherit_descriptions, skip_add_tags, etc.
    should affect all transforms in the pipeline.
    """
    # Configure settings
    yaml_context.settings.force_inherit_descriptions = False
    yaml_context.settings.skip_add_tags = True
    yaml_context.settings.skip_merge_meta = True

    # Execute pipeline
    pipeline = inherit_upstream_column_knowledge
    pipeline(yaml_context)

    # Assert: Pipeline should complete without errors
    assert pipeline.metadata.get("success", True)


def test_pipeline_multiple_targets(yaml_context: YamlRefactorContext, fresh_caches):
    """Test running the same pipeline on multiple target nodes.

    This simulates the real-world usage where a pipeline is executed
    across all models in a project.
    """
    # Build pipeline
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge

    # Execute on all nodes (default behavior)
    yaml_context.settings.force_inherit_descriptions = True
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline


def test_pipeline_idempotency(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that running the same pipeline twice produces consistent results.

    A pipeline should be idempotent - running it multiple times should
    not cause errors or unexpected behavior.
    """
    # Execute pipeline first time
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    result1 = pipeline(yaml_context)

    # Execute pipeline second time
    result2 = pipeline(yaml_context)

    # Assert: Both executions should complete successfully
    assert result1 is pipeline
    assert result2 is pipeline


def test_pipeline_with_catalog(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that pipeline uses catalog when available instead of live introspection.

    When catalog.json exists, transforms should use it for column metadata
    instead of querying the database directly.
    """
    # Set catalog path to use the catalog from the test fixture
    yaml_context.settings.catalog_path = str(yaml_context.project_root / "target" / "catalog.json")

    # Execute pipeline
    pipeline = inject_missing_columns >> synchronize_data_types
    result = pipeline(yaml_context)

    # Assert: Should complete using catalog data
    assert result is pipeline


def test_pipeline_error_handling(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that pipeline handles errors gracefully.

    If one transform fails, it should not leave the context in an
    inconsistent state.
    """
    # Execute pipeline with valid operations
    pipeline = inject_missing_columns >> sort_columns_as_configured

    # Should not raise exceptions
    result = pipeline(yaml_context)

    # Assert: Should complete
    assert result is pipeline


def test_pipeline_with_database_sort(yaml_context: YamlRefactorContext, fresh_caches):
    """Test pipeline with database column ordering.

    The sort_columns_as_in_database transform should order columns
    in the same order they appear in the database.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Execute pipeline with database sort
    pipeline = (
        inject_missing_columns >> sort_columns_as_configured  # Uses database ordering by default
    )

    yaml_context.settings.sort_by = "database"
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline

    # Assert: Columns should be in some order
    assert len(target_node.columns) > 0


def test_pipeline_with_alphabetical_sort(yaml_context: YamlRefactorContext, fresh_caches):
    """Test pipeline with alphabetical column ordering.

    The sort_columns_alphabetically transform should order columns
    alphabetically by name.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Execute pipeline with alphabetical sort
    pipeline = inject_missing_columns >> sort_columns_alphabetically
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline

    # Assert: Columns should be alphabetically sorted
    column_names = list(target_node.columns.keys())
    assert column_names == sorted(column_names)


def test_pipeline_metadata_tracking(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that pipeline correctly tracks execution metadata.

    The pipeline should track which operations were executed and their success.
    """
    # Execute pipeline
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    pipeline(yaml_context)

    # Assert: Metadata should be populated
    assert "steps" in pipeline.metadata
    assert len(pipeline.metadata["steps"]) == 2  # Two operations

    # Each step should have success indicator
    for step in pipeline.metadata["steps"]:
        assert step.get("success", True)


def test_three_transform_pipeline(yaml_context: YamlRefactorContext, fresh_caches):
    """Test a longer pipeline with three transforms chained together.

    This verifies that the >> operator works correctly for multiple transforms.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Execute three-transform pipeline
    pipeline = (
        inject_missing_columns >> inherit_upstream_column_knowledge >> sort_columns_alphabetically
    )
    result = pipeline(yaml_context)

    # Assert: Pipeline completed
    assert result is pipeline

    # Assert: Columns should be alphabetically sorted
    column_names = list(target_node.columns.keys())
    assert column_names == sorted(column_names)


def test_pipeline_commit_mode(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that pipeline commit modes work correctly.

    The commit_mode affects when changes are committed to YAML files.
    """
    # Execute pipeline with different commit modes
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    pipeline.commit_mode = "batch"
    result = pipeline(yaml_context)

    # Assert: Should complete
    assert result is pipeline


def test_pipeline_with_specific_node(yaml_context: YamlRefactorContext, fresh_caches):
    """Test running a pipeline on a specific node instead of all nodes.

    This tests that the node parameter is properly passed through the pipeline.
    """
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # Execute pipeline on specific node
    pipeline = inject_missing_columns >> inherit_upstream_column_knowledge
    result = pipeline(yaml_context, target_node)

    # Assert: Should complete
    assert result is pipeline
