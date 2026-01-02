# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.inheritance import _get_node_yaml
from dbt_osmosis.core.schema.writer import commit_yamls
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.sync_operations import sync_node_to_yaml


@pytest.fixture(scope="function")
def fresh_caches():
    """
    Patches the internal caches so each test starts with a fresh state.
    """
    with (
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_sync_node_to_yaml(yaml_context: YamlRefactorContext, fresh_caches):
    """
    For a single node, we can confirm that sync_node_to_yaml runs without error,
    using the real file or generating one if missing (in dry_run mode).
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_sync_node_to_yaml_versioned(yaml_context: YamlRefactorContext, fresh_caches):
    """Test syncing a versioned node to YAML."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_commit_yamls_no_write(yaml_context: YamlRefactorContext):
    """
    Since dry_run=True, commit_yamls should not actually write anything to disk.
    We just ensure no exceptions are raised.
    """
    commit_yamls(
        yaml_handler=yaml_context.yaml_handler,
        yaml_handler_lock=yaml_context.yaml_handler_lock,
        dry_run=yaml_context.settings.dry_run,
        mutation_tracker=yaml_context.register_mutations,
    )


def test_preserve_unrendered_descriptions(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Test that when use_unrendered_descriptions is True, descriptions containing
    doc blocks ({{ doc(...) }} or {% docs %}{% enddocs %}) are preserved instead
    of being replaced with the rendered version from the manifest.

    This addresses GitHub issue #219.
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML for this node
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Find a column to test with (use "status" which has a doc block reference)
    original_columns = original_yaml.get("columns", [])
    status_col = None
    for col in original_columns:
        if col.get("name") == "status":
            status_col = col
            break

    # The status column should have a doc reference in the YAML
    if status_col and "{{ doc(" in status_col.get("description", ""):
        original_description = status_col["description"]

        # Enable use_unrendered_descriptions
        yaml_context.settings.use_unrendered_descriptions = True
        yaml_context.settings.force_inherit_descriptions = True

        # Sync the node
        with (
            mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
            mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        ):
            sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the status column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_status_col = None
        for col in updated_columns:
            if col.get("name") == "status":
                updated_status_col = col
                break

        # The description should still contain the doc reference (unrendered)
        assert updated_status_col is not None
        assert "{{ doc(" in updated_status_col.get("description", ""), (
            "Expected unrendered doc reference to be preserved when "
            "use_unrendered_descriptions is True"
        )
        # Verify the exact doc reference is preserved
        assert updated_status_col["description"] == original_description
