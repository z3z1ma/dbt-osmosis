# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.inheritance import _get_node_yaml
from dbt_osmosis.core.schema.writer import commit_yamls
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.sync_operations import sync_node_to_yaml


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_sync_node_to_yaml(yaml_context: YamlRefactorContext, fresh_caches):
    """For a single node, we can confirm that sync_node_to_yaml runs without error,
    using the real file or generating one if missing (in dry_run mode).
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_sync_node_to_yaml_versioned(yaml_context: YamlRefactorContext, fresh_caches):
    """Test syncing a versioned node to YAML."""
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_commit_yamls_no_write(yaml_context: YamlRefactorContext):
    """Since dry_run=True, commit_yamls should not actually write anything to disk.
    We just ensure no exceptions are raised.
    """
    commit_yamls(
        yaml_handler=yaml_context.yaml_handler,
        yaml_handler_lock=yaml_context.yaml_handler_lock,
        dry_run=yaml_context.settings.dry_run,
        mutation_tracker=yaml_context.register_mutations,
    )


def test_preserve_unrendered_descriptions(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that when use_unrendered_descriptions is True, descriptions containing
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


def test_prefer_yaml_values_preserves_var_jinja(yaml_context: YamlRefactorContext, fresh_caches):
    """Test that when prefer_yaml_values is True, fields containing {{ var() }}
    jinja templates are preserved instead of being replaced with rendered values.

    This addresses GitHub issue #266.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML and add a mock policy_tags field with var()
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Simulate a column with policy_tags containing {{ var() }}
    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "order_id":
            # Add a policy_tags field with unrendered jinja
            col["policy_tags"] = ['{{ var("policy_tag_order_id") }}']
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values
        yaml_context.settings.prefer_yaml_values = True

        with (
            mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
            mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        ):
            sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the order_id column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_order_id_col = None
        for col in updated_columns:
            if col.get("name") == "order_id":
                updated_order_id_col = col
                break

        # The policy_tags should still contain the unrendered var() reference
        assert updated_order_id_col is not None
        assert "policy_tags" in updated_order_id_col
        policy_tags = updated_order_id_col["policy_tags"]
        assert any("{{ var(" in str(tag) for tag in policy_tags), (
            "Expected unrendered {{ var() }} reference to be preserved when "
            "prefer_yaml_values is True"
        )


def test_prefer_yaml_values_preserves_env_var_jinja(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that when prefer_yaml_values is True, fields containing {{ env_var() }}
    jinja templates are preserved.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML and add a mock field with env_var()
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Simulate a column with meta containing env_var()
    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "customer_id":
            # Add a meta field with unrendered jinja
            col["meta"] = {"pii": "{{ env_var('PII_LEVEL') }}"}
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values
        yaml_context.settings.prefer_yaml_values = True

        with (
            mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
            mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        ):
            sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Find the customer_id column in the updated YAML
        updated_columns = updated_yaml.get("columns", [])
        updated_customer_id_col = None
        for col in updated_columns:
            if col.get("name") == "customer_id":
                updated_customer_id_col = col
                break

        # The meta.pii should still contain the unrendered env_var() reference
        assert updated_customer_id_col is not None
        assert "meta" in updated_customer_id_col
        assert "{{ env_var(" in updated_customer_id_col["meta"].get("pii", ""), (
            "Expected unrendered {{ env_var() }} reference to be preserved when "
            "prefer_yaml_values is True"
        )


def test_prefer_yaml_values_preserves_all_jinja_patterns(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that prefer_yaml_values preserves all types of jinja templates including:
    - {{ doc() }}
    - {{ var() }}
    - {{ env_var() }}
    - {% docs %}{% enddocs %}
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Get the original YAML
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    # Add multiple columns with different jinja patterns
    original_columns = original_yaml.get("columns", [])
    test_cases = [
        ("order_id", "policy_tags", ["{{ var('policy_tag') }}"]),
        ("customer_id", "meta", {"classification": "{{ env_var('CLASS') }}"}),
        ("status", "description", "{% docs status_desc %}{% enddocs %}"),
    ]

    for col_name, field_key, field_value in test_cases:
        for col in original_columns:
            if col.get("name") == col_name:
                col[field_key] = field_value
                break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        # Enable prefer_yaml_values
        yaml_context.settings.prefer_yaml_values = True

        with (
            mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
            mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        ):
            sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Verify all jinja patterns were preserved
        updated_columns = updated_yaml.get("columns", [])
        for col_name, field_key, field_value in test_cases:
            col = next((c for c in updated_columns if c.get("name") == col_name), None)
            assert col is not None, f"Column {col_name} not found"
            assert field_key in col, f"Field {field_key} not in column {col_name}"

            # Check that jinja patterns are preserved
            if isinstance(field_value, list):
                updated_value = col[field_key]
                assert any("{{ var(" in str(v) for v in updated_value), (
                    f"Expected {{ var() }} to be preserved in {col_name}.{field_key}"
                )
            elif isinstance(field_value, dict):
                updated_value = col[field_key]
                assert any("{{ env_var(" in str(v) for v in updated_value.values()), (
                    f"Expected {{ env_var() }} to be preserved in {col_name}.{field_key}"
                )
            else:
                updated_value = col[field_key]
                assert "{% docs " in updated_value, (
                    f"Expected {{% docs %}} to be preserved in {col_name}.{field_key}"
                )


def test_add_inheritance_for_specified_keys_still_works(
    yaml_context: YamlRefactorContext,
    fresh_caches,
):
    """Test that --add-inheritance-for-specified-keys still works for granular control.
    This ensures backward compatibility with existing functionality.
    """
    from dbt_osmosis.core.inheritance import _get_node_yaml

    # Use add_inheritance_for_specified_keys to inherit policy_tags
    yaml_context.settings.add_inheritance_for_specified_keys = ["policy_tags"]

    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.orders"]

    # Add a policy_tags field to the source YAML
    original_yaml = _get_node_yaml(yaml_context, node)
    assert original_yaml is not None

    original_columns = original_yaml.get("columns", [])
    for col in original_columns:
        if col.get("name") == "order_id":
            # Add a policy_tags field with var()
            col["policy_tags"] = ["{{ var('upstream_policy') }}"]
            break

    # Write the modified YAML back
    import io

    buffer = io.StringIO()
    yaml_context.yaml_handler.dump(original_yaml, buffer)
    with mock.patch("builtins.open", mock.mock_open(read_data=buffer.getvalue())):
        with (
            mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
            mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        ):
            sync_node_to_yaml(yaml_context, node, commit=False)

        # Get the updated YAML
        updated_yaml = _get_node_yaml(yaml_context, node)
        assert updated_yaml is not None

        # Verify that policy_tags was inherited (unrendered)
        updated_columns = updated_yaml.get("columns", [])
        updated_order_id_col = None
        for col in updated_columns:
            if col.get("name") == "order_id":
                updated_order_id_col = col
                break

        assert updated_order_id_col is not None
        assert "policy_tags" in updated_order_id_col
        assert any("{{ var(" in str(tag) for tag in updated_order_id_col["policy_tags"]), (
            "Expected unrendered {{ var() }} to be inherited via add-inheritance-for-specified-keys"
        )
