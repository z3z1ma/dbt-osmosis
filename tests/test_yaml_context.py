from unittest import mock

from dbt_osmosis.core.config import _reload_manifest
from dbt_osmosis.core.introspection import _COLUMN_LIST_CACHE, get_columns
from dbt_osmosis.core.path_management import create_missing_source_yamls
from dbt_osmosis.core.restructuring import apply_restructure_plan, draft_restructure_delta_plan
from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.transforms import inherit_upstream_column_knowledge

# Note: The yaml_context fixture is defined in conftest.py


# Sanity tests


def test_reload_manifest(yaml_context: YamlRefactorContext):
    _reload_manifest(yaml_context.project)


def test_create_missing_source_yamls(yaml_context: YamlRefactorContext):
    create_missing_source_yamls(yaml_context)


def test_draft_restructure_delta_plan(yaml_context: YamlRefactorContext):
    assert draft_restructure_delta_plan(yaml_context) is not None


def test_apply_restructure_plan(yaml_context: YamlRefactorContext):
    plan = draft_restructure_delta_plan(yaml_context)
    apply_restructure_plan(yaml_context, plan, confirm=False)


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext):
    inherit_upstream_column_knowledge(yaml_context)


# Column type + settings tests


def _customer_column_types(yaml_context: YamlRefactorContext) -> dict[str, str]:
    node = next(n for n in yaml_context.project.manifest.nodes.values() if n.name == "customers")
    assert node

    columns = get_columns(yaml_context, node)
    assert columns

    column_types = dict({name: meta.type for name, meta in columns.items()})
    assert column_types
    return column_types


def test_get_columns_meta(yaml_context: YamlRefactorContext):
    with mock.patch.dict(_COLUMN_LIST_CACHE, {}, clear=True):
        assert _customer_column_types(yaml_context) == {
            # in DuckDB decimals always have presision and scale
            "customer_average_value": "DECIMAL(18,3)",
            "customer_id": "INTEGER",
            "customer_lifetime_value": "DOUBLE",
            "first_name": "VARCHAR",
            "first_order": "DATE",
            "last_name": "VARCHAR",
            "most_recent_order": "DATE",
            "number_of_orders": "BIGINT",
        }


def test_get_columns_meta_char_length(yaml_context: YamlRefactorContext):
    """Test string_length setting uses catalog types (VARCHAR)."""
    # Update the context settings for this test
    yaml_context.settings.string_length = True
    with mock.patch.dict(_COLUMN_LIST_CACHE, {}, clear=True):
        # Catalog returns VARCHAR, not character varying(256)
        assert _customer_column_types(yaml_context) == {
            "customer_average_value": "DECIMAL(18,3)",
            "customer_id": "INTEGER",
            "customer_lifetime_value": "DOUBLE",
            "first_name": "VARCHAR",  # Catalog type
            "first_order": "DATE",
            "last_name": "VARCHAR",  # Catalog type
            "most_recent_order": "DATE",
            "number_of_orders": "BIGINT",
        }


def test_get_columns_meta_numeric_precision(yaml_context: YamlRefactorContext):
    """Test numeric_precision_and_scale setting."""
    yaml_context.settings.numeric_precision_and_scale = True
    with mock.patch.dict(_COLUMN_LIST_CACHE, {}, clear=True):
        assert _customer_column_types(yaml_context) == {
            # in DuckDB decimals always have presision and scale
            "customer_average_value": "DECIMAL(18,3)",
            "customer_id": "INTEGER",
            "customer_lifetime_value": "DOUBLE",
            "first_name": "VARCHAR",
            "first_order": "DATE",
            "last_name": "VARCHAR",
            "most_recent_order": "DATE",
            "number_of_orders": "BIGINT",
        }
