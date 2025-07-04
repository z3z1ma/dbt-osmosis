from unittest import mock

import pytest

from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorContext,
    YamlRefactorSettings,
    _reload_manifest,
    apply_restructure_plan,
    create_dbt_project_context,
    create_missing_source_yamls,
    draft_restructure_delta_plan,
    get_columns,
    inherit_upstream_column_knowledge,
)


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    # initializing the context is a sanity test in and of itself
    c = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    c.vars = {"dbt-osmosis": {}}
    project = create_dbt_project_context(c)
    context = YamlRefactorContext(
        project, settings=YamlRefactorSettings(use_unrendered_descriptions=True, dry_run=True)
    )
    return context


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
    with mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}):
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


def test_get_columns_meta_char_length():
    yaml_context = YamlRefactorContext(
        project=create_dbt_project_context(
            DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
        ),
        settings=YamlRefactorSettings(string_length=True, dry_run=True),
    )
    # Patch both possible cache locations to ensure isolation
    with (
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
    ):
        assert (
            _customer_column_types(yaml_context)
            == {
                # in DuckDB decimals always have presision and scale
                "customer_average_value": "DECIMAL(18,3)",
                "customer_id": "INTEGER",
                "customer_lifetime_value": "DOUBLE",
                "first_name": "character varying(256)",  # DuckDB returns detailed type when string_length=True
                "first_order": "DATE",
                "last_name": "character varying(256)",  # DuckDB returns detailed type when string_length=True
                "most_recent_order": "DATE",
                "number_of_orders": "BIGINT",
            }
        )


def test_get_columns_meta_numeric_precision():
    yaml_context = YamlRefactorContext(
        project=create_dbt_project_context(
            DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
        ),
        settings=YamlRefactorSettings(numeric_precision_and_scale=True, dry_run=True),
    )
    with mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}):
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
