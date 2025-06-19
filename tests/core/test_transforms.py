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
    Minimal test that runs the inheritance logic on all matched nodes in the real project.
    """
    inherit_upstream_column_knowledge(yaml_context)


def test_inject_missing_columns(yaml_context: YamlRefactorContext, fresh_caches):
    """
    If the DB has columns the YAML/manifest doesn't, we inject them.
    We run on all matched nodes to ensure no errors.
    """
    inject_missing_columns(yaml_context)


def test_remove_columns_not_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    If the manifest has columns the DB does not, we remove them.
    Typically, your real project might not have any extra columns, so this is a sanity test.
    """
    remove_columns_not_in_database(yaml_context)


def test_sort_columns_as_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Sort columns in the order the DB sees them.
    With duckdb, this is minimal but we can still ensure no errors.
    """
    sort_columns_as_in_database(yaml_context)


def test_sort_columns_alphabetically(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Check that sort_columns_alphabetically doesn't blow up in real project usage.
    """
    sort_columns_alphabetically(yaml_context)


def test_sort_columns_as_configured(yaml_context: YamlRefactorContext, fresh_caches):
    """
    By default, the sort_by is 'database', but let's confirm it doesn't blow up.
    """
    sort_columns_as_configured(yaml_context)


def test_synchronize_data_types(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Synchronizes data types with the DB.
    """
    synchronize_data_types(yaml_context)
