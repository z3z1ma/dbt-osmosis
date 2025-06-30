# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.sync_operations import sync_node_to_yaml
from dbt_osmosis.core.schema.writer import commit_yamls


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
