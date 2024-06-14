import pytest

from dbt_osmosis.core.osmosis import DbtYamlManager


@pytest.fixture
def yaml_manager():
    return DbtYamlManager(project_dir="demo_duckdb", profiles_dir="demo_duckdb", dry_run=True)


def test_list(yaml_manager):
    yaml_manager.list()


def test_bootstrap_sources(yaml_manager):
    yaml_manager.bootstrap_sources()


def test_draft_project_structure_update_plan(yaml_manager):
    yaml_manager.draft_project_structure_update_plan()
