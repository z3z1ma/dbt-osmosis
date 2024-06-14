import pytest

from dbt_osmosis.core.osmosis import DbtYamlManager


@pytest.fixture(scope="module")
def yaml_manager() -> DbtYamlManager:
    return DbtYamlManager(project_dir="demo_duckdb", profiles_dir="demo_duckdb", dry_run=True)


def test_initialize_adapter(yaml_manager: DbtYamlManager):
    yaml_manager.initialize_adapter()


def test_list(yaml_manager: DbtYamlManager):
    yaml_manager.list()


def test_test(yaml_manager: DbtYamlManager):
    yaml_manager.test()


def test_run(yaml_manager: DbtYamlManager):
    yaml_manager.run()


def test_build(yaml_manager: DbtYamlManager):
    yaml_manager.build()


def test_parse_project(yaml_manager: DbtYamlManager):
    yaml_manager.parse_project()


def test_safe_parse_project(yaml_manager: DbtYamlManager):
    yaml_manager.safe_parse_project()


def test_bootstrap_sources(yaml_manager: DbtYamlManager):
    yaml_manager.bootstrap_sources()


def test_draft_project_structure_update_plan(yaml_manager: DbtYamlManager):
    yaml_manager.draft_project_structure_update_plan()


def test_commit_project_restructure_to_disk(yaml_manager: DbtYamlManager):
    yaml_manager.commit_project_restructure_to_disk()


def test_propagate_documentation_downstream(yaml_manager: DbtYamlManager):
    yaml_manager.propagate_documentation_downstream()
