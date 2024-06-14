from dbt_osmosis.core.osmosis import DbtYamlManager

manager = DbtYamlManager(project_dir="demo_duckdb", profiles_dir="demo_duckdb", dry_run=True)

def test_list():    
    manager.list()

def test_bootstrap_sources():
    manager.bootstrap_sources()

def test_draft_project_structure_update_plan():
    manager.draft_project_structure_update_plan()
