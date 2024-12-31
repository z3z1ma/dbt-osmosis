# TODO: refactor this test
# from pathlib import Path
#
# import pytest
# from dbt.contracts.results import CatalogKey
#
# from dbt_osmosis.core.osmosis import DbtYamlManager
#
#
# @pytest.fixture(scope="module")
# def yaml_manager() -> DbtYamlManager:
#     return DbtYamlManager(project_dir="demo_duckdb", profiles_dir="demo_duckdb", dry_run=True)
#
#
# def test_initialize_adapter(yaml_manager: DbtYamlManager):
#     yaml_manager.initialize_adapter()
#
#
# def test_list(yaml_manager: DbtYamlManager):
#     yaml_manager.list()
#
#
# def test_test(yaml_manager: DbtYamlManager):
#     yaml_manager.test()
#
#
# def test_run(yaml_manager: DbtYamlManager):
#     yaml_manager.run()
#
#
# def test_build(yaml_manager: DbtYamlManager):
#     yaml_manager.build()
#
#
# def test_parse_project(yaml_manager: DbtYamlManager):
#     yaml_manager.parse_project()
#
#
# def test_safe_parse_project(yaml_manager: DbtYamlManager):
#     yaml_manager.safe_parse_project()
#
#
# def test_bootstrap_sources(yaml_manager: DbtYamlManager):
#     yaml_manager.bootstrap_sources()
#
#
# def test_draft_project_structure_update_plan(yaml_manager: DbtYamlManager):
#     yaml_manager.draft_project_structure_update_plan()
#
#
# def test_commit_project_restructure_to_disk(yaml_manager: DbtYamlManager):
#     yaml_manager.commit_project_restructure_to_disk()
#
#
# def test_propagate_documentation_downstream(yaml_manager: DbtYamlManager):
#     yaml_manager.propagate_documentation_downstream()
#
#
# def _customer_column_types(yaml_manager: DbtYamlManager) -> dict[str, str]:
#     node = next(n for n in yaml_manager.manifest.nodes.values() if n.name == "customers")
#     assert node
#
#     catalog_key = yaml_manager.get_catalog_key(node)
#     columns = yaml_manager.get_columns_meta(catalog_key)
#     assert columns
#
#     column_types = dict({name: meta.type for name, meta in columns.items()})
#     assert column_types
#     return column_types
#
#
# def test_get_columns_meta(yaml_manager: DbtYamlManager):
#     assert _customer_column_types(yaml_manager) == {
#         # in DuckDB decimals always have presision and scale
#         "customer_average_value": "DECIMAL(18,3)",
#         "customer_id": "INTEGER",
#         "customer_lifetime_value": "DOUBLE",
#         "first_name": "VARCHAR",
#         "first_order": "DATE",
#         "last_name": "VARCHAR",
#         "most_recent_order": "DATE",
#         "number_of_orders": "BIGINT",
#     }
#
#
# def test_get_columns_meta_char_length():
#     yaml_manager = DbtYamlManager(
#         project_dir="demo_duckdb", profiles_dir="demo_duckdb", char_length=True, dry_run=True
#     )
#     assert _customer_column_types(yaml_manager) == {
#         # in DuckDB decimals always have presision and scale
#         "customer_average_value": "DECIMAL(18,3)",
#         "customer_id": "INTEGER",
#         "customer_lifetime_value": "DOUBLE",
#         "first_name": "character varying(256)",
#         "first_order": "DATE",
#         "last_name": "character varying(256)",
#         "most_recent_order": "DATE",
#         "number_of_orders": "BIGINT",
#     }
#
#
# def test_get_columns_meta_numeric_precision():
#     yaml_manager = DbtYamlManager(
#         project_dir="demo_duckdb", profiles_dir="demo_duckdb", numeric_precision=True, dry_run=True
#     )
#     assert _customer_column_types(yaml_manager) == {
#         # in DuckDB decimals always have presision and scale
#         "customer_average_value": "DECIMAL(18,3)",
#         "customer_id": "INTEGER",
#         "customer_lifetime_value": "DOUBLE",
#         "first_name": "VARCHAR",
#         "first_order": "DATE",
#         "last_name": "VARCHAR",
#         "most_recent_order": "DATE",
#         "number_of_orders": "BIGINT",
#     }
