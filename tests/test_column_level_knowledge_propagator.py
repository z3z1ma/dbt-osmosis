# TODO: refactor this test
# import json
# from pathlib import Path
#
# import dbt.version
# import pytest
# from dbt.contracts.graph.manifest import Manifest
# from packaging.version import Version
#
# from dbt_osmosis.core.column_level_knowledge_propagator import (
#     ColumnLevelKnowledgePropagator,
#     _build_node_ancestor_tree,
#     _inherit_column_level_knowledge,
# )
#
# dbt_version = Version(dbt.version.get_installed_version().to_version_string(skip_matcher=True))
#
#
# def load_manifest() -> Manifest:
#     manifest_path = Path(__file__).parent.parent / "demo_duckdb/target/manifest.json"
#     with manifest_path.open("r") as f:
#         manifest_text = f.read()
#         manifest_dict = json.loads(manifest_text)
#     return Manifest.from_dict(manifest_dict)
#
#
# def test_build_node_ancestor_tree():
#     manifest = load_manifest()
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
#     expect = {
#         "generation_0": [
#             "model.jaffle_shop_duckdb.stg_customers",
#             "model.jaffle_shop_duckdb.stg_orders",
#             "model.jaffle_shop_duckdb.stg_payments",
#         ],
#         "generation_1": [
#             "seed.jaffle_shop_duckdb.raw_customers",
#             "seed.jaffle_shop_duckdb.raw_orders",
#             "seed.jaffle_shop_duckdb.raw_payments",
#         ],
#     }
#     assert _build_node_ancestor_tree(manifest, target_node) == expect
#
#
# def test_inherit_column_level_knowledge():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["seed.jaffle_shop_duckdb.raw_orders"].columns[
#         "status"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#
#     expect = {
#         "customer_id": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_customers",
#             "generation": "generation_0",
#             "name": "customer_id",
#             "description": "THIS COLUMN IS UPDATED FOR TESTING",
#             "data_type": "INTEGER",
#             "constraints": [],
#             "quote": None,
#         },
#         "first_name": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_customers",
#             "generation": "generation_0",
#             "name": "first_name",
#             "data_type": "VARCHAR",
#             "constraints": [],
#             "quote": None,
#         },
#         "last_name": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_customers",
#             "generation": "generation_0",
#             "name": "last_name",
#             "data_type": "VARCHAR",
#             "constraints": [],
#             "quote": None,
#         },
#         "rank": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_customers",
#             "generation": "generation_0",
#             "name": "rank",
#             "data_type": "VARCHAR",
#             "constraints": [],
#             "quote": None,
#         },
#         "order_id": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_orders",
#             "generation": "generation_0",
#             "name": "order_id",
#             "data_type": "INTEGER",
#             "constraints": [],
#             "quote": None,
#         },
#         "order_date": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_orders",
#             "generation": "generation_0",
#             "name": "order_date",
#             "data_type": "DATE",
#             "constraints": [],
#             "quote": None,
#         },
#         "status": {
#             "progenitor": "seed.jaffle_shop_duckdb.raw_orders",
#             "generation": "generation_1",
#             "name": "status",
#             "description": "THIS COLUMN IS UPDATED FOR TESTING",
#             "data_type": "VARCHAR",
#             "constraints": [],
#             "quote": None,
#         },
#         "payment_id": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_payments",
#             "generation": "generation_0",
#             "name": "payment_id",
#             "data_type": "INTEGER",
#             "constraints": [],
#             "quote": None,
#         },
#         "payment_method": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_payments",
#             "generation": "generation_0",
#             "name": "payment_method",
#             "data_type": "VARCHAR",
#             "constraints": [],
#             "quote": None,
#         },
#         "amount": {
#             "progenitor": "model.jaffle_shop_duckdb.stg_payments",
#             "generation": "generation_0",
#             "name": "amount",
#             "data_type": "DOUBLE",
#             "constraints": [],
#             "quote": None,
#         },
#     }
#     if dbt_version >= Version("1.9.0"):
#         for key in expect.keys():
#             expect[key]["granularity"] = None
#
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
#     family_tree = _build_node_ancestor_tree(manifest, target_node)
#     placeholders = [""]
#     assert _inherit_column_level_knowledge(manifest, family_tree, placeholders) == expect
#
#
# def test_update_undocumented_columns_with_prior_knowledge():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value"
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#     manifest.nodes[target_node_name].columns["customer_id"].tags = set(
#         [
#             "my_tag3",
#             "my_tag4",
#         ]
#     )
#     manifest.nodes[target_node_name].columns["customer_id"].meta = {
#         "my_key": "my_old_value",
#         "my_new_key": "my_new_value",
#     }
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#
#     assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_id"].meta == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#     }
#     assert set(target_node.columns["customer_id"].tags) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#
#
# def test_update_undocumented_columns_with_prior_knowledge_skip_add_tags():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value"
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=True,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert yaml_file_model_section["columns"][0]["meta"] == {"my_key": "my_value"}
#     assert "tags" not in yaml_file_model_section["columns"][0]
#
#     assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_id"].meta == {"my_key": "my_value"}
#     assert set(target_node.columns["customer_id"].tags) == set([])
#
#
# def test_update_undocumented_columns_with_prior_knowledge_skip_merge_meta():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value"
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=True,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert "meta" not in yaml_file_model_section["columns"][0]
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])
#
#     assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_id"].meta == {}
#     assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])
#
#
# def test_update_undocumented_columns_with_prior_knowledge_add_progenitor_to_meta():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value"
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=True,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])
#
#     assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_id"].meta == {
#         "my_key": "my_value",
#         "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
#     }
#     assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])
#
#
# def test_update_undocumented_columns_with_prior_knowledge_with_osmosis_keep_description():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value",
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     column_description_not_updated = (
#         "This column will not be updated as it has the 'osmosis_keep_description' attribute"
#     )
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#
#     manifest.nodes[target_node_name].columns[
#         "customer_id"
#     ].description = column_description_not_updated
#     manifest.nodes[target_node_name].columns["customer_id"].tags = set(
#         [
#             "my_tag3",
#             "my_tag4",
#         ]
#     )
#     manifest.nodes[target_node_name].columns["customer_id"].meta = {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#     }
#
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=True,
#         skip_merge_meta=True,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert yaml_file_model_section["columns"][0]["description"] == column_description_not_updated
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag3", "my_tag4"])
#
#     assert target_node.columns["customer_id"].description == column_description_not_updated
#     assert target_node.columns["customer_id"].meta == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#     }
#     assert set(target_node.columns["customer_id"].tags) == set(["my_tag3", "my_tag4"])
#
#
# def test_update_undocumented_columns_with_prior_knowledge_add_progenitor_to_meta_and_osmosis_keep_description():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value",
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     column_description_not_updated = (
#         "This column will not be updated as it has the 'osmosis_keep_description' attribute"
#     )
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#
#     manifest.nodes[target_node_name].columns[
#         "customer_id"
#     ].description = column_description_not_updated
#     manifest.nodes[target_node_name].columns["customer_id"].meta = {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#     }
#
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=True,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert yaml_file_model_section["columns"][0]["description"] == column_description_not_updated
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#         "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])
#
#     assert target_node.columns["customer_id"].description == column_description_not_updated
#     assert target_node.columns["customer_id"].meta == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#         "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
#     }
#     assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])
#
#
# def test_update_undocumented_columns_with_prior_knowledge_with_add_inheritance_for_specified_keys():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "customer_id"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
#         "my_key": "my_value"
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"]._extra = {
#         "policy_tags": ["my_policy_tag1"],
#     }
#
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#     manifest.nodes[target_node_name].columns["customer_id"].tags = set(
#         [
#             "my_tag3",
#             "my_tag4",
#         ]
#     )
#     manifest.nodes[target_node_name].columns["customer_id"].meta = {
#         "my_key": "my_old_value",
#         "my_new_key": "my_new_value",
#     }
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_id",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=False,
#         add_inheritance_for_specified_keys=["policy_tags"],
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#     assert set(yaml_file_model_section["columns"][0]["policy_tags"]) == set(["my_policy_tag1"])
#
#     assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_id"].meta == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#     }
#     assert set(target_node.columns["customer_id"].tags) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#     assert set(target_node.columns["customer_id"]._extra["policy_tags"]) == set(["my_policy_tag1"])
#
#
# def test_update_undocumented_columns_with_osmosis_prefix_meta_with_prior_knowledge():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "rank"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["rank"].meta = {
#         "my_key": "my_value",
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["rank"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#     manifest.nodes[target_node_name].columns["customer_rank"].tags = set(
#         [
#             "my_tag3",
#             "my_tag4",
#         ]
#     )
#     manifest.nodes[target_node_name].columns["customer_rank"].meta = {
#         "my_key": "my_old_value",
#         "my_new_key": "my_new_value",
#         "osmosis_prefix": "customer_",
#     }
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_rank",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=False,
#         skip_merge_meta=False,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_rank"
#     assert (
#         yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
#     )
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#         "osmosis_prefix": "customer_",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#
#     assert target_node.columns["customer_rank"].description == "THIS COLUMN IS UPDATED FOR TESTING"
#     assert target_node.columns["customer_rank"].meta == {
#         "my_key": "my_value",
#         "my_new_key": "my_new_value",
#         "osmosis_prefix": "customer_",
#     }
#     assert set(target_node.columns["customer_rank"].tags) == set(
#         ["my_tag1", "my_tag2", "my_tag3", "my_tag4"]
#     )
#
#
# def test_update_undocumented_columns_with_osmosis_prefix_meta_with_prior_knowledge_with_osmosis_keep_description():
#     manifest = load_manifest()
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
#         "rank"
#     ].description = "THIS COLUMN IS UPDATED FOR TESTING"
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["rank"].meta = {
#         "my_key": "my_value",
#     }
#     manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["rank"].tags = [
#         "my_tag1",
#         "my_tag2",
#     ]
#
#     column_description_not_updated = (
#         "This column will not be updated as it has the 'osmosis_keep_description' attribute"
#     )
#     target_node_name = "model.jaffle_shop_duckdb.customers"
#
#     manifest.nodes[target_node_name].columns[
#         "customer_rank"
#     ].description = column_description_not_updated
#     manifest.nodes[target_node_name].columns["customer_rank"].tags = set(
#         [
#             "my_tag3",
#             "my_tag4",
#         ]
#     )
#     manifest.nodes[target_node_name].columns["customer_rank"].meta = {
#         "my_key": "my_value",
#         "osmosis_prefix": "customer_",
#         "osmosis_keep_description": True,
#     }
#
#     target_node = manifest.nodes[target_node_name]
#     knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
#         manifest, target_node, placeholders=[""]
#     )
#     yaml_file_model_section = {
#         "columns": [
#             {
#                 "name": "customer_rank",
#             }
#         ]
#     }
#     undocumented_columns = target_node.columns.keys()
#     ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
#         undocumented_columns,
#         target_node,
#         yaml_file_model_section,
#         knowledge,
#         skip_add_tags=True,
#         skip_merge_meta=True,
#         add_progenitor_to_meta=False,
#     )
#
#     assert yaml_file_model_section["columns"][0]["name"] == "customer_rank"
#     assert yaml_file_model_section["columns"][0]["description"] == column_description_not_updated
#     assert yaml_file_model_section["columns"][0]["meta"] == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#         "osmosis_prefix": "customer_",
#     }
#     assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag3", "my_tag4"])
#
#     assert target_node.columns["customer_rank"].description == column_description_not_updated
#     assert target_node.columns["customer_rank"].meta == {
#         "my_key": "my_value",
#         "osmosis_keep_description": True,
#         "osmosis_prefix": "customer_",
#     }
#     assert set(target_node.columns["customer_rank"].tags) == set(["my_tag3", "my_tag4"])
#
#
# @pytest.mark.parametrize("use_unrendered_descriptions", [True, False])
# def test_use_unrendered_descriptions(use_unrendered_descriptions):
#     manifest = load_manifest()
#     # changing directory, assuming that I need to carry profile_dir through as this doesn't work outside of the dbt project
#     project_dir = Path(__file__).parent.parent / "demo_duckdb"
#     target_node = manifest.nodes["model.jaffle_shop_duckdb.orders"]
#     placeholders = [""]
#     family_tree = _build_node_ancestor_tree(manifest, target_node)
#     knowledge = _inherit_column_level_knowledge(
#         manifest,
#         family_tree,
#         placeholders,
#         project_dir,
#         use_unrendered_descriptions=use_unrendered_descriptions,
#     )
#     if use_unrendered_descriptions:
#         expected = '{{ doc("orders_status") }}'
#     else:
#         expected = "Orders can be one of the following statuses:"
#     assert knowledge["status"]["description"].startswith(
#         expected
#     )  # starts with so I don't have to worry about linux/windows line endings
