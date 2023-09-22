import json

from dbt.contracts.graph.manifest import Manifest

from dbt_osmosis.core.column_level_knowledge_propagator import (
    ColumnLevelKnowledgePropagator,
    _build_node_ancestor_tree,
    _inherit_column_level_knowledge,
)


def load_manifest() -> Manifest:
    manifest_path = "tests/data/manifest.json"
    with open(manifest_path, "r") as f:
        manifest_text = f.read()
        manifest_dict = json.loads(manifest_text)
    return Manifest.from_dict(manifest_dict)


def test_build_node_ancestor_tree():
    manifest = load_manifest()
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    expect = {
        "generation_0": [
            "model.jaffle_shop_duckdb.stg_customers",
            "model.jaffle_shop_duckdb.stg_orders",
            "model.jaffle_shop_duckdb.stg_payments",
        ],
        "generation_1": [
            "seed.jaffle_shop_duckdb.raw_customers",
            "seed.jaffle_shop_duckdb.raw_orders",
            "seed.jaffle_shop_duckdb.raw_payments",
        ],
    }
    assert _build_node_ancestor_tree(manifest, target_node) == expect


def test_inherit_column_level_knowledge():
    manifest = load_manifest()
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"

    expect = {
        "customer_id": {
            "progenitor": "model.jaffle_shop_duckdb.stg_customers",
            "generation": "generation_0",
            "name": "customer_id",
            "description": "THIS COLUMN IS UPDATED FOR TESTING",
            "data_type": "INTEGER",
            "constraints": [],
            "quote": None,
        },
        "first_name": {
            "progenitor": "model.jaffle_shop_duckdb.stg_customers",
            "generation": "generation_0",
            "name": "first_name",
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
        },
        "last_name": {
            "progenitor": "model.jaffle_shop_duckdb.stg_customers",
            "generation": "generation_0",
            "name": "last_name",
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
        },
        "order_id": {
            "progenitor": "model.jaffle_shop_duckdb.stg_orders",
            "generation": "generation_0",
            "name": "order_id",
            "data_type": "INTEGER",
            "constraints": [],
            "quote": None,
        },
        "order_date": {
            "progenitor": "model.jaffle_shop_duckdb.stg_orders",
            "generation": "generation_0",
            "name": "order_date",
            "data_type": "DATE",
            "constraints": [],
            "quote": None,
        },
        "status": {
            "progenitor": "model.jaffle_shop_duckdb.stg_orders",
            "generation": "generation_0",
            "name": "status",
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
        },
        "payment_id": {
            "progenitor": "model.jaffle_shop_duckdb.stg_payments",
            "generation": "generation_0",
            "name": "payment_id",
            "data_type": "INTEGER",
            "constraints": [],
            "quote": None,
        },
        "payment_method": {
            "progenitor": "model.jaffle_shop_duckdb.stg_payments",
            "generation": "generation_0",
            "name": "payment_method",
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
        },
        "amount": {
            "progenitor": "model.jaffle_shop_duckdb.stg_payments",
            "generation": "generation_0",
            "name": "amount",
            "data_type": "DOUBLE",
            "constraints": [],
            "quote": None,
        },
    }
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    family_tree = _build_node_ancestor_tree(manifest, target_node)
    placeholders = [""]
    assert _inherit_column_level_knowledge(manifest, family_tree, placeholders) == expect


def test_update_undocumented_columns_with_prior_knowledge():
    manifest = load_manifest()
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
        "my_key": "my_value"
    }
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
        "my_tag1",
        "my_tag2",
    ]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
        manifest, target_node, placeholders=[""]
    )
    yaml_file_model_section = {
        "columns": [
            {
                "name": "customer_id",
            }
        ]
    }
    undocumented_columns = target_node.columns.keys()
    ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
        undocumented_columns,
        target_node,
        yaml_file_model_section,
        knowledge,
        skip_add_tags=False,
        skip_merge_meta=False,
        add_progenitor_to_meta=False,
    )

    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert yaml_file_model_section["columns"][0]["meta"] == {"my_key": "my_value"}
    assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])

    assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert target_node.columns["customer_id"].meta == {"my_key": "my_value"}
    assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])


def test_update_undocumented_columns_with_prior_knowledge_skip_add_tags():
    manifest = load_manifest()
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
        "my_key": "my_value"
    }
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
        "my_tag1",
        "my_tag2",
    ]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
        manifest, target_node, placeholders=[""]
    )
    yaml_file_model_section = {
        "columns": [
            {
                "name": "customer_id",
            }
        ]
    }
    undocumented_columns = target_node.columns.keys()
    ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
        undocumented_columns,
        target_node,
        yaml_file_model_section,
        knowledge,
        skip_add_tags=True,
        skip_merge_meta=False,
        add_progenitor_to_meta=False,
    )

    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert yaml_file_model_section["columns"][0]["meta"] == {"my_key": "my_value"}
    assert "tags" not in yaml_file_model_section["columns"][0]

    assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert target_node.columns["customer_id"].meta == {"my_key": "my_value"}
    assert set(target_node.columns["customer_id"].tags) == set([])


def test_update_undocumented_columns_with_prior_knowledge_skip_merge_meta():
    manifest = load_manifest()
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
        "my_key": "my_value"
    }
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
        "my_tag1",
        "my_tag2",
    ]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
        manifest, target_node, placeholders=[""]
    )
    yaml_file_model_section = {
        "columns": [
            {
                "name": "customer_id",
            }
        ]
    }
    undocumented_columns = target_node.columns.keys()
    ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
        undocumented_columns,
        target_node,
        yaml_file_model_section,
        knowledge,
        skip_add_tags=False,
        skip_merge_meta=True,
        add_progenitor_to_meta=False,
    )

    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert "meta" not in yaml_file_model_section["columns"][0]
    assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])

    assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert target_node.columns["customer_id"].meta == {}
    assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])


def test_update_undocumented_columns_with_prior_knowledge_add_progenitor_to_meta():
    manifest = load_manifest()
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].meta = {
        "my_key": "my_value"
    }
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"].tags = [
        "my_tag1",
        "my_tag2",
    ]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    knowledge = ColumnLevelKnowledgePropagator.get_node_columns_with_inherited_knowledge(
        manifest, target_node, placeholders=[""]
    )
    yaml_file_model_section = {
        "columns": [
            {
                "name": "customer_id",
            }
        ]
    }
    undocumented_columns = target_node.columns.keys()
    ColumnLevelKnowledgePropagator.update_undocumented_columns_with_prior_knowledge(
        undocumented_columns,
        target_node,
        yaml_file_model_section,
        knowledge,
        skip_add_tags=False,
        skip_merge_meta=False,
        add_progenitor_to_meta=True,
    )

    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert yaml_file_model_section["columns"][0]["meta"] == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }
    assert set(yaml_file_model_section["columns"][0]["tags"]) == set(["my_tag1", "my_tag2"])

    assert target_node.columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert target_node.columns["customer_id"].meta == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }
    assert set(target_node.columns["customer_id"].tags) == set(["my_tag1", "my_tag2"])
