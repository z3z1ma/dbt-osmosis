import json

from dbt.contracts.graph.manifest import Manifest

from dbt_osmosis.core.column_level_knowledge_propagator import (
    _build_node_ancestor_tree,
    _inherit_column_level_knowledge,
)


def test_build_node_ancestor_tree():
    manifest_path = "tests/data/manifest.json"
    with open(manifest_path, "r") as f:
        manifest_text = f.read()
        manifest_dict = json.loads(manifest_text)
    manifest = Manifest.from_dict(manifest_dict)

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
    manifest_path = "tests/data/manifest.json"
    with open(manifest_path, "r") as f:
        manifest_text = f.read()
        manifest_dict = json.loads(manifest_text)

    manifest_dict["nodes"]["model.jaffle_shop_duckdb.stg_customers"]["columns"]["customer_id"][
        "description"
    ] = "THIS COLUMN IS UPDATED FOR TESTING"
    manifest = Manifest.from_dict(manifest_dict)

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
