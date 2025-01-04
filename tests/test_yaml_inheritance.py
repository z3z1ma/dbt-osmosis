# pyright: reportAny=false, reportUnknownMemberType=false, reportPrivateUsage=false
import json
import typing as t
from pathlib import Path
from unittest import mock

import dbt.version
import pytest
from dbt.contracts.graph.manifest import Manifest
from packaging.version import Version

from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorContext,
    YamlRefactorSettings,
    _build_node_ancestor_tree,
    _get_node_yaml,
    create_dbt_project_context,
    inherit_upstream_column_knowledge,
    sync_node_to_yaml,
)

dbt_version = Version(dbt.version.get_installed_version().to_version_string(skip_matcher=True))


@pytest.fixture(scope="function")
def yaml_context() -> YamlRefactorContext:
    # initializing the context is a sanity test in and of itself
    c = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    c.vars = {"dbt-osmosis": {}}
    project = create_dbt_project_context(c)
    context = YamlRefactorContext(
        project,
        settings=YamlRefactorSettings(
            dry_run=True,
        ),
    )
    return context


def load_manifest() -> Manifest:
    manifest_path = Path(__file__).parent.parent / "demo_duckdb/target/manifest.json"
    with manifest_path.open("r") as f:
        manifest_text = f.read()
        manifest_dict = json.loads(manifest_text)
    return Manifest.from_dict(manifest_dict)


@pytest.mark.parametrize(
    "node_id, expected_tree",
    [
        (
            "model.jaffle_shop_duckdb.customers",
            {
                "generation_0": ["model.jaffle_shop_duckdb.customers"],
                "generation_1": [
                    "model.jaffle_shop_duckdb.stg_customers",
                    "model.jaffle_shop_duckdb.stg_orders",
                    "model.jaffle_shop_duckdb.stg_payments",
                ],
                "generation_2": [
                    "seed.jaffle_shop_duckdb.raw_customers",
                    "seed.jaffle_shop_duckdb.raw_orders",
                    "seed.jaffle_shop_duckdb.raw_payments",
                ],
            },
        ),
    ],
)
def test_build_node_ancestor_tree(node_id: str, expected_tree: dict[str, list[str]]):
    """Test the build node ancestor tree functionality."""
    manifest = load_manifest()
    target_node = manifest.nodes[node_id]
    assert _build_node_ancestor_tree(manifest, target_node) == expected_tree


# NOTE: downstream node has the following set in the test body, keep these in mind when creating cases
# local_column.description = "I was steadfast and unyielding"
# local_column.tags = ["baz"]
# local_column.meta = {"c": 3}
@pytest.mark.parametrize(
    "settings, upstream_mutations, downstream_metadata",
    [
        # Case 1a: Add progenitor to meta and force inherit descriptions = false (default)
        (
            {"force_inherit_descriptions": False, "add_progenitor_to_meta": True},
            {
                "stg_customers.customer_id": {
                    "description": "I will be inherited, forcibly so :)",
                    "meta": {"a": 1, "b": 2},
                    "tags": ["foo", "bar"],
                }
            },
            {
                "description": "I was steadfast and unyielding",
                "meta": {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
                },
                "tags": ["foo", "bar", "baz"],
            },
        ),
        # Case 1b: Add progenitor to meta and force inherit descriptions = true
        (
            {"force_inherit_descriptions": True, "add_progenitor_to_meta": True},
            {
                "stg_customers.customer_id": {
                    "description": "I will be inherited, forcibly so :)",
                    "meta": {"a": 1, "b": 2},
                    "tags": ["foo", "bar"],
                }
            },
            {
                "description": "I will be inherited, forcibly so :)",
                "meta": {
                    "a": 1,
                    "b": 2,
                    "c": 3,
                    "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
                },
                "tags": ["foo", "bar", "baz"],
            },
        ),
        # Case 2: Skip add tags and merge meta
        (
            {"skip_add_tags": True, "skip_merge_meta": True},
            {
                "stg_customers.customer_id": {
                    "description": "I will not be inherited, since the customer table documents me",
                    "meta": {"a": 1},
                    "tags": ["foo", "bar"],
                }
            },
            {
                "description": "I was steadfast and unyielding",
                "meta": {"c": 3},
                "tags": ["baz"],
            },
        ),
        # Case 3: Use unrendered descriptions
        (
            {"use_unrendered_descriptions": True, "force_inherit_descriptions": True},
            {
                "stg_customers.customer_id": {
                    "description": "{{ doc('stg_customer_description') }}",
                    "meta": {"d": 4},
                    "tags": ["rendered", "unrendered"],
                }
            },
            {
                "description": "{{ doc('stg_customer_description') }}",
                "meta": {"c": 3, "d": 4},
                "tags": ["rendered", "unrendered", "baz"],
            },
        ),
        # Case 4: Skip add data types but inherit specified keys
        (
            {"skip_add_data_types": True, "add_inheritance_for_specified_keys": ["quote"]},
            {
                "stg_customers.customer_id": {
                    "description": "Keep on, keeping on",
                    "meta": {"e": 5},
                    "tags": ["constrainted"],
                    "quote": True,
                }
            },
            {
                "description": "I was steadfast and unyielding",
                "meta": {"c": 3, "e": 5},
                "tags": ["constrainted", "baz"],
                "quote": True,
            },
        ),
        # Case 5: Output to lowercase
        (
            {"output_to_lower": True},
            {
                "stg_customers.customer_id": {
                    "name": "WTF",
                }
            },
            {
                "name": "wtf",
                "description": "I was steadfast and unyielding",
                "meta": {"c": 3},
                "tags": ["baz"],
            },
        ),
        # Case 6: Add inheritance for any specified keys
        (
            {
                "skip_add_tags": False,
                "skip_merge_meta": False,
                "add_inheritance_for_specified_keys": ["policy_tags"],
                "force_inherit_descriptions": True,
            },
            {
                "stg_customers.customer_id": {
                    "description": "I will prevail",
                    "meta": {"a": 1},
                    "tags": ["foo", "bar"],
                    "_extra": {"policy_tags": ["pii_main"]},
                }
            },
            {
                "description": "I will prevail",
                "meta": {"a": 1, "c": 3},
                "tags": ["foo", "bar", "baz"],
                "policy_tags": ["pii_main"],
            },
        ),
    ],
)
def test_inherit_upstream_column_knowledge_with_various_settings(
    yaml_context: YamlRefactorContext,
    settings: dict[str, t.Any],
    upstream_mutations: dict[str, t.Any],
    downstream_metadata: dict[str, t.Any],
):
    """Test inherit_upstream_column_knowledge with various settings and configurations."""
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    local_column = target_node.columns["customer_id"]
    local_column.description = "I was steadfast and unyielding"
    local_column.tags = ["baz"]
    local_column.meta = {"c": 3}

    # Apply settings
    for key, value in settings.items():
        setattr(yaml_context.settings, key, value)

    # Modify upstream column data
    for column_path, mods in upstream_mutations.items():
        node_id, column_name = column_path.split(".")
        upstream_col = manifest.nodes[f"model.jaffle_shop_duckdb.{node_id}"].columns[column_name]
        for attr, attr_value in mods.items():
            setattr(upstream_col, attr, attr_value)

    # Perform inheritance
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        _ = inherit_upstream_column_knowledge(yaml_context, target_node)
        sync_node_to_yaml(yaml_context, target_node, commit=False)
        yaml_slice = _get_node_yaml(yaml_context, target_node)

    # Assert metadata, description, and tags
    cid = target_node.columns["customer_id"]
    assert cid.description == downstream_metadata["description"]
    assert cid.meta == downstream_metadata["meta"]
    assert sorted(cid.tags) == sorted(downstream_metadata["tags"])

    # Validate YAML output
    assert yaml_slice
    yaml_column = yaml_slice["columns"][0]
    assert yaml_column["description"] == downstream_metadata["description"]
    assert yaml_column["meta"] == downstream_metadata["meta"]
    assert sorted(yaml_column["tags"]) == sorted(downstream_metadata["tags"])


@pytest.mark.parametrize(
    "use_unrendered_descriptions, expected_start",
    [
        (True, '{{ doc("orders_status") }}'),
        (False, "Orders can be one of the following statuses:"),
    ],
)
def test_use_unrendered_descriptions(
    yaml_context: YamlRefactorContext, use_unrendered_descriptions: bool, expected_start: str
):
    """Test the handling of unrendered descriptions."""
    manifest = yaml_context.project.manifest
    target_node = manifest.nodes["model.jaffle_shop_duckdb.orders"]
    yaml_context.settings.use_unrendered_descriptions = use_unrendered_descriptions
    yaml_context.settings.force_inherit_descriptions = True

    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        _ = inherit_upstream_column_knowledge(yaml_context, target_node)
        sync_node_to_yaml(yaml_context, target_node)

    assert target_node.columns["status"].description.startswith(expected_start)


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext):
    manifest = yaml_context.project.manifest
    manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ].description = "THIS COLUMN IS UPDATED FOR TESTING"

    expect: dict[str, t.Any] = {
        "customer_id": {
            "name": "customer_id",
            "description": "THIS COLUMN IS UPDATED FOR TESTING",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers"},
            "data_type": "INTEGER",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "first_name": {
            "name": "first_name",
            "description": "Customer's first name. PII.",
            "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "last_name": {
            "name": "last_name",
            "description": "Customer's last name. PII.",
            "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
            "data_type": "VARCHAR",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "first_order": {
            "name": "first_order",
            "description": "Date (UTC) of a customer's first order",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DATE",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "most_recent_order": {
            "name": "most_recent_order",
            "description": "Date (UTC) of a customer's most recent order",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DATE",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "number_of_orders": {
            "name": "number_of_orders",
            "description": "Count of the number of orders a customer has placed",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "BIGINT",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "customer_lifetime_value": {
            "name": "customer_lifetime_value",
            "description": "",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DOUBLE",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
        "customer_average_value": {
            "name": "customer_average_value",
            "description": "",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DECIMAL(18,3)",
            "constraints": [],
            "quote": None,
            "tags": [],
        },
    }
    if dbt_version >= Version("1.9.0"):
        for column in expect.keys():
            expect[column]["granularity"] = None

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    target_node.columns["customer_id"].description = ""

    yaml_context.placeholders = ("",)
    yaml_context.settings.add_progenitor_to_meta = True

    # Perform inheritance on the node
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        _ = inherit_upstream_column_knowledge(yaml_context, target_node)

    assert {k: v.to_dict() for k, v in target_node.columns.items()} == expect
