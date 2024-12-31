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
    _get_member_yaml,
    create_dbt_project_context,
    inherit_upstream_column_knowledge,
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


def test_build_node_ancestor_tree():
    manifest = load_manifest()
    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    expect = {
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
    }
    assert _build_node_ancestor_tree(manifest, target_node) == expect


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
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "first_name": {
            "name": "first_name",
            "description": "Customer's first name. PII.",
            "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "last_name": {
            "name": "last_name",
            "description": "Customer's last name. PII.",
            "meta": {"osmosis_progenitor": "seed.jaffle_shop_duckdb.raw_customers"},
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "first_order": {
            "name": "first_order",
            "description": "Date (UTC) of a customer's first order",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "most_recent_order": {
            "name": "most_recent_order",
            "description": "Date (UTC) of a customer's most recent order",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "number_of_orders": {
            "name": "number_of_orders",
            "description": "Count of the number of orders a customer has placed",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": None,
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "customer_lifetime_value": {
            "name": "customer_lifetime_value",
            "description": "",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DOUBLE",
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
        "customer_average_value": {
            "name": "customer_average_value",
            "description": "",
            "meta": {"osmosis_progenitor": "model.jaffle_shop_duckdb.customers"},
            "data_type": "DECIMAL(18,3)",
            "constraints": [],
            "quote": None,
            "tags": [],
            "granularity": None,
        },
    }
    if dbt_version >= Version("1.9.0"):
        for column in expect.keys():
            expect[column]["granularity"] = None

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    # NOTE: we will only update empty / placeholders descriptions by design, see force_inherit_descriptions for legacy behavior
    target_node.columns["customer_id"].description = ""

    yaml_context.placeholders = ("",)
    yaml_context.settings.add_progenitor_to_meta = True

    # Perform inheritance on the node
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)

    assert {k: v.to_dict() for k, v in target_node.columns.items()} == expect


def test_inherit_upstream_column_knowledge_with_mutations(yaml_context: YamlRefactorContext):
    yaml_context.settings.force_inherit_descriptions = True  # NOTE: matches legacy behavior

    manifest = yaml_context.project.manifest
    customer_id_column = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ]
    customer_id_column.description = "THIS COLUMN IS UPDATED FOR TESTING"
    customer_id_column.meta = {"my_key": "my_value"}
    customer_id_column.tags = ["my_tag1", "my_tag2"]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    target_node_customer_id = target_node.columns["customer_id"]
    target_node_customer_id.tags = ["my_tag3", "my_tag4"]
    target_node_customer_id.meta = {"my_key": "my_local_value", "my_new_key": "my_new_value"}

    # Perform inheritance on the node
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        yaml_file_model_section = _get_member_yaml(yaml_context, target_node)

    target_node_customer_id = target_node.columns["customer_id"]
    assert target_node_customer_id.description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert (
        target_node_customer_id.meta
        == {
            "my_key": "my_local_value",  # NOTE: keys on the node itself always take precedence, hence `my_key` was not overridden
            "my_new_key": "my_new_value",
        }
    )
    assert sorted(target_node_customer_id.tags) == [
        "my_tag1",
        "my_tag2",
        "my_tag3",
        "my_tag4",
    ]

    assert yaml_file_model_section
    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert yaml_file_model_section["columns"][0]["meta"] == {
        "my_key": "my_local_value",
        "my_new_key": "my_new_value",
    }
    assert sorted(yaml_file_model_section["columns"][0]["tags"]) == [
        "my_tag1",
        "my_tag2",
        "my_tag3",
        "my_tag4",
    ]


def test_inherit_upstream_column_knowledge_skip_add_tags(yaml_context: YamlRefactorContext):
    yaml_context.settings.skip_add_tags = True
    yaml_context.settings.skip_merge_meta = False

    manifest = yaml_context.project.manifest
    customer_id_column = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ]
    customer_id_column.description = "THIS COLUMN IS UPDATED FOR TESTING"
    customer_id_column.meta = {"my_key": "my_value"}
    customer_id_column.tags = ["my_tag1", "my_tag2"]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    target_node_customer_id = target_node.columns["customer_id"]
    target_node_customer_id.description = ""  # NOTE: allow inheritance to update this
    target_node_customer_id.tags = ["my_tag3", "my_tag4"]
    target_node_customer_id.meta = {"my_key": "my_value"}

    # Perform inheritance on the node
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        yaml_file_model_section = _get_member_yaml(yaml_context, target_node)

    target_node_customer_id = target_node.columns["customer_id"]
    assert target_node_customer_id.description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert target_node_customer_id.meta == {"my_key": "my_value"}
    assert (
        sorted(target_node_customer_id.tags) == ["my_tag3", "my_tag4"]
    )  # NOTE: nodes tags are not mutated beyond our original mutation in the manifest node since skip_add_tags is True

    assert yaml_file_model_section
    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    assert yaml_file_model_section["columns"][0]["meta"] == {"my_key": "my_value"}
    # TODO: consider a function which synchronizes a node with its yaml buffer, and then consider if inherit_upstream_column_knowledge should sync nodes
    # in which case it would pick up manual mutations to the node and apply them to the yaml buffer (which could be useful I think)
    assert (
        yaml_file_model_section["columns"][0].get("tags", []) == []
    )  # NOTE: yaml tags do not exist in buffer because we added them artificially to the node and skip_add_tags is True


def test_update_undocumented_columns_with_prior_knowledge_skip_merge_meta(
    yaml_context: YamlRefactorContext,
):
    yaml_context.settings.skip_add_tags = False
    yaml_context.settings.skip_merge_meta = True

    manifest = yaml_context.project.manifest
    stg_customer_columns = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns
    stg_customer_columns["customer_id"].description = "THIS COLUMN IS UPDATED FOR TESTING"
    stg_customer_columns["customer_id"].meta = {"my_upstream_key": "my_upstream_value"}
    stg_customer_columns["customer_id"].tags = ["my_tag1", "my_tag2"]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    target_node_columns = target_node.columns
    target_node_columns["customer_id"].description = ""  # NOTE: allow inheritance to update this
    target_node_columns["customer_id"].tags = ["my_tag3", "my_tag4"]
    target_node_columns["customer_id"].meta = {"my_key": "my_value"}

    # Perform inheritance on the node
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        yaml_file_model_section = _get_member_yaml(yaml_context, target_node)

    assert target_node_columns["customer_id"].description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert (
        target_node_columns["customer_id"].meta == {"my_key": "my_value"}
    )  # NOTE: nodes meta is not mutated beyond our original mutation in the manifest node since skip_merge_meta is True
    assert sorted(target_node_columns["customer_id"].tags) == [
        "my_tag1",
        "my_tag2",
        "my_tag3",
        "my_tag4",
    ]

    assert yaml_file_model_section
    assert yaml_file_model_section["columns"][0]["name"] == "customer_id"
    assert (
        yaml_file_model_section["columns"][0]["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    )
    # TODO: consider a function which synchronizes a node with its yaml buffer, and then consider if inherit_upstream_column_knowledge should sync nodes
    # in which case it would pick up manual mutations to the node and apply them to the yaml buffer (which could be useful I think)
    assert (
        yaml_file_model_section["columns"][0].get("meta", {}) == {}
    )  # NOTE: yaml meta does not exist in buffer because we added it artificially to the node and skip_merge_meta is True
    assert sorted(yaml_file_model_section["columns"][0]["tags"]) == [
        "my_tag1",
        "my_tag2",
        "my_tag3",
        "my_tag4",
    ]


def test_update_undocumented_columns_with_prior_knowledge_add_progenitor_to_meta(
    yaml_context: YamlRefactorContext,
):
    # 1) Setup the manifest + node references
    manifest = yaml_context.project.manifest
    upstream_col = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"]
    upstream_col.description = "THIS COLUMN IS UPDATED FOR TESTING"
    upstream_col.meta = {"my_key": "my_value"}
    upstream_col.tags = ["my_tag1", "my_tag2"]

    target_node = manifest.nodes["model.jaffle_shop_duckdb.customers"]

    # 2) Configure the refactor settings
    yaml_context.settings.skip_add_tags = False
    yaml_context.settings.skip_merge_meta = False
    yaml_context.settings.add_progenitor_to_meta = True
    yaml_context.settings.force_inherit_descriptions = True  # if you want forced overwrite

    # 3) Perform the inheritance
    # The caches are used internally to reduce repeated YAML loads; we patch them with empty dicts
    # to ensure that the test sees a "fresh" environment each time
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        yaml_file_model_section = _get_member_yaml(yaml_context, target_node)

    # 4) Validate the Node
    cid = target_node.columns["customer_id"]
    assert cid.description == "THIS COLUMN IS UPDATED FOR TESTING"
    assert cid.meta == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }
    assert set(cid.tags) == {"my_tag1", "my_tag2"}

    # 5) Validate the in-memory YAML buffer
    assert yaml_file_model_section
    col_0 = yaml_file_model_section["columns"][0]
    assert col_0["name"] == "customer_id"
    assert col_0["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    assert col_0["meta"] == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }
    assert set(col_0["tags"]) == {"my_tag1", "my_tag2"}


# NOTE: this test is currently moot, as the default behavior is to keep the description if its not a placeholder
# but I leave the test in case we want to reintroduce a fine grained osmosis_keep_description meta attr
def test_update_undocumented_columns_with_prior_knowledge_with_osmosis_keep_description(
    yaml_context: YamlRefactorContext,
):
    # 1) Setup
    manifest = yaml_context.project.manifest
    stg_customers_col = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns[
        "customer_id"
    ]
    stg_customers_col.description = "THIS COLUMN IS UPDATED FOR TESTING"
    stg_customers_col.meta = {"my_key": "my_value"}
    stg_customers_col.tags = ["my_tag1", "my_tag2"]

    # The "downstream" node
    column_description_not_updated = (
        "This column will not be updated as it has the 'osmosis_keep_description' attribute"
    )
    target_node_name = "model.jaffle_shop_duckdb.customers"
    target_node = manifest.nodes[target_node_name]

    # The local column
    target_node.columns["customer_id"].description = column_description_not_updated
    target_node.columns["customer_id"].tags = ["my_tag3", "my_tag4"]
    target_node.columns["customer_id"].meta = {
        "my_key": "my_value",
        # "osmosis_keep_description": True, NOTE: this is now the default, and much more sane
    }

    # 2) Configure flags
    yaml_context.settings.skip_add_tags = True
    yaml_context.settings.skip_merge_meta = True
    yaml_context.settings.add_progenitor_to_meta = False
    yaml_context.settings.force_inherit_descriptions = (
        False  # This is the default, but we'll be explicit
    )

    # 3) Perform inheritance
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        yaml_file_model_section = _get_member_yaml(yaml_context, target_node)

    # 4) Assert Node
    cid = target_node.columns["customer_id"]
    assert cid.description == column_description_not_updated
    assert cid.meta == {
        "my_key": "my_value",
        # "osmosis_keep_description": True,
    }
    assert set(cid.tags) == {"my_tag3", "my_tag4"}

    # 5) Assert YAML
    assert yaml_file_model_section
    col_0 = yaml_file_model_section["columns"][0]
    assert col_0["name"] == "customer_id"
    assert col_0["description"] == column_description_not_updated


# NOTE: this test is currently moot, as the default behavior is to keep the description if its not a placeholder
# but I leave the test in case we want to reintroduce a fine grained osmosis_keep_description meta attr
def test_update_undocumented_columns_with_prior_knowledge_add_progenitor_to_meta_and_osmosis_keep_description(
    yaml_context: YamlRefactorContext,
):
    # 1) Setup
    manifest = yaml_context.project.manifest
    upstream = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"]
    upstream.description = "THIS COLUMN IS UPDATED FOR TESTING"
    upstream.meta = {"my_key": "my_value"}
    upstream.tags = ["my_tag1", "my_tag2"]

    column_description_not_updated = (
        "This column will not be updated as the 'force_inherit_descriptions' flag is false"
    )
    target_node_name = "model.jaffle_shop_duckdb.customers"
    target_node = manifest.nodes[target_node_name]

    # The local column is partially documented & has keep desc
    target_node.columns["customer_id"].description = column_description_not_updated
    target_node.columns["customer_id"].meta = {
        "my_key": "my_value",
    }

    # 2) Configure
    yaml_context.settings.skip_add_tags = False
    yaml_context.settings.skip_merge_meta = False
    yaml_context.settings.add_progenitor_to_meta = True
    yaml_context.settings.force_inherit_descriptions = False

    # 3) Inherit
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        model_section = _get_member_yaml(yaml_context, target_node)

    # 4) Assert Node
    cid = target_node.columns["customer_id"]
    # Since we have keep_description, the desc is not overwritten
    assert cid.description == column_description_not_updated
    # meta is merged with the upstream & also has progenitor
    assert cid.meta == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }
    # unify tags
    # upstream => my_tag1, my_tag2
    # local => is empty, so final is upstream
    assert set(cid.tags) == {"my_tag1", "my_tag2"}

    # 5) Assert YAML
    assert model_section
    col_0 = model_section["columns"][0]
    assert col_0["name"] == "customer_id"
    # keep desc
    assert col_0["description"] == column_description_not_updated
    assert col_0["meta"] == {
        "my_key": "my_value",
        "osmosis_progenitor": "model.jaffle_shop_duckdb.stg_customers",
    }


def test_update_undocumented_columns_with_prior_knowledge_with_add_inheritance_for_specified_keys(
    yaml_context: YamlRefactorContext,
):
    # 1) Setup
    manifest = yaml_context.project.manifest
    upstream_col = manifest.nodes["model.jaffle_shop_duckdb.stg_customers"].columns["customer_id"]
    upstream_col.description = "THIS COLUMN IS UPDATED FOR TESTING"
    upstream_col.meta = {"my_key": "my_value"}
    upstream_col.tags = ["my_tag1", "my_tag2"]
    # We'll store an `_extra` dict for a custom key
    upstream_col._extra = {"policy_tags": ["my_policy_tag1"]}

    target_node_name = "model.jaffle_shop_duckdb.customers"
    target_node = manifest.nodes[target_node_name]
    # local partial
    cinfo = target_node.columns["customer_id"]
    cinfo.tags = ["my_tag3", "my_tag4"]
    cinfo.meta = {
        "my_key": "my_old_value",
        "my_new_key": "my_new_value",
    }

    # 2) Configure
    yaml_context.settings.skip_add_tags = False
    yaml_context.settings.skip_merge_meta = False
    yaml_context.settings.add_progenitor_to_meta = False
    # we want to also inherit "policy_tags"
    yaml_context.settings.add_inheritance_for_specified_keys = ["policy_tags"]
    yaml_context.settings.force_inherit_descriptions = True

    # 3) Inherit
    with (
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
    ):
        inherit_upstream_column_knowledge(yaml_context, target_node)
        section = _get_member_yaml(yaml_context, target_node)

    # 4) Assert Node
    cid = target_node.columns["customer_id"]
    assert cid.description == "THIS COLUMN IS UPDATED FOR TESTING"
    # meta is merged
    assert cid.meta == {
        "my_key": "my_old_value",
        "my_new_key": "my_new_value",
    }
    # tags unify
    assert set(cid.tags) == {"my_tag1", "my_tag2", "my_tag3", "my_tag4"}
    # `_extra` might be stored if your code does so. e.g.
    # cid._extra["policy_tags"] == [...], but we skip that if not needed.

    # 5) Assert YAML
    assert section
    col_0 = section["columns"][0]
    assert col_0["name"] == "customer_id"
    assert col_0["description"] == "THIS COLUMN IS UPDATED FOR TESTING"
    assert col_0["meta"] == {
        "my_key": "my_old_value",
        "my_new_key": "my_new_value",
    }
    assert set(col_0["tags"]) == {"my_tag1", "my_tag2", "my_tag3", "my_tag4"}
    # confirm "policy_tags" is added
    assert set(col_0["policy_tags"]) == {"my_policy_tag1"}


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
