"""Behavior-focused tests for column knowledge inheritance.

This module tests the actual USER-FACING BEHAVIOR of column inheritance,
not the internal implementation details. The core value proposition of
dbt-osmosis is "inheriting descriptions downstream" - these tests verify
that behavior works correctly in real-world scenarios.

Key scenarios tested:
- Multi-generation inheritance (A → B → C)
- Empty columns inheriting from upstream
- Partial documentation propagation
- Tag and meta inheritance across generations
- Diamond pattern (multiple upstream sources)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from dbt_osmosis.core.inheritance import (
    _build_column_knowledge_graph,
    _build_node_ancestor_tree,
    _collect_column_variants,
    _get_node_yaml,
    _merge_graph_node_data,
    _versioned_model_yaml_view,
)
from dbt_osmosis.core.schema.reader import _read_yaml
from dbt_osmosis.core.sync_operations import sync_node_to_yaml
from dbt_osmosis.core.transforms import apply_semantic_analysis, inherit_upstream_column_knowledge


def _set_column_progenitor_override(
    yaml_context,
    node_id: str,
    column_name: str,
    *,
    model_default_progenitor: str | None = None,
    column_default_progenitor: str | None = None,
) -> None:
    """Mutate the cached YAML for a model so inheritance reads the requested override."""
    manifest = yaml_context.project.manifest
    node = manifest.nodes[node_id]
    project_dir = Path(yaml_context.project.runtime_cfg.project_root)
    path = project_dir.joinpath(node.patch_path.split("://")[-1])
    yaml_doc = _read_yaml(yaml_context.yaml_handler, yaml_context.yaml_handler_lock, path)
    model = next(model for model in yaml_doc["models"] if model["name"] == node.name)

    if model_default_progenitor is not None:
        model.setdefault("meta", {})["default_progenitor"] = model_default_progenitor

    if column_default_progenitor is not None:
        column = next(column for column in model["columns"] if column["name"] == column_name)
        column.setdefault("meta", {})["column_default_progenitor"] = column_default_progenitor


def _ensure_column_config(column):
    """Ensure a dbt column-like object has a mutable config object for tests."""
    config = getattr(column, "config", None)
    if config is None:
        pytest.skip("dbt ColumnInfo.config is not available in this dbt version")
    if not hasattr(config, "tags"):
        config.tags = []
    if not hasattr(config, "meta"):
        config.meta = {}
    return config


def test_ancestor_tree_detects_cycle_back_to_root_unique_id():
    """A root node revisited through an ancestor cycle should be skipped by full unique ID."""
    root_id = "model.pkg.a"
    parent_id = "model.pkg.b"
    root = SimpleNamespace(
        unique_id=root_id,
        depends_on=SimpleNamespace(nodes=[parent_id]),
    )
    parent = SimpleNamespace(
        unique_id=parent_id,
        depends_on=SimpleNamespace(nodes=[root_id]),
    )
    manifest = SimpleNamespace(nodes={root_id: root, parent_id: parent}, sources={})

    tree = _build_node_ancestor_tree(manifest, root)

    assert tree == {
        "generation_0": [root_id],
        "generation_1": [parent_id],
    }


def test_column_knowledge_processes_numeric_generations_farthest_to_closest(
    yaml_context,
    fresh_caches,
    monkeypatch,
):
    """generation_10 should be treated as farther away than generation_2, not lexicographic."""
    import dbt_osmosis.core.inheritance as inheritance_module

    manifest = yaml_context.project.manifest
    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = "Raw source description"
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "Closer staging description"
    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""

    monkeypatch.setattr(
        inheritance_module,
        "_build_node_ancestor_tree",
        lambda manifest, node: {
            "generation_0": [customers.unique_id],
            "generation_2": [stg_customers.unique_id],
            "generation_10": [raw_customers.unique_id],
        },
    )

    graph = _build_column_knowledge_graph(yaml_context, customers)

    assert graph["first_name"]["description"] == "Closer staging description"


def test_graph_node_tag_merges_preserve_local_then_inherited_order():
    """Top-level and config tags should use order-preserving union, not set order."""
    graph_node = {
        "tags": ["local", "shared"],
        "config": {"tags": ["config_local", "config_shared"]},
    }
    graph_edge = {
        "tags": ["upstream", "shared", "upstream_2"],
        "config": {"tags": ["config_upstream", "config_shared", "config_upstream_2"]},
    }

    _merge_graph_node_data(graph_node, graph_edge)

    assert graph_node["tags"] == ["local", "shared", "upstream", "upstream_2"]
    assert graph_node["config"]["tags"] == [
        "config_local",
        "config_shared",
        "config_upstream",
        "config_upstream_2",
    ]


def test_semantic_analysis_tag_merge_preserves_existing_then_suggested_order(
    yaml_context,
    monkeypatch,
):
    """Semantic tag suggestions should append unseen tags without set reordering."""
    import dbt_osmosis.core.inheritance as inheritance_module
    import dbt_osmosis.core.llm as llm_module

    class FakeColumn:
        def __init__(
            self,
            *,
            description="",
            tags=None,
            meta=None,
            data_type="integer",
        ):
            self.description = description
            self.tags = tags or []
            self.meta = meta or {}
            self.data_type = data_type

        def replace(self, **kwargs):
            values = {
                "description": self.description,
                "tags": self.tags,
                "meta": self.meta,
                "data_type": self.data_type,
            }
            values.update(kwargs)
            return FakeColumn(**values)

    node = SimpleNamespace(
        unique_id="model.pkg.semantic_target",
        name="semantic_target",
        description="",
        raw_sql="select 1 as id",
        columns={"id": FakeColumn(tags=["existing", "shared"])},
    )

    def fake_analyze_column_semantics(**kwargs):
        return {"tags": ["semantic", "shared", "new"], "semantic_type": "primary_key"}

    monkeypatch.setattr(
        inheritance_module, "_build_column_knowledge_graph", lambda context, node: {}
    )
    monkeypatch.setitem(
        fake_analyze_column_semantics.__globals__, "get_llm_client", lambda: object()
    )
    monkeypatch.setattr(llm_module, "analyze_column_semantics", fake_analyze_column_semantics)
    monkeypatch.setattr(
        llm_module,
        "generate_semantic_description",
        lambda **kwargs: "Generated description",
    )

    apply_semantic_analysis(yaml_context, node)

    assert node.columns["id"].tags == ["existing", "shared", "semantic", "new"]


def test_collect_column_variants_honors_project_level_prefix(tmp_path, fresh_caches):
    """Inheritance plugins should receive full context for project-level prefix settings."""
    (tmp_path / "dbt-osmosis.yml").write_text("prefix: raw_\n")

    node = mock.MagicMock()
    node.meta = {}
    node.config.extra = {}
    node.config.meta = {}
    node.unrendered_config = {}
    node.columns = {"raw_customer_id": mock.MagicMock(meta={})}

    context = SimpleNamespace(
        project=SimpleNamespace(
            runtime_cfg=SimpleNamespace(
                project_root=tmp_path,
                vars={},
            ),
        ),
    )

    variants = _collect_column_variants(context, node)

    assert "customer_id" in variants["raw_customer_id"]


def test_multi_generation_inheritance_chain(yaml_context, fresh_caches):
    """Test that documentation propagates through multi-generation model chains.

    Scenario: raw_customers (seed) → stg_customers.v1 → customers

    When raw_customers.seed has column documentation and customers does NOT,
    the documentation should propagate through stg_customers.v1 to customers.

    This tests the core value proposition: descriptions flow downstream across
    multiple hops in the DAG.
    """
    manifest = yaml_context.project.manifest

    # Set up: raw_customers has documentation for first_name, customers does not
    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["id"].description = "Unique customer identifier from source system"
    raw_customers.columns["first_name"].description = "Customer first name from source"

    # Clear the description in stg_customers and customers to test propagation
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["customer_id"].description = ""
    stg_customers.columns["first_name"].description = ""

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["customer_id"].description = ""
    customers.columns["first_name"].description = ""

    # Execute: Run inheritance on customers
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True

    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: first_name should have inherited from raw_customers
    assert customers.columns["first_name"].description == "Customer first name from source"
    # The progenitor should be the original source (raw_customers)
    assert (
        customers.columns["first_name"].meta.get("osmosis_progenitor")
        == "seed.jaffle_shop_duckdb.raw_customers"
    )


def test_empty_column_inherits_from_upstream(yaml_context, fresh_caches):
    """Test that an empty column description inherits from the nearest upstream source.

    This is the most common use case: a downstream model has an undocumented column,
    and it should inherit documentation from its upstream parent.
    """
    manifest = yaml_context.project.manifest

    # Set up: raw_customers (seed) has documentation, customers does not
    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = "Customer's first name - PII sensitive"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""  # Empty - should inherit

    # Execute
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True  # Enable progenitor tracking
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: first_name should inherit from raw_customers
    assert customers.columns["first_name"].description == "Customer's first name - PII sensitive"
    # Progenitor should be the seed (original source)
    assert (
        customers.columns["first_name"].meta.get("osmosis_progenitor")
        == "seed.jaffle_shop_duckdb.raw_customers"
    )


def test_partial_documentation_propagation(yaml_context, fresh_caches):
    """Test that only undocumented columns are inherited, not already documented ones.

    When a model has some documented and some undocumented columns:
    - Documented columns should keep their descriptions
    - Undocumented columns should inherit from upstream
    """
    manifest = yaml_context.project.manifest

    # Set up: stg_customers has documentation for all columns
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "First name from source"
    stg_customers.columns["last_name"].description = "Last name from source"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""  # Empty - should inherit
    customers.columns[
        "last_name"
    ].description = "Customer family name (custom description)"  # Has doc - keep it

    # Execute
    yaml_context.settings.force_inherit_descriptions = False  # Don't override existing docs
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: first_name inherited, last_name kept original
    assert customers.columns["first_name"].description == "First name from source"
    assert customers.columns["last_name"].description == "Customer family name (custom description)"


def test_tag_and_meta_inheritance(yaml_context, fresh_caches):
    """Test that tags and meta fields propagate through inheritance.

    Tags and metadata are as important as descriptions for data governance.
    This test verifies they flow downstream correctly.
    """
    manifest = yaml_context.project.manifest

    # Set up: upstream has tags and meta
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["customer_id"].tags = ["pk", "identifier"]
    stg_customers.columns["customer_id"].meta = {
        "sensitivity": "public",
        "governance": "customer_key",
    }

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["customer_id"].tags = []
    customers.columns["customer_id"].meta = {}

    # Execute
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: tags and meta should be inherited (merged with local)
    assert "pk" in customers.columns["customer_id"].tags
    assert "identifier" in customers.columns["customer_id"].tags
    assert customers.columns["customer_id"].meta.get("sensitivity") == "public"
    assert customers.columns["customer_id"].meta.get("governance") == "customer_key"


def test_config_tag_and_meta_inheritance(yaml_context, fresh_caches):
    """Column config.tags and config.meta should inherit as effective tags/meta."""
    manifest = yaml_context.project.manifest

    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["customer_id"].tags = []
    stg_customers.columns["customer_id"].meta = {}
    stg_customer_id_config = _ensure_column_config(stg_customers.columns["customer_id"])
    stg_customer_id_config.tags = ["config_pk", "config_identifier"]
    stg_customer_id_config.meta = {
        "classification": "restricted",
        "governance": "config_customer_key",
    }

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["customer_id"].tags = []
    customers.columns["customer_id"].meta = {}
    customer_id_config = _ensure_column_config(customers.columns["customer_id"])
    customer_id_config.tags = []
    customer_id_config.meta = {}

    inherit_upstream_column_knowledge(yaml_context, customers)

    assert "config_pk" in customers.columns["customer_id"].tags
    assert "config_identifier" in customers.columns["customer_id"].tags
    assert customers.columns["customer_id"].meta.get("classification") == "restricted"
    assert customers.columns["customer_id"].meta.get("governance") == "config_customer_key"


def test_diamond_pattern_inheritance(yaml_context, fresh_caches):
    """Test inheritance when a column comes from multiple upstream models.

    Scenario: customers model depends on stg_customers, stg_orders, and stg_payments.
    If first_name appears in multiple upstreams, the FIRST documented source should win.

    This tests the "diamond pattern" DAG where multiple models converge.
    """
    manifest = yaml_context.project.manifest

    # Set up: first_name documented in stg_customers
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "First name from customers staging"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""  # Empty - should inherit

    # Execute
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Should inherit from stg_customers
    assert customers.columns["first_name"].description == "First name from customers staging"
    # Progenitor should be set
    assert customers.columns["first_name"].meta.get("osmosis_progenitor") is not None


def test_force_inherit_overrides_existing_descriptions(yaml_context, fresh_caches):
    """Test that force_inherit_descriptions=true overrides existing documentation.

    When force_inherit_descriptions is enabled, even columns that already have
    descriptions should inherit from upstream. This is useful for standardizing
    documentation across the data warehouse.
    """
    manifest = yaml_context.project.manifest

    # Set up: both upstream and downstream have descriptions
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "Standardized first name"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = "Local first name description"

    # Execute with force_inherit_descriptions = True
    yaml_context.settings.force_inherit_descriptions = True
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Should override local description with upstream
    assert customers.columns["first_name"].description == "Standardized first name"


def test_skip_add_tags_preserves_local_tags(yaml_context, fresh_caches):
    """Test that skip_add_tags=true prevents upstream tags from being added.

    When skip_add_tags is enabled, local tags should be preserved and upstream
    tags should NOT be merged in.
    """
    manifest = yaml_context.project.manifest

    # Set up
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["customer_id"].tags = ["upstream_tag"]

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["customer_id"].tags = ["local_tag"]

    # Execute with skip_add_tags = True
    yaml_context.settings.skip_add_tags = True
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Should keep local tags only
    assert customers.columns["customer_id"].tags == ["local_tag"]


def test_skip_merge_meta_preserves_local_meta(yaml_context, fresh_caches):
    """Test that skip_merge_meta=true prevents upstream meta from being merged.

    When skip_merge_meta is enabled, local meta should be preserved and upstream
    meta should NOT be merged in.
    """
    manifest = yaml_context.project.manifest

    # Set up
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["customer_id"].meta = {"upstream_key": "upstream_value"}

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["customer_id"].meta = {"local_key": "local_value"}

    # Execute with skip_merge_meta = True
    yaml_context.settings.skip_merge_meta = True
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Should keep local meta only
    assert customers.columns["customer_id"].meta == {"local_key": "local_value"}


def test_inheritance_with_placeholder_descriptions(yaml_context, fresh_caches):
    """Test that empty descriptions are treated as undocumented.

    This test verifies that columns with empty descriptions inherit from upstream.
    The default placeholder list includes empty string.
    """
    manifest = yaml_context.project.manifest

    # Set up: upstream has good description, downstream has empty description
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "Customer first name"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""  # Empty - should inherit

    # Execute - should inherit because empty string is a placeholder
    yaml_context.settings.force_inherit_descriptions = False  # Don't override existing docs
    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Should inherit because empty description is treated as undocumented
    assert customers.columns["first_name"].description == "Customer first name"


def test_inheritance_across_all_models(yaml_context, fresh_caches):
    """Test that inheritance works correctly when run across all models in the project.

    This is an integration test that verifies inheritance works end-to-end
    for all models in the jaffle_shop project.
    """
    # Execute inheritance on all models
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True

    inherit_upstream_column_knowledge(yaml_context)

    # Verify: Check that inheritance propagated through the DAG
    manifest = yaml_context.project.manifest

    # Verify stg_customers has some columns with progenitor set
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    # At least some columns should have progenitor from seeds
    progenitor_count = sum(
        1
        for col in stg_customers.columns.values()
        if col.meta.get("osmosis_progenitor") == "seed.jaffle_shop_duckdb.raw_customers"
    )
    assert progenitor_count > 0, (
        "At least some columns should have progenitor from raw_customers seed"
    )


def test_progenitor_tracking_across_generations(yaml_context, fresh_caches):
    """Test that osmosis_progenitor correctly tracks the column's origin.

    The progenitor field should always point to the ORIGINAL source of the column,
    not just the immediate parent.
    """
    manifest = yaml_context.project.manifest

    # Set up documentation at the seed level
    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = "Original first name documentation"

    # Clear intermediate descriptions
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = ""

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""

    # Execute inheritance
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True

    inherit_upstream_column_knowledge(yaml_context, customers)

    # Assert: Progenitor should point to the seed, not stg_customers
    assert (
        customers.columns["first_name"].meta.get("osmosis_progenitor")
        == "seed.jaffle_shop_duckdb.raw_customers"
    )


def test_default_progenitor_override_reuses_selected_ancestor_knowledge(
    yaml_context,
    fresh_caches,
):
    """A default_progenitor should inherit the override node's resolved lineage, not its raw column."""
    manifest = yaml_context.project.manifest

    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = "Original first name documentation"
    raw_customers.columns["first_name"].meta = {"classification": "restricted"}
    raw_customers.columns["first_name"].tags = ["pii"]

    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = ""
    stg_customers.columns["first_name"].meta = {}
    stg_customers.columns["first_name"].tags = []

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""
    customers.columns["first_name"].meta = {}
    customers.columns["first_name"].tags = []

    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = True
    _set_column_progenitor_override(
        yaml_context,
        "model.jaffle_shop_duckdb.customers",
        "first_name",
        model_default_progenitor="model.jaffle_shop_duckdb.stg_customers.v1",
    )

    inherit_upstream_column_knowledge(yaml_context, customers)

    assert customers.columns["first_name"].description == "Original first name documentation"
    assert customers.columns["first_name"].meta.get("classification") == "restricted"
    assert "pii" in customers.columns["first_name"].tags
    assert (
        customers.columns["first_name"].meta.get("osmosis_progenitor")
        == "seed.jaffle_shop_duckdb.raw_customers"
    )


def test_column_default_progenitor_override_applies_without_progenitor_tracking(
    yaml_context,
    fresh_caches,
):
    """column_default_progenitor should work even when osmosis_progenitor tracking is disabled."""
    manifest = yaml_context.project.manifest

    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = "Original first name documentation"

    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "Stage-level first name documentation"

    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""

    yaml_context.settings.fusion_compat = False
    yaml_context.settings.dry_run = False
    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.add_progenitor_to_meta = False
    _set_column_progenitor_override(
        yaml_context,
        "model.jaffle_shop_duckdb.customers",
        "first_name",
        column_default_progenitor="seed.jaffle_shop_duckdb.raw_customers",
    )

    inherit_upstream_column_knowledge(yaml_context, customers)
    sync_node_to_yaml(yaml_context, customers, commit=False)
    yaml_slice = _get_node_yaml(yaml_context, customers)
    assert yaml_slice is not None

    yaml_column = next(column for column in yaml_slice["columns"] if column["name"] == "first_name")

    assert customers.columns["first_name"].description == "Original first name documentation"
    assert (
        yaml_column["meta"]["column_default_progenitor"] == "seed.jaffle_shop_duckdb.raw_customers"
    )


def test_get_node_yaml_returns_versioned_block_with_top_level_fallback(
    yaml_context,
    fresh_caches,
):
    """Versioned models should expose selected versions[].columns plus model fallback metadata."""
    manifest = yaml_context.project.manifest
    node = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v2"]
    project_dir = Path(yaml_context.project.runtime_cfg.project_root)
    path = project_dir.joinpath(node.patch_path.split("://")[-1])
    yaml_doc = _read_yaml(yaml_context.yaml_handler, yaml_context.yaml_handler_lock, path)
    model = next(model for model in yaml_doc["models"] if model["name"] == node.name)
    model["description"] = "Top-level fallback description"
    model["tags"] = ["top_level"]
    model["meta"] = {"domain": "customers"}
    version = next(version for version in model["versions"] if version["v"] == 2)
    version["description"] = ""
    version["columns"].insert(0, {"include": "*"})

    yaml_slice = _get_node_yaml(yaml_context, node)

    assert yaml_slice is not None
    assert yaml_slice["name"] == "stg_customers"
    assert yaml_slice["v"] == 2
    assert yaml_slice["description"] == "Top-level fallback description"
    assert yaml_slice["tags"] == ["top_level"]
    assert yaml_slice["meta"] == {"domain": "customers"}
    assert [column["name"] for column in yaml_slice["columns"] if "name" in column] == [
        "id",
        "first_name",
        "last_name",
    ]


def test_versioned_ancestor_unrendered_description_reads_version_columns(
    yaml_context,
    fresh_caches,
):
    """Inheritance should preserve unrendered docs from version-level ancestor columns."""
    manifest = yaml_context.project.manifest
    stg_customers = manifest.nodes["model.jaffle_shop_duckdb.stg_customers.v1"]
    stg_customers.columns["first_name"].description = "Rendered version-level first name"
    raw_customers = manifest.nodes["seed.jaffle_shop_duckdb.raw_customers"]
    raw_customers.columns["first_name"].description = ""
    customers = manifest.nodes["model.jaffle_shop_duckdb.customers"]
    customers.columns["first_name"].description = ""

    project_dir = Path(yaml_context.project.runtime_cfg.project_root)
    path = project_dir.joinpath(stg_customers.patch_path.split("://")[-1])
    yaml_doc = _read_yaml(yaml_context.yaml_handler, yaml_context.yaml_handler_lock, path)
    model = next(model for model in yaml_doc["models"] if model["name"] == stg_customers.name)
    version = next(version for version in model["versions"] if version["v"] == 1)
    version["columns"].insert(0, {"include": "*"})
    first_name = next(column for column in version["columns"] if column.get("name") == "first_name")
    first_name["description"] = "{{ doc('versioned_first_name') }}"

    yaml_context.settings.force_inherit_descriptions = True
    yaml_context.settings.use_unrendered_descriptions = True
    inherit_upstream_column_knowledge(yaml_context, customers)

    assert customers.columns["first_name"].description == "{{ doc('versioned_first_name') }}"


def test_versioned_model_yaml_view_prefers_exact_string_version_match():
    """String versions that normalize numerically should remain distinct."""
    model = {
        "name": "orders",
        "versions": [
            {"v": "1.1", "columns": [{"name": "legacy_id"}]},
            {"v": "1.10", "columns": [{"name": "current_id"}]},
        ],
    }
    member = SimpleNamespace(version="1.10", unique_id="model.pkg.orders.v1.10")

    yaml_slice = _versioned_model_yaml_view(model, member)

    assert yaml_slice is not None
    assert yaml_slice["v"] == "1.10"
    assert yaml_slice["columns"] == [{"name": "current_id"}]

    missing_exact = {"name": "orders", "versions": [{"v": "1.1", "columns": []}]}
    assert _versioned_model_yaml_view(missing_exact, member) is None

    leading_zero = {"name": "orders", "versions": [{"v": "01", "columns": []}]}
    assert _versioned_model_yaml_view(leading_zero, SimpleNamespace(version=1)) is None
