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

from unittest import mock

import pytest

from dbt_osmosis.core.osmosis import inherit_upstream_column_knowledge


@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


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
