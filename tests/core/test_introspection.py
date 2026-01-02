# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for database introspection and column metadata retrieval.

This module contains:
- Behavior tests: test_get_columns_simple() validates end-to-end column retrieval
- Unit tests for utility functions with clear behavior contracts

The utility function tests (_find_first, _maybe_use_precise_dtype, etc.) test
well-defined behaviors of helper functions that have clear input/output contracts.
"""

from unittest import mock

import pytest

from dbt_osmosis.core.introspection import (
    _find_first,
    _get_setting_for_node,
    _maybe_use_precise_dtype,
    get_columns,
    normalize_column_name,
)
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings


@pytest.fixture(scope="function")
def fresh_caches():
    """
    Patches the internal caches so each test starts with a fresh state.
    """
    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
    ):
        yield


def test_get_columns_simple(yaml_context: YamlRefactorContext):
    """
    Tests the get_columns flow on a known table, e.g., 'customers'.
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    cols = get_columns(yaml_context, node)
    assert "customer_id" in cols


def test_find_first():
    """Test the _find_first utility function."""
    data = [1, 2, 3, 4]
    assert _find_first(data, lambda x: x > 2) == 3
    assert _find_first(data, lambda x: x > 4) is None
    assert _find_first(data, lambda x: x > 4, default=999) == 999


@pytest.mark.parametrize(
    "input_col,expected",
    [
        ('"My_Col"', "My_Col"),
        ("my_col", "MY_COL"),
    ],
)
def test_normalize_column_name_snowflake(input_col, expected):
    """Test column name normalization for Snowflake adapter."""
    # For snowflake, if quoted - we preserve case but strip quotes, otherwise uppercase
    assert normalize_column_name(input_col, "snowflake") == expected


def test_normalize_column_name_others():
    """Test column name normalization for other adapters."""
    # For other adapters, we only strip outer quotes but do not uppercase or lowercase for now
    assert normalize_column_name('"My_Col"', "duckdb") == "My_Col"
    assert normalize_column_name("my_col", "duckdb") == "my_col"


def test_maybe_use_precise_dtype_numeric():
    """
    Check that _maybe_use_precise_dtype uses the data_type if numeric_precision_and_scale is enabled.
    """
    from dbt.adapters.base.column import Column

    col = Column("col1", "DECIMAL(18,3)", None)  # data_type and dtype
    settings = YamlRefactorSettings(numeric_precision_and_scale=True)
    result = _maybe_use_precise_dtype(col, settings, node=None)
    assert result == "DECIMAL(18,3)"


def test_maybe_use_precise_dtype_string():
    """
    If string_length is True, we use col.data_type (like 'varchar(256)')
    instead of col.dtype (which might be 'VARCHAR').
    """
    from dbt.adapters.base.column import Column

    col = Column("col1", "VARCHAR(256)", None)
    settings = YamlRefactorSettings(string_length=True)
    result = _maybe_use_precise_dtype(col, settings, node=None)
    assert result == "VARCHAR(256)"


def test_get_setting_for_node_basic():
    """
    Check that _get_setting_for_node can read from node.meta, etc.
    We mock the node to have certain meta fields.
    """
    node = mock.Mock()
    node.config.extra = {}
    node.meta = {
        "dbt-osmosis-options": {
            "test-key": "test-value",
        }
    }
    # key = "test-key", which means we look for 'dbt-osmosis-options' => "test-key"
    val = _get_setting_for_node("test-key", node=node, col=None, fallback=None)
    assert val == "test-value"
