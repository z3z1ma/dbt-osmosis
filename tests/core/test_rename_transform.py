# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false
"""Tests for the enrich_rename_descriptions transform.

All tests use MagicMock — no DuckDB or warehouse connection required.

dbt_column_lineage is stubbed into sys.modules before any imports so the
package does not need to be installed in this virtualenv.

`_get_setting_for_node` lives in dbt_osmosis.core.introspection and is
imported locally inside each transform function body.  We patch the
canonical location (dbt_osmosis.core.introspection._get_setting_for_node)
so all local `from … import` calls resolve the patched version.

`get_column_lineage` is likewise imported locally inside the transform body
from dbt_column_lineage.api, so we patch that canonical location.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub dbt_column_lineage into sys.modules BEFORE importing transforms,
# so the lazy local import inside the transform body always resolves.
# ---------------------------------------------------------------------------
_stub_api = types.ModuleType("dbt_column_lineage.api")
_stub_api.get_column_lineage = MagicMock(return_value=[])  # type: ignore[attr-defined]
_stub_root = types.ModuleType("dbt_column_lineage")
sys.modules.setdefault("dbt_column_lineage", _stub_root)
sys.modules.setdefault("dbt_column_lineage.api", _stub_api)

from dbt_osmosis.core.transforms import (  # noqa: E402
    _LINEAGE_CACHE,
    _find_progenitor_description,
    enrich_rename_descriptions,
)

# ---------------------------------------------------------------------------
# Patch-target constants
# _get_setting_for_node is imported locally in the function body via
#   from dbt_osmosis.core.introspection import _get_setting_for_node
# Patching the canonical location makes every local import resolve the mock.
# ---------------------------------------------------------------------------
_SETTING = "dbt_osmosis.core.introspection._get_setting_for_node"
_GCL = "dbt_column_lineage.api.get_column_lineage"

# ---------------------------------------------------------------------------
# Shared placeholder tuple (must match the real value)
# ---------------------------------------------------------------------------
_PLACEHOLDERS = ("", "Pending further documentation", "No description for this column")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def _make_lineage_result(
    model: str,
    column: str,
    *,
    is_rename: bool = False,
    source_column: str | None = None,
    progenitor_model: str | None = None,
    progenitor_column: str | None = None,
) -> MagicMock:
    r = MagicMock()
    r.model = model
    r.column = column
    r.is_rename = is_rename
    r.source_column = source_column
    r.progenitor_model = progenitor_model
    r.progenitor_column = progenitor_column
    return r


def _make_col(desc: str) -> MagicMock:
    """Create a minimal ColumnInfo-like mock with a working .replace()."""
    col = MagicMock()
    col.description = desc

    def _replace(**kw: str) -> MagicMock:
        return _make_col(kw.get("description", desc))

    col.replace = _replace
    return col


def _make_node(
    name: str = "my_model",
    resource_type: str = "model",
    compiled_code: str | None = "select 1 as id",
    columns: dict[str, str] | None = None,
) -> MagicMock:
    from dbt.artifacts.resources.types import NodeType

    node = MagicMock()
    node.unique_id = f"model.pkg.{name}"
    node.name = name
    node.resource_type = NodeType.Source if resource_type == "source" else NodeType.Model
    node.compiled_code = compiled_code
    node.compiled_sql = None  # ensure getattr fallback returns None too

    node.columns = {col_name: _make_col(desc) for col_name, desc in (columns or {}).items()}
    return node


def _make_context(
    project_dir: str = "/fake/project",
    profiles_dir: str = "/fake/.dbt",
    target: str | None = None,
    force_inherit_descriptions: bool = False,
    manifest_nodes: dict | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.settings.force_inherit_descriptions = force_inherit_descriptions
    ctx.placeholders = _PLACEHOLDERS
    ctx.project.config.project_dir = project_dir
    ctx.project.config.profiles_dir = profiles_dir
    ctx.project.config.target = target
    ctx.project.manifest.nodes = manifest_nodes or {}
    return ctx


# ---------------------------------------------------------------------------
# Fixture: clear the module-level lineage cache before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_lineage_cache():
    _LINEAGE_CACHE.clear()
    yield
    _LINEAGE_CACHE.clear()


# ---------------------------------------------------------------------------
# Matrix: rename_descriptions × force_inherit_descriptions × existing_desc
# ---------------------------------------------------------------------------

@patch(_SETTING)
def test_rename_descriptions_false_unchanged(mock_setting: MagicMock) -> None:
    """rename_descriptions=False → column is never touched, lineage not fetched."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: (
        False if name == "rename-descriptions" else fallback
    )
    node = _make_node(columns={"CUSTOMER_ID": ""})
    ctx = _make_context()

    with patch(_GCL) as mock_gcl:
        enrich_rename_descriptions(ctx, node)

    mock_gcl.assert_not_called()
    assert node.columns["CUSTOMER_ID"].description == ""


@patch(_SETTING)
def test_rename_true_force_false_empty_desc_enriched(mock_setting: MagicMock) -> None:
    """rename_descriptions=True, force=False, empty desc → rename enrichment applied."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: {
        "rename-descriptions": True,
        "force-inherit-descriptions": False,
        "rename-description-prefix": "renamed from {source_col}",
    }.get(name, fallback)

    lineage = _make_lineage_result(
        "my_model", "customer_id",
        is_rename=True, source_column="id",
        progenitor_model="stg_customers", progenitor_column="ID",
    )
    node = _make_node(columns={"customer_id": ""})
    ctx = _make_context()

    with patch(_GCL, return_value=[lineage]):
        enrich_rename_descriptions(ctx, node)

    assert node.columns["customer_id"].description == "renamed from id"


@patch(_SETTING)
def test_rename_true_force_false_nonempty_desc_skipped(mock_setting: MagicMock) -> None:
    """rename_descriptions=True, force=False, non-empty desc → existing desc preserved."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: {
        "rename-descriptions": True,
        "force-inherit-descriptions": False,
        "rename-description-prefix": "renamed from {source_col}",
    }.get(name, fallback)

    lineage = _make_lineage_result(
        "my_model", "customer_id",
        is_rename=True, source_column="id",
    )
    node = _make_node(columns={"customer_id": "Existing real description"})
    ctx = _make_context()

    with patch(_GCL, return_value=[lineage]):
        enrich_rename_descriptions(ctx, node)

    assert node.columns["customer_id"].description == "Existing real description"


@patch(_SETTING)
def test_rename_true_force_true_overwrites(mock_setting: MagicMock) -> None:
    """rename_descriptions=True, force=True, any desc → rename enrichment applied."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: {
        "rename-descriptions": True,
        "force-inherit-descriptions": True,
        "rename-description-prefix": "renamed from {source_col}",
    }.get(name, fallback)

    lineage = _make_lineage_result(
        "my_model", "customer_id",
        is_rename=True, source_column="id",
    )
    node = _make_node(columns={"customer_id": "Existing real description"})
    ctx = _make_context()

    with patch(_GCL, return_value=[lineage]):
        enrich_rename_descriptions(ctx, node)

    assert node.columns["customer_id"].description == "renamed from id"


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

@patch(_SETTING)
def test_source_node_skipped(mock_setting: MagicMock) -> None:
    """Source nodes are silently skipped even when rename-descriptions=True."""
    mock_setting.return_value = True

    with patch(_GCL) as mock_gcl:
        node = _make_node(resource_type="source", columns={"id": ""})
        enrich_rename_descriptions(_make_context(), node)

    mock_gcl.assert_not_called()


@patch(_SETTING)
def test_no_compiled_sql_skipped(mock_setting: MagicMock) -> None:
    """Ephemeral-like nodes without compiled SQL are silently skipped."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: (
        True if name == "rename-descriptions" else fallback
    )

    with patch(_GCL) as mock_gcl:
        node = _make_node(compiled_code=None, columns={"id": ""})
        enrich_rename_descriptions(_make_context(), node)

    mock_gcl.assert_not_called()


# ---------------------------------------------------------------------------
# Prefix template behaviour
# ---------------------------------------------------------------------------

@patch(_SETTING)
def test_prefix_joined_with_progenitor_desc(mock_setting: MagicMock) -> None:
    """Prefix and progenitor description are joined with ' | ' when both exist."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: {
        "rename-descriptions": True,
        "force-inherit-descriptions": False,
        "rename-description-prefix": "renamed from {source_col}",
    }.get(name, fallback)

    progenitor_col = MagicMock()
    progenitor_col.description = "The unique customer identifier"
    progenitor_node = MagicMock()
    progenitor_node.name = "stg_customers"
    progenitor_node.columns = {"ID": progenitor_col}

    lineage = _make_lineage_result(
        "my_model", "customer_id",
        is_rename=True, source_column="id",
        progenitor_model="stg_customers", progenitor_column="ID",
    )
    node = _make_node(columns={"customer_id": ""})
    ctx = _make_context(manifest_nodes={"model.pkg.stg_customers": progenitor_node})

    with patch(_GCL, return_value=[lineage]):
        enrich_rename_descriptions(ctx, node)

    assert node.columns["customer_id"].description == (
        "renamed from id | The unique customer identifier"
    )


@patch(_SETTING)
def test_custom_prefix_template(mock_setting: MagicMock) -> None:
    """Custom rename-description-prefix is applied with {source_col} substituted."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: {
        "rename-descriptions": True,
        "force-inherit-descriptions": False,
        "rename-description-prefix": "aliased: {source_col}",
    }.get(name, fallback)

    lineage = _make_lineage_result(
        "my_model", "cust_id",
        is_rename=True, source_column="customer_id",
    )
    node = _make_node(columns={"cust_id": ""})

    with patch(_GCL, return_value=[lineage]):
        enrich_rename_descriptions(_make_context(), node)

    assert node.columns["cust_id"].description == "aliased: customer_id"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

@patch(_SETTING)
def test_lineage_failure_is_graceful(mock_setting: MagicMock) -> None:
    """If get_column_lineage raises, the node is skipped without propagating."""
    mock_setting.side_effect = lambda name, node, *a, fallback=None, **kw: (
        True if name == "rename-descriptions" else fallback
    )

    with patch(_GCL, side_effect=RuntimeError("adapter timeout")):
        node = _make_node(columns={"id": ""})
        enrich_rename_descriptions(_make_context(), node)  # must not raise

    assert node.columns["id"].description == ""


# ---------------------------------------------------------------------------
# _find_progenitor_description unit tests
# ---------------------------------------------------------------------------

def test_find_progenitor_description_found() -> None:
    col = MagicMock()
    col.description = "The primary key"
    upstream = MagicMock()
    upstream.name = "stg_orders"
    upstream.columns = {"ORDER_ID": col}
    ctx = _make_context(manifest_nodes={"model.pkg.stg_orders": upstream})
    assert _find_progenitor_description(ctx, "stg_orders", "ORDER_ID") == "The primary key"


def test_find_progenitor_description_case_insensitive() -> None:
    col = MagicMock()
    col.description = "The primary key"
    upstream = MagicMock()
    upstream.name = "stg_orders"
    upstream.columns = {"ORDER_ID": col}
    ctx = _make_context(manifest_nodes={"model.pkg.stg_orders": upstream})
    # lowercase lookup must still resolve
    assert _find_progenitor_description(ctx, "stg_orders", "order_id") == "The primary key"


def test_find_progenitor_description_placeholder_returns_none() -> None:
    col = MagicMock()
    col.description = ""  # placeholder
    upstream = MagicMock()
    upstream.name = "stg_orders"
    upstream.columns = {"ORDER_ID": col}
    ctx = _make_context(manifest_nodes={"model.pkg.stg_orders": upstream})
    assert _find_progenitor_description(ctx, "stg_orders", "ORDER_ID") is None


def test_find_progenitor_description_no_model_returns_none() -> None:
    ctx = _make_context()
    assert _find_progenitor_description(ctx, "nonexistent_model", "col") is None
