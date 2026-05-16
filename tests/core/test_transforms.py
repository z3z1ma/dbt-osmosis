# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from collections import OrderedDict
from unittest import mock

import pytest

from dbt_osmosis.core.settings import YamlRefactorContext
from dbt_osmosis.core.transforms import (
    apply_semantic_analysis,
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    remove_columns_not_in_database,
    sort_columns_alphabetically,
    sort_columns_as_configured,
    sort_columns_as_in_database,
    synchronize_data_types,
)


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext, fresh_caches):
    """Minimal test that runs the inheritance logic on all matched nodes in the real project."""
    inherit_upstream_column_knowledge(yaml_context)


def test_inject_missing_columns(yaml_context: YamlRefactorContext, fresh_caches):
    """If the DB has columns the YAML/manifest doesn't, we inject them.
    We run on all matched nodes to ensure no errors.
    """
    inject_missing_columns(yaml_context)


def test_inject_missing_columns_honors_supplementary_file_skip(
    tmp_path,
    fresh_caches,
):
    """Supplementary dbt-osmosis.yml should affect the real inject transform."""
    (tmp_path / "dbt-osmosis.yml").write_text("skip-add-columns: true\n")

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.resource_type = "model"
    mock_node.meta = {}
    mock_node.config.extra = {}
    mock_node.config.meta = {}
    mock_node.unrendered_config = {}
    mock_node.columns = OrderedDict()

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = False
    context.settings.output_to_lower = False
    context.settings.output_to_upper = False
    context.project.runtime_cfg.project_root = tmp_path
    context.project.runtime_cfg.vars = {}
    context.project.runtime_cfg.credentials.type = "postgres"

    incoming_col = mock.MagicMock()
    incoming_col.type = "text"
    incoming_col.comment = "from database"

    with mock.patch(
        "dbt_osmosis.core.introspection.get_columns",
        return_value=OrderedDict([("new_col", incoming_col)]),
    ) as get_columns:
        inject_missing_columns(context, mock_node)

    get_columns.assert_not_called()
    assert mock_node.columns == OrderedDict()


def test_inject_missing_columns_all_nodes_allows_node_false_override(
    tmp_path,
    fresh_caches,
):
    """Project-level skip should not bypass higher-precedence node false overrides."""
    (tmp_path / "dbt-osmosis.yml").write_text("skip-add-columns: true\n")

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.resource_type = "model"
    mock_node.meta = {"dbt-osmosis-skip-add-columns": False}
    mock_node.config.extra = {}
    mock_node.config.meta = {}
    mock_node.unrendered_config = {}
    mock_node.columns = OrderedDict()

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = False
    context.settings.output_to_lower = False
    context.settings.output_to_upper = False
    context.project.runtime_cfg.project_root = tmp_path
    context.project.runtime_cfg.vars = {}
    context.project.runtime_cfg.credentials.type = "postgres"
    context.pool.map.side_effect = lambda func, items: [func(item) for item in items]

    incoming_col = mock.MagicMock()
    incoming_col.type = "text"
    incoming_col.comment = "from database"

    with (
        mock.patch(
            "dbt_osmosis.core.node_filters._iter_candidate_nodes",
            return_value=iter([("model.test.test_model", mock_node)]),
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=OrderedDict([("new_col", incoming_col)]),
        ) as get_columns,
    ):
        inject_missing_columns(context)

    get_columns.assert_called_once_with(context, mock_node)
    assert list(mock_node.columns) == ["new_col"]


def test_remove_columns_not_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """If the manifest has columns the DB does not, we remove them.
    Typically, your real project might not have any extra columns, so this is a sanity test.
    """
    remove_columns_not_in_database(yaml_context)


def test_sort_columns_as_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """Sort columns in the order the DB sees them.
    With duckdb, this is minimal but we can still ensure no errors.
    """
    sort_columns_as_in_database(yaml_context)


@pytest.mark.parametrize(
    ("output_to_lower", "output_to_upper", "yaml_columns", "database_columns", "expected_order"),
    [
        (
            False,
            True,
            ["B", "STALE", "A"],
            ["a", "b"],
            ["A", "B", "STALE"],
        ),
        (
            True,
            False,
            ["b", "stale", "a"],
            ["A", "B"],
            ["a", "b", "stale"],
        ),
    ],
)
def test_sort_columns_as_in_database_honors_output_case_settings(
    fresh_caches,
    output_to_lower: bool,
    output_to_upper: bool,
    yaml_columns: list[str],
    database_columns: list[str],
    expected_order: list[str],
):
    from dbt.contracts.graph.nodes import ColumnInfo
    from dbt_common.contracts.metadata import ColumnMetadata

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = OrderedDict(
        (name, ColumnInfo.from_dict({"name": name, "description": ""})) for name in yaml_columns
    )

    context = mock.MagicMock()
    context.settings.output_to_lower = output_to_lower
    context.settings.output_to_upper = output_to_upper
    context.project.runtime_cfg.credentials.type = "postgres"

    incoming = OrderedDict(
        (
            name,
            ColumnMetadata(name=name, type="TEXT", index=index),
        )
        for index, name in enumerate(database_columns)
    )

    with (
        mock.patch("dbt_osmosis.core.introspection.get_columns", return_value=incoming),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        sort_columns_as_in_database(context, mock_node)

    assert list(mock_node.columns) == expected_order


def test_sort_columns_alphabetically(yaml_context: YamlRefactorContext, fresh_caches):
    """Check that sort_columns_alphabetically doesn't blow up in real project usage."""
    sort_columns_alphabetically(yaml_context)


def test_sort_columns_as_configured(yaml_context: YamlRefactorContext, fresh_caches):
    """By default, the sort_by is 'database', but let's confirm it doesn't blow up."""
    sort_columns_as_configured(yaml_context)


def test_synchronize_data_types(yaml_context: YamlRefactorContext, fresh_caches):
    """Synchronizes data types with the DB."""
    synchronize_data_types(yaml_context)


def test_sort_columns_alphabetically_with_output_to_lower(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting respects output-to-lower setting.

    When output-to-lower is enabled, columns should be sorted based on their
    lowercase form, not their original case. This ensures that after case
    conversion, the columns remain in alphabetical order.

    For example, with columns ["ZEBRA", "apple", "Banana"]:
    - Without fix: Sorted as ["ZEBRA", "Banana", "apple"] (ASCII order)
    - After lower: ["zebra", "banana", "apple"] (WRONG - not alphabetical)
    - With fix: Sorted as ["apple", "Banana", "ZEBRA"] (lowercase order)
    - After lower: ["apple", "banana", "zebra"] (CORRECT - alphabetical)
    """
    # Create a mock node with mixed-case column names that would sort
    # incorrectly in ASCII order
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "ZEBRA": mock.MagicMock(name="ZEBRA"),
        "apple": mock.MagicMock(name="apple"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context with output_to_lower enabled
    context_with_lower = mock.MagicMock()
    context_with_lower.settings.output_to_lower = True
    context_with_lower.settings.output_to_upper = False

    # Sort with output_to_lower enabled
    sort_columns_alphabetically(context_with_lower, mock_node)

    # Verify columns are sorted by lowercase name
    column_names = list(mock_node.columns.keys())
    assert column_names == ["apple", "Banana", "ZEBRA"], (
        f"Columns should be sorted by lowercase name, got {column_names}"
    )


def test_sort_columns_alphabetically_with_output_to_upper(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting respects output-to-upper setting.

    When output-to-upper is enabled, columns should be sorted based on their
    uppercase form, not their original case.
    """
    # Create a mock node with mixed-case column names
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "zebra": mock.MagicMock(name="zebra"),
        "APPLE": mock.MagicMock(name="APPLE"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context with output_to_upper enabled
    context_with_upper = mock.MagicMock()
    context_with_upper.settings.output_to_lower = False
    context_with_upper.settings.output_to_upper = True

    # Sort with output_to_upper enabled
    sort_columns_alphabetically(context_with_upper, mock_node)

    # Verify columns are sorted by uppercase name
    column_names = list(mock_node.columns.keys())
    assert column_names == ["APPLE", "Banana", "zebra"], (
        f"Columns should be sorted by uppercase name, got {column_names}"
    )


def test_sort_columns_alphabetically_without_case_conversion(
    yaml_context: YamlRefactorContext, fresh_caches
):
    """Test that alphabetical sorting works correctly when no case conversion is enabled.

    When neither output-to-lower nor output-to-upper is set, columns should be
    sorted using their original case (standard lexicographic order).
    """
    # Create a mock node with mixed-case column names
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = {
        "ZEBRA": mock.MagicMock(name="ZEBRA"),
        "apple": mock.MagicMock(name="apple"),
        "Banana": mock.MagicMock(name="Banana"),
    }

    # Create a context without case conversion
    context_no_conversion = mock.MagicMock()
    context_no_conversion.settings.output_to_lower = False
    context_no_conversion.settings.output_to_upper = False

    # Sort without case conversion
    sort_columns_alphabetically(context_no_conversion, mock_node)

    # Verify columns are sorted by original name (ASCII order)
    column_names = list(mock_node.columns.keys())
    # In ASCII: uppercase letters come before lowercase
    assert column_names == ["Banana", "ZEBRA", "apple"], (
        f"Columns should be sorted by original name (ASCII order), got {column_names}"
    )


def test_inject_missing_columns_idempotent_with_output_to_upper_on_postgres(fresh_caches):
    """Test that inject_missing_columns is idempotent on non-Snowflake DBs with output-to-upper.

    Scenario (PostgreSQL + output-to-upper):
    1st run: DB returns 'zebra' → injected as 'ZEBRA'
    2nd run: current_columns has 'ZEBRA', incoming has 'zebra'
    → Should NOT re-add the column (idempotent)
    """
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.resource_type = "model"
    # Simulate state after first run: columns stored with uppercase keys
    mock_node.columns = OrderedDict({
        "ZEBRA": ColumnInfo.from_dict({"name": "ZEBRA", "description": "existing"}),
    })

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = True
    context.settings.output_to_lower = False
    context.settings.output_to_upper = True
    context.project.runtime_cfg.credentials.type = "postgres"

    mock_col = mock.MagicMock()
    mock_col.type = None
    mock_col.comment = ""

    # DB returns lowercase (PostgreSQL behavior)
    incoming = OrderedDict([("zebra", mock_col)])

    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        inject_missing_columns(context, mock_node)

    # Should still have exactly one column with original description preserved
    assert list(mock_node.columns.keys()) == ["ZEBRA"]
    assert mock_node.columns["ZEBRA"].description == "existing"


@pytest.mark.parametrize("fusion_compat", [False, True])
def test_inject_missing_columns_preserves_column_config_for_sync(fresh_caches, fusion_compat: bool):
    """Injected ColumnInfo objects must keep dbt's config field and sync without empty config."""
    from collections import OrderedDict

    from dbt_osmosis.core.sync_operations import _sync_doc_section

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.name = "test_model"
    mock_node.schema = "main"
    mock_node.description = ""
    mock_node.resource_type = "model"
    mock_node.columns = OrderedDict()

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = False
    context.settings.scaffold_empty_configs = False
    context.settings.skip_merge_meta = False
    context.settings.prefer_yaml_values = False
    context.settings.use_unrendered_descriptions = False
    context.settings.output_to_lower = False
    context.settings.output_to_upper = False
    context.fusion_compat = fusion_compat
    context.placeholders = set()
    context.read_catalog.return_value = None
    context.project.runtime_cfg.credentials.type = "postgres"

    mock_col = mock.MagicMock()
    mock_col.type = "text"
    mock_col.comment = "from database"
    incoming = OrderedDict([("new_col", mock_col)])

    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        inject_missing_columns(context, mock_node)
        doc_section: dict[str, object] = {}
        _sync_doc_section(context, mock_node, doc_section)

    if hasattr(mock_node.columns["new_col"], "config"):
        assert mock_node.columns["new_col"].config is not None
    assert doc_section["columns"] == [
        {"name": "new_col", "description": "from database", "data_type": "text"}
    ]
    assert "config" not in doc_section["columns"][0]


def test_remove_columns_not_in_database_with_output_to_upper_on_postgres(fresh_caches):
    """Test that remove_columns_not_in_database doesn't incorrectly remove columns
    when output-to-upper is active on a non-Snowflake DB.

    Scenario (PostgreSQL + output-to-upper):
    node.columns has 'ZEBRA' (uppercased), DB returns 'zebra' (lowercase).
    normalize_column_name('ZEBRA', 'postgres') = 'ZEBRA' ≠ 'zebra'
    → Without fix: 'ZEBRA' flagged as extra and removed incorrectly.
    → With fix: case-insensitive comparison prevents incorrect removal.
    """
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = OrderedDict({
        "ZEBRA": ColumnInfo.from_dict({"name": "ZEBRA", "description": "a column"}),
        "APPLE": ColumnInfo.from_dict({"name": "APPLE", "description": "another column"}),
    })

    context = mock.MagicMock()
    context.settings.output_to_lower = False
    context.settings.output_to_upper = True
    context.project.runtime_cfg.credentials.type = "postgres"

    mock_col_z = mock.MagicMock()
    mock_col_z.type = "VARCHAR"
    mock_col_z.comment = ""
    mock_col_a = mock.MagicMock()
    mock_col_a.type = "INTEGER"
    mock_col_a.comment = ""

    # DB returns lowercase (PostgreSQL behavior)
    incoming = OrderedDict([("zebra", mock_col_z), ("apple", mock_col_a)])

    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        remove_columns_not_in_database(context, mock_node)

    # Both columns should be preserved (not removed)
    assert set(mock_node.columns.keys()) == {"ZEBRA", "APPLE"}


def test_remove_columns_not_in_database_removes_truly_extra_columns(fresh_caches):
    """Test that truly extra columns are still removed even with case conversion."""
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.columns = OrderedDict({
        "ZEBRA": ColumnInfo.from_dict({"name": "ZEBRA", "description": ""}),
        "STALE": ColumnInfo.from_dict({"name": "STALE", "description": "removed from DB"}),
    })

    context = mock.MagicMock()
    context.settings.output_to_lower = False
    context.settings.output_to_upper = True
    context.project.runtime_cfg.credentials.type = "postgres"

    mock_col = mock.MagicMock()
    mock_col.type = "VARCHAR"
    mock_col.comment = ""

    # DB only has 'zebra', not 'stale'
    incoming = OrderedDict([("zebra", mock_col)])

    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        remove_columns_not_in_database(context, mock_node)

    # STALE should be removed, ZEBRA should remain
    assert list(mock_node.columns.keys()) == ["ZEBRA"]


def test_synchronize_data_types_with_output_to_upper_on_postgres(fresh_caches):
    """Test that synchronize_data_types matches columns correctly when output-to-upper
    is active on a non-Snowflake DB.

    Scenario (PostgreSQL + output-to-upper):
    node.columns has 'ZEBRA', DB returns column 'zebra' with type 'varchar'.
    normalize_column_name('ZEBRA', 'postgres') = 'ZEBRA', but incoming key is 'zebra'.
    → Without fix: lookup fails, data type not synced.
    → With fix: case-insensitive fallback finds the match.
    """
    from collections import OrderedDict

    from dbt.contracts.graph.nodes import ColumnInfo

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    col = ColumnInfo.from_dict({"name": "ZEBRA", "description": "", "data_type": ""})
    mock_node.columns = OrderedDict({"ZEBRA": col})

    context = mock.MagicMock()
    context.settings.skip_add_data_types = False
    context.settings.output_to_lower = False
    context.settings.output_to_upper = True
    context.project.runtime_cfg.credentials.type = "postgres"

    mock_col = mock.MagicMock()
    mock_col.type = "varchar"
    mock_col.comment = ""

    incoming = OrderedDict([("zebra", mock_col)])

    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        synchronize_data_types(context, mock_node)

    # Data type should be synced and uppercased (output-to-upper)
    assert mock_node.columns["ZEBRA"].data_type == "VARCHAR"


def test_inject_missing_columns_applies_output_to_lower(fresh_caches):
    """Test that inject_missing_columns converts new column keys to lowercase
    when output-to-lower is enabled.

    This fixes the issue where Snowflake returns uppercase column names,
    which are injected as uppercase keys. Subsequent alphabetical sorting
    uses lowercase comparison, but the keys remain uppercase, causing
    incorrect sort order on the first run.
    """
    from collections import OrderedDict

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.resource_type = "model"
    mock_node.columns = OrderedDict()

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = False
    context.settings.output_to_lower = True
    context.settings.output_to_upper = False
    context.project.runtime_cfg.credentials.type = "snowflake"

    # Simulate database returning uppercase column names (Snowflake behavior)
    mock_col_a = mock.MagicMock()
    mock_col_a.type = "VARCHAR"
    mock_col_a.comment = ""
    mock_col_b = mock.MagicMock()
    mock_col_b.type = "INTEGER"
    mock_col_b.comment = ""
    mock_col_c = mock.MagicMock()
    mock_col_c.type = "BOOLEAN"
    mock_col_c.comment = ""

    incoming = OrderedDict([("ZEBRA", mock_col_a), ("APPLE", mock_col_b), ("BANANA", mock_col_c)])

    # Patch at dbt_osmosis.core.introspection (source module) because
    # inject_missing_columns uses local imports (from ... import get_columns),
    # which re-resolve the module attribute on each call.
    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        inject_missing_columns(context, mock_node)

    # Keys should be lowercase
    column_keys = list(mock_node.columns.keys())
    assert all(k == k.lower() for k in column_keys), (
        f"All column keys should be lowercase, got {column_keys}"
    )
    assert set(column_keys) == {"zebra", "apple", "banana"}


def test_inject_missing_columns_with_lower_then_sort_alphabetically(fresh_caches):
    """End-to-end test: inject with output-to-lower followed by alphabetical sort
    should produce correctly ordered lowercase keys on the first run.
    """
    from collections import OrderedDict

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.test_model"
    mock_node.resource_type = "model"
    mock_node.columns = OrderedDict()

    context = mock.MagicMock()
    context.settings.skip_add_columns = False
    context.settings.skip_add_source_columns = False
    context.settings.skip_add_data_types = True
    context.settings.output_to_lower = True
    context.settings.output_to_upper = False
    context.project.runtime_cfg.credentials.type = "snowflake"

    mock_col = mock.MagicMock()
    mock_col.type = None
    mock_col.comment = ""

    incoming = OrderedDict([
        ("ZEBRA", mock_col),
        ("APPLE", mock_col),
        ("BANANA", mock_col),
    ])

    # Patch at source module; see comment in test_inject_missing_columns_applies_output_to_lower
    with (
        mock.patch(
            "dbt_osmosis.core.introspection.get_columns",
            return_value=incoming,
        ),
        mock.patch(
            "dbt_osmosis.core.introspection.resolve_setting",
            side_effect=lambda *args, fallback=None, **kw: fallback,
        ),
    ):
        inject_missing_columns(context, mock_node)

    # Now sort alphabetically
    sort_columns_alphabetically(context, mock_node)

    column_keys = list(mock_node.columns.keys())
    assert column_keys == ["apple", "banana", "zebra"], (
        f"Columns should be alphabetically sorted lowercase on first run, got {column_keys}"
    )


def test_apply_semantic_analysis_accumulates_description_tags_and_meta(monkeypatch):
    from dbt.contracts.graph.nodes import ColumnInfo

    def fake_analyze_column_semantics(**kwargs):
        return {
            "semantic_type": "identifier",
            "tags": ["existing_tag", "primary_key"],
            "meta": {
                "semantic_type": "identifier",
                "owner": "suggested owner",
            },
        }

    def fake_generate_semantic_description(**kwargs):
        return "Generated customer identifier."

    monkeypatch.setitem(fake_analyze_column_semantics.__globals__, "get_llm_client", object)
    monkeypatch.setattr(
        "dbt_osmosis.core.llm.analyze_column_semantics",
        fake_analyze_column_semantics,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.llm.generate_semantic_description",
        fake_generate_semantic_description,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.inheritance._build_column_knowledge_graph",
        lambda context, node: {},
    )

    column = ColumnInfo.from_dict({
        "name": "customer_id",
        "description": "short",
        "tags": ["existing_tag"],
        "meta": {"owner": "existing owner"},
    })
    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.customers"
    mock_node.name = "customers"
    mock_node.description = ""
    mock_node.raw_sql = ""
    mock_node.columns = OrderedDict({"customer_id": column})

    apply_semantic_analysis(mock.MagicMock(), mock_node)

    updated_column = mock_node.columns["customer_id"]
    assert updated_column.description == "Generated customer identifier."
    assert updated_column.tags == ["existing_tag", "primary_key"]
    assert updated_column.meta == {
        "semantic_type": "identifier",
        "owner": "existing owner",
    }


def test_apply_semantic_analysis_continues_after_column_failure(monkeypatch):
    from dbt.contracts.graph.nodes import ColumnInfo

    def fake_analyze_column_semantics(**kwargs):
        if kwargs["column_name"] == "bad_column":
            raise RuntimeError("analysis failed")
        return {"semantic_type": "dimension"}

    def fake_generate_semantic_description(**kwargs):
        return "Generated description."

    monkeypatch.setitem(fake_analyze_column_semantics.__globals__, "get_llm_client", object)
    monkeypatch.setattr(
        "dbt_osmosis.core.llm.analyze_column_semantics",
        fake_analyze_column_semantics,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.llm.generate_semantic_description",
        fake_generate_semantic_description,
    )
    monkeypatch.setattr(
        "dbt_osmosis.core.inheritance._build_column_knowledge_graph",
        lambda context, node: {},
    )

    mock_node = mock.MagicMock()
    mock_node.unique_id = "model.test.customers"
    mock_node.name = "customers"
    mock_node.description = ""
    mock_node.raw_sql = ""
    mock_node.columns = OrderedDict({
        "bad_column": ColumnInfo.from_dict({
            "name": "bad_column",
            "description": "bad",
        }),
        "good_column": ColumnInfo.from_dict({
            "name": "good_column",
            "description": "good",
        }),
    })

    apply_semantic_analysis(mock.MagicMock(), mock_node)

    assert mock_node.columns["bad_column"].description == "bad"
    assert mock_node.columns["good_column"].description == "Generated description."
