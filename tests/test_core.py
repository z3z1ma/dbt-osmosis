# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import json
import os
import threading
import time
from pathlib import Path
from unittest import mock

import dbt.version
import pytest
from dbt.artifacts.resources.types import NodeType
from dbt.contracts.graph.manifest import Manifest
from packaging.version import Version

from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    FuzzyCaseMatching,
    FuzzyPrefixMatching,
    RestructureDeltaPlan,
    RestructureOperation,
    YamlRefactorContext,
    YamlRefactorSettings,
    _find_first,
    _get_setting_for_node,
    _maybe_use_precise_dtype,
    _reload_manifest,
    _topological_sort,
    apply_restructure_plan,
    commit_yamls,
    compile_sql_code,
    config_to_namespace,
    create_dbt_project_context,
    create_missing_source_yamls,
    discover_profiles_dir,
    discover_project_dir,
    draft_restructure_delta_plan,
    execute_sql_code,
    get_columns,
    get_plugin_manager,
    get_table_ref,
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    normalize_column_name,
    pretty_print_plan,
    remove_columns_not_in_database,
    sort_columns_alphabetically,
    sort_columns_as_configured,
    sort_columns_as_in_database,
    sync_node_to_yaml,
    synchronize_data_types,
)

dbt_version = Version(dbt.version.get_installed_version().to_version_string(skip_matcher=True))


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    """
    Creates a YamlRefactorContext for the real 'demo_duckdb' project,
    which must contain a valid dbt_project.yml, profiles, and manifest.
    """
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    # Typically, we can place dbt-osmosis config in `vars`,
    # if you want to skip or enable certain features, do it here:
    cfg.vars = {"dbt-osmosis": {}}

    project_context = create_dbt_project_context(cfg)
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,  # Avoid writing to disk in most tests
            use_unrendered_descriptions=True,
        ),
    )
    return context


@pytest.fixture(scope="function")
def fresh_caches():
    """
    Patches the internal caches so each test starts with a fresh state.
    """
    with (
        mock.patch("dbt_osmosis.core.osmosis._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.osmosis._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_discover_project_dir(tmp_path):
    """
    Ensures discover_project_dir falls back properly if no environment
    variable is set and no dbt_project.yml is found in parents.
    """
    # If the current CWD has no dbt_project.yml, it should default to it.
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        found = discover_project_dir()
        assert str(tmp_path.resolve()) == found
    finally:
        os.chdir(original_cwd)


def test_discover_profiles_dir(tmp_path):
    """
    Ensures discover_profiles_dir falls back to ~/.dbt
    if no DBT_PROFILES_DIR is set and no local profiles.yml is found.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        found = discover_profiles_dir()
        # It's not necessarily the same as tmp_path, so we just ensure it returns ~/.dbt
        assert str(Path.home() / ".dbt") == found
    finally:
        os.chdir(original_cwd)


def test_config_to_namespace():
    """
    Tests that DbtConfiguration is properly converted to argparse.Namespace.
    """
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb", target="dev")
    ns = config_to_namespace(cfg)
    assert ns.project_dir == "demo_duckdb"
    assert ns.profiles_dir == "demo_duckdb"
    assert ns.target == "dev"


def test_reload_manifest(yaml_context: YamlRefactorContext):
    """
    Basic check that _reload_manifest doesn't raise, given a real project.
    """
    _reload_manifest(yaml_context.project)


def test_create_missing_source_yamls(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Creates missing source YAML files if any are declared in dbt-osmosis sources
    but do not exist in the manifest. Typically, might be none in your project.
    """
    create_missing_source_yamls(yaml_context)


def test_draft_restructure_delta_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Ensures we can generate a restructure plan for real models and sources.
    Usually, this plan might be empty if everything lines up already.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    assert plan is not None


def test_apply_restructure_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Applies the restructure plan for the real project (in dry_run mode).
    Should not raise errors even if the plan is empty or small.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    apply_restructure_plan(yaml_context, plan, confirm=False)


def test_inherit_upstream_column_knowledge(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Minimal test that runs the inheritance logic on all matched nodes in the real project.
    """
    inherit_upstream_column_knowledge(yaml_context)


def test_get_columns_simple(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Tests the get_columns flow on a known table, e.g., 'customers'.
    Adjust this if your project has a different set of models.
    """
    # Let's find a model named "customers" from your jaffle_shop_duckdb project:
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    ref = get_table_ref(node)
    cols = get_columns(yaml_context, ref)
    assert "customer_id" in cols


def test_inject_missing_columns(yaml_context: YamlRefactorContext, fresh_caches):
    """
    If the DB has columns the YAML/manifest doesn't, we inject them.
    We run on all matched nodes to ensure no errors.
    """
    inject_missing_columns(yaml_context)


def test_remove_columns_not_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    If the manifest has columns the DB does not, we remove them.
    Typically, your real project might not have any extra columns, so this is a sanity test.
    """
    remove_columns_not_in_database(yaml_context)


def test_sort_columns_as_in_database(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Sort columns in the order the DB sees them.
    With duckdb, this is minimal but we can still ensure no errors.
    """
    sort_columns_as_in_database(yaml_context)


def test_sort_columns_alphabetically(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Check that sort_columns_alphabetically doesn't blow up in real project usage.
    """
    sort_columns_alphabetically(yaml_context)


def test_sort_columns_as_configured(yaml_context: YamlRefactorContext, fresh_caches):
    """
    By default, the sort_by is 'database', but let's confirm it doesn't blow up.
    """
    sort_columns_as_configured(yaml_context)


def test_synchronize_data_types(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Synchronizes data types with the DB.
    """
    synchronize_data_types(yaml_context)


def test_sync_node_to_yaml(yaml_context: YamlRefactorContext, fresh_caches):
    """
    For a single node, we can confirm that sync_node_to_yaml runs without error,
    using the real file or generating one if missing (in dry_run mode).
    """
    node = yaml_context.project.manifest.nodes["model.jaffle_shop_duckdb.customers"]
    sync_node_to_yaml(yaml_context, node, commit=False)


def test_commit_yamls_no_write(yaml_context: YamlRefactorContext):
    """
    Since dry_run=True, commit_yamls should not actually write anything to disk.
    We just ensure no exceptions are raised.
    """
    commit_yamls(yaml_context)


def test_find_first():
    data = [1, 2, 3, 4]
    assert _find_first(data, lambda x: x > 2) == 3
    assert _find_first(data, lambda x: x > 4) is None
    assert _find_first(data, lambda x: x > 4, default=999) == 999


def test_topological_sort():
    # We'll simulate a trivial adjacency-like approach:
    node_a = mock.MagicMock()
    node_b = mock.MagicMock()
    node_c = mock.MagicMock()
    node_a.depends_on_nodes = ["node_b"]  # a depends on b
    node_b.depends_on_nodes = ["node_c"]  # b depends on c
    node_c.depends_on_nodes = []
    input_list = [
        ("node_a", node_a),
        ("node_b", node_b),
        ("node_c", node_c),
    ]
    sorted_nodes = _topological_sort(input_list)
    # We expect node_c -> node_b -> node_a
    assert [uid for uid, _ in sorted_nodes] == ["node_c", "node_b", "node_a"]


@pytest.mark.parametrize(
    "input_col,expected",
    [
        ('"My_Col"', "My_Col"),
        ("my_col", "MY_COL"),
    ],
)
def test_normalize_column_name_snowflake(input_col, expected):
    # For snowflake, if quoted - we preserve case but strip quotes, otherwise uppercase
    assert normalize_column_name(input_col, "snowflake") == expected


def test_normalize_column_name_others():
    # For other adapters, we only strip outer quotes but do not uppercase or lowercase for now
    assert normalize_column_name('"My_Col"', "duckdb") == "My_Col"
    assert normalize_column_name("my_col", "duckdb") == "my_col"


def load_demo_manifest() -> Manifest:
    """
    Helper for verifying certain known nodes.
    """
    manifest_path = Path("demo_duckdb/target/manifest.json")
    assert manifest_path.is_file(), "Must have a compiled manifest.json in demo_duckdb/target"
    with manifest_path.open("r") as f:
        return Manifest.from_dict(json.load(f))


def test_real_manifest_contains_customers():
    """
    Quick test ensuring your 'demo_duckdb' project manifest includes 'customers' node
    in the expected location (model.jaffle_shop_duckdb.customers).
    """
    manifest = load_demo_manifest()
    assert "model.jaffle_shop_duckdb.customers" in manifest.nodes


def test_compile_sql_code_no_jinja(yaml_context: YamlRefactorContext):
    """
    Check compile_sql_code with a plain SELECT (no Jinja).
    We should skip calling the 'process_node' logic and the returned node
    should have the raw SQL as is.
    """
    raw_sql = "SELECT 1 AS mycol"
    with mock.patch("dbt_osmosis.core.osmosis.process_node") as mock_process:
        node = compile_sql_code(yaml_context.project, raw_sql)
        mock_process.assert_not_called()
    assert node.raw_code == raw_sql
    assert node.compiled_code is None


def test_compile_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """
    Compile SQL that has Jinja statements, ensuring 'process_node' is invoked and
    we get a compiled node.
    """
    raw_sql = "SELECT {{ 1 + 1 }} AS mycol"
    with (
        mock.patch("dbt_osmosis.core.osmosis.process_node") as mock_process,
        mock.patch("dbt_osmosis.core.osmosis.SqlCompileRunner.compile") as mock_compile,
    ):
        node_mock = mock.Mock()
        node_mock.raw_code = raw_sql
        node_mock.compiled_code = "SELECT 2 AS mycol"
        mock_compile.return_value = node_mock

        compiled_node = compile_sql_code(yaml_context.project, raw_sql)
        mock_process.assert_called_once()
        mock_compile.assert_called_once()
        assert compiled_node.compiled_code == "SELECT 2 AS mycol"


def test_execute_sql_code_no_jinja(yaml_context: YamlRefactorContext):
    """
    If there's no jinja, 'execute_sql_code' calls adapter.execute directly with raw_sql.
    """
    raw_sql = "SELECT 42 AS meaning"
    with mock.patch.object(yaml_context.project._adapter, "execute") as mock_execute:
        mock_execute.return_value = ("OK", mock.Mock(rows=[(42,)]))
        resp, table = execute_sql_code(yaml_context.project, raw_sql)
        mock_execute.assert_called_with(raw_sql, auto_begin=False, fetch=True)
    assert resp == "OK"
    assert table.rows[0] == (42,)


def test_execute_sql_code_with_jinja(yaml_context: YamlRefactorContext):
    """
    If there's Jinja, we compile first, then execute the compiled code.
    """
    raw_sql = "SELECT {{ 2 + 2 }} AS four"
    with (
        mock.patch.object(yaml_context.project._adapter, "execute") as mock_execute,
        mock.patch("dbt_osmosis.core.osmosis.compile_sql_code") as mock_compile,
    ):
        mock_execute.return_value = ("OK", mock.Mock(rows=[(4,)]))

        node_mock = mock.Mock()
        node_mock.compiled_code = "SELECT 4 AS four"
        mock_compile.return_value = node_mock

        resp, table = execute_sql_code(yaml_context.project, raw_sql)
        mock_compile.assert_called_once()
        assert resp == "OK"
        assert table.rows[0] == (4,)


def test_pretty_print_plan(caplog):
    """
    Test pretty_print_plan logs the correct output for each operation.
    """
    import logging

    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "my_model"}]},
            ),
            RestructureOperation(
                file_path=Path("sources/another_file.yml"),
                content={"sources": [{"name": "my_source"}]},
                superseded_paths={Path("old_file.yml"): []},
            ),
        ]
    )
    test_logger = logging.getLogger("test_logger")
    with mock.patch("dbt_osmosis.core.logger.LOGGER", test_logger):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            pretty_print_plan(plan)
    logs = caplog.text
    assert "Restructure plan includes => 2 operations" in logs
    assert "CREATE or MERGE => models/some_file.yml" in logs
    assert "['old_file.yml'] -> sources/another_file.yml" in logs


def test_missing_osmosis_config_error(yaml_context: YamlRefactorContext):
    """
    Ensures MissingOsmosisConfig is raised if there's no path template
    for a model. We'll mock the node config so we remove any 'dbt-osmosis' key.
    """
    node = None
    # Find some real model node
    for n in yaml_context.project.manifest.nodes.values():
        if n.resource_type == NodeType.Model:
            node = n
            break
    assert node, "No model found in your demo_duckdb project"

    # We'll forcibly remove the dbt_osmosis config from node.config.extra
    old = node.config.extra.pop("dbt-osmosis", None)
    node.unrendered_config.pop("dbt-osmosis", None)

    from dbt_osmosis.core.osmosis import MissingOsmosisConfig, _get_yaml_path_template

    with pytest.raises(MissingOsmosisConfig):
        _ = _get_yaml_path_template(yaml_context, node)

    node.config.extra["dbt-osmosis"] = old
    node.unrendered_config["dbt-osmosis"] = old


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


def test_adapter_ttl_expiration(yaml_context: YamlRefactorContext):
    """
    Check that if the TTL is expired, we refresh the connection in DbtProjectContext.adapter.
    We patch time.time to simulate a large jump.
    """
    project_ctx = yaml_context.project
    old_adapter = project_ctx.adapter
    # Force we have an entry in _connection_created_at
    thread_id = threading.get_ident()
    project_ctx._connection_created_at[thread_id] = time.time() - 999999  # artificially old

    with (
        mock.patch.object(old_adapter.connections, "release") as mock_release,
        mock.patch.object(old_adapter.connections, "clear_thread_connection") as mock_clear,
    ):
        new_adapter = project_ctx.adapter
        # The underlying object is the same instance, but the connection is re-acquired
        assert new_adapter == old_adapter
        mock_release.assert_called_once()
        mock_clear.assert_called_once()


def test_plugin_manager_hooks():
    """
    Ensure FuzzyCaseMatching and FuzzyPrefixMatching are registered by default,
    and that get_candidates works as expected.
    """
    pm = get_plugin_manager()
    # We can search for the classes
    plugins = pm.get_plugins()
    has_case = any(isinstance(p, FuzzyCaseMatching) for p in plugins)
    has_prefix = any(isinstance(p, FuzzyPrefixMatching) for p in plugins)
    assert has_case
    assert has_prefix

    # We'll manually trigger the hook
    # Typically: pm.hook.get_candidates(name="my_col", node=<some node>, context=<ctx>)
    results = pm.hook.get_candidates(name="my_col", node=None, context=None)
    # results is a list of lists from each plugin => flatten them
    combined = [variant for sublist in results for variant in sublist]
    # Expect e.g. my_col => MY_COL => myCol => MyCol from FuzzyCaseMatching
    # FuzzyPrefixMatching might do nothing unless we set prefix
    assert "my_col" in combined
    assert "MY_COL" in combined
    assert "myCol" in combined
    assert "MyCol" in combined


def test_apply_restructure_plan_confirm_prompt(
    yaml_context: YamlRefactorContext, fresh_caches, capsys
):
    """
    We test apply_restructure_plan with confirm=True, mocking input to 'n' to skip it.
    This ensures we handle user input logic.
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "m1"}]},
            )
        ]
    )

    with mock.patch("builtins.input", side_effect=["n"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        assert "Skipping restructure plan." in captured.err


def test_apply_restructure_plan_confirm_yes(
    yaml_context: YamlRefactorContext, fresh_caches, capsys
):
    """
    Same as above, but we input 'y' so it actually proceeds with the plan.
    (No real writing occurs due to dry_run=True).
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/whatever.yml"),
                content={"models": [{"name": "m2"}]},
            )
        ]
    )

    with mock.patch("builtins.input", side_effect=["y"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        assert "Committing all restructure changes and reloading" in captured.err
        # We don't expect "Skipping restructure plan."
        # Instead, it should apply

    # Because it's a dry_run, no file is written. But we can see it didn't skip.


def test_create_yaml_instance_settings():
    """
    Quick check that create_yaml_instance returns a configured YAML object with custom indenting.
    """
    from dbt_osmosis.core.osmosis import create_yaml_instance

    y = create_yaml_instance(indent_mapping=4, indent_sequence=2, indent_offset=0)
    assert y.map_indent == 4
    assert y.sequence_indent == 2
    assert y.sequence_dash_offset == 0
    assert y.width == 100  # default
    assert y.preserve_quotes is False
