# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import json
from pathlib import Path

import dbt.version
from dbt.contracts.graph.manifest import Manifest
from packaging.version import Version

# Test that all major functions can still be imported from the main osmosis module
# This ensures backwards compatibility
from dbt_osmosis.core.osmosis import (
    DbtConfiguration,
    YamlRefactorSettings,
    compile_sql_code,
    create_dbt_project_context,
    execute_sql_code,
    get_columns,
    inherit_upstream_column_knowledge,
    inject_missing_columns,
    sync_node_to_yaml,
)

dbt_version = Version(dbt.version.get_installed_version().to_version_string(skip_matcher=True))


def load_demo_manifest(manifest_path: Path) -> Manifest:
    """Helper for verifying certain known nodes."""
    assert manifest_path.is_file(), f"Must have a compiled manifest.json at {manifest_path}"
    with manifest_path.open("r") as f:
        return Manifest.from_dict(json.load(f))


def test_real_manifest_contains_customers(demo_manifest_path: Path):
    """Quick test ensuring your 'demo_duckdb' project manifest includes 'customers' node
    in the expected location (model.jaffle_shop_duckdb.customers).
    """
    manifest = load_demo_manifest(demo_manifest_path)
    assert "model.jaffle_shop_duckdb.customers" in manifest.nodes


def test_backwards_compatibility_imports():
    """Test that all major functions are available from the main osmosis module.
    This ensures that existing code using osmosis imports continues to work.
    """
    # Test that key functions are callable (basic smoke test)
    assert callable(create_dbt_project_context)
    assert callable(get_columns)
    assert callable(compile_sql_code)
    assert callable(execute_sql_code)
    assert callable(sync_node_to_yaml)
    assert callable(inject_missing_columns)
    assert callable(inherit_upstream_column_knowledge)

    # Test that classes can be instantiated
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    assert cfg.project_dir == "demo_duckdb"

    settings = YamlRefactorSettings(dry_run=True)
    assert settings.dry_run is True
