# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import json
import os
import threading
import time
from pathlib import Path
from unittest import mock

from dbt_osmosis.core.config import (
    DbtConfiguration,
    _detect_fusion_manifest,
    _reload_manifest,
    config_to_namespace,
    discover_profiles_dir,
    discover_project_dir,
)
from dbt_osmosis.core.settings import YamlRefactorContext


def test_discover_project_dir(tmp_path):
    """Ensures discover_project_dir falls back properly if no environment
    variable is set and no dbt_project.yml is found in parents.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        found = discover_project_dir()
        assert str(tmp_path.resolve()) == found
    finally:
        os.chdir(original_cwd)


def test_discover_profiles_dir(tmp_path):
    """Ensures discover_profiles_dir falls back to ~/.dbt
    if no DBT_PROFILES_DIR is set and no local profiles.yml is found.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        found = discover_profiles_dir()
        assert str(Path.home() / ".dbt") == found
    finally:
        os.chdir(original_cwd)


def test_config_to_namespace():
    """Tests that DbtConfiguration is properly converted to argparse.Namespace."""
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb", target="dev")
    ns = config_to_namespace(cfg)
    assert ns.project_dir == "demo_duckdb"
    assert ns.profiles_dir == "demo_duckdb"
    assert ns.target == "dev"


def test_reload_manifest(yaml_context: YamlRefactorContext):
    """Basic check that _reload_manifest doesn't raise, given a real project."""
    _reload_manifest(yaml_context.project)


def test_adapter_ttl_expiration(yaml_context: YamlRefactorContext):
    """Check that if the TTL is expired, we refresh the connection in DbtProjectContext.adapter.
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


class TestDetectFusionManifest:
    """Tests for _detect_fusion_manifest() Fusion detection logic."""

    def test_no_manifest_no_binary(self, tmp_path):
        """No manifest and no Fusion binary → returns False."""
        with mock.patch("shutil.which", return_value=None):
            assert _detect_fusion_manifest(str(tmp_path)) is False

    def test_no_manifest_but_dbtf_binary(self, tmp_path):
        """No manifest but dbtf binary on PATH → returns True."""
        with mock.patch(
            "shutil.which", side_effect=lambda cmd: "/usr/bin/dbtf" if cmd == "dbtf" else None
        ):
            assert _detect_fusion_manifest(str(tmp_path)) is True

    def test_no_manifest_but_dbt_fusion_binary(self, tmp_path):
        """No manifest but dbt-fusion binary on PATH → returns True."""
        with mock.patch(
            "shutil.which",
            side_effect=lambda cmd: "/usr/bin/dbt-fusion" if cmd == "dbt-fusion" else None,
        ):
            assert _detect_fusion_manifest(str(tmp_path)) is True

    def test_dbt_core_manifest_v12(self, tmp_path):
        """dbt-core manifest (v12) → returns False."""
        target = tmp_path / "target"
        target.mkdir()
        manifest = {
            "metadata": {
                "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json",
                "dbt_version": "1.11.2",
            },
        }
        (target / "manifest.json").write_text(json.dumps(manifest))
        assert _detect_fusion_manifest(str(tmp_path)) is False

    def test_fusion_manifest_v20(self, tmp_path):
        """Fusion manifest (v20) → returns True."""
        target = tmp_path / "target"
        target.mkdir()
        manifest = {
            "metadata": {
                "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v20.json",
                "dbt_version": "2025.1.0",
            },
        }
        (target / "manifest.json").write_text(json.dumps(manifest))
        assert _detect_fusion_manifest(str(tmp_path)) is True

    def test_future_manifest_v13(self, tmp_path):
        """Any manifest version > 12 triggers Fusion detection."""
        target = tmp_path / "target"
        target.mkdir()
        manifest = {
            "metadata": {
                "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v13.json",
            },
        }
        (target / "manifest.json").write_text(json.dumps(manifest))
        assert _detect_fusion_manifest(str(tmp_path)) is True

    def test_malformed_manifest(self, tmp_path):
        """Malformed manifest.json → returns False gracefully."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "manifest.json").write_text("not valid json{{{")
        assert _detect_fusion_manifest(str(tmp_path)) is False

    def test_manifest_missing_metadata(self, tmp_path):
        """Manifest with no metadata section → returns False."""
        target = tmp_path / "target"
        target.mkdir()
        (target / "manifest.json").write_text(json.dumps({"nodes": {}}))
        assert _detect_fusion_manifest(str(tmp_path)) is False

    def test_manifest_empty_schema_version(self, tmp_path):
        """Manifest with empty dbt_schema_version → returns False."""
        target = tmp_path / "target"
        target.mkdir()
        manifest = {"metadata": {"dbt_schema_version": ""}}
        (target / "manifest.json").write_text(json.dumps(manifest))
        assert _detect_fusion_manifest(str(tmp_path)) is False
