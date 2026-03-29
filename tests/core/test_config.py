# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dbt_osmosis.core.config import (
    DbtConfiguration,
    _detect_fusion_manifest,
    _reload_manifest,
    create_dbt_project_context,
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


def test_discover_profiles_dir_finds_project_root_profiles_from_subdirectory(tmp_path):
    """Subdirectory invocation should still find a project-local profiles.yml."""
    project_root = tmp_path / "demo_project"
    nested_dir = project_root / "models" / "staging"
    nested_dir.mkdir(parents=True)
    (project_root / "dbt_project.yml").write_text("name: demo_project\nversion: '1.0'\n")
    (project_root / "profiles.yml").write_text("demo_project: {}\n")

    original_cwd = os.getcwd()
    os.chdir(nested_dir)
    try:
        found = discover_profiles_dir()
        assert str(project_root.resolve()) == found
    finally:
        os.chdir(original_cwd)


def test_discover_profiles_dir_prefers_current_directory_over_project_root(tmp_path):
    """A local profiles.yml should win before falling back to the project root."""
    project_root = tmp_path / "demo_project"
    nested_dir = project_root / "models" / "staging"
    nested_dir.mkdir(parents=True)
    (project_root / "dbt_project.yml").write_text("name: demo_project\nversion: '1.0'\n")
    (project_root / "profiles.yml").write_text("demo_project: {}\n")
    (nested_dir / "profiles.yml").write_text("nested_profile: {}\n")

    original_cwd = os.getcwd()
    os.chdir(nested_dir)
    try:
        found = discover_profiles_dir()
        assert str(nested_dir.resolve()) == found
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


def test_create_dbt_project_context_accepts_interface_registered_adapter():
    """Bootstrap succeeds when dbt-core-interface owns the factory registration."""
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    adapter = mock.Mock()
    project = SimpleNamespace(
        runtime_config=SimpleNamespace(
            credentials=SimpleNamespace(type="duckdb"),
            adapter=adapter,
        ),
        manifest=mock.Mock(),
        project_name="demo_duckdb",
    )
    factory = SimpleNamespace(adapters={"duckdb": adapter})
    context = mock.sentinel.context

    with (
        mock.patch("dbt.adapters.factory.FACTORY", factory),
        mock.patch("dbt_osmosis.core.config._detect_fusion_manifest", return_value=False),
        mock.patch("dbt_osmosis.core.config.InterfaceDbtProject.from_config", return_value=project),
        mock.patch("dbt_osmosis.core.config.DbtProjectContext.from_project", return_value=context),
        mock.patch("dbt_osmosis.core.config.importlib.import_module", side_effect=ImportError),
    ):
        result = create_dbt_project_context(cfg)

    assert result is context


def test_create_dbt_project_context_registers_project_adapter_when_factory_missing():
    """Bootstrap binds the current project adapter when the factory is empty."""
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    adapter = mock.Mock()
    project = SimpleNamespace(
        runtime_config=SimpleNamespace(
            credentials=SimpleNamespace(type="duckdb"),
            adapter=adapter,
        ),
        manifest=mock.Mock(),
        project_name="demo_duckdb",
    )
    factory = SimpleNamespace(adapters={})

    with (
        mock.patch("dbt.adapters.factory.FACTORY", factory),
        mock.patch("dbt_osmosis.core.config._detect_fusion_manifest", return_value=False),
        mock.patch("dbt_osmosis.core.config.InterfaceDbtProject.from_config", return_value=project),
        mock.patch(
            "dbt_osmosis.core.config.DbtProjectContext.from_project",
            return_value=mock.sentinel.context,
        ),
        mock.patch("dbt_osmosis.core.config.importlib.import_module", side_effect=ImportError),
    ):
        result = create_dbt_project_context(cfg)

    assert result is mock.sentinel.context
    assert factory.adapters["duckdb"] is adapter


def test_create_dbt_project_context_replaces_stale_factory_adapter():
    """Bootstrap replaces stale factory state with the current project adapter."""
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    project_adapter = mock.Mock()
    registered_adapter = mock.Mock()
    project = SimpleNamespace(
        runtime_config=SimpleNamespace(
            credentials=SimpleNamespace(type="duckdb"),
            adapter=project_adapter,
        ),
        manifest=mock.Mock(),
        project_name="demo_duckdb",
    )
    factory = SimpleNamespace(adapters={"duckdb": registered_adapter})

    with (
        mock.patch("dbt.adapters.factory.FACTORY", factory),
        mock.patch("dbt_osmosis.core.config._detect_fusion_manifest", return_value=False),
        mock.patch("dbt_osmosis.core.config.InterfaceDbtProject.from_config", return_value=project),
        mock.patch(
            "dbt_osmosis.core.config.DbtProjectContext.from_project",
            return_value=mock.sentinel.context,
        ),
        mock.patch("dbt_osmosis.core.config.importlib.import_module", side_effect=ImportError),
    ):
        result = create_dbt_project_context(cfg)

    assert result is mock.sentinel.context
    registered_adapter.cleanup_connections.assert_called_once_with()
    assert factory.adapters["duckdb"] is project_adapter


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

    def test_no_manifest_returns_false(self, tmp_path):
        """Without a manifest, there is no project-local Fusion evidence."""
        assert _detect_fusion_manifest(str(tmp_path)) is False

    def test_no_manifest_ignores_fusion_binaries_on_path(self, tmp_path):
        """Installed Fusion binaries alone are not evidence about this project."""
        with mock.patch(
            "shutil.which",
            side_effect=lambda cmd: "/usr/bin/dbtf"
            if cmd == "dbtf"
            else "/usr/bin/dbt-fusion"
            if cmd == "dbt-fusion"
            else None,
        ) as mock_which:
            assert _detect_fusion_manifest(str(tmp_path)) is False
        mock_which.assert_not_called()

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
