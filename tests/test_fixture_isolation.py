"""Regression tests for hermetic demo fixture copying and smoke scripts."""

from __future__ import annotations

from pathlib import Path

import ruamel.yaml

from tests.conftest import _create_temp_project_copy


def _write_minimal_duckdb_profile(project_dir: Path) -> None:
    project_dir.mkdir(parents=True)
    (project_dir / "profiles.yml").write_text(
        "\n".join([
            "jaffle_shop:",
            "  target: test",
            "  outputs:",
            "    test:",
            "      type: duckdb",
            '      path: "test.db"',
            "      threads: 1",
            "",
        ]),
    )


def _load_test_profile_path(project_dir: Path) -> Path:
    yaml = ruamel.yaml.YAML(typ="safe")
    profile = yaml.load((project_dir / "profiles.yml").read_text())
    return Path(profile["jaffle_shop"]["outputs"]["test"]["path"])


def test_temp_project_copy_rewrites_duckdb_profile_paths_inside_copy(tmp_path: Path) -> None:
    """Copied profiles should not resolve DuckDB files from caller cwd or repo root."""
    source_dir = tmp_path / "repo" / "demo_duckdb"
    _write_minimal_duckdb_profile(source_dir)

    project_dir = _create_temp_project_copy(source_dir, tmp_path / "work")

    copied_path = _load_test_profile_path(project_dir)
    assert copied_path.is_absolute()
    assert copied_path == project_dir / "test.db"
    assert copied_path != source_dir.parent / "test.db"


def test_temp_project_copy_excludes_generated_and_local_database_artifacts_by_default(
    tmp_path: Path,
) -> None:
    """Fixture copies should not inherit ignored build outputs or local databases."""
    source_dir = tmp_path / "demo_duckdb"
    _write_minimal_duckdb_profile(source_dir)
    (source_dir / "models").mkdir()
    (source_dir / "models" / "customers.sql").write_text("select 1")

    for relative_path in [
        "target/manifest.json",
        "logs/dbt.log",
        "dbt_packages/package/dbt_project.yml",
        ".pytest_cache/v/cache/nodeids",
        ".cache/tool/output.json",
        ".env",
        ".DS_Store",
        "models/__pycache__/compiled.pyc",
        "models/debug.log",
        "test.db",
        "db.sqlite3",
        "jaffle_shop.duckdb",
        "local.duckdb",
    ]:
        artifact = source_dir / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("generated")

    project_dir = _create_temp_project_copy(source_dir, tmp_path / "work")

    assert (project_dir / "models" / "customers.sql").is_file()
    for relative_path in [
        "target",
        "logs",
        "dbt_packages",
        ".pytest_cache",
        ".cache",
        ".env",
        ".DS_Store",
        "models/__pycache__",
        "models/debug.log",
        "test.db",
        "db.sqlite3",
        "jaffle_shop.duckdb",
        "local.duckdb",
    ]:
        assert not (project_dir / relative_path).exists(), relative_path


def test_core_manifest_fixture_does_not_depend_on_source_tree_target() -> None:
    """Core manifest support should parse an isolated temp copy, not demo_duckdb/target."""
    core_conftest = Path("tests/core/conftest.py").read_text()

    assert "manifest_requires_refresh" not in core_conftest
    assert "demo_duckdb/target" not in core_conftest
    assert "autouse=True" not in core_conftest


def test_integration_script_uses_temp_copy_without_destructive_git_cleanup() -> None:
    """The integration smoke script should mutate only a temporary demo copy."""
    script = Path("demo_duckdb/integration_tests.sh").read_text()

    assert "mktemp" in script
    assert "git checkout" not in script
    assert "git clean" not in script
    assert "uv run dbt-osmosis" not in script
    assert "--target test" in script


def test_ci_integration_smoke_delegates_to_safe_script_without_git_cleanup() -> None:
    """CI should not inline destructive source-tree restore commands."""
    workflow = Path(".github/workflows/tests.yml").read_text()

    assert "demo_duckdb/integration_tests.sh" in workflow
    assert "git checkout --" not in workflow
    assert "git clean" not in workflow
    assert "test ! -e test.db" in workflow
    assert "test ! -e demo_duckdb/target" in workflow


def test_ci_and_taskfile_parse_demo_copy_instead_of_source_tree() -> None:
    """Parse smokes should not recreate source demo_duckdb/target artifacts."""
    workflow = Path(".github/workflows/tests.yml").read_text()
    taskfile = Path("Taskfile.yml").read_text()

    assert "dbt parse --project-dir demo_duckdb" not in workflow
    assert "dbt parse --project-dir demo_duckdb" not in taskfile
    assert "create_temp_project_copy" in workflow
    assert "task parse-demo" in taskfile
