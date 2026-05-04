"""Pytest configuration for tests/core directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import create_temp_project_copy, run_dbt_command


def _build_isolated_demo_manifest(
    temp_dir: Path,
    source_dir: Path = Path("demo_duckdb"),
    target: str = "test",
) -> Path:
    """Parse the demo project in an isolated copy and return its manifest path."""
    project_dir = create_temp_project_copy(source_dir, temp_dir)
    manifest_path = project_dir / "target" / "manifest.json"

    print("\n" + "=" * 60)
    print(f"Parsing isolated demo manifest in {project_dir}")
    print("=" * 60)

    run_dbt_command([
        "parse",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--target",
        target,
    ])

    if not manifest_path.exists():
        raise RuntimeError(f"Manifest file not created at {manifest_path}")

    print(f"✓ Isolated manifest generated at {manifest_path}")
    print("=" * 60 + "\n")
    return manifest_path


@pytest.fixture(scope="session")
def demo_manifest_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped manifest path built from a temp demo project copy."""
    return _build_isolated_demo_manifest(tmp_path_factory.mktemp("demo_manifest"))
