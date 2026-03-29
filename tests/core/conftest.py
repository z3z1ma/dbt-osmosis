"""Pytest configuration for tests/core directory.

This module ensures the demo_duckdb manifest exists before running tests
that depend on it (e.g., test_real_manifest_contains_customers).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support import manifest_requires_refresh, run_dbt_command


def _ensure_manifest_exists(project_dir: Path = Path("demo_duckdb")) -> None:
    """Ensure demo_duckdb/target/manifest.json exists and matches current fixture inputs.

    This is a lightweight alternative to the full dbt run seed+run pipeline.
    dbt parse generates the manifest without executing any SQL, and we rerun it
    whenever fixture inputs are newer than the compiled artifact.
    """
    manifest_path = project_dir / "target" / "manifest.json"

    if not manifest_requires_refresh(manifest_path, project_dir):
        return

    print("\n" + "=" * 60)
    if manifest_path.exists():
        print("Manifest is stale - running dbt parse to refresh it")
    else:
        print("Manifest not found - running dbt parse to generate it")
    print("=" * 60)

    run_dbt_command([
        "parse",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
    ])

    if not manifest_path.exists():
        raise RuntimeError(f"Manifest file not created at {manifest_path}")

    print(f"✓ Manifest generated at {manifest_path}")

    print("=" * 60 + "\n")


@pytest.fixture(scope="session", autouse=True)
def ensure_demo_manifest() -> None:
    """Session-scoped fixture that ensures demo_duckdb manifest exists.

    This fixture runs automatically (autouse=True) before any tests in this
    directory (tests/core/). It checks if demo_duckdb/target/manifest.json
    exists, and it reruns dbt parse whenever the compiled artifact is stale.

    This is a lightweight operation that only runs once per test session.
    The manifest stays cached in the source tree for subsequent test runs until
    demo project inputs change.
    """
    _ensure_manifest_exists()
