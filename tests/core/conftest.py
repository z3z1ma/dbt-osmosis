"""Pytest configuration for tests/core directory.

This module ensures the demo_duckdb manifest exists before running tests
that depend on it (e.g., test_real_manifest_contains_customers).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _ensure_manifest_exists() -> None:
    """Ensure demo_duckdb/target/manifest.json exists by running dbt parse if needed.

    This is a lightweight alternative to the full dbt run seed+run pipeline.
    dbt parse generates the manifest without executing any SQL.
    """
    manifest_path = Path("demo_duckdb/target/manifest.json")

    if manifest_path.exists():
        return

    from dbt.cli.main import dbtRunner

    print("\n" + "=" * 60)
    print("Manifest not found - running dbt parse to generate it")
    print("=" * 60)

    result = dbtRunner().invoke([
        "parse",
        "--project-dir",
        "demo_duckdb",
        "--profiles-dir",
        "demo_duckdb",
    ])

    if result.success:
        print(f"✓ Manifest generated at {manifest_path}")
    else:
        raise RuntimeError(
            f"dbt parse failed: {result.exception if hasattr(result, 'exception') else 'Unknown error'}",
        )

    print("=" * 60 + "\n")


@pytest.fixture(scope="session", autouse=True)
def ensure_demo_manifest() -> None:
    """Session-scoped fixture that ensures demo_duckdb manifest exists.

    This fixture runs automatically (autouse=True) before any tests in this
    directory (tests/core/). It checks if demo_duckdb/target/manifest.json
    exists and runs dbt parse if it doesn't.

    This is a lightweight operation that only runs once per test session.
    The manifest is cached in the source tree for subsequent test runs.
    """
    _ensure_manifest_exists()
