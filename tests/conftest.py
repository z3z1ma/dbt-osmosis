"""Shared pytest configuration and fixtures for dbt-osmosis tests.

This module provides a session-scoped fixture that sets up an in-memory
DuckDB database with the demo_duckdb project data. The in-memory database
avoids file locking issues that occur when multiple test modules try to
access the same DuckDB file concurrently.
"""

from __future__ import annotations

import gc
import time
from typing import Iterator

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings


def _run_dbt_commands(project_dir: str, profiles_dir: str) -> None:
    """Run dbt seed and dbt run to populate the in-memory database.

    Uses dbt's CLI runner to execute seed and run commands. The in-memory
    database is shared across all connections within the same process.
    """
    from dbt.cli.main import dbtRunner

    start = time.time()
    # Run dbt seed to load CSV files into the database
    seed_result = dbtRunner().invoke([
        "seed",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        "test",
    ])
    elapsed = time.time() - start
    if seed_result.success:
        print(f"✓ dbt seed completed successfully ({elapsed:.2f}s)")
    else:
        print(
            f"dbt seed failed: {seed_result.exception if hasattr(seed_result, 'exception') else 'Unknown error'}"
        )

    start = time.time()
    # Run dbt run to create models
    run_result = dbtRunner().invoke([
        "run",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        "test",
    ])
    elapsed = time.time() - start
    if run_result.success:
        print(f"✓ dbt run completed successfully ({elapsed:.2f}s)")
    else:
        print(
            f"dbt run failed: {run_result.exception if hasattr(run_result, 'exception') else 'Unknown error'}"
        )


@pytest.fixture(scope="function")
def yaml_context() -> Iterator[YamlRefactorContext]:
    """Creates a YamlRefactorContext with an in-memory DuckDB database.

    This function-scoped fixture:
    1. Creates a DbtConfiguration pointing to the test profile (:memory:)
    2. Runs dbt seed and dbt run to populate the in-memory database
    3. Creates and yields a YamlRefactorContext
    4. Properly closes connections on teardown
    5. Cleans up cached manifest.json after test to prevent test pollution

    Using :memory: avoids DuckDB file locking issues and ensures each test
    has a fresh manifest, preventing state pollution between tests.

    Yields:
        YamlRefactorContext: Configured context with populated database
    """
    from pathlib import Path

    project_dir = "demo_duckdb"
    profiles_dir = "demo_duckdb"

    # Use the test profile with in-memory database
    cfg = DbtConfiguration(
        project_dir=project_dir,
        profiles_dir=profiles_dir,
        target="test",  # Uses profiles_test.yml with :memory:
    )
    cfg.vars = {"dbt-osmosis": {}}

    # Run dbt seed and run BEFORE creating the project context
    # This ensures the in-memory database is populated
    print("\n=== Setting up in-memory DuckDB database ===")
    _run_dbt_commands(project_dir, profiles_dir)
    print("=== Database setup complete ===\n")

    # Create the project context (will use the populated in-memory database)
    start = time.time()
    project_context = create_dbt_project_context(cfg)
    print(f"✓ create_dbt_project_context took {time.time() - start:.2f}s")

    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,
            use_unrendered_descriptions=True,
        ),
    )

    yield context

    # Teardown: Explicitly close connections and clean up manifest cache
    print("\n=== Tearing down test database connections ===")
    try:
        if hasattr(project_context, "_project") and project_context._project is not None:
            if hasattr(project_context._project, "adapter"):
                adapter = project_context._project.adapter
                if hasattr(adapter, "connections") and hasattr(adapter.connections, "close"):
                    adapter.connections.close()
                    print("✓ Adapter connections closed")
    except Exception as e:
        print(f"Warning: Error during teardown: {e}")

    # Delete the DbtProject reference to clear the WeakValueDictionary cache
    try:
        if hasattr(project_context, "_project"):
            del project_context._project
            # Trigger garbage collection to clear WeakValueDictionary entries
            gc.collect()
            print("✓ DbtProject reference deleted and garbage collected")
    except Exception as e:
        print(f"Warning: Error deleting DbtProject reference: {e}")

    # Clean up cached manifest to prevent test pollution
    try:
        target_dir = Path(project_dir) / "target"
        if target_dir.exists():
            manifest_file = target_dir / "manifest.json"
            if manifest_file.exists():
                manifest_file.unlink()
                print("✓ Cached manifest.json removed")
    except Exception as e:
        print(f"Warning: Error removing manifest cache: {e}")

    print("=== Teardown complete ===\n")


# Function-scoped fixture for tests that need fresh caches
@pytest.fixture(scope="function")
def fresh_caches():
    """Patches the internal caches so each test starts with a fresh state."""
    from unittest import mock

    with (
        mock.patch("dbt_osmosis.core.introspection._COLUMN_LIST_CACHE", {}),
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield
