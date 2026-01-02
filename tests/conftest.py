"""Shared pytest configuration and fixtures for dbt-osmosis tests.

This module provides fixtures for both DuckDB and PostgreSQL databases,
allowing comprehensive multi-database testing of dbt-osmosis functionality.
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Iterator

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings


def _run_dbt_commands(project_dir: str, profiles_dir: str, target: str = "test") -> None:
    """Run dbt seed and dbt run to populate the database.

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
        target,
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
        target,
    ])
    elapsed = time.time() - start
    if run_result.success:
        print(f"✓ dbt run completed successfully ({elapsed:.2f}s)")
    else:
        print(
            f"dbt run failed: {run_result.exception if hasattr(run_result, 'exception') else 'Unknown error'}"
        )


def _create_temp_project_copy(source_dir: Path, temp_dir: Path) -> Path:
    """Create a copy of the source project in a temporary directory."""
    project_dir = temp_dir / source_dir.name
    shutil.copytree(source_dir, project_dir)
    return project_dir


@pytest.fixture(scope="function")
def yaml_context() -> Iterator[YamlRefactorContext]:
    """Creates a YamlRefactorContext with an in-memory DuckDB database.

    This function-scoped fixture:
    1. Copies the demo_duckdb directory to a unique temp directory (avoids DbtProject cache)
    2. Runs dbt seed and dbt run to populate the in-memory database
    3. Creates and yields a YamlRefactorContext
    4. Properly closes connections on teardown
    5. Cleans up the temp directory and cached manifest.json

    Using a unique temp directory for each test ensures complete isolation from
    the DbtProject WeakValueDictionary cache.

    Yields:
        YamlRefactorContext: Configured context with populated database
    """
    # Create a unique temp directory for this test
    temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_test_"))
    source_dir = Path("demo_duckdb")
    project_dir = _create_temp_project_copy(source_dir, temp_dir)

    try:
        # Use the test profile with in-memory database
        cfg = DbtConfiguration(
            project_dir=str(project_dir),
            profiles_dir=str(project_dir),
            target="test",  # Uses profiles_test.yml with :memory:
        )
        cfg.vars = {"dbt-osmosis": {}}

        # Run dbt seed and run BEFORE creating the project context
        # This ensures the in-memory database is populated
        print("\n=== Setting up in-memory DuckDB database ===")
        _run_dbt_commands(str(project_dir), str(project_dir))
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

    finally:
        # Teardown: Clean up temp directory
        print(f"\n=== Cleaning up temp directory {temp_dir} ===")
        try:
            # Close connections first
            if "project_context" in locals():
                if hasattr(project_context, "_project") and project_context._project is not None:
                    if hasattr(project_context._project, "adapter"):
                        adapter = project_context._project.adapter
                        if hasattr(adapter, "connections") and hasattr(
                            adapter.connections, "close"
                        ):
                            try:
                                adapter.connections.close()
                                print("✓ Adapter connections closed")
                            except Exception as e:
                                print(f"Warning: Error closing connections: {e}")
        except Exception as e:
            print(f"Warning: Error during connection cleanup: {e}")

        # Delete the DbtProject reference and trigger GC
        try:
            if "project_context" in locals() and hasattr(project_context, "_project"):
                del project_context._project
                gc.collect()
                print("✓ DbtProject reference deleted and garbage collected")
        except Exception as e:
            print(f"Warning: Error deleting DbtProject reference: {e}")

        # Remove the temp directory
        try:
            shutil.rmtree(temp_dir)
            print(f"✓ Removed temp directory {temp_dir}")
        except Exception as e:
            print(f"Warning: Error removing temp directory: {e}")

        print("=== Teardown complete ===\n")


@pytest.fixture(scope="function")
def postgres_yaml_context() -> Iterator[YamlRefactorContext]:
    """Creates a YamlRefactorContext with PostgreSQL database.

    This function-scoped fixture:
    1. Creates a temporary copy of the demo_duckdb project
    2. Uses PostgreSQL database if POSTGRES_URL is set in environment
    3. Sets up dbt configuration for PostgreSQL
    4. Runs dbt seed and dbt run to populate the PostgreSQL database
    5. Creates and yields a YamlRefactorContext
    6. Properly closes connections and cleans up on teardown

    This fixture is skipped if POSTGRES_URL is not set in the environment.

    Yields:
        YamlRefactorContext: Configured context with PostgreSQL database
    """
    postgres_url = os.environ.get("POSTGRES_URL")
    if not postgres_url:
        pytest.skip("POSTGRES_URL environment variable not set. Skipping PostgreSQL tests.")

    # Create a unique temp directory for this test
    temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_postgres_test_"))
    source_dir = Path("demo_duckdb")
    project_dir = _create_temp_project_copy(source_dir, temp_dir)

    try:
        # Use PostgreSQL target
        cfg = DbtConfiguration(
            project_dir=str(project_dir),
            profiles_dir=str(project_dir),
            target="postgres",  # Will use profiles_postgres.yml
        )
        cfg.vars = {"dbt-osmosis": {}}

        # Set up PostgreSQL database
        print("\n=== Setting up PostgreSQL database ===")
        print(f"Using PostgreSQL URL: {postgres_url}")

        # Run dbt seed and run with PostgreSQL
        _run_dbt_commands(str(project_dir), str(project_dir), target="postgres")
        print("=== PostgreSQL database setup complete ===\n")

        # Create the project context
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

    finally:
        # Teardown: Clean up temp directory
        print(f"\n=== Cleaning up PostgreSQL test environment {temp_dir} ===")
        try:
            # Close connections first
            if "project_context" in locals():
                if hasattr(project_context, "_project") and project_context._project is not None:
                    if hasattr(project_context._project, "adapter"):
                        adapter = project_context._project.adapter
                        if hasattr(adapter, "connections") and hasattr(
                            adapter.connections, "close"
                        ):
                            try:
                                adapter.connections.close()
                                print("✓ PostgreSQL adapter connections closed")
                            except Exception as e:
                                print(f"Warning: Error closing connections: {e}")
        except Exception as e:
            print(f"Warning: Error during connection cleanup: {e}")

        # Delete the DbtProject reference and trigger GC
        try:
            if "project_context" in locals() and hasattr(project_context, "_project"):
                del project_context._project
                gc.collect()
                print("✓ DbtProject reference deleted and garbage collected")
        except Exception as e:
            print(f"Warning: Error deleting DbtProject reference: {e}")

        # Remove the temp directory
        try:
            shutil.rmtree(temp_dir)
            print(f"✓ Removed temp directory {temp_dir}")
        except Exception as e:
            print(f"Warning: Error removing temp directory: {e}")

        print("=== PostgreSQL test teardown complete ===\n")


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
