"""Shared pytest configuration and fixtures for dbt-osmosis tests.

This module provides DuckDB-backed fixtures for exercising dbt-osmosis tests.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from tests.support import create_temp_project_copy, run_dbt_command


def _run_dbt_commands(project_dir: str, profiles_dir: str, target: str = "test") -> None:
    """Run dbt seed and dbt run to populate the database.

    Uses dbt's CLI runner to execute seed and run commands. The in-memory
    database is shared across all connections within the same process.
    """
    # Run dbt seed to load CSV files into the database.
    run_dbt_command([
        "seed",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        target,
    ])

    # Run dbt run to create models.
    run_dbt_command([
        "run",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        target,
    ])

    # Generate catalog to use for introspection instead of live database queries
    # This avoids connection issues when DbtProjectContext creates a new connection pool
    run_dbt_command([
        "docs",
        "generate",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        target,
    ])


def _create_temp_project_copy(
    source_dir: Path,
    temp_dir: Path,
    exclude_target: bool | None = None,
    *,
    include_generated_artifacts: bool = False,
) -> Path:
    """Create a copy of the source project in a temporary directory.

    Args:
        source_dir: Source directory to copy
        temp_dir: Temporary directory to copy into
        exclude_target: Legacy compatibility switch. If set, generated
            artifacts are included only when this is False.
        include_generated_artifacts: If True, intentionally copy generated
            outputs such as target/ and test.db.

    Returns:
        Path: Path to the copied project directory

    """
    if exclude_target is not None:
        include_generated_artifacts = not exclude_target

    return create_temp_project_copy(
        source_dir,
        temp_dir,
        include_generated_artifacts=include_generated_artifacts,
    )


@pytest.fixture(scope="session")
def built_duckdb_template() -> Iterator[Path]:
    """Builds a dbt project template once per test session.

    This session-scoped fixture:
    1. Creates a session-scoped temporary directory
    2. Copies the demo_duckdb project to it
    3. Runs dbt seed, dbt run, and dbt docs generate ONCE
    4. Yields the path to the built template project
    5. Cleans up the template directory at session end

    The resulting template includes:
    - Populated test.db file (DuckDB database)
    - Generated catalog.json (for introspection)
    - Generated manifest.json
    - All other dbt artifacts in target/

    Yields:
        Path: Path to the built template project directory

    """
    # Create a session-scoped temp directory for the template
    template_temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_template_"))
    source_dir = Path("demo_duckdb")
    # Exclude target directory to avoid copying cached manifest.json with wrong paths
    template_project_dir = _create_temp_project_copy(
        source_dir,
        template_temp_dir,
        exclude_target=True,
    )

    try:
        # Build the database and generate artifacts ONCE per session
        print("\n" + "=" * 60)
        print("SESSION SETUP: Building dbt project template")
        print("=" * 60)

        _run_dbt_commands(str(template_project_dir), str(template_project_dir))

        # Verify the database file was created
        db_file = template_project_dir / "test.db"
        if not db_file.exists():
            raise RuntimeError(f"Database file not created at {db_file}")
        print(f"✓ Template database created: {db_file} ({db_file.stat().st_size} bytes)")

        manifest_file = template_project_dir / "target" / "manifest.json"
        if not manifest_file.exists():
            raise RuntimeError(f"Manifest file not created at {manifest_file}")
        print(f"✓ Template manifest created: {manifest_file}")

        # Verify catalog.json was created
        catalog_file = template_project_dir / "target" / "catalog.json"
        if not catalog_file.exists():
            raise RuntimeError(f"Catalog file not created at {catalog_file}")
        print(f"✓ Template catalog created: {catalog_file}")

        print("=" * 60)
        print("SESSION SETUP: Template build complete")
        print("=" * 60 + "\n")

        yield template_project_dir

    finally:
        # Clean up the template directory at session end
        print("\n" + "=" * 60)
        print(f"SESSION TEARDOWN: Removing template directory {template_temp_dir}")
        print("=" * 60)
        try:
            shutil.rmtree(template_temp_dir)
            print("✓ Removed template directory")
        except Exception as e:
            print(f"Warning: Error removing template directory: {e}")
        print("=" * 60 + "\n")


@pytest.fixture(scope="function")
def yaml_context(built_duckdb_template: Path) -> Iterator[YamlRefactorContext]:
    """Creates a YamlRefactorContext with a DuckDB database from the session template.

    This function-scoped fixture:
    1. Copies the session-built template project to a unique temp directory (avoids DbtProject cache)
    2. Uses the pre-built test.db file copied from the template
    3. Creates and yields a YamlRefactorContext
    4. Properly closes connections on teardown
    5. Cleans up the temp directory

    Using a unique temp directory for each test ensures complete isolation from
    the DbtProject WeakValueDictionary cache while reusing the pre-built database.

    Args:
        built_duckdb_template: Path to the session-built template project

    Yields:
        YamlRefactorContext: Configured context with populated database

    """
    # Create a unique temp directory for this test
    temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_test_"))

    # Copy the ENTIRE template project to preserve isolation
    project_dir = _create_temp_project_copy(
        built_duckdb_template,
        temp_dir,
        include_generated_artifacts=True,
    )

    # Verify the database file was copied
    db_file = project_dir / "test.db"
    if not db_file.exists():
        raise RuntimeError(f"Database file not copied to {db_file}")

    try:
        # Use the test profile with file-based database
        cfg = DbtConfiguration(
            project_dir=str(project_dir),
            profiles_dir=str(project_dir),
            target="test",  # Uses profiles.yml with test.db
        )
        cfg.vars = {"dbt-osmosis": {}}

        # Create the project context (will use the copied test.db)
        start = time.time()
        project_context = create_dbt_project_context(cfg)
        print(f"✓ create_dbt_project_context took {time.time() - start:.2f}s")

        # Set catalog_path to use the catalog.json copied from template
        catalog_path = str(project_dir / "target" / "catalog.json")

        context = YamlRefactorContext(
            project_context,
            settings=YamlRefactorSettings(
                dry_run=True,
                use_unrendered_descriptions=True,
                catalog_path=catalog_path,
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
                            adapter.connections,
                            "close",
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


# Function-scoped fixture for tests that need fresh caches
@pytest.fixture(scope="function")
def fresh_caches():
    """Clear production cache instances so each test starts with a fresh state."""
    from dbt_osmosis.core.introspection import _COLUMN_LIST_CACHE, _COLUMN_LIST_CACHE_LOCK
    from dbt_osmosis.core.schema.reader import (
        _YAML_BUFFER_CACHE,
        _YAML_BUFFER_CACHE_LOCK,
        _YAML_ORIGINAL_CACHE,
    )

    def clear_caches() -> None:
        with _COLUMN_LIST_CACHE_LOCK:
            _COLUMN_LIST_CACHE.clear()
        with _YAML_BUFFER_CACHE_LOCK:
            _YAML_BUFFER_CACHE.clear()
            _YAML_ORIGINAL_CACHE.clear()

    clear_caches()
    try:
        yield
    finally:
        clear_caches()
