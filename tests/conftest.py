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
from collections.abc import Iterator
from pathlib import Path

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
            f"dbt seed failed: {seed_result.exception if hasattr(seed_result, 'exception') else 'Unknown error'}",
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
            f"dbt run failed: {run_result.exception if hasattr(run_result, 'exception') else 'Unknown error'}",
        )

    # Generate catalog to use for introspection instead of live database queries
    # This avoids connection issues when DbtProjectContext creates a new connection pool
    start = time.time()
    docs_result = dbtRunner().invoke([
        "docs",
        "generate",
        "--project-dir",
        project_dir,
        "--profiles-dir",
        profiles_dir,
        "--target",
        target,
    ])
    elapsed = time.time() - start
    if docs_result.success:
        print(f"✓ dbt docs generate completed successfully ({elapsed:.2f}s)")
    else:
        print(
            f"dbt docs generate failed: {docs_result.exception if hasattr(docs_result, 'exception') else 'Unknown error'}",
        )


def _create_temp_project_copy(
    source_dir: Path,
    temp_dir: Path,
    exclude_target: bool = False,
) -> Path:
    """Create a copy of the source project in a temporary directory.

    Args:
        source_dir: Source directory to copy
        temp_dir: Temporary directory to copy into
        exclude_target: If True, exclude the target/ directory to avoid
            copying cached manifest.json with wrong paths

    Returns:
        Path: Path to the copied project directory

    """
    project_dir = temp_dir / source_dir.name

    # Function to exclude target directory and other build artifacts
    def _ignore_filter(src: str, names: list[str]) -> list[str]:
        # Get the relative path from the source directory
        src_path = Path(src).relative_to(source_dir)
        # Exclude target directory (contains cached manifest)
        if exclude_target and "target" in names and src_path == Path():
            return ["target"]
        return []

    shutil.copytree(source_dir, project_dir, ignore=_ignore_filter)
    return project_dir


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

        # Change to the project directory before running dbt commands
        # This is necessary because DuckDB uses relative paths based on CWD
        old_cwd = os.getcwd()
        os.chdir(str(template_project_dir))
        try:
            _run_dbt_commands(str(template_project_dir), str(template_project_dir))
        finally:
            os.chdir(old_cwd)

        # Verify the database file was created
        db_file = template_project_dir / "test.db"
        if not db_file.exists():
            raise RuntimeError(f"Database file not created at {db_file}")
        print(f"✓ Template database created: {db_file} ({db_file.stat().st_size} bytes)")

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


@pytest.fixture(scope="session")
def built_postgres_template() -> Iterator[Path | None]:
    """Builds a PostgreSQL-backed dbt project template once per test session.

    This session-scoped fixture:
    1. Checks if POSTGRES_URL is set (skips if not)
    2. Creates a session-scoped temporary directory
    3. Copies the demo_duckdb project to it
    4. Creates a profiles_postgres.yml pointing to POSTGRES_URL
    5. Runs dbt seed, dbt run, and dbt docs generate ONCE against PostgreSQL
    6. Yields the path to the built template project
    7. Cleans up the template directory at session end

    This fixture is skipped if POSTGRES_URL is not set in the environment.

    Yields:
        Path | None: Path to the built template project directory, or None if skipped

    """
    postgres_url = os.environ.get("POSTGRES_URL")
    if not postgres_url:
        print("\n=== POSTGRES_URL not set, skipping PostgreSQL template build ===\n")
        yield None
        return

    # Create a session-scoped temp directory for the template
    template_temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_postgres_template_"))
    source_dir = Path("demo_duckdb")
    # Exclude target directory to avoid copying cached manifest.json with wrong paths
    template_project_dir = _create_temp_project_copy(
        source_dir,
        template_temp_dir,
        exclude_target=True,
    )

    try:
        # Create profiles_postgres.yml pointing to the PostgreSQL database
        postgres_profile_content = f"""
jaffle_shop:
  target: postgres
  outputs:
    postgres:
      type: postgres
      host: {os.environ.get("POSTGRES_HOST", "localhost")}
      user: {os.environ.get("POSTGRES_USER", "postgres")}
      password: {os.environ.get("POSTGRES_PASSWORD", "")}
      port: {int(os.environ.get("POSTGRES_PORT", 5432))}
      dbname: {os.environ.get("POSTGRES_DBNAME", "postgres")}
      schema: {os.environ.get("POSTGRES_SCHEMA", "public")}
      threads: 1
"""
        profile_path = template_project_dir / "profiles_postgres.yml"
        profile_path.write_text(postgres_profile_content)
        print(f"✓ Created {profile_path}")

        # Build the database and generate artifacts ONCE per session
        print("\n" + "=" * 60)
        print("SESSION SETUP: Building PostgreSQL dbt project template")
        print(f"Using PostgreSQL: {postgres_url}")
        print("=" * 60)

        # Change to the project directory before running dbt commands
        # This is necessary because dbt uses relative paths based on CWD
        old_cwd = os.getcwd()
        os.chdir(str(template_project_dir))
        try:
            _run_dbt_commands(
                str(template_project_dir),
                str(template_project_dir),
                target="postgres",
            )
        finally:
            os.chdir(old_cwd)

        print("=" * 60)
        print("SESSION SETUP: PostgreSQL template build complete")
        print("=" * 60 + "\n")

        yield template_project_dir

    finally:
        # Clean up the template directory at session end
        print("\n" + "=" * 60)
        print(f"SESSION TEARDOWN: Removing PostgreSQL template directory {template_temp_dir}")
        print("=" * 60)
        try:
            shutil.rmtree(template_temp_dir)
            print("✓ Removed PostgreSQL template directory")
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
    project_dir = _create_temp_project_copy(built_duckdb_template, temp_dir)

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


@pytest.fixture(scope="function")
def postgres_yaml_context(built_postgres_template: Path | None) -> Iterator[YamlRefactorContext]:
    """Creates a YamlRefactorContext with PostgreSQL database from the session template.

    This function-scoped fixture:
    1. Checks if POSTGRES_URL is set (skips if not)
    2. Copies the session-built template project to a unique temp directory
    3. Creates and yields a YamlRefactorContext
    4. Properly closes connections and cleans up on teardown

    This fixture is skipped if POSTGRES_URL is not set in the environment.
    PostgreSQL database is reused across tests (no per-test database reset).

    Args:
        built_postgres_template: Path to the session-built PostgreSQL template project, or None

    Yields:
        YamlRefactorContext: Configured context with PostgreSQL database

    """
    if built_postgres_template is None:
        pytest.skip("POSTGRES_URL environment variable not set. Skipping PostgreSQL tests.")

    # Create a unique temp directory for this test
    temp_dir = Path(tempfile.mkdtemp(prefix="dbt_osmosis_postgres_test_"))

    # Copy the ENTIRE template project to preserve isolation
    project_dir = _create_temp_project_copy(built_postgres_template, temp_dir)

    try:
        # Use PostgreSQL target
        cfg = DbtConfiguration(
            project_dir=str(project_dir),
            profiles_dir=str(project_dir),
            target="postgres",  # Uses profiles_postgres.yml created in template
        )
        cfg.vars = {"dbt-osmosis": {}}

        # Create the project context (will connect to PostgreSQL)
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
                            adapter.connections,
                            "close",
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
