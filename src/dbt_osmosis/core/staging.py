"""AI-powered staging model generation for dbt-osmosis.

This module provides orchestration for automatically generating staging models
from source tables using AI. It integrates with the dbt project context to
discover sources and generate appropriate staging SQL and YAML documentation.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path

from dbt.contracts.graph.nodes import SourceDefinition

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.config import DbtProjectContext
from dbt_osmosis.core.llm import (
    StagingModelSpec,
    generate_staging_model_spec,
)
from dbt_osmosis.core.settings import YamlRefactorSettings

__all__ = [
    "StagingGenerationResult",
    "generate_staging_for_source",
    "generate_staging_for_all_sources",
    "write_staging_files",
]


@dataclass
class StagingGenerationResult:
    """Result of staging model generation.

    Attributes:
        source_name: Name of the source table
        spec: The generated staging model specification
        sql_path: Path where the SQL file will be written
        yaml_path: Path where the YAML file will be written
        error: Any error that occurred during generation
    """

    source_name: str
    spec: StagingModelSpec | None = None
    sql_path: Path | None = None
    yaml_path: Path | None = None
    error: Exception | None = None


def _get_source_table_columns(
    project: DbtProjectContext,
    source_name: str,
    table_name: str,
) -> list[dict[str, t.Any]]:
    """Get column definitions for a source table from the database.

    Args:
        project: The dbt project context
        source_name: Name of the source (e.g., 'raw_stripe')
        table_name: Name of the table within the source

    Returns:
        List of column definitions with name, data_type, and optional description
    """
    from dbt_osmosis.core.introspection import get_columns

    # Construct the full table name for introspection
    # Sources may have different quoting patterns
    adapter = project._project.adapter

    # Get columns from the database via introspection
    try:
        # Try to get the relation for the source
        relation_name = f"{source_name}_{table_name}"
        columns = get_columns(
            adapter,
            project._project.runtime_config,
            relation_name,
        )

        # Convert to dict format expected by LLM
        column_defs = []
        for col_name, col_meta in columns.items():
            column_defs.append({
                "name": col_name,
                "data_type": col_meta.type or "unknown",
                "description": col_meta.comment or "",
            })

        return column_defs
    except Exception as e:
        logger.warning(
            ":warning: Could not introspect source %s.%s: %s", source_name, table_name, e
        )
        return []


def _get_source_table_from_manifest(
    project: DbtProjectContext,
    source_name: str,
    table_name: str,
) -> SourceDefinition | None:
    """Get a source table definition from the dbt manifest.

    Args:
        project: The dbt project context
        source_name: Name of the source (e.g., 'raw_stripe')
        table_name: Name of the table within the source

    Returns:
        SourceDefinition if found, None otherwise
    """
    manifest = project.manifest

    # Search for the source in the manifest
    for source in manifest.sources.values():
        if source.source_name == source_name and source.name == table_name:
            return source

    return None


def _infer_staging_model_path(
    project: DbtProjectContext,
    settings: YamlRefactorSettings,
    staging_name: str,
) -> tuple[Path, Path]:
    """Infer the file paths for a staging model.

    Args:
        project: The dbt project context
        settings: YAML refactor settings
        staging_name: Name of the staging model (e.g., 'stg_customers')

    Returns:
        Tuple of (sql_path, yaml_path) for the staging model files
    """
    project_dir = Path(project.config.project_dir)

    # Default: models/staging/ directory
    staging_dir = project_dir / "models" / "staging"

    # Check if there's a custom staging pattern configured
    # For now, use a simple convention
    sql_path = staging_dir / f"{staging_name}.sql"
    yaml_path = staging_dir / f"{staging_name}.yml"

    return sql_path, yaml_path


def generate_staging_for_source(
    project: DbtProjectContext,
    settings: YamlRefactorSettings,
    source_name: str,
    table_name: str,
    source_type: str = "source",
    temperature: float = 0.3,
) -> StagingGenerationResult:
    """Generate a staging model for a single source table.

    Args:
        project: The dbt project context
        settings: YAML refactor settings
        source_name: Name of the source (e.g., 'raw_stripe')
        table_name: Name of the table within the source
        source_type: Type of source ('source', 'seed', or 'model')
        temperature: LLM temperature for generation

    Returns:
        StagingGenerationResult with the generated specification or error
    """
    full_source_name = f"{source_name}.{table_name}"

    logger.info(":robot: Generating staging model for source => %s", full_source_name)

    try:
        # Get column definitions from manifest or database
        source_def = _get_source_table_from_manifest(project, source_name, table_name)

        table_description = ""
        columns = []

        if source_def:
            # Use column definitions from manifest
            table_description = source_def.description or ""
            for col_name, col_def in source_def.columns.items():
                columns.append({
                    "name": col_name,
                    "data_type": col_def.data_type or "unknown",
                    "description": col_def.description or "",
                })
            logger.info(":page_facing_up: Found %d columns in manifest", len(columns))
        else:
            # Fall back to database introspection
            logger.info(":mag: Source not in manifest, introspecting database...")
            columns = _get_source_table_columns(project, source_name, table_name)

        if not columns:
            raise ValueError(f"Could not find columns for source {full_source_name}")

        # Generate staging specification
        spec = generate_staging_model_spec(
            source_name=full_source_name,
            columns=columns,
            table_description=table_description,
            source_type=source_type,
            temperature=temperature,
        )

        # Infer file paths
        sql_path, yaml_path = _infer_staging_model_path(project, settings, spec.staging_name)

        logger.info(":white_check_mark: Generated staging spec => %s", spec.staging_name)

        return StagingGenerationResult(
            source_name=full_source_name,
            spec=spec,
            sql_path=sql_path,
            yaml_path=yaml_path,
        )

    except Exception as e:
        logger.error(":boom: Error generating staging for %s: %s", full_source_name, e)
        return StagingGenerationResult(
            source_name=full_source_name,
            error=e,
        )


def generate_staging_for_all_sources(
    project: DbtProjectContext,
    settings: YamlRefactorSettings,
    source_pattern: str | None = None,
    exclude_patterns: list[str] | None = None,
    temperature: float = 0.3,
) -> list[StagingGenerationResult]:
    """Generate staging models for all or filtered source tables.

    Args:
        project: The dbt project context
        settings: YAML refactor settings
        source_pattern: Optional pattern to filter source names (e.g., 'raw_*')
        exclude_patterns: Optional list of patterns to exclude
        temperature: LLM temperature for generation

    Returns:
        List of StagingGenerationResult for each source table
    """
    results = []
    manifest = project.manifest

    # Group sources by source_name and table_name
    sources_to_process: dict[tuple[str, str], SourceDefinition] = {}
    for source in manifest.sources.values():
        key = (source.source_name, source.name)

        # Apply include/exclude filters
        if source_pattern:
            if not _matches_pattern(source.source_name, source_pattern):
                continue

        if exclude_patterns:
            if any(_matches_pattern(source.source_name, pat) for pat in exclude_patterns):
                continue

        sources_to_process[key] = source

    logger.info(":mag: Found %d source tables to process", len(sources_to_process))

    for (source_name, table_name), source_def in sources_to_process.items():
        result = generate_staging_for_source(
            project=project,
            settings=settings,
            source_name=source_name,
            table_name=table_name,
            source_type="source",
            temperature=temperature,
        )
        results.append(result)

    return results


def _matches_pattern(text: str, pattern: str) -> bool:
    """Check if text matches a simple glob pattern.

    Args:
        text: The text to check
        pattern: The glob pattern (supports * wildcards)

    Returns:
        True if the text matches the pattern
    """
    import fnmatch

    return fnmatch.fnmatch(text, pattern)


def write_staging_files(
    result: StagingGenerationResult,
    dry_run: bool = False,
) -> None:
    """Write the generated staging model files to disk.

    Args:
        result: The staging generation result
        dry_run: If True, skip writing files
    """
    if result.error or result.spec is None:
        logger.warning(
            ":warning: Skipping files for %s (error: %s)", result.source_name, result.error
        )
        return

    spec = result.spec

    if result.sql_path:
        logger.info(":writing_hand: Writing SQL => %s", result.sql_path)
        if not dry_run:
            result.sql_path.parent.mkdir(parents=True, exist_ok=True)
            result.sql_path.write_text(spec.to_sql(), encoding="utf-8")

    if result.yaml_path:
        # Generate YAML documentation
        yaml_content = _generate_staging_yaml(spec)

        logger.info(":writing_hand: Writing YAML => %s", result.yaml_path)
        if not dry_run:
            result.yaml_path.parent.mkdir(parents=True, exist_ok=True)
            result.yaml_path.write_text(yaml_content, encoding="utf-8")


def _generate_staging_yaml(spec: StagingModelSpec) -> str:
    """Generate YAML documentation for a staging model.

    Args:
        spec: The staging model specification

    Returns:
        YAML content as a string
    """
    lines = [
        "---",
        "version: 2",
        "models:",
        f"  - name: {spec.staging_name}",
        f'    description: "{spec.description}"',
    ]

    if spec.columns:
        lines.append("    columns:")

        for col in spec.columns:
            col_entry = f"      - name: {col.new_name}"
            if col.description:
                col_entry += f'\n        description: "{col.description}"'

            # Add data_type if available (would need to be inferred from columns)
            # For now, skip data_type in YAML as it's optional

            lines.append(col_entry)

    return "\n".join(lines) + "\n"
