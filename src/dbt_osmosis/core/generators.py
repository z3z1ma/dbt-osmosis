"""Wrapper module for dbt-core-interface generators and documentation checker.

This module provides a unified interface to dbt-core-interface's source generation,
staging model generation, and documentation checking capabilities, adapted for
use in dbt-osmosis.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.config import DbtProjectContext
from dbt_osmosis.core.settings import YamlRefactorSettings
from dbt_osmosis.core.staging import (
    StagingGenerationResult,
    generate_staging_for_source,
    write_staging_files,
)

__all__ = [
    "check_documentation",
    "DocumentationCheckResult",
    "generate_sources_from_database",
]


@dataclass
class SourceGenerationResult:
    """Result of source generation.

    Attributes:
        source_name: Name of the source generated
        table_count: Number of tables discovered
        yaml_content: Generated YAML content
        yaml_path: Path where YAML should be written
        error: Any error that occurred during generation
    """

    source_name: str
    table_count: int
    yaml_content: str
    yaml_path: Path
    error: Exception | None = None


@dataclass
class DocumentationCheckResult:
    """Result of documentation completeness check.

    Attributes:
        total_models: Total number of models checked
        models_with_descriptions: Number of models with descriptions
        models_without_descriptions: Number of models without descriptions
        total_columns: Total number of columns checked
        documented_columns: Number of columns with descriptions
        undocumented_columns: Number of columns without descriptions
        gaps: List of documentation gaps found
    """

    total_models: int
    models_with_descriptions: int
    models_without_descriptions: int
    total_columns: int
    documented_columns: int
    undocumented_columns: int
    gaps: list[t.Any]  # DocumentationGap from dbt-core-interface


def generate_sources_from_database(
    context: DbtProjectContext,
    source_name: str = "raw",
    schema_name: str | None = None,
    exclude_schemas: list[str] | None = None,
    exclude_tables: list[str] | None = None,
    quote_identifiers: bool = False,
    output_path: Path | None = None,
) -> SourceGenerationResult:
    """Generate source definitions from database introspection.

    This function uses dbt-core-interface's SourceGenerator to discover
    tables in the database and generate dbt source YAML definitions.

    Args:
        context: The dbt project context
        source_name: Name for the source (default: "raw")
        schema_name: Specific schema to scan (None = all schemas in database)
        exclude_schemas: Schemas to exclude from scanning
        exclude_tables: Tables to exclude from generation
        quote_identifiers: Whether to quote identifiers in generated YAML
        output_path: Path where YAML file should be written (default: models/sources/{source_name}.yml)

    Returns:
        SourceGenerationResult with generated YAML and metadata

    Raises:
        Exception: If source generation fails
    """
    from dbt_core_interface.source_generator import (
        SourceGenerationOptions,
        SourceGenerationStrategy,
        SourceGenerator,
        to_yaml,
    )

    logger.info(":factory: Generating sources from database introspection...")

    try:
        # Create source generator
        source_gen = SourceGenerator(project=context._project)

        # Configure generation options
        options = SourceGenerationOptions(
            strategy=SourceGenerationStrategy.SPECIFIC_SCHEMA
            if schema_name
            else SourceGenerationStrategy.ALL_SCHEMAS,
            schema_name=schema_name,
            source_name=source_name,
            include_descriptions=True,
            infer_descriptions=True,
            exclude_schemas=exclude_schemas or [],
            exclude_tables=exclude_tables or [],
            quote_identifiers=quote_identifiers,
        )

        # Generate sources
        source_defs = source_gen.generate_sources(options=options)

        if not source_defs:
            logger.warning(":warning: No sources found with given configuration")
            return SourceGenerationResult(
                source_name=source_name,
                table_count=0,
                yaml_content="",
                yaml_path=output_path or Path.cwd() / "models" / "sources" / f"{source_name}.yml",
            )

        # Generate YAML content
        yaml_content = to_yaml(source_defs=source_defs, quote_identifiers=quote_identifiers)

        # Determine output path
        if output_path is None:
            project_root = Path(context.config.project_dir)
            output_path = project_root / "models" / "sources" / f"{source_name}.yml"
        else:
            output_path = Path(output_path)

        total_tables = sum(len(source_def.tables) for source_def in source_defs)

        logger.info(
            ":white_check_mark: Generated source '%s' with %d tables",
            source_name,
            total_tables,
        )

        return SourceGenerationResult(
            source_name=source_name,
            table_count=total_tables,
            yaml_content=yaml_content,
            yaml_path=output_path,
        )

    except Exception as e:
        logger.error(":boom: Error generating sources: %s", e)
        raise


def generate_staging_from_source(
    context: DbtProjectContext,
    source_name: str,
    table_name: str,
    use_ai: bool = False,
    staging_path: Path | None = None,
) -> StagingGenerationResult:
    """Generate a staging model from a source table.

    This function can use either dbt-core-interface's StagingGenerator
    (for deterministic generation) or dbt-osmosis's AI-based generation
    (for intelligent staging with business logic).

    Args:
        context: The dbt project context
        source_name: Name of the source (e.g., "raw")
        table_name: Name of the table in the source
        use_ai: If True, use AI-based generation; otherwise use interface generator
        staging_path: Directory where staging models should be written
                     (default: models/staging/)

    Returns:
        StagingGenerationResult with generated SQL and YAML

    Raises:
        Exception: If staging generation fails
    """
    from dbt_core_interface.staging_generator import (
        NamingConvention,
        StagingModelConfig,
        generate_staging_model_from_source,
    )

    logger.info(
        ":robot: Generating staging model for %s.%s (AI=%s)...", source_name, table_name, use_ai
    )

    try:
        # Get source definition from manifest
        source_def = _get_source_definition(context, source_name, table_name)
        if source_def is None:
            raise ValueError(f"Source {source_name}.{table_name} not found in manifest")

        if use_ai:
            # Generate with AI
            staging_spec = generate_staging_for_source(
                project=context,
                settings=YamlRefactorSettings(
                    output_to_lower=False,
                    output_to_upper=False,
                ),
                source_name=source_name,
                table_name=table_name,
                source_type="source",
                temperature=0.3,
            )

            if staging_spec.error or not staging_spec.spec:
                raise staging_spec.error or Exception("Failed to generate staging spec")

            spec = staging_spec.spec
            staging_path = staging_path or Path(context.config.project_dir) / "models" / "staging"
            staging_path.mkdir(parents=True, exist_ok=True)

            sql_path = staging_path / f"{spec.staging_name}.sql"
            yaml_path = staging_path / f"{spec.staging_name}.yml"

            columns_yaml = "\n".join(
                f"  - name: {col.new_name}\n    description: {col.description}"
                for col in spec.columns
            )

            yaml_content = dedent(f"""\
            version: 2

            models:
              - name: {spec.staging_name}
                description: {spec.description}
                columns:
            {columns_yaml}
            """)

            result = StagingGenerationResult(
                source_name=f"{source_name}.{table_name}",
                staging_name=spec.staging_name,
                spec=spec,
                sql_content=spec.to_sql(),
                yaml_content=yaml_content,
                sql_path=sql_path,
                yaml_path=yaml_path,
            )

            write_staging_files(result, dry_run=False)

            logger.info(
                ":white_check_mark: Generated staging model %s (AI-based)",
                spec.staging_name,
            )

            return result
        else:
            # Use dbt-core-interface generator
            config = StagingModelConfig(
                source_name=source_name,
                table_name=table_name,
                materialization="view",
                naming_convention=NamingConvention.SNAKE_CASE,
                generate_tests=False,  # Don't auto-generate tests
                generate_documentation=True,
            )

            result_dict = generate_staging_model_from_source(
                source=source_def,
                manifest=context.manifest,
                config=config,
            )

            # Determine output paths
            if staging_path is None:
                project_root = Path(context.config.project_dir)
                staging_path = project_root / "models" / "staging"

            staging_path.mkdir(parents=True, exist_ok=True)

            staging_name = result_dict.get("staging_name", f"stg_{table_name}")
            sql_path = staging_path / f"{staging_name}.sql"
            yaml_path = staging_path / f"{staging_name}.yml"

            # Write files
            sql_path.write_text(result_dict.get("sql", ""), encoding="utf-8")

            if "yaml" in result_dict:
                yaml_path.write_text(result_dict["yaml"], encoding="utf-8")

            logger.info(
                ":white_check_mark: Generated staging model %s (interface-based)",
                staging_name,
            )

            return StagingGenerationResult(
                source_name=f"{source_name}.{table_name}",
                staging_name=staging_name,
                spec=None,
                sql_content=result_dict.get("sql", ""),
                yaml_content=result_dict.get("yaml", ""),
                sql_path=sql_path,
                yaml_path=yaml_path,
            )

    except Exception as e:
        logger.error(":boom: Error generating staging model: %s", e)
        raise


def check_documentation(
    context: DbtProjectContext,
    model_filter: str | None = None,
    min_model_length: int = 10,
    min_column_length: int = 5,
) -> DocumentationCheckResult:
    """Check documentation completeness across the dbt project.

    This function uses dbt-core-interface's DocumentationChecker to analyze
    model and column documentation, identifying gaps and completeness.

    Args:
        context: The dbt project context
        model_filter: Optional model name filter (None = check all models)
        min_model_length: Minimum length for model descriptions
        min_column_length: Minimum length for column descriptions

    Returns:
        DocumentationCheckResult with coverage statistics and gaps

    Raises:
        Exception: If documentation check fails
    """
    from dbt_core_interface.doc_checker import DocumentationChecker

    logger.info(":mag: Checking documentation completeness...")

    try:
        # Create documentation checker
        doc_checker = DocumentationChecker(
            min_model_description_length=min_model_length,
            min_column_description_length=min_column_length,
        )

        # Run check
        report = doc_checker.check_project(
            manifest=context.manifest,
            project_name=context._project.project_name,
            model_name_filter=model_filter,
        )

        # Convert to simplified result
        result = DocumentationCheckResult(
            total_models=report.total_models,
            models_with_descriptions=report.models_with_descriptions,
            models_without_descriptions=report.models_without_descriptions,
            total_columns=report.total_columns,
            documented_columns=report.documented_columns,
            undocumented_columns=report.undocumented_columns,
            gaps=report.all_gaps,
        )

        coverage_percent = (
            (result.documented_columns / result.total_columns * 100)
            if result.total_columns > 0
            else 0.0
        )

        logger.info(
            ":white_check_mark: Documentation check complete: %.1f%% coverage (%d/%d columns)",
            coverage_percent,
            result.documented_columns,
            result.total_columns,
        )

        return result

    except Exception as e:
        logger.error(":boom: Error checking documentation: %s", e)
        raise


def _get_source_definition(
    context: DbtProjectContext,
    source_name: str,
    table_name: str,
) -> t.Any | None:
    """Get a source definition from the manifest.

    Args:
        context: The dbt project context
        source_name: Name of the source
        table_name: Name of the table

    Returns:
        SourceDefinition if found, None otherwise
    """
    manifest = context.manifest

    for source in manifest.sources.values():
        if source.source_name == source_name and source.name == table_name:
            return source

    return None
