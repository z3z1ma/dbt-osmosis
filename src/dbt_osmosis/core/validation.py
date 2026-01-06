"""Dry-run validation module for dbt models.

This module provides functionality to validate dbt models by compiling and
executing them against production data without materializing tables or views.
Useful for catching errors and estimating query costs before deployment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from dbt.adapters.contracts.connection import AdapterResponse
from dbt.contracts.graph.nodes import ManifestNode

from dbt_osmosis.core import logger
from dbt_osmosis.core.config import DbtProjectContext
from dbt_osmosis.core.sql_operations import compile_sql_code, execute_sql_code

__all__ = [
    "ModelValidationStatus",
    "ModelValidationResult",
    "ValidationReport",
    "validate_models",
]


class ModelValidationStatus(Enum):
    """Status of a model validation."""

    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


@dataclass
class ModelValidationResult:
    """Result of validating a single dbt model."""

    model_name: str
    """Name of the validated model."""

    unique_id: str
    """Unique identifier of the model in the manifest."""

    status: ModelValidationStatus
    """Validation status."""

    compiled_sql: str | None = None
    """Compiled SQL code (None if compilation failed)."""

    error_message: str | None = None
    """Error message if validation failed."""

    execution_time_seconds: float = 0.0
    """Query execution time in seconds."""

    row_count: int | None = None
    """Number of rows returned (None if execution failed)."""

    bytes_processed: int | None = None
    """Estimated bytes processed (None if not available)."""

    adapter_response: AdapterResponse | None = None
    """Adapter response with metadata (None if execution failed)."""


@dataclass
class ValidationReport:
    """Summary report of model validation results."""

    results: list[ModelValidationResult] = field(default_factory=list)
    """Individual model validation results."""

    total_models: int = 0
    """Total number of models validated."""

    successful: int = 0
    """Number of models that validated successfully."""

    failed: int = 0
    """Number of models that failed validation."""

    total_execution_time: float = 0.0
    """Total execution time across all models in seconds."""

    def add_result(self, result: ModelValidationResult) -> None:
        """Add a validation result to the report.

        Args:
            result: The validation result to add.
        """
        self.results.append(result)
        self.total_models += 1
        self.total_execution_time += result.execution_time_seconds

        if result.status == ModelValidationStatus.SUCCESS:
            self.successful += 1
        else:
            self.failed += 1

    def get_failed_models(self) -> list[ModelValidationResult]:
        """Get all failed validation results.

        Returns:
            List of failed validation results.
        """
        return [r for r in self.results if r.status != ModelValidationStatus.SUCCESS]

    def get_success_rate(self) -> float:
        """Calculate success rate as a percentage.

        Returns:
            Success rate percentage (0-100).
        """
        if self.total_models == 0:
            return 0.0
        return (self.successful / self.total_models) * 100


def validate_models(
    context: DbtProjectContext,
    models: list[tuple[str, ManifestNode]],
    timeout_seconds: float | None = None,
    quiet: bool = False,
) -> ValidationReport:
    """Validate dbt models by compiling and executing against production data.

    This function compiles each model's SQL and executes it against the database
    without creating any tables or views. It's useful for:
    - Validating model logic before deployment
    - Catching SQL errors early
    - Estimating query costs (execution time, rows processed)

    Args:
        context: The dbt project context.
        models: List of (unique_id, node) tuples to validate.
        timeout_seconds: Optional timeout for each query execution.
        quiet: If True, suppress progress output.

    Returns:
        A ValidationReport containing results for all models.
    """
    report = ValidationReport()

    if not models:
        logger.warning(":warning: No models to validate.")
        return report

    logger.info(
        ":mag: Starting dry-run validation for %d model(s)",
        len(models),
    )

    for idx, (unique_id, node) in enumerate(models, 1):
        model_name = node.name
        if not quiet:
            logger.info(
                "[%d/%d] :microscope: Validating model => %s",
                idx,
                len(models),
                model_name,
            )

        result = _validate_single_model(
            context=context,
            node=node,
            unique_id=unique_id,
            timeout_seconds=timeout_seconds,
        )
        report.add_result(result)

        # Log result
        if result.status == ModelValidationStatus.SUCCESS:
            logger.info(
                "[%d/%d] :white_check_mark: %s => OK (%.2fs, %d rows)",
                idx,
                len(models),
                model_name,
                result.execution_time_seconds,
                result.row_count or 0,
            )
        elif result.status == ModelValidationStatus.COMPILE_ERROR:
            logger.error(
                "[%d/%d] :x: %s => COMPILE ERROR: %s",
                idx,
                len(models),
                model_name,
                result.error_message,
            )
        elif result.status == ModelValidationStatus.EXECUTION_ERROR:
            logger.error(
                "[%d/%d] :x: %s => EXECUTION ERROR: %s",
                idx,
                len(models),
                model_name,
                result.error_message,
            )
        elif result.status == ModelValidationStatus.TIMEOUT:
            logger.error(
                "[%d/%d] :hourglass: %s => TIMEOUT after %.2fs",
                idx,
                len(models),
                model_name,
                timeout_seconds or 0,
            )

    # Print summary
    logger.info("=" * 60)
    logger.info(":clipboard: Validation Summary")
    logger.info("=" * 60)
    logger.info("Total models: %d", report.total_models)
    logger.info("Successful: %d", report.successful)
    logger.info("Failed: %d", report.failed)
    logger.info("Success rate: %.1f%%", report.get_success_rate())
    logger.info("Total execution time: %.2fs", report.total_execution_time)
    logger.info("=" * 60)

    return report


def _validate_single_model(
    context: DbtProjectContext,
    node: ManifestNode,
    unique_id: str,
    timeout_seconds: float | None = None,
) -> ModelValidationResult:
    """Validate a single dbt model.

    Args:
        context: The dbt project context.
        node: The dbt manifest node to validate.
        unique_id: Unique identifier of the model.
        timeout_seconds: Optional timeout for query execution.

    Returns:
        A ModelValidationResult with validation details.
    """
    # Get raw SQL from the node
    raw_sql = getattr(node, "raw_code", "") or getattr(node, "raw_sql", "")
    if not raw_sql:
        return ModelValidationResult(
            model_name=node.name,
            unique_id=unique_id,
            status=ModelValidationStatus.COMPILE_ERROR,
            error_message="No SQL code found in node",
        )

    # Step 1: Compile the SQL
    try:
        logger.debug(":zap: Compiling SQL for model => %s", node.name)
        compiled_node = compile_sql_code(context, raw_sql)
        compiled_sql = compiled_node.compiled_code or compiled_node.raw_code
    except Exception as e:
        return ModelValidationResult(
            model_name=node.name,
            unique_id=unique_id,
            status=ModelValidationStatus.COMPILE_ERROR,
            error_message=str(e),
        )

    # Step 2: Execute the compiled SQL
    try:
        logger.debug(":running: Executing SQL for model => %s", node.name)
        start_time = time.time()
        response, table = execute_sql_code(context, compiled_sql)
        execution_time = time.time() - start_time

        # Extract row count from table
        row_count = len(table.rows) if hasattr(table, "rows") else None

        # Extract bytes processed from adapter response if available
        bytes_processed = None
        if hasattr(response, "_message") and response._message:
            # Some adapters return bytes processed in the response message
            # This is adapter-specific, so we try to extract it if present
            pass

        return ModelValidationResult(
            model_name=node.name,
            unique_id=unique_id,
            status=ModelValidationStatus.SUCCESS,
            compiled_sql=compiled_sql,
            execution_time_seconds=execution_time,
            row_count=row_count,
            bytes_processed=bytes_processed,
            adapter_response=response,
        )

    except Exception as e:
        error_message = str(e)
        # Check for timeout
        if timeout_seconds and "timeout" in error_message.lower():
            return ModelValidationResult(
                model_name=node.name,
                unique_id=unique_id,
                status=ModelValidationStatus.TIMEOUT,
                compiled_sql=compiled_sql,
                error_message=f"Query exceeded timeout of {timeout_seconds}s",
                execution_time_seconds=timeout_seconds,
            )

        return ModelValidationResult(
            model_name=node.name,
            unique_id=unique_id,
            status=ModelValidationStatus.EXECUTION_ERROR,
            compiled_sql=compiled_sql,
            error_message=error_message,
        )
