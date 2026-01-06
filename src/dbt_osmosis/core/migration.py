"""Migration planning and SQL generation for dbt-osmosis.

This module provides functionality to generate safe migration SQL for schema
changes detected by the SchemaDiff engine. It supports:

- Automatic SQL generation for safe changes
- Rollback script generation
- Multi-dialect SQL (Snowflake, Postgres, BigQuery, DuckDB, etc.)
- Breaking change validation
- Migration plan export (SQL, JSON, markdown)
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from dbt.contracts.graph.nodes import ResultNode

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

from dbt_osmosis.core.diff import (
    ColumnAdded,
    ColumnRemoved,
    ColumnRenamed,
    ColumnTypeChanged,
    SchemaChange,
    SchemaDiffResult,
)

__all__ = [
    "MigrationPlan",
    "MigrationStep",
    "MigrationPlanner",
    "MigrationFormat",
]


class MigrationFormat(Enum):
    """Output format for migration plans."""

    SQL = "sql"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class MigrationStep:
    """A single migration step with SQL and rollback.

    Attributes:
        description: Human-readable description
        sql: Forward migration SQL
        rollback_sql: Rollback SQL
        change: The SchemaChange this step addresses
        is_breaking: Whether this step requires manual review
    """

    description: str
    sql: str
    rollback_sql: str
    change: SchemaChange
    is_breaking: bool = False

    def __str__(self) -> str:
        breaking = " [BREAKING]" if self.is_breaking else ""
        return f"{self.description}{breaking}"


@dataclass
class MigrationPlan:
    """A complete migration plan for a node.

    Attributes:
        node_id: The dbt node unique_id
        node_name: Human-readable node name
        steps: Migration steps in execution order
        created_at: Timestamp when plan was created
    """

    node_id: str
    node_name: str
    steps: list[MigrationStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_breaking_changes(self) -> bool:
        """Whether this plan contains breaking changes."""
        return any(step.is_breaking for step in self.steps)

    @property
    def safe_steps(self) -> list[MigrationStep]:
        """Steps that are safe to apply automatically."""
        return [s for s in self.steps if not s.is_breaking]

    @property
    def breaking_steps(self) -> list[MigrationStep]:
        """Steps that require manual review."""
        return [s for s in self.steps if s.is_breaking]

    def to_sql(self, *, include_rollback: bool = True) -> str:
        """Generate SQL migration script.

        Args:
            include_rollback: Whether to include rollback comments

        Returns:
            SQL script as string
        """
        lines: list[str] = [
            f"-- Migration Plan for {self.node_name}",
            f"-- Generated: {self.created_at.isoformat()}",
            f"-- Steps: {len(self.steps)} ({len(self.breaking_steps)} breaking)",
            "",
            "BEGIN;",
            "",
        ]

        for i, step in enumerate(self.steps, 1):
            lines.extend([
                f"-- Step {i}: {step.description}",
            ])
            if step.is_breaking:
                lines.append("-- WARNING: BREAKING CHANGE - REVIEW REQUIRED")
            lines.extend([
                step.sql,
                "",
            ])

        lines.extend([
            "COMMIT;",
            "",
        ])

        if include_rollback:
            lines.extend([
                "-- Rollback Script",
                "-- Run to undo this migration",
                "",
                "BEGIN;",
                "",
            ])

            for i, step in enumerate(reversed(self.steps), 1):
                lines.extend([
                    f"-- Rollback Step {i}: {step.description}",
                    step.rollback_sql,
                    "",
                ])

            lines.extend([
                "COMMIT;",
                "",
            ])

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown documentation for the migration plan.

        Returns:
            Markdown formatted string
        """
        lines: list[str] = [
            f"# Migration Plan: {self.node_name}",
            "",
            f"**Node ID:** `{self.node_id}`",
            f"**Generated:** {self.created_at.isoformat()}",
            f"**Total Steps:** {len(self.steps)}",
            "",
            "## Summary",
            "",
            f"- Safe changes: {len(self.safe_steps)}",
            f"- Breaking changes: {len(self.breaking_steps)}",
            "",
        ]

        if self.steps:
            lines.extend([
                "## Steps",
                "",
            ])

            for i, step in enumerate(self.steps, 1):
                status = "⚠️ **BREAKING**" if step.is_breaking else "✅ **Safe**"
                lines.extend([
                    f"### Step {i}: {step.description} {status}",
                    "",
                    "```sql",
                    step.sql,
                    "```",
                    "",
                    "**Rollback:**",
                    "",
                    "```sql",
                    step.rollback_sql,
                    "```",
                    "",
                ])

        return "\n".join(lines)

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dict representation of the migration plan
        """
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "created_at": self.created_at.isoformat(),
            "summary": {
                "total_steps": len(self.steps),
                "safe_steps": len(self.safe_steps),
                "breaking_steps": len(self.breaking_steps),
            },
            "steps": [
                {
                    "description": step.description,
                    "sql": step.sql,
                    "rollback_sql": step.rollback_sql,
                    "is_breaking": step.is_breaking,
                    "change_type": step.change.category.value,
                }
                for step in self.steps
            ],
        }


class MigrationPlanner:
    """Generate migration SQL from schema changes.

    This planner takes schema diff results and generates SQL migration
    scripts with rollback support for multiple database dialects.

    Example:
        >>> from dbt_osmosis.core.migration import MigrationPlanner
        >>> planner = MigrationPlanner(context)
        >>> plan = planner.plan_for_result(diff_result)
        >>> print(plan.to_sql())
    """

    def __init__(
        self,
        context: YamlRefactorContextProtocol,
        *,
        dry_run: bool = False,
        format: MigrationFormat = MigrationFormat.SQL,
    ) -> None:
        """Initialize the migration planner.

        Args:
            context: The YamlRefactorContext instance
            dry_run: If True, generates SQL but doesn't apply changes
            format: Output format for migration plans
        """
        self._context = context
        self._dry_run = dry_run
        self._format = format
        self._dialect = self._detect_dialect()

    def plan_for_result(self, result: SchemaDiffResult) -> MigrationPlan:
        """Generate a migration plan for a single diff result.

        Args:
            result: SchemaDiffResult from SchemaDiff

        Returns:
            MigrationPlan with SQL for all changes
        """
        steps: list[MigrationStep] = []

        # Sort changes by dependency order
        # 1. Column renames (before drops)
        # 2. Column additions
        # 3. Type changes
        # 4. Column removals
        sorted_changes = self._sort_changes_by_dependency(result.changes)

        for change in sorted_changes:
            step = self._plan_change(change, result.node)
            if step:
                steps.append(step)

        return MigrationPlan(
            node_id=result.node.unique_id,
            node_name=result.node.name,
            steps=steps,
        )

    def plan_for_results(
        self,
        results: dict[str, SchemaDiffResult],
    ) -> dict[str, MigrationPlan]:
        """Generate migration plans for multiple diff results.

        Args:
            results: Dict of node_id -> SchemaDiffResult

        Returns:
            Dict of node_id -> MigrationPlan
        """
        plans = {}
        for node_id, result in results.items():
            plans[node_id] = self.plan_for_result(result)
        return plans

    def export_plan(
        self,
        plan: MigrationPlan,
        output_path: Path,
        *,
        format: MigrationFormat | None = None,
    ) -> None:
        """Export a migration plan to a file.

        Args:
            plan: The migration plan to export
            output_path: Path to write the plan
            format: Output format (uses default if None)
        """
        fmt = format or self._format
        output_path = Path(output_path)

        if fmt == MigrationFormat.SQL:
            content = plan.to_sql()
        elif fmt == MigrationFormat.JSON:
            import json

            content = json.dumps(plan.to_dict(), indent=2)
        elif fmt == MigrationFormat.MARKDOWN:
            content = plan.to_markdown()
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    def _detect_dialect(self) -> str:
        """Detect the database dialect from the context.

        Returns:
            Dialect identifier (snowflake, postgres, bigquery, duckdb, etc.)
        """
        credentials_type = self._context.project.runtime_cfg.credentials.type
        # Normalize dialect names
        dialect_map = {
            "snowflake": "snowflake",
            "postgres": "postgres",
            "postgresql": "postgres",
            "redshift": "postgres",
            "bigquery": "bigquery",
            "duckdb": "duckdb",
            "databricks": "databricks",
            "spark": "spark",
            "sqlserver": "sqlserver",
            "mssql": "sqlserver",
        }
        return dialect_map.get(credentials_type.lower(), credentials_type.lower())

    def _sort_changes_by_dependency(
        self,
        changes: list[SchemaChange],
    ) -> list[SchemaChange]:
        """Sort changes to respect dependencies.

        Column renames must happen before drops, etc.

        Args:
            changes: List of schema changes

        Returns:
            Sorted list of changes
        """
        from dbt_osmosis.core.diff import ChangeCategory

        # Define execution order
        order = {
            ChangeCategory.COLUMN_RENAMED: 1,
            ChangeCategory.COLUMN_ADDED: 2,
            ChangeCategory.TYPE_CHANGED: 3,
            ChangeCategory.COLUMN_REMOVED: 4,
            ChangeCategory.METADATA_CHANGED: 5,
        }

        return sorted(changes, key=lambda c: order.get(c.category, 99))

    def _plan_change(
        self,
        change: SchemaChange,
        node: ResultNode,
    ) -> MigrationStep | None:
        """Generate SQL for a single schema change.

        Args:
            change: The schema change to plan
            node: The dbt node

        Returns:
            MigrationStep or None if no SQL needed
        """
        from dbt_osmosis.core.diff import ChangeCategory

        table_name = self._quote_identifier(f"{node.schema}.{node.name}")

        if change.category == ChangeCategory.COLUMN_ADDED:
            return self._plan_column_added(t.cast("ColumnAdded", change), table_name)

        if change.category == ChangeCategory.COLUMN_REMOVED:
            return self._plan_column_removed(t.cast("ColumnRemoved", change), table_name)

        if change.category == ChangeCategory.COLUMN_RENAMED:
            return self._plan_column_renamed(t.cast("ColumnRenamed", change), table_name)

        if change.category == ChangeCategory.TYPE_CHANGED:
            return self._plan_type_changed(t.cast("ColumnTypeChanged", change), table_name)

        return None

    def _plan_column_added(self, change: ColumnAdded, table_name: str) -> MigrationStep:
        """Generate SQL for adding a column.

        Args:
            change: ColumnAdded change
            table_name: Fully qualified table name

        Returns:
            MigrationStep for adding the column
        """
        col_name = self._quote_identifier(change.column_name)
        type_clause = f" {change.data_type}" if change.data_type else ""

        sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name}{type_clause};"
        rollback = f"ALTER TABLE {table_name} DROP COLUMN {col_name};"

        return MigrationStep(
            description=f"Add column '{change.column_name}' to {table_name}",
            sql=sql,
            rollback_sql=rollback,
            change=change,
            is_breaking=False,
        )

    def _plan_column_removed(self, change: ColumnRemoved, table_name: str) -> MigrationStep:
        """Generate SQL for removing a column.

        Note: This is marked as breaking for safety since data loss may occur.

        Args:
            change: ColumnRemoved change
            table_name: Fully qualified table name

        Returns:
            MigrationStep for dropping the column
        """
        col_name = self._quote_identifier(change.column_name)

        sql = f"ALTER TABLE {table_name} DROP COLUMN {col_name};"

        # Rollback would need to recreate the column, but we'd lose data
        # Provide a comment-only rollback
        rollback = f"-- ROLLBACK: Cannot restore column '{change.column_name}' without backup"

        return MigrationStep(
            description=f"Drop column '{change.column_name}' from {table_name}",
            sql=sql,
            rollback_sql=rollback,
            change=change,
            is_breaking=True,
        )

    def _plan_column_renamed(self, change: ColumnRenamed, table_name: str) -> MigrationStep:
        """Generate SQL for renaming a column.

        Args:
            change: ColumnRenamed change
            table_name: Fully qualified table name

        Returns:
            MigrationStep for renaming the column
        """
        old_col = self._quote_identifier(change.old_name)
        new_col = self._quote_identifier(change.new_name)

        # Different dialects use different syntax
        if self._dialect == "snowflake":
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"
        elif self._dialect in ("postgres", "redshift"):
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"
        elif self._dialect == "bigquery":
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"
        elif self._dialect == "duckdb":
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"
        elif self._dialect == "spark":
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"
        elif self._dialect == "sqlserver":
            sql = f"EXEC sp_rename '{table_name}.{change.old_name}', '{change.new_name}', 'COLUMN';"
        elif self._dialect == "databricks":
            # Delta Lake doesn't support ALTER COLUMN directly, need recreate
            # This is a more complex migration
            sql = f"-- RENAME NOT SUPPORTED: Manual migration required for {table_name}.{old_col} -> {new_col}"
        else:
            # Fallback syntax
            sql = f"ALTER TABLE {table_name} RENAME COLUMN {old_col} TO {new_col};"

        rollback = f"ALTER TABLE {table_name} RENAME COLUMN {new_col} TO {old_col};"

        # For databricks/delta, mark as breaking
        is_breaking = self._dialect == "databricks"

        return MigrationStep(
            description=f"Rename column '{change.old_name}' to '{change.new_name}' in {table_name}",
            sql=sql,
            rollback_sql=rollback,
            change=change,
            is_breaking=is_breaking,
        )

    def _plan_type_changed(self, change: ColumnTypeChanged, table_name: str) -> MigrationStep:
        """Generate SQL for changing a column's data type.

        Args:
            change: ColumnTypeChanged change
            table_name: Fully qualified table name

        Returns:
            MigrationStep for altering the column type
        """
        col_name = self._quote_identifier(change.column_name)

        # Different dialects use different syntax
        if self._dialect == "snowflake":
            sql = (
                f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET DATA TYPE {change.new_type};"
            )
            rollback = (
                f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET DATA TYPE {change.old_type};"
            )
        elif self._dialect in ("postgres", "redshift"):
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.new_type};"
            rollback = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.old_type};"
        elif self._dialect == "bigquery":
            sql = (
                f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET DATA TYPE {change.new_type};"
            )
            rollback = (
                f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET DATA TYPE {change.old_type};"
            )
        elif self._dialect == "duckdb":
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.new_type};"
            rollback = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.old_type};"
        elif self._dialect == "spark":
            # Spark requires USING clause for some type changes
            sql = f"ALTER TABLE {table_name} CHANGE COLUMN {col_name} {col_name} {change.new_type};"
            rollback = (
                f"ALTER TABLE {table_name} CHANGE COLUMN {col_name} {col_name} {change.old_type};"
            )
        elif self._dialect == "sqlserver":
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} {change.new_type};"
            rollback = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} {change.old_type};"
        elif self._dialect == "databricks":
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.new_type};"
            rollback = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.old_type};"
        else:
            # Fallback syntax
            sql = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.new_type};"
            rollback = f"ALTER TABLE {table_name} ALTER COLUMN {col_name} TYPE {change.old_type};"

        return MigrationStep(
            description=f"Change type of column '{change.column_name}' from {change.old_type} to {change.new_type} in {table_name}",
            sql=sql,
            rollback_sql=rollback,
            change=change,
            is_breaking=(change.severity.value == "breaking"),
        )

    def _quote_identifier(self, identifier: str) -> str:
        """Quote a SQL identifier based on the current dialect.

        Args:
            identifier: The identifier to quote

        Returns:
            Quoted identifier
        """
        # Split schema.table if present
        parts = identifier.split(".")

        if self._dialect == "snowflake":
            # Snowflake uses double quotes, case-insensitive without them
            quoted = [f'"{part}"' if not part.startswith('"') else part for part in parts]
            return ".".join(quoted)

        if self._dialect in ("postgres", "redshift", "duckdb"):
            # Postgres uses double quotes, case-sensitive
            quoted = [f'"{part}"' if not part.startswith('"') else part for part in parts]
            return ".".join(quoted)

        if self._dialect == "bigquery":
            # BigQuery uses backticks
            quoted = [f"`{part}`" if not part.startswith("`") else part for part in parts]
            return ".".join(quoted)

        if self._dialect in ("spark", "databricks"):
            # Spark uses backticks
            quoted = [f"`{part}`" if not part.startswith("`") else part for part in parts]
            return ".".join(quoted)

        if self._dialect == "sqlserver":
            # SQL Server uses brackets
            quoted = [
                f"[{part}]" if not (part.startswith("[") and part.endswith("]")) else part
                for part in parts
            ]
            return ".".join(quoted)

        # Default: use double quotes
        quoted = [f'"{part}"' if not part.startswith('"') else part for part in parts]
        return ".".join(quoted)
