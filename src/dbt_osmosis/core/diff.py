"""Schema diff and change detection module for dbt-osmosis.

This module provides functionality to detect and categorize schema changes
between YAML definitions and the actual database schema. It supports:

- Column additions and removals
- Column renames (detected via fuzzy matching)
- Data type changes
- Breaking vs non-breaking change classification
- Change impact assessment
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dbt.contracts.graph.nodes import ColumnInfo, ResultNode  # pyright: ignore[reportPrivateImportUsage]
from dbt_common.contracts.metadata import ColumnMetadata  # pyright: ignore[reportPrivateImportUsage]
from rapidfuzz import fuzz, process

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

__all__ = [
    "ChangeCategory",
    "ChangeSeverity",
    "SchemaChange",
    "ColumnAdded",
    "ColumnRemoved",
    "ColumnRenamed",
    "ColumnTypeChanged",
    "SchemaDiffResult",
    "SchemaDiff",
]


class ChangeCategory(Enum):
    """Category of schema change for grouping and reporting."""

    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    METADATA_CHANGED = "metadata_changed"


class ChangeSeverity(Enum):
    """Severity level of a schema change for impact assessment."""

    SAFE = "safe"  # Non-breaking, can be applied automatically
    MODERATE = "moderate"  # May require review, generally safe
    BREAKING = "breaking"  # Requires manual review and planning


@dataclass(frozen=True)
class SchemaChange:
    """Base class for schema changes.

    Attributes:
        category: The type of change
        severity: Impact severity of the change
        node: The dbt node this change affects
        description: Human-readable description of the change
    """

    category: ChangeCategory
    severity: ChangeSeverity
    node: ResultNode
    description: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.description}"


@dataclass(frozen=True)
class ColumnAdded(SchemaChange):
    """A column that exists in the database but not in YAML.

    This is generally a safe change - adding columns to YAML is non-breaking.
    """

    column_name: str
    data_type: str | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        if self.description == "":
            object.__setattr__(
                self,
                "description",
                f"Column '{self.column_name}' added to {self.node.name}",
            )


@dataclass(frozen=True)
class ColumnRemoved(SchemaChange):
    """A column that exists in YAML but not in the database.

    This is a potentially breaking change - the column may have been dropped
    from the database, or it may be a YAML-only discrepancy.
    """

    column_name: str
    data_type: str | None = None

    def __post_init__(self) -> None:
        if self.description == "":
            object.__setattr__(
                self,
                "description",
                f"Column '{self.column_name}' removed from database in {self.node.name}",
            )


@dataclass(frozen=True)
class ColumnRenamed(SchemaChange):
    """A column that was renamed (detected via fuzzy matching).

    This is detected when a column in YAML closely matches a column in the
    database, but the names don't match exactly.
    """

    old_name: str
    new_name: str
    similarity_score: float
    data_type: str | None = None

    def __post_init__(self) -> None:
        if self.description == "":
            object.__setattr__(
                self,
                "description",
                f"Column '{self.old_name}' renamed to '{self.new_name}' in {self.node.name} "
                f"(similarity: {self.similarity_score:.1%})",
            )


@dataclass(frozen=True)
class ColumnTypeChanged(SchemaChange):
    """A column whose data type changed between YAML and database.

    The severity depends on the type change:
    - SAFE: precision/width changes (e.g., varchar(50) -> varchar(100))
    - MODERATE: compatible type changes (e.g., int -> bigint)
    - BREAKING: incompatible type changes (e.g., int -> text)
    """

    column_name: str
    old_type: str
    new_type: str

    def __post_init__(self) -> None:
        if self.description == "":
            object.__setattr__(
                self,
                "description",
                f"Column '{self.column_name}' type changed from {self.old_type} to {self.new_type} in {self.node.name}",
            )


@dataclass(frozen=True)
class SchemaDiffResult:
    """Result of a schema diff operation.

    Attributes:
        node: The dbt node that was compared
        yaml_columns: Columns defined in YAML
        database_columns: Columns from database introspection
        changes: List of detected changes
        summary: Summary statistics
    """

    node: ResultNode
    yaml_columns: dict[str, ColumnInfo]
    database_columns: dict[str, ColumnMetadata]
    changes: list[SchemaChange] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def summary(self) -> dict[str, int]:
        """Summary of changes by category."""
        summary: dict[str, int] = {}
        for change in self.changes:
            key = change.category.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return len(self.changes) > 0

    @property
    def breaking_changes(self) -> list[SchemaChange]:
        """Filter changes to only breaking ones."""
        return [c for c in self.changes if c.severity == ChangeSeverity.BREAKING]

    @property
    def safe_changes(self) -> list[SchemaChange]:
        """Filter changes to only safe ones."""
        return [c for c in self.changes if c.severity == ChangeSeverity.SAFE]


class SchemaDiff:
    """Schema change detection and comparison engine.

    This class compares YAML schema definitions with database introspection
    results to detect and categorize schema changes.

    Example:
        >>> from dbt_osmosis.core.diff import SchemaDiff
        >>> diff = SchemaDiff(context)
        >>> result = diff.compare_node(node)
        >>> for change in result.changes:
        ...     print(change)
    """

    def __init__(
        self,
        context: YamlRefactorContextProtocol,
        *,
        fuzzy_match_threshold: float = 85.0,
        detect_column_renames: bool = True,
    ) -> None:
        """Initialize the schema diff engine.

        Args:
            context: The YamlRefactorContext instance
            fuzzy_match_threshold: Threshold for detecting column renames (0-100)
            detect_column_renames: Whether to enable fuzzy matching for renames
        """
        self._context = context
        self._fuzzy_match_threshold = fuzzy_match_threshold
        self._detect_column_renames = detect_column_renames  # type: ignore[assignment]

    def compare_node(self, node: ResultNode) -> SchemaDiffResult:
        """Compare a single node's YAML schema with database schema.

        Args:
            node: The dbt node to compare

        Returns:
            SchemaDiffResult with detected changes
        """
        from dbt_osmosis.core.introspection import get_columns, normalize_column_name

        # Get YAML columns
        yaml_columns: dict[str, ColumnInfo] = node.columns

        # Get database columns
        database_columns = get_columns(self._context, node)

        # Normalize column names for comparison
        credentials_type = self._context.project.runtime_cfg.credentials.type
        yaml_col_names = {
            normalize_column_name(c.name, credentials_type) for c in yaml_columns.values()
        }
        db_col_names = set(database_columns.keys())

        # Detect changes
        changes: list[SchemaChange] = []

        # Find added columns (in DB but not in YAML)
        added_columns = db_col_names - yaml_col_names
        for col_name in added_columns:
            col_meta = database_columns[col_name]
            changes.append(
                ColumnAdded(
                    category=ChangeCategory.COLUMN_ADDED,
                    severity=ChangeSeverity.SAFE,
                    node=node,
                    description="",
                    column_name=col_name,
                    data_type=col_meta.type,
                    comment=col_meta.comment,
                )
            )

        # Find removed columns (in YAML but not in DB)
        removed_columns = yaml_col_names - db_col_names
        for col_name in removed_columns:
            # Get the original column info (before normalization)
            original_col = next(
                (
                    c
                    for c in yaml_columns.values()
                    if normalize_column_name(c.name, credentials_type) == col_name
                ),
                None,
            )
            changes.append(
                ColumnRemoved(
                    category=ChangeCategory.COLUMN_REMOVED,
                    severity=ChangeSeverity.MODERATE,
                    node=node,
                    description="",
                    column_name=col_name,
                    data_type=original_col.data_type if original_col else None,
                )
            )

        # Detect column renames via fuzzy matching
        if self._detect_column_renames and added_columns and removed_columns:
            renames = self._detect_column_renames(  # pyright: ignore[reportArgumentType]
                list(removed_columns),
                list(added_columns),
                database_columns,
            )
            # Replace added/removed with rename if we found a match
            changes = [
                c
                for c in changes
                if not (
                    isinstance(c, ColumnAdded) and c.column_name in {r.new_name for r in renames}
                )
            ]
            changes = [
                c
                for c in changes
                if not (
                    isinstance(c, ColumnRemoved) and c.column_name in {r.old_name for r in renames}
                )
            ]
            changes.extend(renames)

        # Detect type changes for common columns
        common_columns = yaml_col_names & db_col_names
        for col_name in common_columns:
            yaml_col = next(
                (
                    c
                    for c in yaml_columns.values()
                    if normalize_column_name(c.name, credentials_type) == col_name
                ),
                None,
            )
            db_col = database_columns[col_name]

            if yaml_col and db_col and yaml_col.data_type != db_col.type:
                severity = self._classify_type_change(yaml_col.data_type or "unknown", db_col.type)
                changes.append(
                    ColumnTypeChanged(
                        category=ChangeCategory.TYPE_CHANGED,
                        severity=severity,
                        node=node,
                        description="",
                        column_name=col_name,
                        old_type=yaml_col.data_type or "unknown",
                        new_type=db_col.type,
                    )
                )

        return SchemaDiffResult(
            node=node,
            yaml_columns=yaml_columns,
            database_columns=database_columns,  # pyright: ignore[reportArgumentType]
            changes=changes,
        )

    def compare_all(
        self,
        nodes: t.Iterable[ResultNode] | None = None,
    ) -> dict[str, SchemaDiffResult]:
        """Compare multiple nodes.

        Args:
            nodes: Iterable of nodes to compare. If None, uses context nodes.

        Returns:
            Dict mapping node unique_id to SchemaDiffResult
        """
        if nodes is None:
            from dbt_osmosis.core.node_filters import _iter_candidate_nodes

            nodes = [n for _, n in _iter_candidate_nodes(self._context)]

        results = {}
        for node in nodes:
            result = self.compare_node(node)
            if result.has_changes:
                results[node.unique_id] = result

        return results

    def _detect_column_renames(
        self,
        removed: list[str],
        added: list[str],
        database_columns: dict[str, ColumnMetadata],
    ) -> list[ColumnRenamed]:
        """Detect column renames using fuzzy string matching.

        Args:
            removed: Column names in YAML but not in database
            added: Column names in database but not in YAML
            database_columns: Database column metadata for type info

        Returns:
            List of ColumnRenamed changes
        """
        renames: list[ColumnRenamed] = []
        matched_added: set[str] = set()

        for old_name in removed:
            # Use fuzzy matching to find potential rename
            match = process.extractOne(
                old_name,
                added,
                scorer=fuzz.WRatio,
                score_cutoff=int(self._fuzzy_match_threshold),
            )

            if match and match[1] >= self._fuzzy_match_threshold:
                new_name = match[0]
                similarity = match[1]
                matched_added.add(new_name)

                renames.append(
                    ColumnRenamed(
                        category=ChangeCategory.COLUMN_RENAMED,
                        severity=ChangeSeverity.SAFE,
                        node=self._context.current_node
                        if hasattr(self._context, "current_node")
                        else next(iter(self._context.manifest.nodes.values())),  # type: ignore
                        description="",
                        old_name=old_name,
                        new_name=new_name,
                        similarity_score=similarity,
                        data_type=database_columns[new_name].type,
                    )
                )

        return renames

    def _classify_type_change(self, old_type: str, new_type: str) -> ChangeSeverity:
        """Classify the severity of a data type change.

        Args:
            old_type: Original data type
            new_type: New data type

        Returns:
            ChangeSeverity classification
        """
        # Normalize types for comparison
        old_norm = old_type.lower().replace(" ", "")
        new_norm = new_type.lower().replace(" ", "")

        # Same type = safe
        if old_norm == new_norm:
            return ChangeSeverity.SAFE

        # Type family changes (breaking)
        type_families = {
            "integer": {"int", "integer", "smallint", "bigint", "tinyint"},
            "float": {"float", "double", "real", "doubleprecision"},
            "text": {"text", "varchar", "char", "character", "string", "clob"},
            "boolean": {"bool", "boolean", "bit"},
            "timestamp": {"timestamp", "datetime", "timestamptz"},
            "date": {"date"},
            "numeric": {"numeric", "decimal", "number", "dec"},
        }

        # Check if types are in the same family
        for family, types in type_families.items():
            if any(t in old_norm for t in types) and any(t in new_norm for t in types):
                # Same family - generally safe (e.g., int -> bigint, varchar(50) -> varchar(100))
                # But narrowing is potentially breaking
                if self._is_type_narrowing(old_norm, new_norm):
                    return ChangeSeverity.MODERATE
                return ChangeSeverity.SAFE

        # Different families = breaking
        return ChangeSeverity.BREAKING

    def _is_type_narrowing(self, old_type: str, new_type: str) -> bool:
        """Check if a type change narrows precision (potentially breaking).

        Args:
            old_type: Original data type
            new_type: New data type

        Returns:
            True if the new type is narrower than the old type
        """
        # Extract precision/scale for numeric types
        import re

        def extract_precision(type_str: str) -> tuple[str, int | None, int | None]:
            """Extract base type, precision, and scale from a type string."""
            match = re.match(r"(\w+)(?:\((\d+)(?:,(\d+))?\))?", type_str.lower())
            if match:
                base = match.group(1)
                precision = int(match.group(2)) if match.group(2) else None
                scale = int(match.group(3)) if match.group(3) else None
                return base, precision, scale
            return type_str.lower(), None, None

        old_base, old_prec, old_scale = extract_precision(old_type)
        new_base, new_prec, new_scale = extract_precision(new_type)

        # Check for precision narrowing (e.g., varchar(100) -> varchar(50))
        if old_base == new_base:
            if old_prec and new_prec and new_prec < old_prec:
                return True
            if old_scale and new_scale and new_scale < old_scale:
                return True

        # Check for integer narrowing (e.g., bigint -> int -> smallint)
        narrowing_order = ["bigint", "int", "integer", "smallint", "tinyint"]
        if old_base in narrowing_order and new_base in narrowing_order:
            return narrowing_order.index(old_base) < narrowing_order.index(new_base)

        return False
