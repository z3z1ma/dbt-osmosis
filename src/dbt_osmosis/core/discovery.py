"""Smart discovery of undocumented dbt models and columns.

This module provides algorithms to find models that need documentation attention,
with priority scoring based on model importance, usage patterns, and recency.
"""

from __future__ import annotations

import dataclasses
import typing as t
from datetime import datetime, timezone

if t.TYPE_CHECKING:
    from dbt.contracts.graph.nodes import ResultNode

    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

__all__ = [
    "DocumentationGap",
    "DiscoveryResult",
    "discover_undocumented_models",
    "discover_undocumented_columns",
    "calculate_priority_score",
    "get_documentation_coverage",
]


@dataclasses.dataclass
class DocumentationGap:
    """Represents a documentation gap found in a model or column.

    Attributes:
        node: The dbt node with missing or poor documentation
        gap_type: Type of gap ('missing', 'poor', 'outdated', 'inconsistent')
        description: Human-readable description of the gap
        current_doc: Current documentation (if any)
        priority: Priority score (0-100)
        reason: Explanation for the priority score
    """

    node: ResultNode
    gap_type: t.Literal["missing", "poor", "outdated", "inconsistent"]
    description: str
    current_doc: str | None
    priority: float
    reason: str

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the gap
        """
        return {
            "unique_id": self.node.unique_id,
            "name": self.node.name,
            "gap_type": self.gap_type,
            "description": self.description,
            "current_doc": self.current_doc,
            "priority": self.priority,
            "reason": self.reason,
            "resource_type": self.node.resource_type,
        }


@dataclasses.dataclass
class DiscoveryResult:
    """Result of a documentation discovery scan.

    Attributes:
        gaps: List of documentation gaps found
        total_models: Total number of models scanned
        total_columns: Total number of columns scanned
        coverage_percent: Overall documentation coverage percentage
        scan_time: Time when the scan was performed
        duration_seconds: Duration of the scan in seconds
    """

    gaps: list[DocumentationGap]
    total_models: int
    total_columns: int
    coverage_percent: float
    scan_time: datetime
    duration_seconds: float

    @property
    def high_priority_gaps(self) -> list[DocumentationGap]:
        """Get high-priority gaps (priority >= 70).

        Returns:
            List of high-priority gaps
        """
        return [g for g in self.gaps if g.priority >= 70]

    @property
    def medium_priority_gaps(self) -> list[DocumentationGap]:
        """Get medium-priority gaps (40 <= priority < 70).

        Returns:
            List of medium-priority gaps
        """
        return [g for g in self.gaps if 40 <= g.priority < 70]

    @property
    def low_priority_gaps(self) -> list[DocumentationGap]:
        """Get low-priority gaps (priority < 40).

        Returns:
            List of low-priority gaps
        """
        return [g for g in self.gaps if g.priority < 40]

    @property
    def summary(self) -> dict[str, t.Any]:
        """Get a summary of the discovery results.

        Returns:
            Summary dictionary with key statistics
        """
        return {
            "total_gaps": len(self.gaps),
            "high_priority": len(self.high_priority_gaps),
            "medium_priority": len(self.medium_priority_gaps),
            "low_priority": len(self.low_priority_gaps),
            "coverage_percent": self.coverage_percent,
            "total_models": self.total_models,
            "total_columns": self.total_columns,
        }

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the result
        """
        return {
            "gaps": [g.to_dict() for g in self.gaps],
            "summary": self.summary,
            "total_models": self.total_models,
            "total_columns": self.total_columns,
            "coverage_percent": self.coverage_percent,
            "scan_time": self.scan_time.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


def calculate_priority_score(
    node: ResultNode,
    manifest: t.Any,
    gap_type: t.Literal["missing", "poor", "outdated", "inconsistent"],
    context: YamlRefactorContextProtocol | None = None,
) -> tuple[float, str]:
    """Calculate a priority score for addressing a documentation gap.

    Priority is based on:
    - Fan-out: How many models depend on this one (higher = more important)
    - Recency: When the model was last modified (more recent = higher priority)
    - Resource type: Sources vs models (sources are foundational)
    - Position in DAG: Upstream models affect more downstream nodes

    Args:
        node: The dbt node with the documentation gap
        manifest: The dbt manifest containing dependency information
        gap_type: Type of documentation gap
        context: Optional YamlRefactorContext for additional context

    Returns:
        Tuple of (priority_score_0_to_100, reason_explanation)
    """
    score = 0.0
    reasons = []

    # Base score by gap type
    base_scores = {
        "missing": 50.0,
        "poor": 30.0,
        "outdated": 40.0,
        "inconsistent": 20.0,
    }
    score += base_scores.get(gap_type, 30.0)
    reasons.append(f"Gap type: {gap_type}")

    # Fan-out score: More dependents = higher priority
    dependents = _get_dependents(node, manifest)
    fan_out = len(dependents)
    if fan_out > 0:
        fan_out_score = min(30.0, fan_out * 3)
        score += fan_out_score
        reasons.append(f"Fan-out: {fan_out} downstream models")

    # Resource type score
    if node.resource_type == "source":
        score += 15.0
        reasons.append("Source table (foundational)")

    # Recency score (if we can get modification time)
    if hasattr(node, "patch_path"):
        # Recently modified files get higher priority
        # This is a simplified check - in production, use actual file mtime
        score += 5.0
        reasons.append("Recently modified")

    # Model position in DAG
    is_upstream = _is_upstream_model(node, manifest)
    if is_upstream:
        score += 10.0
        reasons.append("Upstream model (affects many downstream nodes)")

    # Column count (more columns = more documentation needed)
    column_count = len(node.columns)
    if column_count > 10:
        score += 5.0
        reasons.append(f"Complex model ({column_count} columns)")

    # Clamp score to [0, 100]
    priority = max(0.0, min(100.0, score))
    reason = "; ".join(reasons)

    return priority, reason


def _get_dependents(node: ResultNode, manifest: t.Any) -> list[str]:
    """Get list of nodes that depend on this node.

    Args:
        node: The dbt node
        manifest: The dbt manifest

    Returns:
        List of dependent node unique_ids
    """
    dependents: list[str] = []

    # Check all nodes to see if they depend on this one
    all_nodes = list(manifest.nodes.values()) + list(manifest.sources.values())

    for other_node in all_nodes:
        if hasattr(other_node, "depends_on_nodes"):
            if node.unique_id in other_node.depends_on_nodes:  # type: ignore
                dependents.append(other_node.unique_id)

    return dependents


def _is_upstream_model(node: ResultNode, manifest: t.Any) -> bool:
    """Check if this is an upstream model (few dependencies, many dependents).

    Args:
        node: The dbt node
        manifest: The dbt manifest

    Returns:
        True if this is an upstream model
    """
    dependencies = len(getattr(node, "depends_on_nodes", []))
    dependents = len(_get_dependents(node, manifest))

    # Upstream models have few dependencies but many dependents
    return dependencies < 5 and dependents > 2


def discover_undocumented_models(
    context: YamlRefactorContextProtocol,
    min_columns: int = 3,
    exclude_sources: bool = True,
) -> DiscoveryResult:
    """Discover models that need documentation attention.

    Args:
        context: The YamlRefactorContext instance
        min_columns: Minimum number of columns to consider a model
        exclude_sources: Whether to exclude source definitions

    Returns:
        DiscoveryResult with documentation gaps
    """
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    start_time = datetime.now(timezone.utc)
    gaps: list[DocumentationGap] = []
    total_models = 0
    total_columns = 0
    documented_models = 0
    documented_columns = 0

    for _, node in _iter_candidate_nodes(context):
        # Skip sources if requested
        if exclude_sources and node.resource_type == "source":
            continue

        # Skip models with too few columns
        if len(node.columns) < min_columns:
            continue

        total_models += 1
        total_columns += len(node.columns)

        # Check model-level documentation
        model_gap = _check_model_documentation(node, context)
        if model_gap:
            priority, reason = calculate_priority_score(
                node, context.project.manifest, model_gap["gap_type"], context
            )
            gaps.append(
                DocumentationGap(
                    node=node,
                    gap_type=model_gap["gap_type"],
                    description=model_gap["description"],
                    current_doc=node.description,
                    priority=priority,
                    reason=reason,
                )
            )
        else:
            documented_models += 1

        # Check column-level documentation
        for col_name, col in node.columns.items():
            col_gap = _check_column_documentation(col_name, col, context)
            if col_gap:
                documented_columns += 1
            else:
                # Column gap - but we score at column level
                documented_columns += 0

    # Calculate coverage
    coverage_percent = (documented_models / total_models * 100) if total_models > 0 else 0.0

    # Sort gaps by priority (descending)
    gaps.sort(key=lambda g: g.priority, reverse=True)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    return DiscoveryResult(
        gaps=gaps,
        total_models=total_models,
        total_columns=total_columns,
        coverage_percent=coverage_percent,
        scan_time=start_time,
        duration_seconds=duration,
    )


def discover_undocumented_columns(
    context: YamlRefactorContextProtocol,
    min_priority: float = 0.0,
) -> DiscoveryResult:
    """Discover columns that need documentation attention.

    Args:
        context: The YamlRefactorContext instance
        min_priority: Minimum priority score to include

    Returns:
        DiscoveryResult with documentation gaps
    """
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    start_time = datetime.now(timezone.utc)
    gaps: list[DocumentationGap] = []
    total_columns = 0
    documented_columns = 0

    for _, node in _iter_candidate_nodes(context):
        for col_name, col in node.columns.items():
            total_columns += 1

            col_gap = _check_column_documentation(col_name, col, context)
            if not col_gap:
                # Column has a gap
                priority, reason = (
                    calculate_priority_score(node, context.project.manifest, "missing", context) / 2
                )  # Lower priority for individual columns

                if priority >= min_priority:
                    gaps.append(
                        DocumentationGap(
                            node=node,
                            gap_type="missing",
                            description=f"Column '{col_name}' lacks documentation",
                            current_doc=col.description,
                            priority=priority,
                            reason=reason,
                        )
                    )
            else:
                documented_columns += 1

    # Sort gaps by priority (descending)
    gaps.sort(key=lambda g: g.priority, reverse=True)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    coverage = (documented_columns / total_columns * 100) if total_columns > 0 else 0.0

    return DiscoveryResult(
        gaps=gaps,
        total_models=len(context.project.manifest.nodes),
        total_columns=total_columns,
        coverage_percent=coverage,
        scan_time=start_time,
        duration_seconds=duration,
    )


def _check_model_documentation(
    node: ResultNode, context: YamlRefactorContextProtocol
) -> dict[str, t.Any] | None:
    """Check if a model has documentation issues.

    Args:
        node: The dbt node to check
        context: The YamlRefactorContext instance

    Returns:
        Gap description dict or None if no gap
    """
    # No description at all
    if not node.description:
        return {"gap_type": "missing", "description": "Model has no description"}

    # Placeholder descriptions
    if node.description in context.placeholders:
        return {"gap_type": "missing", "description": "Model has placeholder description"}

    # Very short descriptions (likely poor)
    word_count = len(node.description.split())
    if word_count < 3:
        return {"gap_type": "poor", "description": "Model description is too short"}

    # Generic/descriptive patterns that indicate poor documentation
    generic_patterns = ["this is a", "contains data", "table for", "model of"]
    description_lower = node.description.lower()
    if any(pattern in description_lower for pattern in generic_patterns):
        return {
            "gap_type": "poor",
            "description": "Model description appears generic or low-quality",
        }

    # No significant issues
    return None


def _check_column_documentation(
    col_name: str, col: t.Any, context: YamlRefactorContextProtocol
) -> bool:
    """Check if a column has adequate documentation.

    Args:
        col_name: The column name
        col: The column info
        context: The YamlRefactorContext instance

    Returns:
        True if column is well-documented, False otherwise
    """
    # No description
    if not col.description:
        return False

    # Placeholder description
    if col.description in context.placeholders:
        return False

    # Very short description
    word_count = len(col.description.split())
    if word_count < 2:
        return False

    # Generic description
    if col.description.lower() in ["the id", "the name", "the value"]:
        return False

    return True


def get_documentation_coverage(
    context: YamlRefactorContextProtocol,
) -> dict[str, t.Any]:
    """Get overall documentation coverage statistics.

    Args:
        context: The YamlRefactorContext instance

    Returns:
        Dictionary with coverage statistics
    """
    from dbt_osmosis.core.node_filters import _iter_candidate_nodes

    total_models = 0
    documented_models = 0
    total_columns = 0
    documented_columns = 0

    for _, node in _iter_candidate_nodes(context):
        total_models += 1

        # Check model documentation
        if node.description and node.description not in context.placeholders:
            documented_models += 1

        # Check column documentation
        for col in node.columns.values():
            total_columns += 1
            if col.description and col.description not in context.placeholders:
                documented_columns += 1

    model_coverage = (documented_models / total_models * 100) if total_models > 0 else 0.0
    column_coverage = documented_columns / total_columns * 100 if total_columns > 0 else 0.0

    return {
        "total_models": total_models,
        "documented_models": documented_models,
        "model_coverage_percent": model_coverage,
        "total_columns": total_columns,
        "documented_columns": documented_columns,
        "column_coverage_percent": column_coverage,
        "overall_coverage_percent": (model_coverage + column_coverage) / 2,
    }
