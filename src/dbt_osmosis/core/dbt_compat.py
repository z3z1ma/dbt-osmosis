"""Compatibility layer for dbt version differences.

This module provides an abstraction layer for handling structural differences
between dbt versions, particularly the meta/tags namespace change in dbt 1.10.

In dbt < 1.10:
    - meta and tags are top-level fields on nodes and columns
    - column.meta, column.tags

In dbt >= 1.10:
    - meta and tags moved to the config namespace
    - column.config.meta, column.config.tags
"""

from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    from dbt_osmosis.core.dbt_protocols import YamlRefactorContextProtocol

# A type hint for dbt node or column dictionaries
DbtNode = t.Dict[str, t.Any]

__all__ = [
    "get_meta",
    "get_tags",
    "set_meta",
    "set_tags",
]


def get_meta(context: YamlRefactorContextProtocol, node_or_column: DbtNode) -> t.Dict[str, t.Any]:
    """Safely retrieves the 'meta' dictionary from a dbt node or column dictionary.

    For dbt 1.10+, merges both config.meta (canonical) and top-level meta (legacy).
    This handles cases where:
    - Meta was set via ColumnInfo objects (top-level)
    - Meta was set via compat layer (both locations)
    - Meta exists in config.meta from previous inheritance (canonical)

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary

    Returns:
        The merged meta dictionary, empty dict if not set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        # Merge both config.meta and top-level meta (top-level takes precedence)
        config_meta = node_or_column.get("config", {}).get("meta", {})
        top_level_meta = node_or_column.get("meta", {})
        return {**config_meta, **top_level_meta}
    else:
        return node_or_column.get("meta", {})


def get_tags(context: YamlRefactorContextProtocol, node_or_column: DbtNode) -> t.List[str]:
    """Safely retrieves the 'tags' list from a dbt node or column dictionary.

    For dbt 1.10+, merges both config.tags (canonical) and top-level tags (legacy).
    This handles cases where:
    - Tags were set via ColumnInfo objects (top-level)
    - Tags were set via compat layer (both locations)
    - Tags exist in config.tags from previous inheritance (canonical)

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary

    Returns:
        The merged tags list (unique), empty list if not set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        # Merge both config.tags and top-level tags (deduplicated)
        config_tags = node_or_column.get("config", {}).get("tags", [])
        top_level_tags = node_or_column.get("tags", [])
        return list(set(config_tags) | set(top_level_tags))
    else:
        return node_or_column.get("tags", [])


def set_meta(
    context: YamlRefactorContextProtocol,
    node_or_column: DbtNode,
    new_meta: t.Dict[str, t.Any],
) -> None:
    """Safely sets the 'meta' dictionary on a dbt node or column dictionary.

    For dbt 1.10+, this sets meta in BOTH locations (config.meta and top-level meta)
    for compatibility with code that expects meta at the top level.

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary to modify
        new_meta: The meta dictionary to set
    """
    # Always set config.meta for dbt 1.10+ (canonical location in dbt 1.10)
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        config = node_or_column.setdefault("config", {})
        config["meta"] = new_meta
        # Also set top-level meta for compatibility with code that expects it there
        # This is important for code like kwargs.get("meta") checks
        node_or_column["meta"] = new_meta
    else:
        node_or_column["meta"] = new_meta


def set_tags(
    context: YamlRefactorContextProtocol,
    node_or_column: DbtNode,
    new_tags: t.List[str],
) -> None:
    """Safely sets the 'tags' list on a dbt node or column dictionary.

    For dbt 1.10+, this sets tags in BOTH locations (config.tags and top-level tags)
    for compatibility with code that expects tags at the top level.

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary to modify
        new_tags: The tags list to set
    """
    # Always set config.tags for dbt 1.10+ (canonical location in dbt 1.10)
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        config = node_or_column.setdefault("config", {})
        config["tags"] = new_tags
        # Also set top-level tags for compatibility with code that expects it there
        node_or_column["tags"] = new_tags
    else:
        node_or_column["tags"] = new_tags
