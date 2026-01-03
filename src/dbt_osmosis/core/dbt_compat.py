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

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary

    Returns:
        The meta dictionary, empty dict if not set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        return node_or_column.get("config", {}).get("meta", {})
    else:
        return node_or_column.get("meta", {})


def get_tags(context: YamlRefactorContextProtocol, node_or_column: DbtNode) -> t.List[str]:
    """Safely retrieves the 'tags' list from a dbt node or column dictionary.

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary

    Returns:
        The tags list, empty list if not set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        return node_or_column.get("config", {}).get("tags", [])
    else:
        return node_or_column.get("tags", [])


def set_meta(
    context: YamlRefactorContextProtocol,
    node_or_column: DbtNode,
    new_meta: t.Dict[str, t.Any],
) -> None:
    """Safely sets the 'meta' dictionary on a dbt node or column dictionary.

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary to modify
        new_meta: The meta dictionary to set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        config = node_or_column.setdefault("config", {})
        config["meta"] = new_meta
    else:
        node_or_column["meta"] = new_meta


def set_tags(
    context: YamlRefactorContextProtocol,
    node_or_column: DbtNode,
    new_tags: t.List[str],
) -> None:
    """Safely sets the 'tags' list on a dbt node or column dictionary.

    Args:
        context: The YAML refactor context
        node_or_column: A dbt node or column dictionary to modify
        new_tags: The tags list to set
    """
    if (
        hasattr(context.project, "is_dbt_v1_10_or_greater")
        and context.project.is_dbt_v1_10_or_greater
    ):
        config = node_or_column.setdefault("config", {})
        config["tags"] = new_tags
    else:
        node_or_column["tags"] = new_tags
