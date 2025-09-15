from __future__ import annotations

import typing as t

if t.TYPE_CHECKING:
    # This now correctly references the full context object
    from dbt_osmosis.core.settings import YamlRefactorContext

# A type hint for dbt node or column dictionaries
DbtNode = t.Dict[str, t.Any]


# GETTERS: Functions for READING properties from dbt objects
# ==========================================================

def get_meta(context: YamlRefactorContext, node_or_column: DbtNode) -> t.Dict[str, t.Any]:
    """
    Safely retrieves the 'meta' dictionary from a dbt node or column dictionary.
    """
    # UPDATED: Changed context to context.project
    if context.project.is_dbt_v1_10_or_greater:
        return node_or_column.get("config", {}).get("meta", {})
    else:
        return node_or_column.get("meta", {})


def get_tags(context: YamlRefactorContext, node_or_column: DbtNode) -> t.List[str]:
    """
    Safely retrieves the 'tags' list from a dbt node or column dictionary.
    """
    # UPDATED: Changed context to context.project
    if context.project.is_dbt_v1_10_or_greater:
        return node_or_column.get("config", {}).get("tags", [])
    else:
        return node_or_column.get("tags", [])


# SETTERS: Functions for WRITING properties to dbt objects
# =========================================================

def set_meta(context: YamlRefactorContext, node_or_column: DbtNode, new_meta: t.Dict[str, t.Any]):
    """
    Safely sets the 'meta' dictionary on a dbt node or column dictionary.
    """
    # UPDATED: Changed context to context.project
    if context.project.is_dbt_v1_10_or_greater:
        config = node_or_column.setdefault("config", {})
        config["meta"] = new_meta
    else:
        node_or_column["meta"] = new_meta


def set_tags(context: YamlRefactorContext, node_or_column: DbtNode, new_tags: t.List[str]):
    """
    Safely sets the 'tags' list on a dbt node or column dictionary.
    """
    # UPDATED: Changed context to context.project
    if context.project.is_dbt_v1_10_or_greater:
        config = node_or_column.setdefault("config", {})
        config["tags"] = new_tags
    else:
        node_or_column["tags"] = new_tags