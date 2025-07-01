from __future__ import annotations

import re
import typing as t
from functools import lru_cache

import pluggy
from dbt.contracts.graph.nodes import ResultNode

import dbt_osmosis.core.logger as logger

__all__ = [
    "_hookspec",
    "hookimpl",
    "get_candidates",
    "FuzzyCaseMatching",
    "FuzzyPrefixMatching",
    "get_plugin_manager",
]

_hookspec = pluggy.HookspecMarker("dbt-osmosis")
hookimpl = pluggy.HookimplMarker("dbt-osmosis")


@_hookspec
def get_candidates(name: str, node: ResultNode, context: t.Any) -> list[str]:  # pyright: ignore[reportUnusedParameter]
    """Get a list of candidate names for a column."""
    raise NotImplementedError


class FuzzyCaseMatching:
    @hookimpl
    def get_candidates(self, name: str, node: ResultNode, context: t.Any) -> list[str]:
        """Get a list of candidate names for a column based on case variants."""
        _ = node, context
        variants = [
            name.lower(),  # lowercase
            name.upper(),  # UPPERCASE
            cc := re.sub("_(.)", lambda m: m.group(1).upper(), name),  # camelCase
            cc[0].upper() + cc[1:],  # PascalCase
        ]
        logger.debug(":lower_upper_case: FuzzyCaseMatching variants => %s", variants)
        return variants


class FuzzyPrefixMatching:
    @hookimpl
    def get_candidates(self, name: str, node: ResultNode, context: t.Any) -> list[str]:
        """Get a list of candidate names for a column excluding a prefix."""
        _ = context
        variants = []
        from dbt_osmosis.core.introspection import _get_setting_for_node

        p = _get_setting_for_node("prefix", node, name)
        if p:
            mut_name = name.removeprefix(p)
            logger.debug(
                ":scissors: FuzzyPrefixMatching => removing prefix '%s' => %s", p, mut_name
            )
            variants.append(mut_name)
        return variants


@lru_cache(maxsize=None)
def get_plugin_manager():
    """Get the pluggy plugin manager for dbt-osmosis."""
    manager = pluggy.PluginManager("dbt-osmosis")
    _ = manager.register(FuzzyCaseMatching())
    _ = manager.register(FuzzyPrefixMatching())
    _ = manager.load_setuptools_entrypoints("dbt-osmosis")
    return manager
