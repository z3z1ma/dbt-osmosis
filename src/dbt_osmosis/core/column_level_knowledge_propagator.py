from typing import Any, Dict, Iterable, List, Optional

from dbt_osmosis.core.column_level_knowledge import (
    ColumnLevelKnowledge,
    Knowledge,
    delete_if_value_is_empty,
    get_prior_knowledge,
)
from dbt_osmosis.core.log_controller import logger
from dbt_osmosis.vendored.dbt_core_interface.project import ColumnInfo, ManifestNode


def _build_node_ancestor_tree(
    manifest: ManifestNode,
    node: ManifestNode,
    family_tree: Optional[Dict[str, List[str]]] = None,
    members_found: Optional[List[str]] = None,
    depth: int = 0,
) -> Dict[str, List[str]]:
    """Recursively build dictionary of parents in generational order"""
    if family_tree is None:
        family_tree = {}
    if members_found is None:
        members_found = []
    if not hasattr(node, "depends_on"):
        return family_tree
    for parent in getattr(node.depends_on, "nodes", []):
        member = manifest.nodes.get(parent, manifest.sources.get(parent))
        if member and parent not in members_found:
            family_tree.setdefault(f"generation_{depth}", []).append(parent)
            members_found.append(parent)
            # Recursion
            family_tree = _build_node_ancestor_tree(
                manifest, member, family_tree, members_found, depth + 1
            )
    return family_tree


def _inherit_column_level_knowledge(
    manifest: ManifestNode,
    family_tree: Dict[str, Any],
    placeholders: List[str],
) -> Knowledge:
    """Inherit knowledge from ancestors in reverse insertion order to ensure that the most
    recent ancestor is always the one to inherit from
    """
    knowledge: Knowledge = {}
    for generation in reversed(family_tree):
        for ancestor in family_tree[generation]:
            member: ManifestNode = manifest.nodes.get(ancestor, manifest.sources.get(ancestor))
            if not member:
                continue
            for name, info in member.columns.items():
                knowledge_default = {"progenitor": ancestor, "generation": generation}
                knowledge.setdefault(name, knowledge_default)
                deserialized_info = info.to_dict()
                # Handle Info:
                # 1. tags are additive
                # 2. descriptions are overriden
                # 3. meta is merged
                # 4. tests are ignored until I am convinced those shouldn't be
                #       hand curated with love
                if deserialized_info["description"] in placeholders:
                    deserialized_info.pop("description", None)
                deserialized_info["tags"] = list(
                    set(deserialized_info.pop("tags", []) + knowledge[name].get("tags", []))
                )
                if not deserialized_info["tags"]:
                    deserialized_info.pop("tags")  # poppin' tags like Macklemore
                deserialized_info["meta"] = {
                    **knowledge[name].get("meta", {}),
                    **deserialized_info["meta"],
                }
                if not deserialized_info["meta"]:
                    deserialized_info.pop("meta")
                knowledge[name].update(deserialized_info)
    return knowledge


class ColumnLevelKnowledgePropagator:
    @staticmethod
    def get_node_columns_with_inherited_knowledge(
        manifest: ManifestNode,
        node: ManifestNode,
        placeholders: List[str],
    ) -> Knowledge:
        """Build a knowledgebase for the model based on iterating through ancestors"""
        family_tree = _build_node_ancestor_tree(manifest, node)
        knowledge = _inherit_column_level_knowledge(manifest, family_tree, placeholders)
        return knowledge

    @staticmethod
    def _get_original_knowledge(node: ManifestNode, column: str) -> ColumnLevelKnowledge:
        original_knowledge: ColumnLevelKnowledge = {
            "description": None,
            "tags": set(),
            "meta": {},
        }
        if column in node.columns:
            original_knowledge["description"] = node.columns[column].description
            original_knowledge["meta"] = node.columns[column].meta
            original_knowledge["tags"] = node.columns[column].tags
        return original_knowledge

    @staticmethod
    def _merge_prior_knowledge_with_original_knowledge(
        prior_knowledge: ColumnLevelKnowledge,
        original_knowledge: ColumnLevelKnowledge,
        add_progenitor_to_meta: bool,
        progenitor: str,
    ) -> None:
        if "tags" in prior_knowledge:
            prior_knowledge["tags"] = list(
                set(prior_knowledge["tags"] + list(original_knowledge["tags"]))
            )
        else:
            prior_knowledge["tags"] = original_knowledge["tags"]

        if "meta" in prior_knowledge:
            prior_knowledge["meta"] = {
                **original_knowledge["meta"],
                **prior_knowledge["meta"],
            }
        else:
            prior_knowledge["meta"] = original_knowledge["meta"]

        if add_progenitor_to_meta and progenitor:
            prior_knowledge["meta"]["osmosis_progenitor"] = progenitor

        if original_knowledge["meta"].get("osmosis_keep_description", None):
            prior_knowledge["description"] = original_knowledge["description"]

        for k in ["tags", "meta"]:
            delete_if_value_is_empty(prior_knowledge, k)

    @staticmethod
    def update_undocumented_columns_with_prior_knowledge(
        undocumented_columns: Iterable[str],
        node: ManifestNode,
        yaml_file_model_section: Dict[str, Any],
        knowledge: Knowledge,
        skip_add_tags: bool,
        skip_merge_meta: bool,
        add_progenitor_to_meta: bool,
    ) -> int:
        """Update undocumented columns with prior knowledge in node and model simultaneously
        THIS MUTATES THE NODE AND MODEL OBJECTS so that state is always accurate"""
        inheritables = ["description"]
        if not skip_add_tags:
            inheritables.append("tags")
        if not skip_merge_meta:
            inheritables.append("meta")

        changes_committed = 0
        for column in undocumented_columns:
            prior_knowledge: ColumnLevelKnowledge = get_prior_knowledge(knowledge, column)
            progenitor = prior_knowledge.pop("progenitor", None)
            prior_knowledge: ColumnLevelKnowledge = {
                k: v for k, v in prior_knowledge.items() if k in inheritables
            }

            ColumnLevelKnowledgePropagator._merge_prior_knowledge_with_original_knowledge(
                prior_knowledge,
                ColumnLevelKnowledgePropagator._get_original_knowledge(node, column),
                add_progenitor_to_meta,
                progenitor,
            )
            if not prior_knowledge:
                continue

            if column not in node.columns:
                node.columns[column] = ColumnInfo.from_dict({"name": column, **prior_knowledge})
            else:
                node.columns[column] = ColumnInfo.from_dict(
                    dict(node.columns[column].to_dict(), **prior_knowledge)
                )
            for model_column in yaml_file_model_section["columns"]:
                if model_column["name"] == column:
                    model_column.update(prior_knowledge)
            changes_committed += 1
            logger().info(
                ":light_bulb: Column %s is inheriting knowledge from the lineage of progenitor"
                " (%s) for model %s",
                column,
                progenitor,
                node.unique_id,
            )
            logger().info(prior_knowledge)
        return changes_committed
