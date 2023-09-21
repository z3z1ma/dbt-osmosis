from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from dbt_osmosis.vendored.dbt_core_interface.project import (
    ManifestNode,
)


ColumnLevelKnowledge = Dict[str, Any]
Knowledge = Dict[str, ColumnLevelKnowledge]


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
