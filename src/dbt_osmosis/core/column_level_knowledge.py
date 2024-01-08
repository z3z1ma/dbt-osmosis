import re
from typing import Any, Dict

ColumnLevelKnowledge = Dict[str, Any]
Knowledge = Dict[str, ColumnLevelKnowledge]


def delete_if_value_is_empty(prior_knowledge: ColumnLevelKnowledge, key: str) -> None:
    if not prior_knowledge[key]:
        del prior_knowledge[key]


def get_prior_knowledge(
    knowledge: Knowledge,
    column: str,
) -> ColumnLevelKnowledge:
    camel_column = re.sub("_(.)", lambda m: m.group(1).upper(), column)
    prior_knowledge_candidates = list(
        filter(
            lambda k: k,
            [
                knowledge.get(column),
                knowledge.get(column.lower()),
                knowledge.get(camel_column),
            ],
        )
    )
    sorted_prior_knowledge_candidates_sources = sorted(
        [k for k in prior_knowledge_candidates if k["progenitor"].startswith("source")],
        key=lambda k: k["generation"],
        reverse=True,
    )
    sorted_prior_knowledge_candidates_models = sorted(
        [k for k in prior_knowledge_candidates if k["progenitor"].startswith("model")],
        key=lambda k: k["generation"],
        reverse=True,
    )
    sorted_prior_knowledge_candidates = (
        sorted_prior_knowledge_candidates_sources + sorted_prior_knowledge_candidates_models
    )
    prior_knowledge = (
        sorted_prior_knowledge_candidates[0] if sorted_prior_knowledge_candidates else {}
    )
    return prior_knowledge
