# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import pytest
from dbt.artifacts.resources.types import NodeType

from dbt_osmosis.core.path_management import MissingOsmosisConfig, _get_yaml_path_template
from dbt_osmosis.core.settings import YamlRefactorContext


def test_missing_osmosis_config_error(yaml_context: YamlRefactorContext):
    """
    Ensures MissingOsmosisConfig is raised if there's no path template
    for a model. We'll mock the node config so we remove any 'dbt-osmosis' key.
    """
    node = None
    # Find some real model node
    for n in yaml_context.project.manifest.nodes.values():
        if n.resource_type == NodeType.Model:
            node = n
            break
    assert node, "No model found in your demo_duckdb project"

    # We'll forcibly remove the dbt_osmosis config from node.config.extra
    old = node.config.extra.pop("dbt-osmosis", None)
    node.unrendered_config.pop("dbt-osmosis", None)

    with pytest.raises(MissingOsmosisConfig):
        _ = _get_yaml_path_template(yaml_context, node)

    node.config.extra["dbt-osmosis"] = old
    node.unrendered_config["dbt-osmosis"] = old
