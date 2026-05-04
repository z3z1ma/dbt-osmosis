# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Behavior tests for YAML path resolution.

These tests validate the behavior of path resolution based on dbt_project.yml configuration.
Tests use the public API get_target_yaml_path() to verify that:
- MissingOsmosisConfig is raised for nodes without dbt-osmosis config
- Path templates work correctly for different node types
- Template variables like {node.source_name} are available for the right node types
"""

import pytest
from dbt.artifacts.resources.types import NodeType

from dbt_osmosis.core.path_management import (
    MissingOsmosisConfig,
    PathResolutionError,
    get_target_yaml_path,
)
from dbt_osmosis.core.settings import YamlRefactorContext


def test_missing_osmosis_config_error(yaml_context: YamlRefactorContext):
    """Behavior test: Ensures MissingOsmosisConfig is raised if there's no path template
    for a model. Tests the public API get_target_yaml_path() rather than internal function.
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
    old_unrendered = node.unrendered_config.pop("dbt-osmosis", None)

    # Use public API - should raise MissingOsmosisConfig
    with pytest.raises(MissingOsmosisConfig):
        get_target_yaml_path(yaml_context, node)

    # Restore original config
    if old is not None:
        node.config.extra["dbt-osmosis"] = old
    if old_unrendered is not None:
        node.unrendered_config["dbt-osmosis"] = old_unrendered


def test_source_name_in_path_template(yaml_context: YamlRefactorContext):
    """Ensures that {node.source_name} is available in path templates for source nodes.
    Regression test for GitHub issue #242.
    """
    source_node = None
    for n in yaml_context.project.manifest.sources.values():
        if n.resource_type == NodeType.Source:
            source_node = n
            break
    if not source_node:
        pytest.skip("No source found in your demo_duckdb project")

    # Configure a path template that uses {node.source_name}
    yaml_context.source_definitions[source_node.source_name] = {
        "path": "sources/{node.source_name}.yml",
    }

    target_path = get_target_yaml_path(yaml_context, source_node)
    assert target_path.name == f"{source_node.source_name}.yml"
    assert "sources" in str(target_path)


def test_source_name_not_available_for_models(yaml_context: YamlRefactorContext):
    """Ensures that using {node.source_name} in a path template for a non-source node
    (model, seed) raises a clear PathResolutionError.
    """
    model_node = None
    for n in yaml_context.project.manifest.nodes.values():
        if n.resource_type == NodeType.Model:
            model_node = n
            break
    assert model_node, "No model found in your demo_duckdb project"

    # Configure a path template that uses {node.source_name} (invalid for models)
    old = model_node.config.extra.get("dbt-osmosis")
    model_node.config.extra["dbt-osmosis"] = "models/{node.source_name}.yml"

    # Should raise a user-facing path error because source_name is not available for models
    with pytest.raises(PathResolutionError, match="source_name") as exc_info:
        get_target_yaml_path(yaml_context, model_node)
    message = str(exc_info.value)
    assert model_node.unique_id in message
    assert "models/{node.source_name}.yml" in message

    # Restore original config
    if old is not None:
        model_node.config.extra["dbt-osmosis"] = old
    else:
        model_node.config.extra.pop("dbt-osmosis", None)


def test_missing_path_template_placeholder_raises_path_resolution_error(
    yaml_context: YamlRefactorContext,
):
    """Missing format keys in user templates should not leak raw KeyError."""
    model_node = None
    for n in yaml_context.project.manifest.nodes.values():
        if n.resource_type == NodeType.Model:
            model_node = n
            break
    assert model_node, "No model found in your demo_duckdb project"

    old = model_node.config.extra.get("dbt-osmosis")
    template = "models/{missing_placeholder}.yml"
    model_node.config.extra["dbt-osmosis"] = template

    with pytest.raises(PathResolutionError, match="missing_placeholder") as exc_info:
        get_target_yaml_path(yaml_context, model_node)
    message = str(exc_info.value)
    assert model_node.unique_id in message
    assert template in message

    if old is not None:
        model_node.config.extra["dbt-osmosis"] = old
    else:
        model_node.config.extra.pop("dbt-osmosis", None)
