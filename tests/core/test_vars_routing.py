# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

"""Tests for vars-based per-folder YAML routing (fusion-compatible).

These tests validate _resolve_vars_routing which allows users to specify
YAML path templates under vars.dbt-osmosis.models instead of +dbt-osmosis
config keys in dbt_project.yml. This is needed for dbt-fusion compatibility
since fusion rejects unknown + prefixed config keys.
"""

from __future__ import annotations

import typing as t
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from dbt.artifacts.resources.types import NodeType

from dbt_osmosis.core.path_management import (
    MissingOsmosisConfig,
    _resolve_vars_routing,
)


def _make_node(
    fqn: list[str],
    resource_type: NodeType = NodeType.Model,
    name: str | None = None,
) -> MagicMock:
    """Create a minimal mock node with the given FQN."""
    node = MagicMock()
    node.fqn = fqn
    node.resource_type = resource_type
    node.name = name or fqn[-1]
    node.config = MagicMock()
    node.config.extra = {}
    node.unrendered_config = {}
    node.meta = {}
    return node


def _make_context(vars_dict: dict[str, t.Any]) -> MagicMock:
    """Create a minimal mock context with the given project vars."""
    context = MagicMock()
    context.project.runtime_cfg.vars.to_dict.return_value = vars_dict
    return context


class TestResolveVarsRouting:
    """Tests for _resolve_vars_routing."""

    def test_models_single_folder_match(self):
        """Matches a model in staging/ to vars.dbt-osmosis.models.staging."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                },
            },
        })
        node = _make_node(["project", "staging", "oem_raw", "stg_oem_raw__mme"])
        result = _resolve_vars_routing(context, node)
        assert result == "_stg_{parent}__models.yml"

    def test_models_nested_folder_most_specific_wins(self):
        """When both 'staging' and 'staging.oem_raw' exist, the deeper match wins."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                    "staging.oem_raw": "_stg_oem_raw__models.yml",
                },
            },
        })
        node = _make_node(["project", "staging", "oem_raw", "stg_oem_raw__mme"])
        result = _resolve_vars_routing(context, node)
        assert result == "_stg_oem_raw__models.yml"

    def test_models_falls_back_to_parent_folder(self):
        """Model in staging/oem_raw/ falls back to 'staging' when no 'staging.oem_raw' key."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                    "intermediate": "_int_{parent}__models.yml",
                },
            },
        })
        node = _make_node(["project", "staging", "oem_raw", "stg_oem_raw__mme"])
        result = _resolve_vars_routing(context, node)
        assert result == "_stg_{parent}__models.yml"

    def test_models_intermediate_match(self):
        """Matches intermediate models correctly."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                    "intermediate": "_int_{parent}__models.yml",
                    "marts": "_marts_{parent}__models.yml",
                },
            },
        })
        node = _make_node(["project", "intermediate", "int_oil_prices"])
        result = _resolve_vars_routing(context, node)
        assert result == "_int_{parent}__models.yml"

    def test_models_no_match_returns_none(self):
        """Returns None when no folder key matches the node's FQN."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                },
            },
        })
        node = _make_node(["project", "marts", "fct_oil_price"])
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_models_empty_routing_returns_none(self):
        """Returns None when models routing dict is empty."""
        context = _make_context({"dbt-osmosis": {"models": {}}})
        node = _make_node(["project", "staging", "my_model"])
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_models_missing_routing_returns_none(self):
        """Returns None when vars.dbt-osmosis exists but has no models key."""
        context = _make_context({"dbt-osmosis": {"sources": {}}})
        node = _make_node(["project", "staging", "my_model"])
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_no_osmosis_vars_returns_none(self):
        """Returns None when vars has no dbt-osmosis section."""
        context = _make_context({"some_other_var": "value"})
        node = _make_node(["project", "staging", "my_model"])
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_underscore_variant_key(self):
        """Supports dbt_osmosis (underscore) as alternative to dbt-osmosis (kebab)."""
        context = _make_context({
            "dbt_osmosis": {
                "models": {
                    "staging": "_stg_{parent}__models.yml",
                },
            },
        })
        node = _make_node(["project", "staging", "oem_raw", "stg_oem_raw__mme"])
        result = _resolve_vars_routing(context, node)
        assert result == "_stg_{parent}__models.yml"

    def test_kebab_preferred_over_underscore(self):
        """When both dbt-osmosis and dbt_osmosis exist, kebab-case wins."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {"staging": "kebab_wins.yml"},
            },
            "dbt_osmosis": {
                "models": {"staging": "underscore_loses.yml"},
            },
        })
        node = _make_node(["project", "staging", "my_model"])
        result = _resolve_vars_routing(context, node)
        assert result == "kebab_wins.yml"

    def test_seeds_string_value(self):
        """Seeds routing with a single string applies to all seeds."""
        context = _make_context({
            "dbt-osmosis": {
                "seeds": "_seeds__models.yml",
            },
        })
        node = _make_node(
            ["project", "my_seed"],
            resource_type=NodeType.Seed,
            name="my_seed",
        )
        result = _resolve_vars_routing(context, node)
        assert result == "_seeds__models.yml"

    def test_seeds_dict_per_folder(self):
        """Seeds routing with a dict supports per-folder routing."""
        context = _make_context({
            "dbt-osmosis": {
                "seeds": {
                    "reference": "_ref_seeds.yml",
                    "test_data": "_test_seeds.yml",
                },
            },
        })
        node = _make_node(
            ["project", "reference", "dim_countries"],
            resource_type=NodeType.Seed,
            name="dim_countries",
        )
        result = _resolve_vars_routing(context, node)
        assert result == "_ref_seeds.yml"

    def test_seeds_not_matched_for_models(self):
        """Model nodes do not match against seeds routing."""
        context = _make_context({
            "dbt-osmosis": {
                "seeds": "_seeds__models.yml",
            },
        })
        node = _make_node(["project", "staging", "my_model"], resource_type=NodeType.Model)
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_models_not_matched_for_seeds(self):
        """Seed nodes do not match against models routing."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {"staging": "_stg.yml"},
            },
        })
        node = _make_node(["project", "my_seed"], resource_type=NodeType.Seed)
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_vars_exception_returns_none(self):
        """Returns None gracefully when vars.to_dict() throws."""
        context = MagicMock()
        context.project.runtime_cfg.vars.to_dict.side_effect = RuntimeError("boom")
        node = _make_node(["project", "staging", "my_model"])
        result = _resolve_vars_routing(context, node)
        assert result is None

    def test_short_fqn_model_at_root(self):
        """Handles models with only 2 FQN parts (project + model, no folder)."""
        context = _make_context({
            "dbt-osmosis": {
                "models": {"staging": "_stg.yml"},
            },
        })
        node = _make_node(["project", "root_model"])
        result = _resolve_vars_routing(context, node)
        assert result is None


class TestGetYamlPathTemplateWithVars:
    """Integration tests: _get_yaml_path_template falls through to vars routing."""

    def test_vars_routing_used_when_config_extra_empty(self, yaml_context):
        """When a node has no +dbt-osmosis config, vars routing is used."""
        from dbt_osmosis.core.path_management import _get_yaml_path_template

        node = None
        for n in yaml_context.project.manifest.nodes.values():
            if n.resource_type == NodeType.Model and len(n.fqn) > 2:
                node = n
                break
        if not node:
            pytest.skip("No model with subfolder FQN found in demo_duckdb project")

        # Remove config.extra dbt-osmosis key
        old_extra = node.config.extra.pop("dbt-osmosis", None)
        old_unrendered = node.unrendered_config.pop("dbt-osmosis", None)
        old_meta = node.meta.pop("dbt-osmosis", None)

        # Set up vars routing that matches this node's folder
        folder = node.fqn[1] if len(node.fqn) > 2 else None
        if folder:
            vars_dict = yaml_context.project.runtime_cfg.vars.to_dict()
            osmosis_vars = vars_dict.setdefault("dbt-osmosis", {})
            models_routing = osmosis_vars.setdefault("models", {})
            models_routing[folder] = "_{model}.yml"

            result = _get_yaml_path_template(yaml_context, node)
            assert result == "_{model}.yml"

            # Cleanup
            del models_routing[folder]
            if not models_routing:
                del osmosis_vars["models"]
        else:
            pytest.skip("Model has no folder in FQN")

        # Restore
        if old_extra is not None:
            node.config.extra["dbt-osmosis"] = old_extra
        if old_unrendered is not None:
            node.unrendered_config["dbt-osmosis"] = old_unrendered
        if old_meta is not None:
            node.meta["dbt-osmosis"] = old_meta

    def test_config_extra_takes_priority_over_vars(self, yaml_context):
        """config.extra +dbt-osmosis still wins over vars routing."""
        from dbt_osmosis.core.path_management import _get_yaml_path_template

        node = None
        for n in yaml_context.project.manifest.nodes.values():
            if (
                n.resource_type == NodeType.Model
                and "dbt-osmosis" in n.config.extra
                and len(n.fqn) > 2
            ):
                node = n
                break
        if not node:
            pytest.skip("No model with dbt-osmosis config and subfolder FQN found")

        # Add vars routing for same folder
        folder = node.fqn[1] if len(node.fqn) > 2 else None
        if folder:
            vars_dict = yaml_context.project.runtime_cfg.vars.to_dict()
            osmosis_vars = vars_dict.setdefault("dbt-osmosis", {})
            models_routing = osmosis_vars.setdefault("models", {})
            models_routing[folder] = "vars_should_not_win.yml"

            result = _get_yaml_path_template(yaml_context, node)
            # config.extra value should win, not vars
            assert result != "vars_should_not_win.yml"

            # Cleanup
            del models_routing[folder]
        else:
            pytest.skip("Model has no folder in FQN")
