# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

import logging
from pathlib import Path
from unittest import mock

import pytest

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.restructuring import (
    apply_restructure_plan,
    draft_restructure_delta_plan,
    pretty_print_plan,
    RestructureOperation,
    RestructureDeltaPlan,
)
from dbt_osmosis.core.path_management import create_missing_source_yamls


@pytest.fixture(scope="module")
def yaml_context() -> YamlRefactorContext:
    """
    Creates a YamlRefactorContext for the real 'demo_duckdb' project.
    """
    cfg = DbtConfiguration(project_dir="demo_duckdb", profiles_dir="demo_duckdb")
    cfg.vars = {"dbt-osmosis": {}}

    project_context = create_dbt_project_context(cfg)
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,
            use_unrendered_descriptions=True,
        ),
    )
    return context


@pytest.fixture(scope="function")
def fresh_caches():
    """
    Patches the internal caches so each test starts with a fresh state.
    """
    with (
        mock.patch("dbt_osmosis.core.schema.reader._YAML_BUFFER_CACHE", {}),
    ):
        yield


def test_create_missing_source_yamls(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Creates missing source YAML files if any are declared in dbt-osmosis sources
    but do not exist in the manifest. Typically, might be none in your project.
    """
    create_missing_source_yamls(yaml_context)


def test_draft_restructure_delta_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Ensures we can generate a restructure plan for real models and sources.
    Usually, this plan might be empty if everything lines up already.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    assert plan is not None


def test_apply_restructure_plan(yaml_context: YamlRefactorContext, fresh_caches):
    """
    Applies the restructure plan for the real project (in dry_run mode).
    Should not raise errors even if the plan is empty or small.
    """
    plan = draft_restructure_delta_plan(yaml_context)
    apply_restructure_plan(yaml_context, plan, confirm=False)


def test_pretty_print_plan(caplog):
    """
    Test pretty_print_plan logs the correct output for each operation.
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "my_model"}]},
            ),
            RestructureOperation(
                file_path=Path("sources/another_file.yml"),
                content={"sources": [{"name": "my_source"}]},
                superseded_paths={Path("old_file.yml"): []},
            ),
        ]
    )
    test_logger = logging.getLogger("test_logger")
    with mock.patch("dbt_osmosis.core.logger.LOGGER", test_logger):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            pretty_print_plan(plan)
    logs = caplog.text
    assert "Restructure plan includes => 2 operations" in logs
    assert "CREATE or MERGE => models/some_file.yml" in logs
    assert "['old_file.yml'] -> sources/another_file.yml" in logs


def test_apply_restructure_plan_confirm_prompt(
    yaml_context: YamlRefactorContext, fresh_caches, capsys
):
    """
    We test apply_restructure_plan with confirm=True, mocking input to 'n' to skip it.
    This ensures we handle user input logic.
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/some_file.yml"),
                content={"models": [{"name": "m1"}]},
            )
        ]
    )

    with mock.patch("builtins.input", side_effect=["n"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        assert "Skipping restructure plan." in captured.err


def test_apply_restructure_plan_confirm_yes(
    yaml_context: YamlRefactorContext, fresh_caches, capsys
):
    """
    Same as above, but we input 'y' so it actually proceeds with the plan.
    (No real writing occurs due to dry_run=True).
    """
    plan = RestructureDeltaPlan(
        operations=[
            RestructureOperation(
                file_path=Path("models/whatever.yml"),
                content={"models": [{"name": "m2"}]},
            )
        ]
    )

    with mock.patch("builtins.input", side_effect=["y"]):
        apply_restructure_plan(yaml_context, plan, confirm=True)
        captured = capsys.readouterr()
        # Check for the key message parts that appear in the log output
        assert "Committing all restructure changes" in captured.err
        assert "reloading manifest" in captured.err
