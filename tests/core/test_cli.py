# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from unittest import mock

import pytest
from click.testing import CliRunner

from dbt_osmosis.cli.main import cli
from dbt_osmosis.core.settings import YamlRefactorContext


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    """Create a Click CliRunner for testing CLI commands."""
    return CliRunner()


def test_cli_group(runner: CliRunner) -> None:
    """Test that the main CLI group is accessible and shows help."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "dbt-osmosis" in result.output
    assert "yaml" in result.output
    assert "sql" in result.output
    assert "test" in result.output
    assert "workbench" in result.output


def test_yaml_group(runner: CliRunner) -> None:
    """Test that the yaml command group is accessible."""
    result = runner.invoke(cli, ["yaml", "--help"])
    assert result.exit_code == 0
    assert "Manage, document, and organize dbt YAML files" in result.output
    assert "refactor" in result.output
    assert "organize" in result.output
    assert "document" in result.output


def test_sql_group(runner: CliRunner) -> None:
    """Test that the sql command group is accessible."""
    result = runner.invoke(cli, ["sql", "--help"])
    assert result.exit_code == 0
    assert "Execute and compile dbt SQL statements" in result.output
    assert "run" in result.output
    assert "compile" in result.output


def test_yaml_refactor_help(runner: CliRunner) -> None:
    """Test that the yaml refactor command shows help with all options."""
    result = runner.invoke(cli, ["yaml", "refactor", "--help"])
    assert result.exit_code == 0
    assert "--project-dir" in result.output
    assert "--profiles-dir" in result.output
    assert "--dry-run" in result.output
    assert "--check" in result.output
    assert "--synthesize" in result.output
    assert "--fusion-compat" in result.output
    assert "discovered project root" in result.output


def test_fusion_compat_flag_in_yaml_commands(runner: CliRunner) -> None:
    """Test that --fusion-compat and --no-fusion-compat flags appear in yaml commands."""
    for cmd in ["refactor", "organize", "document"]:
        result = runner.invoke(cli, ["yaml", cmd, "--help"])
        assert result.exit_code == 0
        assert "--fusion-compat" in result.output, (
            f"--fusion-compat flag missing from yaml {cmd} command"
        )


def test_yaml_organize_help(runner: CliRunner) -> None:
    """Test that the yaml organize command shows help."""
    result = runner.invoke(cli, ["yaml", "organize", "--help"])
    assert result.exit_code == 0
    assert "--project-dir" in result.output
    assert "--auto-apply" in result.output


def test_yaml_document_help(runner: CliRunner) -> None:
    """Test that the yaml document command shows help."""
    result = runner.invoke(cli, ["yaml", "document", "--help"])
    assert result.exit_code == 0
    assert "--project-dir" in result.output
    assert "--synthesize" in result.output
    assert "--use-unrendered-descriptions" in result.output


def test_sql_run_help(runner: CliRunner) -> None:
    """Test that the sql run command shows help."""
    result = runner.invoke(cli, ["sql", "run", "--help"])
    assert result.exit_code == 0
    assert "SQL" in result.output


def test_test_group(runner: CliRunner) -> None:
    """Test that the test command group is accessible."""
    result = runner.invoke(cli, ["test", "--help"])
    assert result.exit_code == 0
    assert "Suggest and generate dbt tests" in result.output
    assert "suggest" in result.output


def test_test_suggest_help(runner: CliRunner) -> None:
    """Test that the test suggest command shows help."""
    result = runner.invoke(cli, ["test", "suggest", "--help"])
    assert result.exit_code == 0
    assert "--project-dir" in result.output
    assert "--use-ai" in result.output
    assert "--pattern-only" in result.output
    assert "--format" in result.output


def test_sql_compile_help(runner: CliRunner) -> None:
    """Test that the sql compile command shows help."""
    result = runner.invoke(cli, ["sql", "compile", "--help"])
    assert result.exit_code == 0
    assert "SQL" in result.output


def test_sql_compile_plain_sql_outputs_sql(
    runner: CliRunner, yaml_context: YamlRefactorContext
) -> None:
    """Plain SQL compile should print executable SQL, not None."""
    project_dir = str(yaml_context.project.runtime_cfg.project_root)

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context",
        return_value=yaml_context.project,
    ):
        result = runner.invoke(
            cli,
            [
                "sql",
                "compile",
                "--project-dir",
                project_dir,
                "--profiles-dir",
                project_dir,
                "select 1",
            ],
        )

    assert result.exit_code == 0
    assert result.output.strip().splitlines()[-1] == "select 1"


def test_workbench_help(runner: CliRunner) -> None:
    """Test that the workbench command shows help."""
    result = runner.invoke(cli, ["workbench", "--help"])
    assert result.exit_code == 0
    assert "discovered project root" in result.output
    assert "--host" in result.output
    assert "--port" in result.output


def test_test_llm_command(runner: CliRunner) -> None:
    """Test the test_llm command exists and is callable."""
    result = runner.invoke(cli, ["test-llm"])
    # Should not crash, should provide helpful output about missing env vars
    assert (
        result.exit_code in [0, 1]
        or "LLM_PROVIDER" in result.output
        or "openai" in result.output.lower()
    )


def test_version_option(runner: CliRunner) -> None:
    """Test that --version works."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_formatter_flag_in_yaml_commands(runner: CliRunner) -> None:
    """Test that --formatter flag appears in help for all yaml subcommands."""
    for cmd in ["refactor", "organize", "document"]:
        result = runner.invoke(cli, ["yaml", cmd, "--help"])
        assert result.exit_code == 0
        assert "--formatter" in result.output, f"--formatter flag missing from yaml {cmd} command"


def test_invalid_command(runner: CliRunner) -> None:
    """Test that invalid commands produce a helpful error."""
    result = runner.invoke(cli, ["invalid-command"])
    assert result.exit_code != 0
    assert "No such command" in result.output
