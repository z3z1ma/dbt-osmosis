from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from dbt_osmosis.cli.main import cli


def test_generate_group_shows_help():
    """Test that generate command group shows help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    print("Exit code:", result.exit_code)
    print("Output:")
    print(result.output)
    print("\n---")

    # Verify help contains all expected commands
    assert result.exit_code == 0, "Exit code should be 0"
    assert "Generate dbt artifacts: sources, staging models, and more" in result.output, (
        "Should show group description"
    )
    assert "sources" in result.output, "Should list sources command"
    assert "staging" in result.output, "Should list staging command"
    assert "model" in result.output, "Should list model command"
    assert "query" in result.output, "Should list query command"


def test_generate_staging_dry_run_does_not_write_files(tmp_path: Path):
    """Dry-run staging generation should print content without creating files."""
    runner = CliRunner()
    sql_path = tmp_path / "models" / "staging" / "stg_users.sql"
    yaml_path = tmp_path / "models" / "staging" / "stg_users.yml"
    mock_result = mock.Mock(
        staging_name="stg_users",
        sql_content="select 1 as user_id",
        yaml_content="version: 2\nmodels: []\n",
        sql_path=sql_path,
        yaml_path=yaml_path,
    )

    with mock.patch("dbt_osmosis.cli.main.create_dbt_project_context", return_value=mock.Mock()):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_staging_from_source", return_value=mock_result
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "staging",
                    "raw",
                    "users",
                    "--dry-run",
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code == 0
    assert "Generated SQL:" in result.output
    assert "select 1 as user_id" in result.output
    assert "Generated YAML:" in result.output
    assert "version: 2" in result.output
    assert not sql_path.exists()
    assert not yaml_path.exists()


if __name__ == "__main__":
    test_generate_group_shows_help()
