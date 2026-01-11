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


if __name__ == "__main__":
    test_generate_group_shows_help()
