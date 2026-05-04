from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ruamel.yaml
from click.testing import CliRunner

from dbt_osmosis.cli.main import cli


def _mock_project(project_root: Path):
    return SimpleNamespace(
        manifest=SimpleNamespace(nodes={}, sources={}),
        runtime_cfg=SimpleNamespace(project_root=project_root),
        config=SimpleNamespace(project_dir=str(project_root)),
    )


def _mock_model_spec() -> dict[str, object]:
    return {
        "model_name": "stg_safe_model",
        "description": "New generated description",
        "materialized": "view",
        "sql": "select 1 as id",
        "columns": [{"name": "id", "description": "Generated id"}],
    }


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


def test_generate_model_refuses_existing_schema_without_overwrite(tmp_path: Path):
    """Existing schema YAML should fail closed unless overwrite is explicit."""
    runner = CliRunner()
    schema_path = tmp_path / "models" / "stg_safe_model.yml"
    output_path = tmp_path / "models" / "stg_safe_model.sql"
    schema_path.parent.mkdir(parents=True)
    original_yaml = "version: 2\nmodels:\n  - name: existing_model\n"
    schema_path.write_text(original_yaml, encoding="utf-8")

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=_mock_model_spec()
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--output-path",
                    str(output_path),
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert schema_path.read_text(encoding="utf-8") == original_yaml


def test_generate_model_overwrite_preserves_unmanaged_top_level_sections(tmp_path: Path):
    """Explicit overwrite should preserve YAML sections outside dbt-osmosis ownership."""
    runner = CliRunner()
    schema_path = tmp_path / "models" / "stg_safe_model.yml"
    output_path = tmp_path / "models" / "stg_safe_model.sql"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        "version: 2\n"
        "models:\n"
        "  - name: old_model\n"
        "semantic_models:\n"
        "  - name: preserved_semantic_model\n"
        "    model: ref('old_model')\n",
        encoding="utf-8",
    )

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=_mock_model_spec()
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--overwrite",
                    "--output-path",
                    str(output_path),
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code == 0, result.output
    parsed = ruamel.yaml.YAML().load(schema_path)
    assert parsed["models"][0]["name"] == "stg_safe_model"
    assert parsed["semantic_models"][0]["name"] == "preserved_semantic_model"


def test_generate_model_refuses_schema_path_outside_project_root(tmp_path: Path):
    """Generated schema YAML paths should stay inside the dbt project root."""
    runner = CliRunner()
    outside_schema_path = tmp_path.parent / f"{tmp_path.name}_outside.yml"
    output_path = tmp_path / "models" / "stg_safe_model.sql"

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=_mock_model_spec()
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--output-path",
                    str(output_path),
                    "--schema-yml",
                    str(outside_schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "outside the dbt project root" in result.output
    assert not outside_schema_path.exists()


def test_generate_model_refuses_sql_output_path_outside_project_root(tmp_path: Path):
    """Generated SQL output paths should also stay inside the dbt project root."""
    runner = CliRunner()
    outside_output_path = tmp_path.parent / f"{tmp_path.name}_outside.sql"
    schema_path = tmp_path / "models" / "stg_safe_model.yml"

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=_mock_model_spec()
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--output-path",
                    str(outside_output_path),
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "outside the dbt project root" in result.output
    assert not outside_output_path.exists()
    assert not schema_path.exists()


def test_generate_model_refuses_traversal_in_generated_model_name(tmp_path: Path):
    """Auto-derived SQL paths should validate generated model names too."""
    runner = CliRunner()
    outside_sql_path = tmp_path.parent / "outside_generated.sql"
    schema_path = tmp_path / "models" / "safe_schema.yml"
    model_spec = _mock_model_spec()
    model_spec["model_name"] = "../../outside_generated"

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch("dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=model_spec):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "outside the dbt project root" in result.output
    assert not outside_sql_path.exists()
    assert not schema_path.exists()


def test_deprecated_nl_generate_refuses_traversal_in_generated_model_name(tmp_path: Path):
    """Deprecated nl generate should use the same default SQL path guard."""
    runner = CliRunner()
    outside_sql_path = tmp_path.parent / "outside_generated.sql"
    schema_path = tmp_path / "models" / "safe_schema.yml"
    model_spec = _mock_model_spec()
    model_spec["model_name"] = "../../outside_generated"

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch("dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=model_spec):
            result = runner.invoke(
                cli,
                [
                    "nl",
                    "generate",
                    "build a safe model",
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "outside the dbt project root" in result.output
    assert not outside_sql_path.exists()
    assert not schema_path.exists()


def test_generate_model_dry_run_refuses_existing_schema_without_overwrite(tmp_path: Path):
    """Dry-run should match real-run overwrite refusal for existing schema YAML."""
    runner = CliRunner()
    schema_path = tmp_path / "models" / "stg_safe_model.yml"
    output_path = tmp_path / "models" / "stg_safe_model.sql"
    schema_path.parent.mkdir(parents=True)
    original_yaml = "version: 2\nmodels: []\n"
    schema_path.write_text(original_yaml, encoding="utf-8")

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_dbt_model_from_nl", return_value=_mock_model_spec()
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "model",
                    "build a safe model",
                    "--dry-run",
                    "--output-path",
                    str(output_path),
                    "--schema-yml",
                    str(schema_path),
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert "Planned writes:" not in result.output
    assert schema_path.read_text(encoding="utf-8") == original_yaml
    assert not output_path.exists()


def test_generate_sources_refuses_existing_yaml_without_overwrite(tmp_path: Path):
    """Generated source YAML should share the fail-closed overwrite policy."""
    runner = CliRunner()
    yaml_path = tmp_path / "models" / "sources" / "raw.yml"
    yaml_path.parent.mkdir(parents=True)
    original_yaml = "version: 2\nsources: []\n"
    yaml_path.write_text(original_yaml, encoding="utf-8")
    mock_result = mock.Mock(
        yaml_content="version: 2\nsources:\n  - name: raw\n",
        yaml_path=yaml_path,
    )

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
        with mock.patch(
            "dbt_osmosis.cli.main.generate_sources_from_database", return_value=mock_result
        ):
            result = runner.invoke(
                cli,
                [
                    "generate",
                    "sources",
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert yaml_path.read_text(encoding="utf-8") == original_yaml


def test_generate_staging_dry_run_reports_planned_writes(tmp_path: Path):
    """Dry-run should list SQL and YAML files that a real run would write."""
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

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
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

    assert result.exit_code == 0, result.output
    assert "Planned writes:" in result.output
    assert str(sql_path) in result.output
    assert str(yaml_path) in result.output
    assert not sql_path.exists()
    assert not yaml_path.exists()


def test_generate_staging_refuses_existing_yaml_without_overwrite(tmp_path: Path):
    """Generated staging YAML should share the fail-closed overwrite policy."""
    runner = CliRunner()
    sql_path = tmp_path / "models" / "staging" / "stg_users.sql"
    yaml_path = tmp_path / "models" / "staging" / "stg_users.yml"
    yaml_path.parent.mkdir(parents=True)
    original_yaml = "version: 2\nmodels: []\n"
    yaml_path.write_text(original_yaml, encoding="utf-8")
    mock_result = mock.Mock(
        staging_name="stg_users",
        sql_content="select 1 as user_id",
        yaml_content="version: 2\nmodels:\n  - name: stg_users\n",
        sql_path=sql_path,
        yaml_path=yaml_path,
    )

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
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
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "--overwrite" in result.output
    assert yaml_path.read_text(encoding="utf-8") == original_yaml
    assert not sql_path.exists()


def test_generate_staging_refuses_sql_path_outside_project_root_without_yaml(
    tmp_path: Path,
):
    """Staging SQL path validation should not depend on YAML content being present."""
    runner = CliRunner()
    outside_sql_path = tmp_path.parent / f"{tmp_path.name}_outside.sql"
    mock_result = mock.Mock(
        staging_name="stg_users",
        sql_content="select 1 as user_id",
        yaml_content="",
        sql_path=outside_sql_path,
        yaml_path=None,
    )

    with mock.patch(
        "dbt_osmosis.cli.main.create_dbt_project_context", return_value=_mock_project(tmp_path)
    ):
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
                    "--project-dir",
                    str(tmp_path),
                    "--profiles-dir",
                    str(tmp_path),
                ],
            )

    assert result.exit_code != 0
    assert "outside the dbt project root" in result.output
    assert not outside_sql_path.exists()


if __name__ == "__main__":
    test_generate_group_shows_help()
