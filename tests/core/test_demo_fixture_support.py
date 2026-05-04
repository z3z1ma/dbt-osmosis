# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false
"""Regression tests for demo fixture compatibility with the declared dbt support line."""

from __future__ import annotations

import os
from types import SimpleNamespace
from pathlib import Path

import pytest
import ruamel.yaml

import tests.conftest as test_conftest
import tests.core.conftest as core_conftest

from tests.conftest import _run_dbt_commands
from tests.support import manifest_requires_refresh


DEMO_PROJECT_DIR = Path("demo_duckdb")
GENERIC_TESTS_REQUIRING_ARGUMENTS = {"accepted_values", "relationships"}


def test_demo_generic_tests_keep_legacy_argument_shape_for_oldest_supported_dbt() -> None:
    """Demo schema files must remain parseable on the oldest supported dbt-core line.

    The repository still supports dbt-core 1.8.x, whose generic-test macros do not yet
    accept the newer nested ``arguments`` shape. Until that floor is raised, keep the demo
    fixture on the legacy top-level argument layout even though newer dbt versions emit a
    deprecation warning during parse.
    """
    yaml = ruamel.yaml.YAML(typ="safe")
    offenders: list[str] = []

    for schema_path in DEMO_PROJECT_DIR.rglob("*.yml"):
        if not schema_path.is_file() or "target" in schema_path.parts:
            continue

        raw_doc = yaml.load(schema_path.read_text())
        if not isinstance(raw_doc, dict):
            continue

        for model in raw_doc.get("models", []):
            if not isinstance(model, dict):
                continue

            model_name = model.get("name", "<unknown-model>")
            for column in model.get("columns", []):
                if not isinstance(column, dict):
                    continue

                column_name = column.get("name", "<unknown-column>")
                test_entries = [
                    *(column.get("tests") or []),
                    *(column.get("data_tests") or []),
                ]
                for test_entry in test_entries:
                    if not isinstance(test_entry, dict) or len(test_entry) != 1:
                        continue

                    test_name, config = next(iter(test_entry.items()))
                    if test_name not in GENERIC_TESTS_REQUIRING_ARGUMENTS:
                        continue

                    if not isinstance(config, dict) or "arguments" in config:
                        offenders.append(
                            f"{schema_path}:{model_name}.{column_name}.{test_name}",
                        )

    assert not offenders, (
        "Demo schema generic tests must keep legacy top-level arguments while dbt-core 1.8 remains supported: "
        + ", ".join(sorted(offenders))
    )


def test_run_dbt_commands_fails_fast_on_first_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixture bootstrap must stop at the first failed dbt command."""

    class FakeRunner:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        def invoke(self, args: list[str]) -> SimpleNamespace:
            self.calls.append(args)
            return SimpleNamespace(success=False, exception=RuntimeError("seed exploded"))

    fake_runner = FakeRunner()
    monkeypatch.setattr("dbt.cli.main.dbtRunner", lambda: fake_runner)

    with pytest.raises(RuntimeError, match=r"dbt seed failed after .*seed exploded"):
        _run_dbt_commands("demo_duckdb", "demo_duckdb")

    assert fake_runner.calls == [
        [
            "seed",
            "--project-dir",
            "demo_duckdb",
            "--profiles-dir",
            "demo_duckdb",
            "--target",
            "test",
        ],
    ]


def test_postgres_fixture_branch_is_retired() -> None:
    """The test harness should not advertise an unexercised PostgreSQL fixture path."""

    assert not hasattr(test_conftest, "built_postgres_template")
    assert not hasattr(test_conftest, "postgres_yaml_context")


def test_manifest_requires_refresh_when_fixture_input_is_newer(tmp_path: Path) -> None:
    """Source-tree core tests should re-parse when demo fixture inputs change."""
    project_dir = tmp_path / "demo_project"
    models_dir = project_dir / "models"
    target_dir = project_dir / "target"
    models_dir.mkdir(parents=True)
    target_dir.mkdir()

    manifest_path = target_dir / "manifest.json"
    model_path = models_dir / "orders.sql"

    manifest_path.write_text("{}")
    model_path.write_text("select 1")

    old_manifest_time = manifest_path.stat().st_mtime - 10
    newer_model_time = manifest_path.stat().st_mtime + 10
    _ = os.utime(manifest_path, (old_manifest_time, old_manifest_time))
    _ = os.utime(model_path, (newer_model_time, newer_model_time))

    assert manifest_requires_refresh(manifest_path, project_dir) is True


def test_manifest_requires_refresh_when_manifest_is_current(tmp_path: Path) -> None:
    """Fresh manifests should not trigger redundant dbt parse work."""
    project_dir = tmp_path / "demo_project"
    models_dir = project_dir / "models"
    target_dir = project_dir / "target"
    models_dir.mkdir(parents=True)
    target_dir.mkdir()

    manifest_path = target_dir / "manifest.json"
    model_path = models_dir / "orders.sql"

    model_path.write_text("select 1")
    manifest_path.write_text("{}")

    older_model_time = model_path.stat().st_mtime - 10
    newer_manifest_time = manifest_path.stat().st_mtime + 10
    _ = os.utime(model_path, (older_model_time, older_model_time))
    _ = os.utime(manifest_path, (newer_manifest_time, newer_manifest_time))

    assert manifest_requires_refresh(manifest_path, project_dir) is False


def test_isolated_demo_manifest_parse_uses_temp_project_not_source_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Core manifest support should parse a copied project and leave source target untouched."""
    source_dir = tmp_path / "demo_duckdb"
    source_dir.mkdir()
    (source_dir / "profiles.yml").write_text(
        "\n".join([
            "jaffle_shop:",
            "  target: test",
            "  outputs:",
            "    test:",
            "      type: duckdb",
            '      path: "test.db"',
            "      threads: 1",
            "",
        ]),
    )
    (source_dir / "dbt_project.yml").write_text("name: jaffle_shop\nprofile: jaffle_shop\n")

    calls: list[list[str]] = []

    def fake_run_dbt_command(args: list[str]) -> None:
        calls.append(args)
        project_dir = Path(args[args.index("--project-dir") + 1])
        assert project_dir != source_dir
        manifest_path = project_dir / "target" / "manifest.json"
        manifest_path.parent.mkdir()
        manifest_path.write_text("{}")

    monkeypatch.setattr(core_conftest, "run_dbt_command", fake_run_dbt_command)

    manifest_path = core_conftest._build_isolated_demo_manifest(tmp_path / "work", source_dir)

    assert manifest_path == tmp_path / "work" / "demo_duckdb" / "target" / "manifest.json"
    assert manifest_path.is_file()
    assert not (source_dir / "target" / "manifest.json").exists()
    assert calls == [
        [
            "parse",
            "--project-dir",
            str(tmp_path / "work" / "demo_duckdb"),
            "--profiles-dir",
            str(tmp_path / "work" / "demo_duckdb"),
            "--target",
            "test",
        ],
    ]
