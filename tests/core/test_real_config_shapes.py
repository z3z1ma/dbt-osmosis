from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from packaging.version import parse as parse_version

from dbt_osmosis.core.config import DbtConfiguration, create_dbt_project_context
from dbt_osmosis.core.introspection import PropertyAccessor, SettingsResolver
from dbt_osmosis.core.settings import YamlRefactorContext, YamlRefactorSettings
from dbt_osmosis.core.sync_operations import _sync_doc_section
from dbt_osmosis.core.transforms import inject_missing_columns
from tests.support import create_temp_project_copy, run_dbt_command


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_NODE_ID = "model.jaffle_shop_duckdb.real_config_shape"


@dataclass
class ParsedConfigShapeFixture:
    project_dir: Path
    context: YamlRefactorContext
    node: Any

    def close(self) -> None:
        self.context.close()

    def inject_missing_columns(self) -> dict[str, Any]:
        inject_missing_columns(self.context, self.node)
        doc_section: dict[str, Any] = {}
        _sync_doc_section(self.context, self.node, doc_section)
        return doc_section


def _installed_dbt_version() -> str:
    from dbt.version import get_installed_version

    return str(get_installed_version()).lstrip("=")


def _skip_if_column_config_shape_is_unsupported() -> str:
    dbt_version = _installed_dbt_version()
    print(f"dbt-core installed version: {dbt_version}")
    if parse_version(dbt_version) < parse_version("1.10.0"):
        pytest.skip(f"dbt-core {dbt_version} does not expose dbt 1.10 column config shapes")
    return dbt_version


def _write_real_config_shape_fixture(project_dir: Path) -> None:
    model_dir = project_dir / "models" / "c10_config_shapes"
    model_dir.mkdir(parents=True, exist_ok=True)

    (model_dir / "real_config_shape.sql").write_text(
        """
{{ config(
    meta={"sql_config_meta": "from_config_macro"},
    dbt_osmosis_skip_add_columns=false,
) }}

select
    1::integer as configured_col,
    2::integer as warehouse_only_col
""".lstrip(),
    )
    (model_dir / "real_config_shape.yml").write_text(
        """
version: 2

models:
  - name: real_config_shape
    description: Real parsed dbt 1.10 config-shape fixture
    config:
      meta:
        dbt-osmosis-output-to-lower: true
      dbt-osmosis-skip-add-tags: true
    columns:
      - name: configured_col
        description: Configured column from real dbt parse
        data_type: integer
        config:
          meta:
            dbt-osmosis-string-length: true
          tags:
            - configured-column-tag
""".lstrip(),
    )


def _parsed_config_shape_fixture(
    tmp_path: Path,
    *,
    build_database: bool = False,
) -> ParsedConfigShapeFixture:
    _skip_if_column_config_shape_is_unsupported()
    project_dir = create_temp_project_copy(REPO_ROOT / "demo_duckdb", tmp_path)
    _write_real_config_shape_fixture(project_dir)

    base_args = [
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
        "--target",
        "test",
    ]
    run_dbt_command(["parse", *base_args])
    if build_database:
        run_dbt_command(["run", "--select", "real_config_shape", *base_args])

    project_context = create_dbt_project_context(
        DbtConfiguration(
            project_dir=str(project_dir),
            profiles_dir=str(project_dir),
            target="test",
        ),
    )
    context = YamlRefactorContext(
        project_context,
        settings=YamlRefactorSettings(
            dry_run=True,
            skip_add_data_types=False,
        ),
    )
    return ParsedConfigShapeFixture(
        project_dir=project_dir,
        context=context,
        node=context.manifest.nodes[FIXTURE_NODE_ID],
    )


def test_real_dbt_parse_exposes_dbt_110_config_shapes(tmp_path: Path) -> None:
    fixture = _parsed_config_shape_fixture(tmp_path)
    try:
        node = fixture.node

        assert node.config.meta["dbt-osmosis-output-to-lower"] is True
        assert node.config.meta["sql_config_meta"] == "from_config_macro"
        assert node.config.extra["dbt-osmosis-skip-add-tags"] is True
        assert node.config.extra["dbt_osmosis_skip_add_columns"] is False
        assert node.unrendered_config["dbt_osmosis_skip_add_columns"] is False

        column = node.columns["configured_col"]
        assert column.config.meta["dbt-osmosis-string-length"] is True
        assert "configured-column-tag" in column.config.tags

        resolver = SettingsResolver()
        assert resolver.resolve("string-length", node, column_name="configured_col", fallback=False)

        accessor = PropertyAccessor(fixture.context)
        assert accessor.get_meta(node, column_name="configured_col") == {
            "dbt-osmosis-string-length": True,
        }
        assert accessor.get("tags", node, column_name="configured_col") == [
            "configured-column-tag",
        ]
    finally:
        fixture.close()


def test_real_dbt_missing_column_injection_keeps_syncable_config(tmp_path: Path) -> None:
    fixture = _parsed_config_shape_fixture(tmp_path, build_database=True)
    try:
        assert "warehouse_only_col" not in fixture.node.columns

        injected_doc_section = fixture.inject_missing_columns()

        injected_column = fixture.node.columns["warehouse_only_col"]
        assert hasattr(injected_column, "config")
        assert injected_column.config is not None
        assert injected_column.to_dict(omit_none=True)["name"] == "warehouse_only_col"

        columns_by_name = {column["name"]: column for column in injected_doc_section["columns"]}
        assert columns_by_name["configured_col"]["config"] == {
            "meta": {"dbt-osmosis-string-length": True},
            "tags": ["configured-column-tag"],
        }
        assert columns_by_name["warehouse_only_col"] == {
            "name": "warehouse_only_col",
            "data_type": "integer",
        }
    finally:
        fixture.close()
