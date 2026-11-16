"""Real CLI evidence for residual #443 and #452 symptoms."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ruamel.yaml import YAML


def _run(project: Path, module: str, *args: str, success: bool = True) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *args,
            "--project-dir",
            str(project),
            "--profiles-dir",
            str(project),
        ],
        env=dict(os.environ, DBT_SEND_ANONYMOUS_USAGE_STATS="false"),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == success, output
    return output


def _snapshot(*roots: Path) -> dict:
    return {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for root in roots for p in root.rglob("*.yml")
    }


def _profile(project: Path) -> None:
    (project / "profiles.yml").write_text(
        "analytics:\n  target: test\n  outputs:\n    test:\n"
        f"      type: duckdb\n      path: {project / 'warehouse.duckdb'}\n      threads: 1\n"
    )


@pytest.mark.parametrize("phantom", ["absent", "empty", "populated"])
def test_introspection_enabled_cross_package_dry_run(tmp_path, phantom):
    project, package = tmp_path / "analytics", tmp_path / "patch_owner"
    for root in (project, package):
        (root / "seeds").mkdir(parents=True)
        (root / "seeds" / "exceptions.csv").write_text("id\n1\n")
    (package / "dbt_project.yml").write_text(
        "name: patch_owner\nversion: '1.0'\nconfig-version: 2\n"
    )
    patch = (
        "version: 2\nseeds:\n  - name: exceptions\n"
        "    description: Package-owned documentation\n"
        "    columns:\n      - name: id\n        description: Package-owned column\n"
    )
    (package / "seeds" / "seeds.yml").write_text(patch)
    (project / "dbt_project.yml").write_text(
        "name: analytics\nversion: '1.0'\nconfig-version: 2\nprofile: analytics\n"
        "seeds:\n  patch_owner:\n    exceptions:\n      +enabled: false\n"
        "vars:\n  dbt-osmosis:\n    seeds: schema.yml\n"
    )
    _profile(project)
    (project / "packages.yml").write_text("packages:\n  - local: ../patch_owner\n")
    _run(project, "dbt.cli.main", "deps")
    _run(project, "dbt.cli.main", "seed")
    manifest = json.loads((project / "target" / "manifest.json").read_text())
    assert manifest["nodes"]["seed.analytics.exceptions"]["patch_path"] == (
        "patch_owner://seeds/seeds.yml"
    )
    if phantom != "absent":
        (project / "seeds" / "seeds.yml").write_text(
            patch if phantom == "populated" else "version: 2\nseeds: []\n"
        )
    before = _snapshot(project, package)
    output = _run(
        project, "dbt_osmosis", "yaml", "refactor", "--dry-run", success=phantom != "populated"
    )
    assert _snapshot(project, package) == before
    assert not (project / "seeds" / "schema.yml").exists()
    if phantom == "populated":
        assert "two schema.yml entries" in output
    else:
        _run(project, "dbt.cli.main", "parse", "--no-partial-parse")


@pytest.mark.parametrize("unrendered", [False, True])
def test_cli_diamond_inherits_after_real_database_introspection(tmp_path, unrendered):
    project = tmp_path / "analytics"
    models = project / "models"
    models.mkdir(parents=True)
    (project / "dbt_project.yml").write_text(
        "name: analytics\nversion: '1.0'\nconfig-version: 2\nprofile: analytics\n"
        "models:\n  analytics:\n    +dbt-osmosis: schema.yml\n"
    )
    _profile(project)
    (models / "lut_classes.sql").write_text("select 1 as class_id\n")
    for year in (2020, 2021):
        (models / f"int_historical_data_{year}.sql").write_text(
            "select l.class_id from {{ ref('lut_classes') }} l "
            "join (select 1 as class_id) facts on l.class_id = facts.class_id\n"
        )
    (models / "int_big_table.sql").write_text(
        "select * from {{ ref('int_historical_data_2020') }} union all "
        "select * from {{ ref('int_historical_data_2021') }}\n"
    )
    (models / "docs.md").write_text(
        "{% docs class_documentation %}Shared class documentation{% enddocs %}\n"
    )
    (models / "schema.yml").write_text(
        "version: 2\nmodels:\n  - name: lut_classes\n    columns:\n"
        "      - name: class_id\n        description: \"{{ doc('class_documentation') }}\"\n"
    )
    _run(project, "dbt.cli.main", "run")
    before = _snapshot(project)
    dry_output = _run(project, "dbt_osmosis", "yaml", "refactor", "--dry-run", "--auto-apply")
    assert _snapshot(project) == before
    assert "Circular dependency detected" not in dry_output
    args = ["yaml", "refactor", "--auto-apply"]
    if unrendered:
        args.append("--use-unrendered-descriptions")
    output = _run(project, "dbt_osmosis", *args)
    assert "Circular dependency detected" not in output
    data = YAML(typ="safe").load(models / "schema.yml")
    nodes = {node["name"]: node for node in data["models"]}
    expected = "{{ doc('class_documentation') }}" if unrendered else "Shared class documentation"
    for name in ("int_historical_data_2020", "int_historical_data_2021", "int_big_table"):
        columns = {col["name"]: col for col in nodes[name]["columns"]}
        assert columns["class_id"]["description"] == expected
    _run(project, "dbt.cli.main", "parse", "--no-partial-parse")
