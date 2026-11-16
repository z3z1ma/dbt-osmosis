"""Exercise opt-in rename inheritance from a fresh CLI parse, without a warehouse."""

import subprocess
import sys

import pytest
from ruamel.yaml import YAML


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_compiles_rename_and_preserves_docs_reference(tmp_path, dry_run):
    yaml = YAML()
    (tmp_path / "models").mkdir()
    (tmp_path / "dbt_project.yml").write_text(
        "name: lineage_cli\nversion: '1.0'\nconfig-version: 2\nprofile: lineage_cli\n"
        "models:\n  lineage_cli:\n    +dbt-osmosis: '{model}.yml'\n"
        "    +meta:\n      dbt-osmosis-options:\n        use-sqlglot-lineage: true\n"
        "        use-unrendered-descriptions: true\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "lineage_cli:\n  target: test\n  outputs:\n    test:\n"
        "      type: duckdb\n      path: ':memory:'\n      threads: 1\n"
    )
    (tmp_path / "models" / "renamed.sql").write_text(
        "with s as (select * from {{ source('raw', 'input') }}) select original as renamed from s"
    )
    (tmp_path / "models" / "docs.md").write_text(
        "{% docs original %}Original documentation{% enddocs %}"
    )
    (tmp_path / "models" / "sources.yml").write_text(
        "version: 2\nsources:\n  - name: raw\n    schema: main\n    tables:\n"
        "      - name: input\n        columns:\n          - name: original\n"
        "            description: \"{{ doc('original') }}\"\n"
    )
    output = tmp_path / "models" / "renamed.yml"
    output.write_text(
        "version: 2\nmodels:\n  - name: renamed\n    columns:\n      - name: renamed\n"
    )
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*.yml")}
    args = [
        sys.executable,
        "-m",
        "dbt_osmosis",
        "yaml",
        "refactor",
        "--auto-apply",
        "--disable-introspection",
        "--project-dir",
        str(tmp_path),
        "--profiles-dir",
        str(tmp_path),
    ]
    if dry_run:
        args.append("--dry-run")
    result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    if dry_run:
        assert {
            p: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in tmp_path.rglob("*.yml")
            if p.is_file()
        } == before
    else:
        assert (
            yaml.load(output)["models"][0]["columns"][0]["description"] == "{{ doc('original') }}"
        )
        first = output.read_bytes()
        result = subprocess.run(args, capture_output=True, text=True, check=False, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert output.read_bytes() == first
