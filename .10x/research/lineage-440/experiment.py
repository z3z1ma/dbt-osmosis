"""Offline research probe; never imports or changes osmosis runtime behavior."""

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.lineage import lineage

CASES = {
    "direct": "select s.old_name as renamed from {{ source('raw', 'input') }} s",
    "cte_star": "with s as (select * from {{ source('raw', 'input') }}), r as (select old_name as renamed from s) select * from r",
    "transform": "with r as (select lower(old_name) as renamed from {{ source('raw', 'input') }}) select renamed from r",
    "multiple": "select coalesce(a.old_name, b.old_name) as renamed from {{ source('raw', 'input') }} a join {{ source('raw', 'other') }} b on a.id=b.id",
    "ambiguous": "select old_name as renamed from {{ source('raw', 'input') }} a join {{ source('raw', 'other') }} b on a.id=b.id",
}
SCHEMA = {
    "probe": {"raw": {name: {"old_name": "TEXT", "id": "INT"} for name in ("input", "other")}}
}


def probe(sql, schema):
    if not sql:
        return {"status": "unavailable"}
    try:
        root = lineage(
            "renamed", sql, schema=schema, dialect="duckdb", validate_qualify_columns=True
        )
        nodes = list(root.walk())
        leaves = [n for n in nodes if not n.downstream]
        # Every projection in the lineage path must preserve a direct column.
        direct = all(
            isinstance(n.expression, exp.Table) or isinstance(n.expression.unalias(), exp.Column)
            for n in nodes
        )
        known_leaf = len(leaves) == 1 and isinstance(leaves[0].expression, exp.Table)
        return {
            "status": "candidate" if direct and known_leaf else "reject",
            "leaves": [n.name for n in leaves],
            "direct_path": direct,
        }
    except SqlglotError as exc:
        return {"status": "unresolved", "error": type(exc).__name__}


def main():
    with tempfile.TemporaryDirectory(prefix="osmosis-lineage-") as temp:
        root = Path(temp)
        (root / "models").mkdir()
        (root / "dbt_project.yml").write_text(
            "name: lineage_probe\nversion: 1.0.0\nconfig-version: 2\nprofile: lineage_probe\n"
        )
        (root / "profiles.yml").write_text(
            "lineage_probe:\n  target: test\n  outputs:\n    test:\n      type: duckdb\n      path: probe.duckdb\n      threads: 1\n"
        )
        (root / "models" / "sources.yml").write_text(
            "version: 2\nsources:\n  - name: raw\n    schema: raw\n    tables:\n      - name: input\n      - name: other\n"
        )
        for name, sql in CASES.items():
            (root / "models" / f"{name}.sql").write_text(sql)
        executable = str(Path(sys.executable).parent / "dbt")
        result = subprocess.run(
            [
                executable,
                "compile",
                "--no-introspect",
                "--project-dir",
                temp,
                "--profiles-dir",
                temp,
            ],
            cwd=temp,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
            env={**os.environ, "DBT_SEND_ANONYMOUS_USAGE_STATS": "false"},
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        manifest = json.loads((root / "target" / "manifest.json").read_text())
        compiled = {
            n["name"]: n["compiled_code"]
            for n in manifest["nodes"].values()
            if n["resource_type"] == "model"
        }
        observations = {name: probe(sql, SCHEMA) for name, sql in compiled.items()}
        observations["missing_schema_star"] = probe(compiled["cte_star"], {})
        observations["missing_sql"] = probe(None, SCHEMA)
        expected = {
            "direct": "candidate",
            "cte_star": "candidate",
            "transform": "reject",
            "multiple": "reject",
            "ambiguous": "unresolved",
            "missing_schema_star": "reject",
            "missing_sql": "unavailable",
        }
        for name, status in expected.items():
            assert observations[name]["status"] == status, (name, observations[name])
        assert observations["direct"]["leaves"] == ["s.old_name"]
        assert observations["cte_star"]["leaves"] == ["input.old_name"]
        assert observations["ambiguous"]["error"] == "OptimizeError"
        print(
            json.dumps(
                {
                    "versions": {
                        p: importlib.metadata.version(p)
                        for p in ("dbt-core", "dbt-duckdb", "sqlglot")
                    },
                    "compiled_sql": compiled,
                    "observations": observations,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
