"""Exercise package-owned seed patches through real dbt parsing and the CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run(project: Path, module: str, *args: str) -> None:
    env = dict(os.environ, DBT_SEND_ANONYMOUS_USAGE_STATS="false")
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
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("existing_yaml", [False, True])
def test_refactor_preserves_package_owned_seed_patch(tmp_path, dry_run, existing_yaml):
    project = tmp_path / "analytics"
    package = project / "dbt_packages" / "patch_owner"
    for root in (project, package):
        (root / "seeds").mkdir(parents=True)
        (root / "seeds" / "exceptions.csv").write_text("id\n1\n")
    (package / "dbt_project.yml").write_text(
        "name: patch_owner\nversion: '1.0'\nconfig-version: 2\n"
    )
    (package / "seeds" / "seeds.yml").write_text(
        "version: 2\nseeds:\n  - name: exceptions\n"
        "    description: Package-owned documentation\n"
        "    columns:\n      - name: id\n        description: Package-owned column\n"
    )
    (project / "dbt_project.yml").write_text(
        "name: analytics\nversion: '1.0'\nconfig-version: 2\nprofile: analytics\n"
        "seeds:\n  patch_owner:\n    exceptions:\n      +enabled: false\n"
        "vars:\n  dbt-osmosis:\n    seeds: schema.yml\n"
    )
    (project / "profiles.yml").write_text(
        "analytics:\n  target: test\n  outputs:\n    test:\n"
        "      type: duckdb\n      path: ':memory:'\n      threads: 1\n"
    )
    if existing_yaml:
        (project / "seeds" / "seeds.yml").write_text("version: 2\nseeds: []\n")

    _run(project, "dbt.cli.main", "parse", "--no-partial-parse")
    manifest = json.loads((project / "target" / "manifest.json").read_text())
    seed = manifest["nodes"]["seed.analytics.exceptions"]
    assert seed["patch_path"] == "patch_owner://seeds/seeds.yml"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")}

    args = ["yaml", "refactor", "--auto-apply", "--disable-introspection"]
    if dry_run:
        args.append("--dry-run")
    _run(project, "dbt_osmosis", *args)

    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in project.rglob("*.yml")} == before
    _run(project, "dbt.cli.main", "parse", "--no-partial-parse")
