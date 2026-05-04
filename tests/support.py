from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import ruamel.yaml

_GENERATED_FIXTURE_DIRS = {
    ".cache",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "dbt_packages",
    "logs",
    "target",
}
_LOCAL_ARTIFACT_FILENAMES = {
    ".coverage",
    ".DS_Store",
    ".env",
    ".envrc",
    "db.sqlite3",
    "db.sqlite3-journal",
    "test.db",
}
_LOCAL_ARTIFACT_SUFFIXES = (
    ".cover",
    ".db",
    ".db.wal",
    ".duckdb",
    ".duckdb.wal",
    ".log",
    ".pyc",
    ".pyo",
)

_DBT_PARSE_INPUT_SUFFIXES = {".sql", ".yml", ".yaml", ".csv", ".md"}
_DBT_PARSE_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dbt_packages",
    "logs",
    "target",
}


def _is_generated_fixture_artifact(relative_path: Path) -> bool:
    """Return True for local artifacts that should not be copied by default."""
    if any(part in _GENERATED_FIXTURE_DIRS for part in relative_path.parts):
        return True

    name = relative_path.name
    if name in _LOCAL_ARTIFACT_FILENAMES:
        return True
    return name.endswith(_LOCAL_ARTIFACT_SUFFIXES)


def rewrite_duckdb_profile_paths(project_dir: Path) -> None:
    """Rewrite DuckDB profile paths to absolute paths inside ``project_dir``."""
    profiles_path = project_dir / "profiles.yml"
    if not profiles_path.exists():
        return

    yaml = ruamel.yaml.YAML()
    profile_data = yaml.load(profiles_path.read_text())
    if not isinstance(profile_data, dict):
        return

    changed = False
    for profile_config in profile_data.values():
        if not isinstance(profile_config, dict):
            continue
        outputs = profile_config.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for output_config in outputs.values():
            if not isinstance(output_config, dict):
                continue
            if output_config.get("type") != "duckdb" or "path" not in output_config:
                continue

            configured_path = str(output_config["path"])
            if configured_path.startswith(":"):
                continue
            database_path = Path(configured_path)
            rewritten_path = (
                project_dir / database_path.name
                if database_path.is_absolute()
                else project_dir / database_path
            ).resolve()
            if configured_path != str(rewritten_path):
                output_config["path"] = str(rewritten_path)
                changed = True

    if changed:
        with profiles_path.open("w") as f:
            yaml.dump(profile_data, f)


def create_temp_project_copy(
    source_dir: Path,
    temp_dir: Path,
    *,
    include_generated_artifacts: bool = False,
    rewrite_profiles: bool = True,
) -> Path:
    """Copy a dbt project into ``temp_dir`` with hermetic DuckDB profile paths."""
    import shutil

    source_dir = source_dir.resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    project_dir = temp_dir / source_dir.name

    def _ignore_filter(src: str, names: list[str]) -> list[str]:
        if include_generated_artifacts:
            return []

        src_path = Path(src).resolve().relative_to(source_dir)
        ignored: list[str] = []
        for name in names:
            relative_path = src_path / name
            if _is_generated_fixture_artifact(relative_path):
                ignored.append(name)
        return ignored

    shutil.copytree(source_dir, project_dir, ignore=_ignore_filter)
    if rewrite_profiles:
        rewrite_duckdb_profile_paths(project_dir)
    return project_dir


class _DbtInvocationResult(Protocol):
    success: bool
    exception: object | None
    result: object | None


def _dbt_command_label(args: Sequence[str]) -> str:
    if len(args) >= 2 and args[0] == "docs" and args[1] == "generate":
        return "docs generate"
    return args[0]


def _dbt_failure_detail(result: _DbtInvocationResult) -> str:
    exception = result.exception
    if exception is not None:
        detail = str(exception)
        if detail:
            return detail
        return repr(exception)

    invocation_result = result.result
    if invocation_result is not None:
        return repr(invocation_result)

    return "Unknown error"


def run_dbt_command(args: Sequence[str]) -> None:
    """Invoke one dbt CLI command and raise immediately on failure."""
    from dbt.cli.main import dbtRunner

    label = _dbt_command_label(args)
    start = time.time()
    result = cast(_DbtInvocationResult, cast(object, dbtRunner().invoke(list(args))))
    elapsed = time.time() - start

    if not result.success:
        raise RuntimeError(
            f"dbt {label} failed after {elapsed:.2f}s: {_dbt_failure_detail(result)}",
        )

    print(f"✓ dbt {label} completed successfully ({elapsed:.2f}s)")


def manifest_requires_refresh(manifest_path: Path, project_dir: Path) -> bool:
    """Return True when dbt parse inputs are newer than the compiled manifest."""
    if not manifest_path.exists():
        return True

    manifest_mtime = manifest_path.stat().st_mtime
    for candidate in project_dir.rglob("*"):
        if not candidate.is_file():
            continue

        relative_path = candidate.relative_to(project_dir)
        if any(part in _DBT_PARSE_EXCLUDED_DIRS for part in relative_path.parts):
            continue
        if candidate.suffix.lower() not in _DBT_PARSE_INPUT_SUFFIXES:
            continue
        if candidate.stat().st_mtime > manifest_mtime:
            return True

    return False
