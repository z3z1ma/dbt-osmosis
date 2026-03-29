from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

_DBT_PARSE_INPUT_SUFFIXES = {".sql", ".yml", ".yaml", ".csv", ".md"}
_DBT_PARSE_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "dbt_packages",
    "logs",
    "target",
}


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
