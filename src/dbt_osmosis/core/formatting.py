"""External YAML formatter integration for dbt-osmosis.

This module provides the ability to run an external formatter command (e.g. prettier,
yamlfmt, yq) on YAML files that were written by dbt-osmosis. The formatter is invoked
once with all written file paths as arguments after osmosis completes its operations.

The formatter command is parsed with shlex.split for safe shell argument handling,
and file paths are appended as positional arguments. This makes the module generic
enough to work with any CLI formatter that accepts file paths as arguments.

Supported formatters (non-exhaustive):
    - prettier --write
    - yamlfmt
    - yq -i '.' (yq in-place identity for normalization)
    - ruamel-yaml-cmd format
    - Any command that accepts file paths as trailing arguments

Example:
    >>> from dbt_osmosis.core.formatting import run_external_formatter
    >>> run_external_formatter("prettier --write", [Path("models/a.yml")], cwd=Path("."))
"""

from __future__ import annotations

import shlex
import subprocess
import typing as t
from pathlib import Path

from dbt_osmosis.core import logger

__all__ = ["run_external_formatter"]


def run_external_formatter(
    formatter_cmd: str,
    files: t.Iterable[Path],
    cwd: Path,
) -> bool:
    """Run an external formatter command on the given YAML files.

    The command string is split using shlex and file paths are appended as
    positional arguments. The formatter is invoked as a single subprocess call
    with all file paths at once.

    Formatter failure (non-zero exit code) is **non-fatal**: a warning is logged
    but no exception is raised. This design choice ensures that osmosis's
    already-written YAML content is preserved even if the formatter crashes or
    is misconfigured.

    Args:
        formatter_cmd: The formatter command string (e.g. "prettier --write").
            Parsed with shlex.split for safe handling of quoted arguments.
        files: An iterable of Path objects for YAML files to format.
        cwd: The working directory for the subprocess.

    Returns:
        True if the formatter ran successfully (exit code 0), False otherwise.
        Also returns True if no files were provided (no-op).

    """
    file_list = [str(f) for f in files]
    if not file_list:
        logger.debug(":white_check_mark: No files to format, skipping external formatter.")
        return True

    try:
        cmd = shlex.split(formatter_cmd) + file_list
    except ValueError as e:
        logger.warning(
            ":warning: Failed to parse formatter command %r: %s",
            formatter_cmd,
            e,
        )
        return False

    logger.info(
        ":art: Running external formatter: %s (%d file(s))",
        formatter_cmd,
        len(file_list),
    )
    logger.debug(":arrow_right: Full command: %s", cmd)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        logger.warning(
            ":warning: Formatter command not found: %r. "
            "Ensure the formatter is installed and available on PATH.",
            cmd[0],
        )
        return False
    except OSError as e:
        logger.warning(
            ":warning: Failed to execute formatter command: %s",
            e,
        )
        return False

    if result.returncode != 0:
        stderr_output = result.stderr.decode("utf-8", errors="replace").strip()
        logger.warning(
            ":warning: External formatter exited with code %d. "
            "YAML files were already written by osmosis and are valid. "
            "Formatter stderr:\n%s",
            result.returncode,
            stderr_output or "(no stderr output)",
        )
        return False

    stdout_output = result.stdout.decode("utf-8", errors="replace").strip()
    if stdout_output:
        logger.debug(":art: Formatter output:\n%s", stdout_output)

    logger.info(
        ":white_check_mark: External formatter completed successfully (%d file(s)).",
        len(file_list),
    )
    return True
