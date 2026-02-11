# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false
"""Tests for the external formatter integration module."""

from __future__ import annotations

from pathlib import Path
from unittest import mock


from dbt_osmosis.core.formatting import run_external_formatter


class TestRunExternalFormatter:
    """Test suite for run_external_formatter function."""

    def test_success(self):
        """Formatter runs successfully with correct command and file paths."""
        files = [Path("models/staging/orders.yml"), Path("models/marts/customers.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=b"Formatted 2 files\n",
                stderr=b"",
            )

            result = run_external_formatter("prettier --write", files, cwd=Path("/project"))

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            cmd = call_args[0][0]
            assert cmd == [
                "prettier",
                "--write",
                "models/staging/orders.yml",
                "models/marts/customers.yml",
            ]
            assert call_args[1]["cwd"] == "/project"
            assert call_args[1]["check"] is False
            assert call_args[1]["capture_output"] is True

    def test_no_files_is_noop(self):
        """No-op when no files are provided."""
        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            result = run_external_formatter("prettier --write", [], cwd=Path("/project"))

            assert result is True
            mock_run.assert_not_called()

    def test_empty_iterable_is_noop(self):
        """No-op when an empty iterable is provided."""
        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            result = run_external_formatter("prettier --write", iter([]), cwd=Path("/project"))

            assert result is True
            mock_run.assert_not_called()

    def test_failure_is_nonfatal(self):
        """Formatter failure (non-zero exit code) does not raise an exception."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(
                returncode=2,
                stdout=b"",
                stderr=b"Error: no matching files\n",
            )

            result = run_external_formatter("prettier --write", files, cwd=Path("/project"))

            assert result is False
            # No exception raised

    def test_command_not_found_is_nonfatal(self):
        """Missing formatter command does not raise an exception."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("No such file or directory: 'nonexistent'")

            result = run_external_formatter("nonexistent --write", files, cwd=Path("/project"))

            assert result is False

    def test_os_error_is_nonfatal(self):
        """OS error during execution does not raise an exception."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            result = run_external_formatter("formatter", files, cwd=Path("/project"))

            assert result is False

    def test_command_parsing_complex(self):
        """Complex command strings are parsed correctly via shlex.split."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter(
                "prettier --write --tab-width 2 --single-quote",
                files,
                cwd=Path("/project"),
            )

            cmd = mock_run.call_args[0][0]
            assert cmd == [
                "prettier",
                "--write",
                "--tab-width",
                "2",
                "--single-quote",
                "models/a.yml",
            ]

    def test_yamlfmt_command(self):
        """yamlfmt formatter is invoked correctly (no special flags needed)."""
        files = [Path("models/a.yml"), Path("models/b.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter("yamlfmt", files, cwd=Path("/project"))

            cmd = mock_run.call_args[0][0]
            assert cmd == ["yamlfmt", "models/a.yml", "models/b.yml"]

    def test_yq_inplace_command(self):
        """yq in-place identity command is parsed correctly."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter("yq -i '.'", files, cwd=Path("/project"))

            cmd = mock_run.call_args[0][0]
            assert cmd == ["yq", "-i", ".", "models/a.yml"]

    def test_frozenset_input(self):
        """frozenset of Paths (as returned by context.written_files) works correctly."""
        files = frozenset({Path("models/a.yml"), Path("models/b.yml")})

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            result = run_external_formatter("prettier --write", files, cwd=Path("/project"))

            assert result is True
            cmd = mock_run.call_args[0][0]
            # First two elements are the command, rest are files (order may vary for frozenset)
            assert cmd[0] == "prettier"
            assert cmd[1] == "--write"
            assert set(cmd[2:]) == {"models/a.yml", "models/b.yml"}

    def test_invalid_shlex_command(self):
        """Malformed command string (unclosed quotes) is handled gracefully."""
        files = [Path("models/a.yml")]

        result = run_external_formatter("prettier --write '", files, cwd=Path("/project"))

        assert result is False

    def test_timeout_is_nonfatal(self):
        """Formatter timeout does not raise an exception."""
        import subprocess

        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["prettier"], timeout=120)

            result = run_external_formatter(
                "prettier --write", files, cwd=Path("/project"), timeout=120
            )

            assert result is False

    def test_timeout_passed_to_subprocess(self):
        """Timeout parameter is forwarded to subprocess.run."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter("prettier --write", files, cwd=Path("/project"), timeout=60)

            assert mock_run.call_args[1]["timeout"] == 60

    def test_timeout_zero_disables(self):
        """Setting timeout=0 passes None to subprocess (no timeout)."""
        files = [Path("models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter("prettier --write", files, cwd=Path("/project"), timeout=0)

            assert mock_run.call_args[1]["timeout"] is None

    def test_paths_converted_to_strings(self):
        """Path objects are properly converted to string arguments."""
        files = [Path("/absolute/path/models/a.yml")]

        with mock.patch("dbt_osmosis.core.formatting.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")

            run_external_formatter("fmt", files, cwd=Path("/project"))

            cmd = mock_run.call_args[0][0]
            assert cmd == ["fmt", "/absolute/path/models/a.yml"]
            # Verify it's a string, not a Path
            assert isinstance(cmd[1], str)
