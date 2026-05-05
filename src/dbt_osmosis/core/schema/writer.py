"""YAML file writing for dbt-osmosis.

Thread-safety:
    - _write_yaml() acquires yaml_handler_lock for the entire write operation
    - Cache invalidation is performed under _YAML_BUFFER_CACHE_LOCK
    - Multiple threads can safely write to different files concurrently
"""

import io
import os
import secrets
import stat
import threading
import typing as t
from pathlib import Path

import ruamel.yaml

from dbt_osmosis.core import logger
from dbt_osmosis.core.schema.parser import _partition_yaml_top_level_sections
from dbt_osmosis.core.schema.reader import (
    _YAML_BUFFER_CACHE,
    _YAML_BUFFER_CACHE_LOCK,
    _YAML_ORIGINAL_CACHE,
    _discard_yaml_caches,
)

__all__ = [
    "_merge_preserved_sections",
    "_write_yaml",
    "commit_yamls",
]


def _merge_preserved_sections(
    filtered_data: dict[str, t.Any], original_data: dict[str, t.Any]
) -> dict[str, t.Any]:
    """Merge preserved top-level sections from original YAML.

    When dbt-osmosis processes a YAML file, it filters out top-level sections that it
    does not manage directly. This function restores every preserved section from the
    original file so mixed schema files do not lose snapshots, exposures, anchors,
    semantic models, or any future dbt keys that dbt-osmosis still ignores.

    Args:
        filtered_data: The processed YAML data (may have models, sources, etc.)
        original_data: The original unfiltered YAML data with unmanaged top-level keys

    Returns:
        A merged dictionary containing both processed and preserved sections.

    """
    # Preserve the original top-level order so anchors defined in unmanaged
    # sections can still precede managed aliases after dbt-osmosis writes.
    merged: dict[str, t.Any] = {}
    _, preserved_sections = _partition_yaml_top_level_sections(original_data)

    for key, value in original_data.items():
        if key in filtered_data:
            merged[key] = filtered_data[key]
        elif key in preserved_sections:
            merged[key] = value
            logger.debug(f":recycle: Restoring preserved section '{key}' from original YAML")

    for key, value in filtered_data.items():
        if key not in merged:
            merged[key] = value

    return merged


def _strip_eof_blank_lines(content: bytes) -> bytes:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content
    newline = "\r\n" if "\r\n" in text else "\n"
    endswith_newline = text.endswith("\n")
    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if not lines:
        return b""
    result = newline.join(lines)
    if endswith_newline:
        result += newline
    return result.encode("utf-8")


def _write_unique_temp_file(path: Path, content: bytes) -> tuple[Path, int]:
    """Write content to a unique temp file in the target directory."""
    for _ in range(100):
        temp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
        try:
            with temp_path.open("xb") as f:
                bytes_written = f.write(content)
            if path.exists():
                temp_path.chmod(stat.S_IMODE(path.stat().st_mode))
            return temp_path, bytes_written
        except FileExistsError:
            continue
        except Exception:
            _cleanup_temp_path(temp_path)
            raise

    raise FileExistsError(f"Unable to create unique temporary file for {path}")


def _cleanup_temp_path(temp_path: Path | None) -> None:
    """Remove a temp file if this write still owns one."""
    if temp_path and temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass


def _write_yaml(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    path: Path,
    data: dict[str, t.Any],
    dry_run: bool = False,
    mutation_tracker: t.Callable[[int], None] | None = None,
    strip_eof_blank_lines: bool = False,
    written_file_tracker: t.Callable[[Path], None] | None = None,
    allow_overwrite: bool = True,
) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache.

    Thread-safety: This function is thread-safe. It acquires yaml_handler_lock
    to ensure exclusive access to the yaml handler, and _YAML_BUFFER_CACHE_LOCK
    for cache invalidation. Multiple threads can safely write to different files.

    Uses a write-validate-replace pattern to prevent data loss:
    1. Write to a unique temporary file in the target directory
    2. Validate write succeeded (file exists and non-empty)
    3. Replace original file via atomic rename
    4. If any step fails, clean up temp file and preserve original

    Note: When dry_run=True, changes are detected and mutation_tracker is called,
    but no files are written to disk. This enables --check to work with --dry-run.
    """
    logger.debug(":page_with_curl: Attempting to write YAML to => %s", path)
    with yaml_handler_lock:
        # Merge preserved sections from original YAML (semantic_models, macros, etc.)
        with _YAML_BUFFER_CACHE_LOCK:
            if path in _YAML_ORIGINAL_CACHE:
                original_content = _YAML_ORIGINAL_CACHE[path]
                data = _merge_preserved_sections(data, original_content)

        if not dry_run:
            if not allow_overwrite and path.exists():
                raise FileExistsError(f"Refusing to overwrite existing YAML file: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)

        original = path.read_bytes() if path.is_file() else b""
        # Use context manager to ensure BytesIO is properly closed
        with io.BytesIO() as staging:
            yaml_handler.dump(data, staging)
            modified = staging.getvalue()
            if strip_eof_blank_lines:
                modified = _strip_eof_blank_lines(modified)
            if modified != original:
                if dry_run:
                    logger.info(":eyes: Would write changes to => %s (dry-run)", path)
                else:
                    logger.info(":writing_hand: Writing changes to => %s", path)

                    # Write to a unique temporary file first for safety
                    temp_path: Path | None = None
                    try:
                        temp_path, bytes_written = _write_unique_temp_file(path, modified)

                        # Validate write succeeded
                        if not temp_path.exists():
                            raise OSError(f"Temporary file not created: {temp_path}")
                        if temp_path.stat().st_size == 0 and len(modified) > 0:
                            raise OSError(f"Temporary file is empty: {temp_path}")
                        if bytes_written != len(modified):
                            raise OSError(
                                f"Write incomplete: expected {len(modified)} bytes, wrote {bytes_written}",
                            )

                        # Atomic replace: only delete original after successful temp write
                        if allow_overwrite:
                            _replace_atomically(temp_path, path)
                        else:
                            try:
                                os.link(temp_path, path)
                            except FileExistsError:
                                raise FileExistsError(
                                    f"Refusing to overwrite existing YAML file: {path}"
                                ) from None
                            finally:
                                _cleanup_temp_path(temp_path)
                            temp_path = None

                        # Clear cache entry only after successful write
                        with _YAML_BUFFER_CACHE_LOCK:
                            _discard_yaml_caches(path)

                        if written_file_tracker:
                            written_file_tracker(path)

                    except Exception as e:
                        _cleanup_temp_path(temp_path)
                        # Re-raise to signal failure
                        logger.error(":boom: Failed to write YAML to => %s: %s", path, e)
                        raise

                # Track mutation regardless of dry_run (enables --check with --dry-run)
                if mutation_tracker:
                    mutation_tracker(1)

            else:
                logger.debug(":white_check_mark: Skipping write => %s (no changes)", path)

            # Clear cache entries after truthful disk outcomes. Dry-run writes
            # always compare against disk but must not pin process-global YAML
            # state for later reads in the same process.
            if dry_run or modified == original:
                with _YAML_BUFFER_CACHE_LOCK:
                    _discard_yaml_caches(path)


def _replace_atomically(temp_path: Path, target_path: Path) -> None:
    """Atomically replace target_path with temp_path.

    This ensures that the target file is never in a partially-written state.
    Works across platforms using the safest available method.
    """
    try:
        # Try atomic rename (works on Unix and Windows with Python 3.3+)
        temp_path.replace(target_path)
    except OSError:
        # Fallback for older systems or special filesystems
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)


def commit_yamls(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    dry_run: bool = False,
    mutation_tracker: t.Callable[[int], None] | None = None,
    strip_eof_blank_lines: bool = False,
    written_file_tracker: t.Callable[[Path], None] | None = None,
) -> None:
    """Commit all files in the yaml buffer cache to disk. Clears the buffer cache and registers mutations.

    Uses the same write-validate-replace pattern as _write_yaml for safety.

    Note: When dry_run=True, changes are detected and mutation_tracker is called,
    but no files are written to disk. This enables --check to work with --dry-run.
    """
    logger.info(":inbox_tray: Committing all YAMLs from buffer cache to disk.")
    with yaml_handler_lock:
        with _YAML_BUFFER_CACHE_LOCK:
            paths = list(_YAML_BUFFER_CACHE.keys())
        for path in paths:
            # Ensure parent directory exists before writing
            if not dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.is_file() else b""
            # Use context manager to ensure BytesIO is properly closed
            with io.BytesIO() as staging:
                with _YAML_BUFFER_CACHE_LOCK:
                    data = _YAML_BUFFER_CACHE[path]
                    # Merge preserved sections from original YAML (semantic_models, macros, etc.)
                    if path in _YAML_ORIGINAL_CACHE:
                        original_content = _YAML_ORIGINAL_CACHE[path]
                        data = _merge_preserved_sections(data, original_content)
                yaml_handler.dump(data, staging)
                modified = staging.getvalue()
                if strip_eof_blank_lines:
                    modified = _strip_eof_blank_lines(modified)
                if modified != original:
                    if dry_run:
                        logger.info(":eyes: Would write changes to => %s (dry-run)", path)
                    else:
                        logger.info(":writing_hand: Writing => %s", path)

                        # Write to a unique temporary file first for safety
                        temp_path: Path | None = None
                        try:
                            temp_path, bytes_written = _write_unique_temp_file(path, modified)

                            # Validate write succeeded
                            if not temp_path.exists():
                                raise OSError(f"Temporary file not created: {temp_path}")
                            if temp_path.stat().st_size == 0 and len(modified) > 0:
                                raise OSError(f"Temporary file is empty: {temp_path}")
                            if bytes_written != len(modified):
                                raise OSError(
                                    f"Write incomplete: expected {len(modified)} bytes, wrote {bytes_written}",
                                )

                            # Atomic replace: only delete original after successful temp write
                            _replace_atomically(temp_path, path)

                            # Clear cache entry only after successful write
                            with _YAML_BUFFER_CACHE_LOCK:
                                _discard_yaml_caches(path)

                            if written_file_tracker:
                                written_file_tracker(path)

                        except Exception as e:
                            _cleanup_temp_path(temp_path)
                            # Re-raise to signal failure
                            logger.error(":boom: Failed to commit YAML to => %s: %s", path, e)
                            raise

                    # Track mutation regardless of dry_run (enables --check with --dry-run)
                    if mutation_tracker:
                        mutation_tracker(1)

                else:
                    logger.debug(":white_check_mark: Skipping => %s (no changes)", path)
                    # Clear cache entry even when no changes (to keep cache consistent)
                    if not dry_run:
                        with _YAML_BUFFER_CACHE_LOCK:
                            _discard_yaml_caches(path)

                # After dry-run mutation reporting, discard every processed
                # buffered path so follow-up reads reflect disk-backed state.
                if dry_run:
                    with _YAML_BUFFER_CACHE_LOCK:
                        _discard_yaml_caches(path)
