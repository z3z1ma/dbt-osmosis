"""YAML file writing for dbt-osmosis.

Thread-safety:
    - _write_yaml() acquires yaml_handler_lock for the entire write operation
    - Cache invalidation is performed under _YAML_BUFFER_CACHE_LOCK
    - Multiple threads can safely write to different files concurrently
"""

import io
import threading
import typing as t
from pathlib import Path

import ruamel.yaml

from dbt_osmosis.core import logger
from dbt_osmosis.core.schema.reader import (
    _YAML_BUFFER_CACHE,
    _YAML_BUFFER_CACHE_LOCK,
    _YAML_ORIGINAL_CACHE,
)

__all__ = [
    "_merge_preserved_sections",
    "_write_yaml",
    "commit_yamls",
]


# Keys that are filtered out by OsmosisYAML but should be preserved when writing
_PRESERVED_KEYS = {"semantic_models", "macros"}


def _merge_preserved_sections(
    filtered_data: dict[str, t.Any], original_data: dict[str, t.Any]
) -> dict[str, t.Any]:
    """Merge preserved sections (semantic_models, macros, etc.) from original YAML.

    When dbt-osmosis processes a YAML file, it filters out sections like semantic_models
    and macros that it shouldn't modify. This function restores those sections from the
    original file so they're not lost when writing back to disk.

    Args:
        filtered_data: The processed YAML data (may have models, sources, etc.)
        original_data: The original unfiltered YAML data (may have semantic_models, macros, etc.)

    Returns:
        A merged dictionary containing both processed and preserved sections.

    """
    # Start with the filtered data (processed content)
    merged = dict(filtered_data)

    # Add back any preserved sections from the original data
    for key in _PRESERVED_KEYS:
        if key in original_data and key not in merged:
            merged[key] = original_data[key]
            logger.debug(f":recycle: Restoring preserved section '{key}' from original YAML")

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


def _write_yaml(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    path: Path,
    data: dict[str, t.Any],
    dry_run: bool = False,
    mutation_tracker: t.Callable[[int], None] | None = None,
    strip_eof_blank_lines: bool = False,
) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache.

    Thread-safety: This function is thread-safe. It acquires yaml_handler_lock
    to ensure exclusive access to the yaml handler, and _YAML_BUFFER_CACHE_LOCK
    for cache invalidation. Multiple threads can safely write to different files.

    Uses a write-validate-replace pattern to prevent data loss:
    1. Write to temporary file (.yml.tmp)
    2. Validate write succeeded (file exists and non-empty)
    3. Replace original file via atomic rename
    4. If any step fails, clean up temp file and preserve original
    """
    logger.debug(":page_with_curl: Attempting to write YAML to => %s", path)
    if not dry_run:
        with yaml_handler_lock:
            # Merge preserved sections from original YAML (semantic_models, macros, etc.)
            with _YAML_BUFFER_CACHE_LOCK:
                if path in _YAML_ORIGINAL_CACHE:
                    original_content = _YAML_ORIGINAL_CACHE[path]
                    data = _merge_preserved_sections(data, original_content)

            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.is_file() else b""
            # Use context manager to ensure BytesIO is properly closed
            with io.BytesIO() as staging:
                yaml_handler.dump(data, staging)
                modified = staging.getvalue()
                if strip_eof_blank_lines:
                    modified = _strip_eof_blank_lines(modified)
                if modified != original:
                    logger.info(":writing_hand: Writing changes to => %s", path)

                    # Write to temporary file first for safety
                    temp_path = path.with_suffix(path.suffix + ".tmp")
                    try:
                        # Write to temp file
                        with temp_path.open("wb") as f:
                            bytes_written = f.write(modified)

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
                            if path in _YAML_BUFFER_CACHE:
                                del _YAML_BUFFER_CACHE[path]
                            if path in _YAML_ORIGINAL_CACHE:
                                del _YAML_ORIGINAL_CACHE[path]

                        if mutation_tracker:
                            mutation_tracker(1)

                    except Exception as e:
                        # Clean up temp file on any error
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except Exception:
                                pass
                        # Re-raise to signal failure
                        logger.error(":boom: Failed to write YAML to => %s: %s", path, e)
                        raise
                else:
                    logger.debug(":white_check_mark: Skipping write => %s (no changes)", path)
                    # Clear cache entry even when no changes (to keep cache consistent)
                    with _YAML_BUFFER_CACHE_LOCK:
                        if path in _YAML_BUFFER_CACHE:
                            del _YAML_BUFFER_CACHE[path]
                        if path in _YAML_ORIGINAL_CACHE:
                            del _YAML_ORIGINAL_CACHE[path]


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
) -> None:
    """Commit all files in the yaml buffer cache to disk. Clears the buffer cache and registers mutations.

    Uses the same write-validate-replace pattern as _write_yaml for safety.
    """
    logger.info(":inbox_tray: Committing all YAMLs from buffer cache to disk.")
    if not dry_run:
        with yaml_handler_lock:
            with _YAML_BUFFER_CACHE_LOCK:
                paths = list(_YAML_BUFFER_CACHE.keys())
            for path in paths:
                # Ensure parent directory exists before writing
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
                        logger.info(":writing_hand: Writing => %s", path)

                        # Write to temporary file first for safety
                        temp_path = path.with_suffix(path.suffix + ".tmp")
                        try:
                            # Write to temp file
                            with temp_path.open("wb") as f:
                                bytes_written = f.write(modified)

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
                                if path in _YAML_BUFFER_CACHE:
                                    del _YAML_BUFFER_CACHE[path]
                                if path in _YAML_ORIGINAL_CACHE:
                                    del _YAML_ORIGINAL_CACHE[path]

                            if mutation_tracker:
                                mutation_tracker(1)

                        except Exception as e:
                            # Clean up temp file on any error
                            if temp_path.exists():
                                try:
                                    temp_path.unlink()
                                except Exception:
                                    pass
                            # Re-raise to signal failure
                            logger.error(":boom: Failed to commit YAML to => %s: %s", path, e)
                            raise
                    else:
                        logger.debug(":white_check_mark: Skipping => %s (no changes)", path)
                        # Clear cache entry even when no changes (to keep cache consistent)
                        with _YAML_BUFFER_CACHE_LOCK:
                            if path in _YAML_BUFFER_CACHE:
                                del _YAML_BUFFER_CACHE[path]
                            if path in _YAML_ORIGINAL_CACHE:
                                del _YAML_ORIGINAL_CACHE[path]
