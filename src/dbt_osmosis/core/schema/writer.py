import io
import threading
import typing as t
from pathlib import Path

import ruamel.yaml

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.schema.reader import _YAML_BUFFER_CACHE, _YAML_BUFFER_CACHE_LOCK

__all__ = [
    "_write_yaml",
    "commit_yamls",
]


def _write_yaml(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    path: Path,
    data: dict[str, t.Any],
    dry_run: bool = False,
    mutation_tracker: t.Callable[[int], None] | None = None,
) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache.

    Uses a write-validate-replace pattern to prevent data loss:
    1. Write to temporary file (.yml.tmp)
    2. Validate write succeeded (file exists and non-empty)
    3. Replace original file via atomic rename
    4. If any step fails, clean up temp file and preserve original
    """
    logger.debug(":page_with_curl: Attempting to write YAML to => %s", path)
    if not dry_run:
        with yaml_handler_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.is_file() else b""
            # Use context manager to ensure BytesIO is properly closed
            with io.BytesIO() as staging:
                yaml_handler.dump(data, staging)
                modified = staging.getvalue()
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
                            raise IOError(f"Temporary file not created: {temp_path}")
                        if temp_path.stat().st_size == 0 and len(modified) > 0:
                            raise IOError(f"Temporary file is empty: {temp_path}")
                        if bytes_written != len(modified):
                            raise IOError(
                                f"Write incomplete: expected {len(modified)} bytes, wrote {bytes_written}"
                            )

                        # Atomic replace: only delete original after successful temp write
                        _replace_atomically(temp_path, path)

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
            with _YAML_BUFFER_CACHE_LOCK:
                if path in _YAML_BUFFER_CACHE:
                    del _YAML_BUFFER_CACHE[path]


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
                original = path.read_bytes() if path.is_file() else b""
                # Use context manager to ensure BytesIO is properly closed
                with io.BytesIO() as staging:
                    with _YAML_BUFFER_CACHE_LOCK:
                        data = _YAML_BUFFER_CACHE[path]
                    yaml_handler.dump(data, staging)
                    modified = staging.getvalue()
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
                                raise IOError(f"Temporary file not created: {temp_path}")
                            if temp_path.stat().st_size == 0 and len(modified) > 0:
                                raise IOError(f"Temporary file is empty: {temp_path}")
                            if bytes_written != len(modified):
                                raise IOError(
                                    f"Write incomplete: expected {len(modified)} bytes, wrote {bytes_written}"
                                )

                            # Atomic replace: only delete original after successful temp write
                            _replace_atomically(temp_path, path)

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
                with _YAML_BUFFER_CACHE_LOCK:
                    del _YAML_BUFFER_CACHE[path]
