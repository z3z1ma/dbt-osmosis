import io
import threading
import typing as t
from pathlib import Path

import ruamel.yaml

import dbt_osmosis.core.logger as logger
from dbt_osmosis.core.schema.reader import _YAML_BUFFER_CACHE

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
    mutation_tracker: t.Optional[t.Callable[[int], None]] = None,
) -> None:
    """Write a yaml file to disk and register a mutation with the context. Clears the path from the buffer cache."""
    logger.debug(":page_with_curl: Attempting to write YAML to => %s", path)
    if not dry_run:
        with yaml_handler_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes() if path.is_file() else b""
            yaml_handler.dump(data, staging := io.BytesIO())
            modified = staging.getvalue()
            if modified != original:
                logger.info(":writing_hand: Writing changes to => %s", path)
                with path.open("wb") as f:
                    _ = f.write(modified)
                    if mutation_tracker:
                        mutation_tracker(1)
            else:
                logger.debug(":white_check_mark: Skipping write => %s (no changes)", path)
            del staging
            if path in _YAML_BUFFER_CACHE:
                del _YAML_BUFFER_CACHE[path]


def commit_yamls(
    yaml_handler: ruamel.yaml.YAML,
    yaml_handler_lock: threading.Lock,
    dry_run: bool = False,
    mutation_tracker: t.Optional[t.Callable[[int], None]] = None,
) -> None:
    """Commit all files in the yaml buffer cache to disk. Clears the buffer cache and registers mutations."""
    logger.info(":inbox_tray: Committing all YAMLs from buffer cache to disk.")
    if not dry_run:
        with yaml_handler_lock:
            for path in list(_YAML_BUFFER_CACHE.keys()):
                original = path.read_bytes() if path.is_file() else b""
                yaml_handler.dump(_YAML_BUFFER_CACHE[path], staging := io.BytesIO())
                modified = staging.getvalue()
                if modified != original:
                    logger.info(":writing_hand: Writing => %s", path)
                    with path.open("wb") as f:
                        logger.info(f"Writing {path}")
                        _ = f.write(modified)
                        if mutation_tracker:
                            mutation_tracker(1)
                else:
                    logger.debug(":white_check_mark: Skipping => %s (no changes)", path)
                del _YAML_BUFFER_CACHE[path]
