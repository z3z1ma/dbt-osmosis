# pyright: reportAny=false
"""Logging module for dbt-osmosis. The module itself can be used as a logger as it proxies calls to the default LOGGER instance."""

from __future__ import annotations

import logging
import typing as t
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

_LOG_FILE_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
_LOG_PATH = Path.home().absolute() / ".dbt-osmosis" / "logs"
_LOGGING_LEVEL = logging.INFO


def get_rotating_log_handler(name: str, path: Path, formatter: str) -> RotatingFileHandler:
    """This handler writes warning and higher level outputs to logs in a home .dbt-osmosis directory rotating them as needed"""
    path.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path / "{log_name}.log".format(log_name=name)),
        maxBytes=int(1e6),
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(formatter))
    handler.setLevel(logging.WARNING)
    return handler


@lru_cache(maxsize=10)
def get_logger(
    name: str = "dbt-osmosis",
    level: int | str = _LOGGING_LEVEL,
    path: Path = _LOG_PATH,
    formatter: str = _LOG_FILE_FORMAT,
) -> logging.Logger:
    """Builds and caches loggers. Can be configured with module level attributes or on a call by call basis.

    Simplifies logger management without having to instantiate separate pointers in each module.

    Args:
        name (str, optional): Logger name, also used for output log file name in `~/.dbt-osmosis/logs` directory.
        level (Union[int, str], optional): Logging level, this is explicitly passed to console handler which effects what level of log messages make it to the console. Defaults to logging.INFO.
        path (Path, optional): Path for output warning level+ log files. Defaults to `~/.dbt-osmosis/logs`
        formatter (str, optional): Format for output log files. Defaults to a "time — name — level — message" format

    Returns:
        logging.Logger: Prepared logger with rotating logs and console streaming. Can be executed directly from function.
    """
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(get_rotating_log_handler(name, path, formatter))
    logger.addHandler(
        RichHandler(
            level=level,
            rich_tracebacks=True,
            markup=True,
            show_time=False,
        )
    )
    logger.propagate = False
    return logger


LOGGER = get_logger()
"""Default logger for dbt-osmosis"""


class LogMethod(t.Protocol):
    """Protocol for logger methods"""

    def __call__(self, msg: str, /, *args: t.Any, **kwds: t.Any) -> t.Any: ...


def __getattr__(name: str) -> LogMethod:
    func = getattr(LOGGER, name)
    return func
