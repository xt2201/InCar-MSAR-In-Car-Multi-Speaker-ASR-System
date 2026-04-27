"""Structured logging utilities using loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def get_logger(
    name: str = "incar_asr",
    log_file: str | Path | None = "outputs/run.log",
    level: str = "INFO",
) -> "loguru.Logger":
    """Configure and return a loguru logger.

    Parameters
    ----------
    name : str
        Logger name (used in log format).
    log_file : str or Path or None
        Path to log file. If None, only logs to stderr.
    level : str
        Minimum log level: DEBUG, INFO, WARNING, ERROR.

    Returns
    -------
    logger : loguru.Logger
        Configured logger instance.
    """
    logger.remove()  # Remove default handler

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{name}</cyan> | "
        "<white>{message}</white>"
    )

    # Console handler
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=fmt,
            level=level,
            rotation="50 MB",
            retention="7 days",
            colorize=False,
        )

    return logger
