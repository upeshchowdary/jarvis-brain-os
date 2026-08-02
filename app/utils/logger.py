"""Logging Subsystem for JARVIS AI Operating System using Loguru.

Configures formatted console logging and rotating file logging for requests,
LLM provider calls, errors, latency, and token usage metrics.
"""

import sys
from pathlib import Path
from loguru import logger
from app.config.settings import settings


def setup_logger() -> None:
    """Initialize and configure Loguru handlers for stdout and file output."""
    # Remove standard default logger
    logger.remove()

    # Log format specification
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console Handler
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL.upper(),
        format=log_format,
        colorize=True,
        backtrace=settings.DEBUG,
        diagnose=settings.DEBUG,
    )

    # Ensure log file directory exists
    log_path = Path(settings.base_dir) / settings.LOG_FILE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Rotating File Handler
    logger.add(
        str(log_path),
        level=settings.LOG_LEVEL.upper(),
        format=log_format,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,  # Thread-safe async logging
    )


# Configure logging immediately upon module import
setup_logger()

__all__ = ["logger", "setup_logger"]
