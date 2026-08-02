"""Brain Structured Logging System using Loguru."""

import sys
from pathlib import Path
from loguru import logger
from brain.brain_config import brain_config


def setup_brain_logger() -> None:
    """Configure Loguru structured logging for brain framework decisions and errors."""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console output
    logger.add(
        sys.stdout,
        level=brain_config.LOG_LEVEL.upper(),
        format=log_format,
        colorize=True,
        backtrace=brain_config.DEBUG,
        diagnose=brain_config.DEBUG,
    )

    # Rotating file log
    log_path = brain_config.base_dir / brain_config.LOG_FILE_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=brain_config.LOG_LEVEL.upper(),
        format=log_format,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
    )


setup_brain_logger()

__all__ = ["logger", "setup_brain_logger"]
