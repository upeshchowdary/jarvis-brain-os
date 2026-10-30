"""Structured logging (loguru) with per-request correlation IDs."""

from __future__ import annotations

import contextvars
import sys
from typing import Any

from loguru import logger

from core.config import settings
from core.paths import logs_dir

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_configured = False


def _patch(record: Any) -> None:
    record["extra"].setdefault("request_id", request_id_var.get())


def setup_logging() -> None:
    """Configure console + rotating file sinks once per process."""
    global _configured
    if _configured:
        return
    _configured = True
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<magenta>{extra[request_id]}</magenta> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.remove()
    logger.configure(patcher=_patch)
    logger.add(sys.stderr, level=settings.LOG_LEVEL.upper(), format=fmt, colorize=True, backtrace=False, diagnose=False)
    logger.add(
        str(logs_dir() / "jarvis.log"),
        level=settings.LOG_LEVEL.upper(),
        format=fmt,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


setup_logging()

__all__ = ["logger", "request_id_var", "setup_logging"]
