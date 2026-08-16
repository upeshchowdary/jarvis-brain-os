"""JARVIS Automation Logger — Structured per-task logging.

Writes to logs/automation.log with rotation. Format:
[timestamp] | LEVEL | TASK | ACTION | TARGET | RESULT
"""

import sys
from pathlib import Path
from loguru import logger as _base_logger

# Create a dedicated automation logger instance
automation_logger = _base_logger.bind(module="automation")

_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)

# Add file sink with rotation
_base_logger.add(
    "logs/automation.log",
    level="DEBUG",
    rotation="50 MB",
    retention="7 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | AUTOMATION | {message}",
    filter=lambda record: record["extra"].get("module") == "automation",
    enqueue=True,  # async-safe
)


def log_action(
    task: str,
    action: str,
    target: str = "",
    result: str = "OK",
    level: str = "INFO",
    retry: int = 0,
    dry_run: bool = False,
) -> None:
    """Log a structured automation action event."""
    prefix = "[DRY-RUN] " if dry_run else ""
    retry_str = f" (retry #{retry})" if retry > 0 else ""
    msg = f"{prefix}[{task}] {action}"
    if target:
        msg += f" -> {target}"
    msg += f" | {result}{retry_str}"

    log_fn = getattr(automation_logger, level.lower(), automation_logger.info)
    log_fn(msg)


def log_task_start(task_id: str, goal: str) -> None:
    automation_logger.info(f"TASK STARTED | id={task_id} | goal={goal!r}")


def log_task_complete(task_id: str, elapsed_ms: float, success: bool) -> None:
    status = "SUCCESS" if success else "FAILED"
    automation_logger.info(f"TASK {status} | id={task_id} | elapsed={elapsed_ms:.0f}ms")


def log_emergency_stop(reason: str = "user request") -> None:
    automation_logger.warning(f"EMERGENCY STOP triggered | reason={reason}")
