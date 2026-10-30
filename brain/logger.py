"""Brain logging uses the shared JARVIS logger (core.logging)."""

from core.logging import logger, setup_logging

setup_brain_logger = setup_logging

__all__ = ["logger", "setup_brain_logger"]
