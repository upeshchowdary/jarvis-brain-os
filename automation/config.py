"""Automation timing configuration.

Safety settings (dry-run, confirmation policy, protected processes) live in
``core.config.settings`` and are enforced by ``automation.gateway`` — there is a
single source of truth. ``automation_config.DRY_RUN`` is an alias kept for
backwards compatibility.
"""

from dataclasses import dataclass

from core.config import settings


@dataclass
class AutomationConfig:
    # ── Retry & timing ──────────────────────────────────────────────
    MAX_RETRIES: int = 2
    ACTION_TIMEOUT: float = 5.0
    TASK_TIMEOUT: float = 30.0
    VERIFICATION_TIMEOUT: float = 1.0
    VISION_TIMEOUT: float = 8.0
    APP_STARTUP_TIMEOUT: float = 1.5
    ELEMENT_SEARCH_TIMEOUT: float = 2.0

    # ── Action timing ───────────────────────────────────────────────
    ACTION_DELAY: float = 0.05
    TYPE_INTERVAL: float = 0.01
    CLICK_DELAY: float = 0.02

    # ── Workflows ───────────────────────────────────────────────────
    MAX_WORKFLOW_STEPS: int = 200
    LOG_SCREENSHOTS: bool = False

    # ── Vision ──────────────────────────────────────────────────────
    SCREENSHOT_MAX_DIM: int = 1280
    OCR_CONFIDENCE_THRESHOLD: float = 0.4

    @property
    def DRY_RUN(self) -> bool:  # noqa: N802
        return settings.DRY_RUN

    @DRY_RUN.setter
    def DRY_RUN(self, value: bool) -> None:  # noqa: N802
        settings.DRY_RUN = bool(value)

    @property
    def CONFIRMATION_LEVEL(self) -> str:  # noqa: N802
        return "medium" if settings.CONFIRM_MEDIUM_RISK else "low"


automation_config = AutomationConfig()
