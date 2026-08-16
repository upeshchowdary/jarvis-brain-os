"""JARVIS Automation Module — Central Configuration.

All timeouts, retry limits, safety levels, and feature flags are defined here.
Do NOT hardcode these values in individual controllers.
"""

from dataclasses import dataclass, field
from typing import Set


@dataclass
class AutomationConfig:
    """Centralized config for the automation agent layer."""

    # ── Retry & Timing ──────────────────────────────────────────────
    MAX_RETRIES: int = 3
    ACTION_TIMEOUT: float = 10.0          # Per-action timeout (seconds)
    TASK_TIMEOUT: float = 120.0           # Overall task timeout (seconds)
    VERIFICATION_TIMEOUT: float = 5.0    # Post-action verification timeout
    VISION_TIMEOUT: float = 15.0         # Max wait for vision response
    APP_STARTUP_TIMEOUT: float = 15.0    # Max wait for app to open
    ELEMENT_SEARCH_TIMEOUT: float = 8.0  # Max wait to find an on-screen element

    # ── Action timing (seconds between actions for reliability) ─────
    ACTION_DELAY: float = 0.3            # Brief pause between actions
    TYPE_INTERVAL: float = 0.03          # Delay between each typed character
    CLICK_DELAY: float = 0.1             # Pause after each click

    # ── Safety ──────────────────────────────────────────────────────
    SAFETY_ENABLED: bool = True
    # "low"    = only block HIGH risk
    # "medium" = block MEDIUM + HIGH risk (require confirmation)
    # "strict" = block everything above LOW
    CONFIRMATION_LEVEL: str = "low"

    # ── Emergency Stop ───────────────────────────────────────────────
    EMERGENCY_STOP_KEY: str = "escape"   # Key to trigger immediate stop

    # ── Dry Run ──────────────────────────────────────────────────────
    DRY_RUN: bool = False                # If True: log actions but don't execute

    # ── Workflow Storage ─────────────────────────────────────────────
    WORKFLOWS_DIR: str = "data/workflows"
    MAX_WORKFLOW_STEPS: int = 200

    # ── Logging ──────────────────────────────────────────────────────
    LOG_FILE: str = "logs/automation.log"
    LOG_SCREENSHOTS: bool = False        # Embed screenshots in workflow JSON
    MAX_LOG_SIZE_MB: int = 50

    # ── Vision ───────────────────────────────────────────────────────
    SCREENSHOT_MAX_DIM: int = 1280       # Resize before sending to vision
    OCR_CONFIDENCE_THRESHOLD: float = 0.4

    # ── High-risk command keywords (always require confirmation) ─────
    HIGH_RISK_SHELL_KEYWORDS: Set[str] = field(default_factory=lambda: {
        "shutdown", "restart", "format", "del ", "rmdir", "rm -rf",
        "diskpart", "fdisk", "reg delete", "net user", "taskkill /f",
        "cipher /w", "sfc /", "bcdedit", "attrib -s -h",
    })


# Global singleton
automation_config = AutomationConfig()
