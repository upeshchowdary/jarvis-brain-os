"""JARVIS Automation Safety Manager.

Classifies every action by risk level before execution.
Blocks HIGH-risk actions unless confirmed.
Never bypasses system authentication, security controls, or MFA.
"""

from enum import Enum
from typing import Set, Optional
from automation.config import automation_config
from automation.automation_logger import automation_logger


class RiskLevel(str, Enum):
    LOW = "low"        # Execute automatically — safe, reversible
    MEDIUM = "medium"  # Configurable — may require confirmation
    HIGH = "high"      # ALWAYS require explicit user confirmation


# ── Action risk registry ─────────────────────────────────────────────────────

_LOW_RISK_ACTIONS: Set[str] = {
    "move_mouse", "scroll", "screenshot", "observe", "ocr", "find_element",
    "open_app", "focus_window", "maximize", "minimize", "restore", "switch_window",
    "navigate_url", "new_tab", "refresh", "back", "forward", "get_page_text",
    "get_position", "list_windows", "get_active_window", "read_file", "list_files",
    "type_text", "press_key", "hotkey",
}

_MEDIUM_RISK_ACTIONS: Set[str] = {
    "click", "double_click", "right_click", "drag",
    "fill_field", "submit_form", "click_button", "click_link",
    "create_file", "write_file", "rename_file", "copy_file",
    "send_message", "post_content",
}

_HIGH_RISK_ACTIONS: Set[str] = {
    "delete_file", "delete_folder", "run_command", "execute_shell",
    "shutdown", "restart", "format_drive", "close_app_force",
    "move_file",  # moving is reversible but can lose data
    "install_software", "change_settings", "modify_registry",
    "change_password", "financial_transaction",
}


class SafetyManager:
    """Evaluates risk level and gates action execution accordingly."""

    def __init__(self) -> None:
        self._pending_confirmation: Optional[str] = None

    def classify(self, action_name: str, target: str = "") -> RiskLevel:
        """Classify an action as LOW / MEDIUM / HIGH risk."""
        a = action_name.lower().strip()

        if a in _HIGH_RISK_ACTIONS:
            return RiskLevel.HIGH

        # Shell command safety: scan for destructive keywords
        if a in ("run_command", "execute_shell"):
            cmd_lower = target.lower()
            for kw in automation_config.HIGH_RISK_SHELL_KEYWORDS:
                if kw in cmd_lower:
                    return RiskLevel.HIGH
            return RiskLevel.MEDIUM  # Shell but no destructive keywords

        if a in _MEDIUM_RISK_ACTIONS:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def is_allowed(
        self,
        action_name: str,
        target: str = "",
        user_confirmed: bool = False,
    ) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).

        Blocks HIGH-risk actions without confirmation.
        Blocks MEDIUM-risk if confirmation_level is 'medium' or 'strict'.
        """
        if not automation_config.SAFETY_ENABLED:
            return True, "Safety disabled."

        risk = self.classify(action_name, target)
        level = automation_config.CONFIRMATION_LEVEL

        if risk == RiskLevel.LOW:
            return True, "Low risk — auto-approved."

        if risk == RiskLevel.HIGH and not user_confirmed:
            reason = (
                f"Action '{action_name}' on '{target}' is HIGH RISK and requires "
                f"explicit confirmation before execution."
            )
            automation_logger.warning(f"BLOCKED HIGH-RISK action: {action_name} | target={target}")
            return False, reason

        if risk == RiskLevel.MEDIUM and level in ("medium", "strict") and not user_confirmed:
            reason = (
                f"Action '{action_name}' on '{target}' requires confirmation "
                f"(confirmation_level='{level}')."
            )
            automation_logger.warning(f"BLOCKED MEDIUM-RISK action: {action_name} | target={target}")
            return False, reason

        return True, f"{risk.value.capitalize()} risk — approved."

    def format_confirmation_request(self, action_name: str, target: str) -> str:
        """Return human-readable confirmation prompt for the user."""
        risk = self.classify(action_name, target)
        emoji = "⚠️" if risk == RiskLevel.MEDIUM else "🚨"
        return (
            f"{emoji} **Confirmation required** ({risk.value.upper()} RISK)\n"
            f"Action: **{action_name}**\n"
            f"Target: `{target}`\n\n"
            f"Reply **yes** to confirm or **no** to cancel."
        )


# Global singleton
safety_manager = SafetyManager()
