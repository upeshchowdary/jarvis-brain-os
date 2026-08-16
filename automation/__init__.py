"""JARVIS Automation Package.

Exports the automation_orchestrator singleton and registers AutomationTool
with the existing ToolRouter.
"""

from automation.orchestrator import automation_orchestrator, AutomationResult
from automation.config import automation_config
from automation.action_executor import trigger_emergency_stop, clear_emergency_stop, is_emergency_stopped
from automation.workflow_recorder import workflow_recorder
from automation.workflow_player import workflow_player
from automation.task_manager import task_manager
from automation.screen_observer import screen_observer
from automation.safety_manager import safety_manager, RiskLevel

__all__ = [
    "automation_orchestrator",
    "AutomationResult",
    "automation_config",
    "trigger_emergency_stop",
    "clear_emergency_stop",
    "is_emergency_stopped",
    "workflow_recorder",
    "workflow_player",
    "task_manager",
    "screen_observer",
    "safety_manager",
    "RiskLevel",
]
