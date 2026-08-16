"""JARVIS Workflow Recorder — Records automation actions to replayable JSON files.

Records: mouse clicks, keyboard, hotkeys, window changes, URLs, and screenshots.
Saves to data/workflows/<name>.json in a Vision-replayable format.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

from automation.config import automation_config
from automation.automation_logger import log_action, automation_logger

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


@dataclass
class WorkflowStep:
    """A single recorded step in a workflow."""
    step_index: int
    action: str                   # open_app | navigate | click | type | hotkey | press | scroll
    target: str = ""              # Window title / URL / text / coordinates
    value: str = ""               # Text typed, key name, scroll amount
    x: Optional[int] = None       # Click X coordinate
    y: Optional[int] = None       # Click Y coordinate
    description: str = ""         # Human description of what this step does
    timestamp: float = field(default_factory=time.time)
    screenshot_b64: str = ""      # Optional screenshot at this step


@dataclass
class Workflow:
    """A complete recorded automation workflow."""
    name: str
    description: str = ""
    trigger_phrases: List[str] = field(default_factory=list)
    required_apps: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    run_count: int = 0
    last_run: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        steps = [WorkflowStep(**s) for s in data.pop("steps", [])]
        wf = cls(**data)
        wf.steps = steps
        return wf


class WorkflowRecorder:
    """Records user automation actions into replayable workflow files."""

    def __init__(self) -> None:
        self._is_recording = False
        self._current_workflow: Optional[Workflow] = None
        self._step_counter = 0
        self._workflows_dir = Path(automation_config.WORKFLOWS_DIR)
        self._workflows_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_workflow(self) -> Optional[Workflow]:
        return self._current_workflow

    def start_recording(
        self,
        name: str,
        description: str = "",
        trigger_phrases: Optional[List[str]] = None,
    ) -> dict:
        """Begin recording a new workflow."""
        if self._is_recording:
            return {"success": False, "error": "Already recording. Stop the current recording first."}

        self._current_workflow = Workflow(
            name=name,
            description=description,
            trigger_phrases=trigger_phrases or [name.lower()],
        )
        self._step_counter = 0
        self._is_recording = True

        automation_logger.info(f"RECORDING STARTED: '{name}'")
        log_action("recorder", "start_recording", name)
        return {"success": True, "action": "start_recording", "workflow": name}

    def record_step(
        self,
        action: str,
        target: str = "",
        value: str = "",
        x: Optional[int] = None,
        y: Optional[int] = None,
        description: str = "",
        include_screenshot: bool = False,
    ) -> None:
        """Record a single automation step during an active recording session."""
        if not self._is_recording or not self._current_workflow:
            return

        screenshot_b64 = ""
        if include_screenshot or automation_config.LOG_SCREENSHOTS:
            try:
                from automation.screen_observer import screen_observer
                import io, base64
                img = screen_observer.capture()
                if img:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=60)
                    screenshot_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass

        step = WorkflowStep(
            step_index=self._step_counter,
            action=action,
            target=target,
            value=value,
            x=x,
            y=y,
            description=description or f"{action} {target}".strip(),
            screenshot_b64=screenshot_b64,
        )
        self._current_workflow.steps.append(step)
        self._step_counter += 1
        log_action("recorder", action, target or value)

    def stop_recording(self) -> dict:
        """Stop recording and return the completed workflow (not saved yet)."""
        if not self._is_recording or not self._current_workflow:
            return {"success": False, "error": "No active recording session."}

        self._is_recording = False
        steps = len(self._current_workflow.steps)
        automation_logger.info(
            f"RECORDING STOPPED: '{self._current_workflow.name}' ({steps} steps)"
        )
        return {
            "success": True,
            "action": "stop_recording",
            "workflow": self._current_workflow.name,
            "steps": steps,
        }

    def save_workflow(self, name: Optional[str] = None) -> dict:
        """Save the current recorded workflow to disk."""
        if not self._current_workflow:
            return {"success": False, "error": "No workflow to save."}

        wf = self._current_workflow
        if name:
            wf.name = name

        filename = wf.name.lower().replace(" ", "_") + ".json"
        path = self._workflows_dir / filename

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(wf.to_dict(), f, indent=2)
            automation_logger.info(f"WORKFLOW SAVED: '{wf.name}' -> {path}")
            return {
                "success": True,
                "action": "save_workflow",
                "name": wf.name,
                "path": str(path),
                "steps": len(wf.steps),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_workflows(self) -> List[str]:
        """Return names of all saved workflow files."""
        return [f.stem for f in self._workflows_dir.glob("*.json")]

    def load_workflow(self, name: str) -> Optional[Workflow]:
        """Load a workflow by name from disk."""
        filename = name.lower().replace(" ", "_") + ".json"
        path = self._workflows_dir / filename
        if not path.exists():
            # Try exact filename
            path = self._workflows_dir / (name + ".json")
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Workflow.from_dict(data)
        except Exception as e:
            automation_logger.error(f"Failed to load workflow '{name}': {e}")
            return None

    def delete_workflow(self, name: str) -> dict:
        """Delete a saved workflow file."""
        filename = name.lower().replace(" ", "_") + ".json"
        path = self._workflows_dir / filename
        if path.exists():
            path.unlink()
            return {"success": True, "deleted": name}
        return {"success": False, "error": f"Workflow '{name}' not found."}


# Global singleton
workflow_recorder = WorkflowRecorder()

