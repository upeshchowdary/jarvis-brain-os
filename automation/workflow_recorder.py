"""Workflow recorder v2 — workflows are human-editable YAML files in data/workflows/.

A workflow file looks like::

    name: send report
    description: Email the weekly report
    params: [name]
    triggers:
      - {type: time, at: "17:30", days: [mon, tue, wed, thu, fri]}
      - {type: file_created, folder: "~/Downloads", pattern: "*.pdf"}
    steps:
      - {action: open_app, target: outlook}
      - {action: click, target: New Email}
      - {action: type, value: "Hi {name}, here is the report."}

Old ``.json`` recordings are still loaded. Steps are replayed through the gateway
(see ``automation.workflow_player``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from automation.automation_logger import automation_logger, log_action
from core.paths import data_dir

try:
    import yaml  # PyYAML (MIT)
except ImportError:  # pragma: no cover - declared dependency
    yaml = None


@dataclass
class WorkflowStep:
    step_index: int
    action: str  # open_app | navigate | click | double_click | type | press | hotkey | scroll | wait | focus_window | fill
    target: str = ""
    value: str = ""
    x: Optional[int] = None
    y: Optional[int] = None
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    screenshot_b64: str = ""

    def to_yaml_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action": self.action}
        for key in ("target", "value", "description"):
            if getattr(self, key):
                d[key] = getattr(self, key)
        if self.x is not None and self.y is not None:
            d["x"], d["y"] = self.x, self.y
        return d


@dataclass
class Workflow:
    name: str
    description: str = ""
    trigger_phrases: List[str] = field(default_factory=list)
    required_apps: List[str] = field(default_factory=list)
    steps: List[WorkflowStep] = field(default_factory=list)
    params: List[str] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    run_count: int = 0
    last_run: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "trigger_phrases": self.trigger_phrases,
            "triggers": self.triggers,
            "steps": [s.to_yaml_dict() for s in self.steps],
            "stats": {"created_at": self.created_at, "run_count": self.run_count, "last_run": self.last_run,
                      "success_rate": round(self.success_rate, 3)},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        data = dict(data)
        stats = data.pop("stats", {}) or {}
        raw_steps = data.pop("steps", []) or []
        steps = []
        for i, s in enumerate(raw_steps):
            s = dict(s)
            s.setdefault("step_index", i)
            steps.append(WorkflowStep(**{k: v for k, v in s.items() if k in WorkflowStep.__dataclass_fields__}))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        wf = cls(**known)
        wf.steps = steps
        for k in ("created_at", "run_count", "last_run", "success_rate"):
            if k in stats:
                setattr(wf, k, stats[k])
        if not wf.params:
            found = set()
            for s in steps:
                found.update(re.findall(r"\{([a-zA-Z_]\w*)\}", f"{s.target} {s.value}"))
            wf.params = sorted(found)
        return wf


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.lower().strip()).strip("_") or "workflow"


class WorkflowRecorder:
    def __init__(self) -> None:
        self._is_recording = False
        self._current_workflow: Optional[Workflow] = None
        self._step_counter = 0

    @property
    def workflows_dir(self) -> Path:
        path = data_dir() / "workflows"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_workflow(self) -> Optional[Workflow]:
        return self._current_workflow

    def start_recording(self, name: str, description: str = "", trigger_phrases: Optional[List[str]] = None) -> dict:
        if self._is_recording:
            return {"success": False, "error": "Already recording. Stop the current recording first."}
        self._current_workflow = Workflow(name=name, description=description, trigger_phrases=trigger_phrases or [name.lower()])
        self._step_counter = 0
        self._is_recording = True
        automation_logger.info(f"RECORDING STARTED: '{name}'")
        return {"success": True, "action": "start_recording", "workflow": name}

    def record_step(self, action: str, target: str = "", value: str = "", x: Optional[int] = None,
                    y: Optional[int] = None, description: str = "", include_screenshot: bool = False) -> None:
        if not self._is_recording or not self._current_workflow:
            return
        step = WorkflowStep(self._step_counter, action, target, value, x, y, description or f"{action} {target or value}".strip())
        self._current_workflow.steps.append(step)
        self._step_counter += 1
        log_action("recorder", action, target or value)

    def stop_recording(self) -> dict:
        if not self._is_recording or not self._current_workflow:
            return {"success": False, "error": "No active recording session."}
        self._is_recording = False
        steps = len(self._current_workflow.steps)
        automation_logger.info(f"RECORDING STOPPED: '{self._current_workflow.name}' ({steps} steps)")
        return {"success": True, "action": "stop_recording", "workflow": self._current_workflow.name, "steps": steps}

    def save(self, wf: Workflow) -> Path:
        path = self.workflows_dir / f"{_slug(wf.name)}.yaml"
        if yaml is not None:
            path.write_text(yaml.safe_dump(wf.to_yaml_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        else:
            path = path.with_suffix(".json")
            path.write_text(json.dumps(wf.to_dict(), indent=2), encoding="utf-8")
        return path

    def save_workflow(self, name: Optional[str] = None) -> dict:
        if not self._current_workflow:
            return {"success": False, "error": "No workflow to save."}
        wf = self._current_workflow
        if name:
            wf.name = name
        try:
            path = self.save(wf)
        except OSError as exc:
            return {"success": False, "error": str(exc)}
        automation_logger.info(f"WORKFLOW SAVED: '{wf.name}' -> {path}")
        return {"success": True, "action": "save_workflow", "name": wf.name, "path": str(path), "steps": len(wf.steps)}

    def _path_for(self, name: str) -> Optional[Path]:
        slug = _slug(name)
        for candidate in (f"{slug}.yaml", f"{slug}.yml", f"{slug}.json", f"{name}.json", f"{name}.yaml"):
            path = self.workflows_dir / candidate
            if path.exists():
                return path
        return None

    def list_workflows(self) -> List[str]:
        names = set()
        for path in self.workflows_dir.iterdir():
            if path.suffix in (".yaml", ".yml", ".json"):
                names.add(path.stem)
        return sorted(names)

    def load_workflow(self, name: str) -> Optional[Workflow]:
        path = self._path_for(name)
        if path is None:
            return None
        try:
            text = path.read_text(encoding="utf-8")
            data = yaml.safe_load(text) if (path.suffix in (".yaml", ".yml") and yaml is not None) else json.loads(text)
            return Workflow.from_dict(data or {})
        except Exception as exc:
            automation_logger.error(f"Failed to load workflow '{name}': {exc}")
            return None

    def record_run(self, name: str, success: bool) -> None:
        wf = self.load_workflow(name)
        if wf is None:
            return
        total_ok = wf.success_rate * wf.run_count + (1 if success else 0)
        wf.run_count += 1
        wf.last_run = time.time()
        wf.success_rate = total_ok / wf.run_count
        try:
            old = self._path_for(name)
            new = self.save(wf)
            if old is not None and old != new and old.suffix == ".json":
                old.unlink()  # migrated to YAML
        except OSError:
            pass

    def delete_workflow(self, name: str) -> dict:
        path = self._path_for(name)
        if path is None:
            return {"success": False, "error": f"Workflow '{name}' not found."}
        path.unlink()
        return {"success": True, "deleted": name}


workflow_recorder = WorkflowRecorder()
