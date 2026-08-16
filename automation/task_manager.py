"""JARVIS Task Manager — Converts natural language into structured task plans.

Implements the Observe → Think → Act loop.
Uses existing BrainManager for NL → plan conversion.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from automation.config import automation_config
from automation.automation_logger import log_action, log_task_start, log_task_complete, automation_logger
from automation.action_executor import is_emergency_stopped


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TaskStep:
    """A single planned step within a task."""
    index: int
    action: str
    target: str = ""
    value: str = ""
    description: str = ""
    verify_after: bool = False
    status: str = "pending"
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class AutomationTask:
    """A complete automation task with goal, steps, and execution state."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    goal: str = ""
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    session_id: str = ""
    result_summary: str = ""
    total_retries: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    elapsed_ms: float = 0.0

    @property
    def is_complete(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.STOPPED)


import re


def _fast_rule_planner(command: str) -> Optional[List[TaskStep]]:
    """Fast sub-millisecond rule-based parser for common automation commands."""
    cmd = command.lower().strip()

    # 1. Open app / Launch app (e.g. "open chrome", "launch spotify", "start notepad")
    m = re.search(
        r"^(?:open|launch|start|run)\s+(?:the\s+)?([a-zA-Z0-9_\s\+\-\.]+?)(?:\s+(?:app|browser|application|program))?$",
        cmd,
    )
    if m:
        target = m.group(1).strip()
        # Avoid matching URL navigation or general conversational phrases
        if target and not any(target.startswith(p) for p in ("http", "www", "file", "tab", "link")):
            return [TaskStep(index=0, action="open_app", target=target, description=f"Open {target}")]

    # 2. Close app / Quit app / Kill app (e.g. "close chrome", "quit spotify")
    m = re.search(
        r"^(?:close|quit|exit|kill|terminate|stop)\s+(?:the\s+)?([a-zA-Z0-9_\s\+\-\.]+?)(?:\s+(?:app|browser|application|program|window))?$",
        cmd,
    )
    if m:
        target = m.group(1).strip()
        if target and not any(target.startswith(p) for p in ("http", "www", "tab", "link", "recording")):
            return [TaskStep(index=0, action="close_app", target=target, description=f"Close {target}")]

    # 3. Search Google / YouTube
    m = re.search(r"^(?:search\s+youtube\s+for|youtube)\s+(.+)$", cmd)
    if m:
        query = m.group(1).strip()
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        return [TaskStep(index=0, action="navigate", target=url, description=f"Search YouTube for: {query}")]

    m = re.search(r"^(?:search\s+google\s+for|search\s+for|google)\s+(.+)$", cmd)
    if m:
        query = m.group(1).strip()
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        return [TaskStep(index=0, action="navigate", target=url, description=f"Search Google for: {query}")]

    # 4. Navigate to URL / Go to website
    m = re.search(r"^(?:go\s+to|navigate\s+to|open\s+website|visit)\s+(?:https?://)?([a-zA-Z0-9_\-\./\?\=\&\#\+]+)$", cmd)
    if m:
        raw_url = m.group(1).strip()
        url = raw_url if raw_url.startswith("http") else f"https://{raw_url}"
        return [TaskStep(index=0, action="navigate", target=url, description=f"Navigate to {url}")]

    # 5. Media & Volume Controls
    if cmd in ("volume up", "increase volume", "louder", "turn up volume"):
        return [TaskStep(index=0, action="press", value="volumeup", description="Volume up")]
    if cmd in ("volume down", "decrease volume", "lower volume", "quieter", "turn down volume"):
        return [TaskStep(index=0, action="press", value="volumedown", description="Volume down")]
    if cmd in ("mute", "unmute", "mute volume", "mute audio"):
        return [TaskStep(index=0, action="press", value="volumemute", description="Mute/unmute volume")]
    if cmd in ("play", "pause", "play pause", "pause music", "play music"):
        return [TaskStep(index=0, action="press", value="playpause", description="Play/pause media")]
    if cmd in ("next track", "next song", "skip track", "next music"):
        return [TaskStep(index=0, action="press", value="nexttrack", description="Next track")]
    if cmd in ("previous track", "prev track", "previous song"):
        return [TaskStep(index=0, action="press", value="prevtrack", description="Previous track")]

    # 6. Window Controls
    if cmd in ("minimize", "minimize window", "minimize all"):
        return [TaskStep(index=0, action="hotkey", value="win+d", description="Minimize windows")]
    if cmd in ("maximize", "maximize window"):
        return [TaskStep(index=0, action="hotkey", value="win+up", description="Maximize window")]

    # 7. Clipboard & Editing Shortcuts
    if cmd in ("copy", "copy text", "copy this"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+c", description="Copy")]
    if cmd in ("paste", "paste text", "paste here"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+v", description="Paste")]
    if cmd in ("select all", "select everything"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+a", description="Select all")]
    if cmd in ("save", "save file", "save document"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+s", description="Save")]
    if cmd in ("undo", "undo action"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+z", description="Undo")]
    if cmd in ("redo", "redo action"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+y", description="Redo")]

    # 8. Browser Tab controls
    if cmd in ("new tab", "open new tab"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+t", description="Open new tab")]
    if cmd in ("close tab", "close current tab"):
        return [TaskStep(index=0, action="hotkey", value="ctrl+w", description="Close tab")]

    # 9. Screenshots
    if cmd in ("screenshot", "take screenshot", "capture screen", "take a screenshot"):
        return [TaskStep(index=0, action="take_screenshot", description="Take screenshot")]

    # 10. Type text / Press key
    m = re.search(r"^(?:type|enter|input)\s+['\"]?(.+?)['\"]?$", cmd)
    if m and not cmd.startswith("type of"):
        text = m.group(1).strip()
        return [TaskStep(index=0, action="type", value=text, description=f"Type '{text}'")]

    m = re.search(r"^(?:press|hit)\s+([a-zA-Z0-9_\+\-]+)$", cmd)
    if m:
        key = m.group(1).strip()
        if "+" in key:
            return [TaskStep(index=0, action="hotkey", value=key, description=f"Press hotkey {key}")]
        return [TaskStep(index=0, action="press", value=key, description=f"Press {key}")]

    return None


class TaskManager:
    """
    Orchestrates the OBSERVE -> THINK -> PLAN -> ACT -> VERIFY loop.
    Converts natural language commands into structured task execution.
    """

    def __init__(self) -> None:
        self._active_task: Optional[AutomationTask] = None
        self._task_history: List[AutomationTask] = []

    @property
    def active_task(self) -> Optional[AutomationTask]:
        return self._active_task

    # ── NL → Task Plan ───────────────────────────────────────────────

    async def plan_from_nl(self, command: str, session_id: str = "") -> AutomationTask:
        """
        Convert a natural language automation command into a structured task plan.
        Uses fast deterministic rules for instant execution (<1ms),
        falling back to LLM-powered planning for complex workflows.
        """
        task = AutomationTask(goal=command, session_id=session_id)

        # ── Step 0: Sub-millisecond Fast Rule Planner ─────────────────
        fast_steps = _fast_rule_planner(command)
        if fast_steps:
            task.goal = command
            task.steps = fast_steps
            automation_logger.info(
                f"[FAST PLAN] Created {len(task.steps)} steps for goal='{task.goal}' in 0.1ms"
            )
            return task

        # ── Step 1: Fallback to LLM Planning for complex multi-step tasks ──
        from brain.brain_manager import brain_manager

        planning_prompt = f"""You are a computer automation planner. 
Convert this user command into a structured JSON automation plan.

User command: "{command}"

Return ONLY valid JSON in this exact format:
{{
  "goal": "brief goal description",
  "steps": [
    {{"action": "open_app", "target": "Chrome", "value": "", "description": "Open Chrome browser"}},
    {{"action": "navigate", "target": "https://google.com", "value": "", "description": "Go to Google"}},
    {{"action": "click", "target": "search box", "value": "", "description": "Click search field"}},
    {{"action": "type", "target": "", "value": "Python jobs", "description": "Type search query"}},
    {{"action": "press", "target": "", "value": "enter", "description": "Submit search"}}
  ]
}}

Valid actions: open_app, close_app, navigate, click, double_click, right_click, 
type, press, hotkey, scroll, fill_field, focus_window, wait, 
read_file, create_file, run_command, take_screenshot.

Be specific. Use exact app names and URLs when possible."""

        task = AutomationTask(goal=command, session_id=session_id)

        try:
            # Use existing brain for planning
            output = await brain_manager.execute_cognitive_pipeline(
                user_query=planning_prompt,
                session_id=f"automation_plan_{session_id}",
                personality="minimal",
            )

            from brain.utils import extract_and_clean_json
            plan_data = extract_and_clean_json(output.response)

            if plan_data and "steps" in plan_data:
                task.goal = plan_data.get("goal", command)
                task.steps = [
                    TaskStep(
                        index=i,
                        action=s.get("action", ""),
                        target=s.get("target", ""),
                        value=s.get("value", ""),
                        description=s.get("description", ""),
                    )
                    for i, s in enumerate(plan_data["steps"])
                ]
                automation_logger.info(
                    f"PLAN CREATED: {len(task.steps)} steps for goal='{task.goal}'"
                )
            else:
                # Fallback: simple single-step task
                automation_logger.warning("Plan parsing failed — creating fallback single-step task.")
                task.steps = [
                    TaskStep(index=0, action="observe", target="screen",
                             description=f"Observe screen for: {command}")
                ]

        except Exception as e:
            automation_logger.error(f"Planning error: {e}")
            task.steps = [
                TaskStep(index=0, action="observe", target="screen",
                         description=command)
            ]

        return task

    # ── Task Execution ───────────────────────────────────────────────

    async def execute_task(self, task: AutomationTask) -> AutomationTask:
        """
        Execute a task through the OBSERVE → THINK → ACT → VERIFY loop.
        """
        self._active_task = task
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        log_task_start(task.id, task.goal)

        # Import controllers here to avoid circular imports at module level
        from automation.mouse_controller import mouse_controller
        from automation.keyboard_controller import keyboard_controller
        from automation.window_controller import window_controller
        from automation.application_controller import application_controller
        from automation.browser_controller import browser_controller
        from automation.system_controller import system_controller
        from automation.screen_observer import screen_observer
        from automation.action_executor import action_executor

        step_results = []

        for step in task.steps:
            if is_emergency_stopped():
                task.status = TaskStatus.STOPPED
                task.result_summary = "Task stopped by emergency stop."
                break

            step.status = "running"
            log_action("taskmanager", step.action, step.target or step.value,
                      result=f"step {step.index + 1}/{len(task.steps)}")

            result = {"success": False, "error": "Unknown action"}

            try:
                a = step.action.lower()

                if a == "open_app":
                    result = await application_controller.open_app(step.target)
                elif a == "close_app":
                    result = await application_controller.close_app(step.target)
                elif a == "navigate":
                    result = await browser_controller.navigate(step.target or step.value)
                elif a == "click":
                    loc = await screen_observer.find_text_location(step.target) if step.target else None
                    if loc:
                        result = await mouse_controller.click(loc[0], loc[1])
                    elif step.target:
                        result = {"success": False, "error": f"Could not find: '{step.target}'"}
                    else:
                        result = {"success": False, "error": "click: no target specified"}
                elif a in ("type", "type_text"):
                    result = await keyboard_controller.type_text(step.value)
                elif a == "press":
                    result = await keyboard_controller.press(step.value)
                elif a == "hotkey":
                    keys = step.value.split("+")
                    result = await keyboard_controller.hotkey(*keys)
                elif a == "scroll":
                    direction = "down" if "down" in step.value else "up"
                    result = await mouse_controller.scroll(3, direction=direction)
                elif a == "fill_field":
                    result = await browser_controller.fill_field(step.target, step.value)
                elif a == "focus_window":
                    result = await window_controller.focus_window(step.target)
                elif a == "take_screenshot":
                    img = screen_observer.capture()
                    result = {"success": img is not None, "action": "take_screenshot"}
                elif a == "observe":
                    state = await screen_observer.analyze(step.target or "describe the screen")
                    result = {"success": True, "description": state.vision_description}
                elif a == "wait":
                    secs = float(step.value or "1.0")
                    await asyncio.sleep(secs)
                    result = {"success": True}
                elif a == "read_file":
                    result = await system_controller.read_file(step.target)
                elif a == "create_file":
                    result = await system_controller.create_file(step.target, step.value)
                elif a == "run_command":
                    result = await system_controller.run_command(step.value or step.target)
                elif a == "double_click":
                    loc = await screen_observer.find_text_location(step.target) if step.target else None
                    if loc:
                        result = await mouse_controller.double_click(loc[0], loc[1])
                    else:
                        result = {"success": False, "error": f"Target not found: '{step.target}'"}
                elif a == "right_click":
                    loc = await screen_observer.find_text_location(step.target) if step.target else None
                    if loc:
                        result = await mouse_controller.right_click(loc[0], loc[1])
                    else:
                        result = {"success": False, "error": f"Target not found: '{step.target}'"}
                else:
                    result = {"success": True, "skipped": True, "action": a}

            except Exception as e:
                result = {"success": False, "error": str(e)}
                automation_logger.error(f"Step {step.index} exception: {e}")

            step.result = result
            step.status = "done" if result.get("success") else "failed"
            step.error = result.get("error", "")
            step_results.append(result)

            # Brief pause between steps
            await asyncio.sleep(automation_config.ACTION_DELAY)

        # ── Determine final task status ───────────────────────────────
        if task.status != TaskStatus.STOPPED:
            failures = [s for s in task.steps if s.status == "failed"]
            task.status = TaskStatus.COMPLETED if not failures else TaskStatus.FAILED

        task.completed_at = time.time()
        task.elapsed_ms = (task.completed_at - task.started_at) * 1000

        # Build human-readable summary
        done = len([s for s in task.steps if s.status == "done"])
        total = len(task.steps)
        task.result_summary = (
            f"Completed {done}/{total} steps for goal: '{task.goal}'."
            + (" All steps succeeded." if task.status == TaskStatus.COMPLETED
               else f" {total - done} step(s) failed.")
        )

        log_task_complete(task.id, task.elapsed_ms, task.status == TaskStatus.COMPLETED)
        self._task_history.append(task)
        self._active_task = None
        return task

    def get_status(self) -> dict:
        """Return current automation task status."""
        if self._active_task:
            t = self._active_task
            return {
                "running": True,
                "task_id": t.id,
                "goal": t.goal,
                "status": t.status,
                "steps_total": len(t.steps),
                "steps_done": len([s for s in t.steps if s.status == "done"]),
            }
        return {"running": False, "last_task": self._task_history[-1].id if self._task_history else None}


# Global singleton
task_manager = TaskManager()
