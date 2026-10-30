"""Task planner for multi-step computer-use goals.

Produces a list of step dicts for the ``computer.run_plan`` gateway operation.
System controls (volume, power, processes …), messaging and shell are *not*
planned here — they are dedicated tools with their own risk levels. The LLM
planner can only emit the UI actions in ``ALLOWED_ACTIONS``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from automation.automation_logger import automation_logger
from automation.engine.context_memory import execution_memory
from automation.web_utils import search_url, site_url
from brain.utils import extract_json

ALLOWED_ACTIONS = {"open_app", "navigate", "click", "double_click", "type", "fill", "press", "hotkey", "scroll", "wait", "focus_window"}
ALLOWED_LEVELS = {"L1_DIRECT_API", "L2_DOM_WEB", "L3_WINDOWS_UIA", "L4_KEYBOARD", "L5_MOUSE", "L6_VISION"}
MAX_STEPS = 12


def _detect_messaging_app(query: str) -> Optional[str]:
    """Explicitly detects which messaging/email app the user wants using strict boundaries."""
    from automation.engine.messaging_handler import _detect_app_from_text

    return _detect_app_from_text(query)


def sanitize_steps(raw: Any) -> List[Dict[str, Any]]:
    """Keep only well-formed, allowed steps (drops shell/run_command and unknown actions)."""
    if isinstance(raw, dict):
        raw = raw.get("steps") or []
    if not isinstance(raw, list):
        return []
    steps: List[Dict[str, Any]] = []
    for item in raw[:MAX_STEPS]:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).lower().strip()
        if action not in ALLOWED_ACTIONS:
            automation_logger.warning(f"[Planner] Dropping disallowed step action '{action}'")
            continue
        step = {
            "action": action,
            "target": str(item.get("target", "") or "")[:300],
            "value": str(item.get("value", "") or "")[:2000],
            "description": str(item.get("description", "") or "")[:200],
        }
        level = str(item.get("level", "") or "")
        if level in ALLOWED_LEVELS:
            step["level"] = level
        if item.get("verify_type") and item.get("verify_type") != "none":
            step["verify_type"] = str(item["verify_type"])
            step["verify_target"] = str(item.get("verify_target", ""))
        steps.append(step)
    return steps


class TaskPlanner:
    """Turns a natural-language computer-use goal into executable steps."""

    async def plan(self, user_instruction: str) -> List[Dict[str, Any]]:
        resolved = execution_memory.resolve_reference(user_instruction)["resolved_text"]
        fast = self.fast_plan(resolved)
        if fast is not None:
            return fast
        return await self._plan_with_llm(resolved)

    def fast_plan(self, query: str) -> Optional[List[Dict[str, Any]]]:
        q = query.strip()
        low = q.lower().rstrip(".!")

        # "search youtube for X" / "play X on youtube" → direct results URL (no typing needed)
        m = re.match(r"^(?:open\s+youtube\s+and\s+)?(?:search|find|look\s+up|play)\s+(?:youtube\s+for\s+|for\s+)?(.+?)\s+on\s+youtube$", low) \
            or re.match(r"^(?:search|find)\s+youtube\s+for\s+(.+)$", low) \
            or re.match(r"^open\s+youtube\s+and\s+(?:search|look\s+up|find|play)\s+(?:for\s+)?(.+)$", low)
        if m:
            return [{"action": "navigate", "target": search_url(m.group(1), "youtube"), "value": "", "description": f"Search YouTube for “{m.group(1)}”"}]

        m = re.match(r"^(?:google|search\s+google\s+for|search\s+the\s+web\s+for|search\s+for)\s+(.+?)(?:\s+on\s+google)?$", low)
        if m:
            return [{"action": "navigate", "target": search_url(m.group(1)), "value": "", "description": f"Google “{m.group(1)}”"}]

        # "open chrome and go to github.com"
        m = re.match(r"^open\s+(chrome|brave|edge|firefox|browser)\s+and\s+(?:open|go\s+to|navigate\s+to|visit)\s+(.+?)(?:\s+in\s+it)?$", low)
        if m:
            url = site_url(m.group(2)) or search_url(m.group(2))
            app = "chrome" if m.group(1) == "browser" else m.group(1)
            return [
                {"action": "open_app", "target": app, "value": ""},
                {"action": "wait", "target": "", "value": "1.5"},
                {"action": "navigate", "target": url, "value": ""},
            ]
        return None

    async def _plan_with_llm(self, query: str) -> List[Dict[str, Any]]:
        from core.llm import llm

        prompt = (
            "You plan desktop/web automation for a Windows PC. Break the goal into at most 10 atomic steps.\n"
            f'Goal: "{query}"\n\n'
            "Allowed actions (nothing else — no shell commands):\n"
            "- open_app (target=app name)\n- navigate (target=URL)\n- click (target=visible text/label)\n"
            "- type (value=text, types into the focused field)\n- fill (target=field label, value=text)\n"
            "- press (value=enter|tab|escape|…)\n- hotkey (value=ctrl+t etc.)\n- scroll (value=up|down)\n"
            "- wait (value=seconds)\n- focus_window (target=window title)\n"
            "Prefer direct URLs (e.g. search result URLs) over typing into search boxes.\n"
            'Reply with a JSON array of {"action","target","value","description"} only.'
        )
        try:
            res = await llm.chat([{"role": "user", "content": prompt}], temperature=0.1, max_tokens=700)
            return sanitize_steps(extract_json(res.content))
        except Exception as exc:
            automation_logger.warning(f"[Planner] LLM planning unavailable: {exc}")
            return []


task_planner = TaskPlanner()
