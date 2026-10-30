"""Task Planner & Graph Engine for JARVIS Automation Engine.

Decomposes complex natural language instructions into an executable TaskGraph
with pre-conditions, verification conditions, and assigned ActionLevels.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Any

from automation.automation_logger import automation_logger
from automation.engine.action_router import ActionLevel, ActionSpec
from automation.engine.context_memory import execution_memory
from automation.engine.verification import VerificationCondition


@dataclass
class TaskNode:
    """A single atomic step in the task graph."""
    id: str
    description: str
    action_spec: ActionSpec
    verification: Optional[VerificationCondition] = None
    dependencies: List[str] = field(default_factory=list)
    is_completed: bool = False
    is_failed: bool = False
    result_data: Any = None


@dataclass
class TaskGraph:
    """Directed graph of task steps representing the overall execution plan."""
    goal: str
    nodes: List[TaskNode] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    current_node_index: int = 0


def _detect_messaging_app(query: str) -> Optional[str]:
    """Explicitly detects which messaging/email app the user wants using strict boundaries."""
    q = query.lower()
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", query) or re.search(r"\b(?:gmail|email|e-mail|mail)\b", q):
        return "gmail"
    if re.search(r"\b(?:whatsapp|wa|wapp|whats\s*app)\b", q):
        return "whatsapp"
    if re.search(r"\b(?:telegram|tg|tele)\b", q):
        return "telegram"
    if re.search(r"\b(?:discord)\b", q):
        return "discord"
    if re.search(r"\b(?:slack)\b", q):
        return "slack"
    return None


class TaskPlanner:
    """Decomposes user instructions into executable, verifiable TaskGraphs."""

    async def plan(self, user_instruction: str) -> TaskGraph:
        """
        Creates an executable TaskGraph from user instruction.
        Resolves pronouns/context references first, then generates atomic steps.
        """
        # 1. Reference & Context Resolution
        resolved_context = execution_memory.resolve_reference(user_instruction)
        effective_query = resolved_context["resolved_text"]

        automation_logger.info(f"[Planner] Original: '{user_instruction}' -> Resolved: '{effective_query}'")

        # 2. Check for fast-path deterministic patterns
        fast_graph = self._try_deterministic_plan(effective_query)
        if fast_graph is not None:
            return fast_graph

        # 3. LLM-based Decomposition
        return await self._plan_with_llm(effective_query)

    def _try_deterministic_plan(self, query: str) -> Optional[TaskGraph]:
        """Fast-path planner for common structured patterns."""
        q_low = query.lower().strip()

        # ── 1. System Control (Window, Volume, Brightness, Power, System Info) ──
        sys_plan = self._try_system_control_plan(q_low, query)
        if sys_plan is not None:
            return sys_plan

        # ── 2. Messaging / Communication detection ──────────────
        msg_app = _detect_messaging_app(q_low)
        is_messaging = (
            msg_app is not None and
            any(w in q_low for w in ("message", "msg", "text", "ask", "tell", "say", "send", "invite", "mail", "email", "dm", "write", "compose"))
        )
        if is_messaging and msg_app:
            nodes = [
                TaskNode(
                    id="step_1_send_message",
                    description=f"Send {msg_app.title()} communication via JARVIS Messaging Engine",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_3_WINDOWS_UIA,
                        action_type="send_message",
                        target=msg_app,
                        value=query,
                    ),
                ),
            ]
            return TaskGraph(goal=query, nodes=nodes)

        # ── 3. Web Navigation & Search Plan ──────────────
        web_plan = self._try_web_plan(q_low, query)
        if web_plan is not None:
            return web_plan

        return None

    def _try_system_control_plan(self, q_low: str, original_query: str) -> Optional[TaskGraph]:
        """Maps natural language system commands directly to L1 system_control actions."""
        # ── Window Management ──
        if any(w in q_low for w in ("minimize this window", "minimize window", "minimize the window", "minimize current window", "minimize active window", "minimize it")) or q_low == "minimize":
            target_app = ""
            m = re.search(r"minimize\s+(?:the\s+)?([a-zA-Z0-9_\-]+)", q_low)
            if m and m.group(1) not in ("this", "the", "window", "current", "active", "it"):
                target_app = m.group(1)
            node = TaskNode(
                id="step_1_minimize",
                description="Minimize window",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="minimize_window",
                    value=target_app,
                    parameters={"target": target_app},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("maximize this window", "maximize window", "maximize the window", "maximize current window", "maximize active window", "maximize it")) or q_low == "maximize":
            target_app = ""
            m = re.search(r"maximize\s+(?:the\s+)?([a-zA-Z0-9_\-]+)", q_low)
            if m and m.group(1) not in ("this", "the", "window", "current", "active", "it"):
                target_app = m.group(1)
            node = TaskNode(
                id="step_1_maximize",
                description="Maximize window",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="maximize_window",
                    value=target_app,
                    parameters={"target": target_app},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("close this window", "close the window", "close active window", "close current window", "close this app")) or q_low in ("close window", "close active"):
            node = TaskNode(
                id="step_1_close_window",
                description="Close active window",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="close_window",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("restore window", "unmaximize", "restore this window")):
            node = TaskNode(
                id="step_1_restore",
                description="Restore window",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="restore_window",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("show desktop", "hide all windows", "minimize all windows", "go to desktop")):
            node = TaskNode(
                id="step_1_show_desktop",
                description="Show Windows desktop",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="show_desktop",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("switch window", "next window", "previous window", "alt tab")):
            node = TaskNode(
                id="step_1_switch_window",
                description="Switch window",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="switch_window",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("list windows", "show open windows", "what windows are open", "list open windows")):
            node = TaskNode(
                id="step_1_list_windows",
                description="List open windows",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="list_windows",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if "snap" in q_low and any(d in q_low for d in ("left", "right", "up", "down", "top", "bottom", "half")):
            d = "left" if "left" in q_low else ("right" if "right" in q_low else ("up" if "up" in q_low else "down"))
            node = TaskNode(
                id="step_1_snap",
                description=f"Snap window {d}",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="snap_window",
                    value=d,
                    parameters={"target": d},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        # ── Audio / Volume ──
        if any(w in q_low for w in ("volume up", "increase volume", "raise volume", "turn up the volume", "turn up volume", "louder")):
            node = TaskNode(
                id="step_1_volume_up",
                description="Increase volume",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="volume_up",
                    parameters={"value": "5"},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("volume down", "decrease volume", "lower volume", "turn down the volume", "turn down volume", "quieter")):
            node = TaskNode(
                id="step_1_volume_down",
                description="Decrease volume",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="volume_down",
                    parameters={"value": "5"},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("mute volume", "mute audio", "mute sound", "unmute volume", "unmute audio", "toggle mute")) or q_low in ("mute", "unmute"):
            node = TaskNode(
                id="step_1_mute",
                description="Toggle mute",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="volume_mute",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("get volume", "what is the volume", "what is volume", "current volume", "check volume", "show volume", "volume status")):
            node = TaskNode(
                id="step_1_get_volume",
                description="Get current system volume level",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="get_volume",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        m_vol = re.search(r"(?:set\s+volume\s+to|volume\s+to|change\s+volume\s+to)\s+(\d+)", q_low)
        if m_vol:
            val = m_vol.group(1)
            node = TaskNode(
                id="step_1_set_volume",
                description=f"Set volume to {val}%",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="set_volume",
                    parameters={"value": val},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])


        # ── Brightness ──
        if any(w in q_low for w in ("brightness up", "increase brightness", "raise brightness", "turn up brightness", "brighter", "make it brighter")):
            node = TaskNode(
                id="step_1_brightness_up",
                description="Increase brightness",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="brightness_up",
                    parameters={"value": "10"},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("brightness down", "decrease brightness", "lower brightness", "turn down brightness", "dim screen", "dim the screen", "dimmer")):
            node = TaskNode(
                id="step_1_brightness_down",
                description="Decrease brightness",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="brightness_down",
                    parameters={"value": "10"},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        m_bright = re.search(r"(?:set\s+brightness\s+to|brightness\s+to)\s+(\d+)", q_low)
        if m_bright:
            val = m_bright.group(1)
            node = TaskNode(
                id="step_1_set_brightness",
                description=f"Set brightness to {val}%",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="set_brightness",
                    parameters={"value": val},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        # ── Power Management ──
        if any(w in q_low for w in ("lock the screen", "lock screen", "lock laptop", "lock pc", "lock computer")) or q_low == "lock":
            node = TaskNode(
                id="step_1_lock",
                description="Lock workstation",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="lock_screen",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("put pc to sleep", "sleep pc", "sleep laptop", "put laptop to sleep")) or q_low == "sleep":
            node = TaskNode(
                id="step_1_sleep",
                description="Put system to sleep",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="sleep",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("turn off pc", "turn off laptop", "shutdown pc", "shutdown laptop", "power off")):
            node = TaskNode(
                id="step_1_shutdown",
                description="Shutdown PC",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="shutdown",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("restart pc", "restart laptop", "reboot pc", "reboot laptop", "restart computer", "reboot")):
            node = TaskNode(
                id="step_1_restart",
                description="Restart PC",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="restart",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        # ── System Utilities & Apps ──
        if any(w in q_low for w in ("open task manager", "open taskmgr", "show task manager")) or q_low == "task manager":
            node = TaskNode(
                id="step_1_taskmgr",
                description="Open Task Manager",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="open_task_manager",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("open settings", "windows settings", "go to settings")):
            node = TaskNode(
                id="step_1_settings",
                description="Open Windows Settings",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="open_settings",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("open file explorer", "open explorer", "open my files", "open files")):
            node = TaskNode(
                id="step_1_explorer",
                description="Open File Explorer",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="open_file_explorer",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("empty recycle bin", "clear recycle bin", "empty trash")):
            node = TaskNode(
                id="step_1_recycle_bin",
                description="Empty Recycle Bin",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="empty_recycle_bin",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("take a screenshot", "take screenshot", "capture screen", "capture my screen")) or q_low == "screenshot":
            node = TaskNode(
                id="step_1_screenshot",
                description="Take screenshot",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="take_screenshot",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("system info", "system status", "hardware status", "pc specs", "laptop specs", "system telemetry")):
            node = TaskNode(
                id="step_1_sysinfo",
                description="Get system telemetry and hardware info",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="system_info",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        # ── WiFi & Network ──
        if any(w in q_low for w in ("turn on wifi", "enable wifi", "wifi on")):
            node = TaskNode(
                id="step_1_wifi_on",
                description="Enable WiFi",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="wifi_on",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("turn off wifi", "disable wifi", "wifi off")):
            node = TaskNode(
                id="step_1_wifi_off",
                description="Disable WiFi",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="wifi_off",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        if any(w in q_low for w in ("wifi status", "wifi info", "check wifi")):
            node = TaskNode(
                id="step_1_wifi_status",
                description="Get WiFi status",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="wifi_status",
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        # ── Process Killing ──
        m_kill = re.search(r"(?:kill\s+process|kill|terminate\s+process|terminate|close\s+process)\s+([a-zA-Z0-9_\-\.]+)", q_low)
        if m_kill and m_kill.group(1) not in ("this", "the", "window", "app", "application"):
            proc_target = m_kill.group(1)
            node = TaskNode(
                id="step_1_kill",
                description=f"Kill process '{proc_target}'",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_1_DIRECT_API,
                    action_type="system_control",
                    target="kill_process",
                    parameters={"target": proc_target},
                ),
            )
            return TaskGraph(goal=original_query, nodes=[node])

        return None


    def _try_web_plan(self, q_low: str, query: str) -> Optional[TaskGraph]:
        """Fast-path planner for web search, YouTube, and browser navigation."""
        # ── YouTube tasks ─────────────────────────────────────────────────────
        if "youtube" in q_low or ("yt" in q_low and any(k in q_low for k in ("search", "channel", "video", "subscribe"))):
            # Extract channel/search term
            channel_match = re.search(r"(?:channel|go to|visit|open)\s+['\"]?([a-zA-Z0-9_\s\-]+?)\s*(?:channel|$)", query, re.IGNORECASE)
            search_match = re.search(r"search\s+(?:for\s+)?['\"]?(.+?)['\"]?$", query, re.IGNORECASE)
            subscribe = "subscribe" in q_low or "sub" in q_low

            if channel_match:
                channel = channel_match.group(1).strip()
                search_query = f"{channel} channel"
            elif search_match:
                search_query = search_match.group(1).strip()
            else:
                # Fallback: use whole query as search
                search_query = re.sub(r"(open|go to|navigate to|search|youtube|and|the)", "", query, flags=re.IGNORECASE).strip()

            nodes = [
                TaskNode(
                    id="step_1_navigate_youtube",
                    description="Navigate to YouTube",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="navigate",
                        target="https://www.youtube.com",
                    ),
                ),
                TaskNode(
                    id="step_2_type_search",
                    description=f"Type '{search_query}' in YouTube search bar",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="type",
                        target="Search",
                        value=search_query,
                    ),
                    dependencies=["step_1_navigate_youtube"],
                ),
                TaskNode(
                    id="step_3_press_enter",
                    description="Submit search",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_4_KEYBOARD,
                        action_type="press",
                        target="enter",
                    ),
                    dependencies=["step_2_type_search"],
                ),
            ]

            if channel_match:
                nodes.append(TaskNode(
                    id="step_4_open_channel",
                    description=f"Click on '{channel}' channel in results",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="click",
                        target=channel,
                    ),
                    dependencies=["step_3_press_enter"],
                ))

            if subscribe:
                nodes.append(TaskNode(
                    id="step_5_subscribe",
                    description="Click Subscribe button",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="click",
                        target="Subscribe",
                    ),
                    dependencies=[nodes[-1].id],
                ))

            return TaskGraph(goal=query, nodes=nodes)

        # ── Google search ──────────────────────────────────────────────────────
        if any(k in q_low for k in ("google search", "search google", "search on google", "search for")):
            m = re.search(r"(?:search|google)\s+(?:for\s+|on\s+google\s+)?['\"]?(.+?)['\"]?$", query, re.IGNORECASE)
            search_term = m.group(1).strip() if m else query
            nodes = [
                TaskNode(
                    id="step_1_navigate_google",
                    description="Navigate to Google",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="navigate",
                        target="https://www.google.com",
                    ),
                ),
                TaskNode(
                    id="step_2_type_search",
                    description=f"Type '{search_term}' in Google search",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="type",
                        target="Search",
                        value=search_term,
                    ),
                    dependencies=["step_1_navigate_google"],
                ),
                TaskNode(
                    id="step_3_press_enter",
                    description="Submit search",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_4_KEYBOARD,
                        action_type="press",
                        target="enter",
                    ),
                    dependencies=["step_2_type_search"],
                ),
            ]
            return TaskGraph(goal=query, nodes=nodes)

        # ── Multi-step Browser Launch & Navigation: open <browser> and (open|go to|visit) <target> ──
        m_browser_nav = re.search(
            r"^open\s+(chrome|brave|edge|firefox|browser)\s+and\s+(?:open|go\s+to|navigate\s+to|visit|search(?:\s+for)?)\s+(?:a\s+)?(.+?)(?:\s+in\s+it)?$",
            q_low,
            re.IGNORECASE
        )
        if m_browser_nav:
            browser_name = m_browser_nav.group(1).strip()
            web_target = m_browser_nav.group(2).strip()
            nodes = [
                TaskNode(
                    id="step_1_open_browser",
                    description=f"Open {browser_name.title()}",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_1_DIRECT_API,
                        action_type="open_app",
                        target=browser_name,
                    ),
                ),
                TaskNode(
                    id="step_2_navigate_url",
                    description=f"Navigate to {web_target}",
                    action_spec=ActionSpec(
                        level=ActionLevel.LEVEL_2_DOM_WEB,
                        action_type="navigate",
                        target=web_target,
                    ),
                    dependencies=["step_1_open_browser"],
                ),
            ]
            return TaskGraph(goal=query, nodes=nodes)

        # ── Direct Navigation: go to <website> / open <website> ──
        m_direct_nav = re.search(
            r"^(?:go\s+to|navigate\s+to|open\s+website|visit)\s+(?:https?://)?([a-zA-Z0-9_\-\./\?\=\&\#\+]+)$",
            q_low,
            re.IGNORECASE
        )
        if m_direct_nav and not any(w in q_low for w in ("whatsapp", "telegram", "settings", "task manager", "explorer", "recycle bin")):
            target_url = m_direct_nav.group(1).strip()
            node = TaskNode(
                id="step_1_navigate",
                description=f"Navigate to {target_url}",
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_2_DOM_WEB,
                    action_type="navigate",
                    target=target_url,
                ),
            )
            return TaskGraph(goal=query, nodes=[node])

        return None


    async def _plan_with_llm(self, query: str) -> TaskGraph:
        """Decomposes arbitrary complex multi-step request using ModelManager."""
        from brain.model_manager import model_manager

        prompt = (
            f"You are the JARVIS Task Planner. Decompose the following computer automation goal "
            f"into an ordered list of atomic executable steps.\n\n"
            f"Goal: \"{query}\"\n\n"
            f"Available action levels:\n"
            f"- L1_DIRECT_API: open_app, run_command, read_file\n"
            f"- L2_DOM_WEB: navigate, click (web text/selector), fill/type (web input)\n"
            f"- L3_WINDOWS_UIA: click (native control name), type (native edit)\n"
            f"- L4_KEYBOARD: press (enter|tab|escape), hotkey (ctrl+t|ctrl+f), type\n"
            f"- L5_MOUSE: click (visible text/OCR target)\n\n"
            f"Output ONLY a JSON array with schema:\n"
            f"[\n"
            f"  {{\n"
            f'    "id": "step_1",\n'
            f'    "description": "...",\n'
            f'    "level": "L1_DIRECT_API|L2_DOM_WEB|L3_WINDOWS_UIA|L4_KEYBOARD|L5_MOUSE",\n'
            f'    "action": "open_app|navigate|click|type|press|hotkey",\n'
            f'    "target": "...",\n'
            f'    "value": "...",\n'
            f'    "verify_type": "text_visible|window_focused|file_exists|none",\n'
            f'    "verify_target": "..."\n'
            f"  }}\n"
            f"]\n"
            f"Output ONLY valid JSON."
        )

        try:
            res = await model_manager.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=600,
            )
            raw = (res.get("content", "") if isinstance(res, dict) else "").strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
            data = json.loads(raw)

            level_map = {
                "L1_DIRECT_API": ActionLevel.LEVEL_1_DIRECT_API,
                "L2_DOM_WEB": ActionLevel.LEVEL_2_DOM_WEB,
                "L3_WINDOWS_UIA": ActionLevel.LEVEL_3_WINDOWS_UIA,
                "L4_KEYBOARD": ActionLevel.LEVEL_4_KEYBOARD,
                "L5_MOUSE": ActionLevel.LEVEL_5_MOUSE,
            }

            nodes: List[TaskNode] = []
            for item in data:
                lvl = level_map.get(item.get("level", "L5_MOUSE"), ActionLevel.LEVEL_5_MOUSE)
                v_type = item.get("verify_type", "none")
                v_cond = None
                if v_type != "none" and item.get("verify_target"):
                    v_cond = VerificationCondition(
                        condition_type=v_type,
                        target=item["verify_target"],
                        timeout_seconds=4.0,
                    )

                node = TaskNode(
                    id=item.get("id", f"step_{len(nodes)+1}"),
                    description=item.get("description", ""),
                    action_spec=ActionSpec(
                        level=lvl,
                        action_type=item.get("action", "click"),
                        target=item.get("target", ""),
                        value=item.get("value", ""),
                    ),
                    verification=v_cond,
                )
                nodes.append(node)

            return TaskGraph(goal=query, nodes=nodes)

        except Exception as e:
            automation_logger.warning(f"[Planner] LLM decomposition error: {e}. Falling back to single step.")
            node = TaskNode(
                id="step_1",
                description=query,
                action_spec=ActionSpec(
                    level=ActionLevel.LEVEL_5_MOUSE,
                    action_type="click",
                    target=query,
                ),
            )
            return TaskGraph(goal=query, nodes=[node])


task_planner = TaskPlanner()
