"""Two-stage intent router.

Stage 1 — exact-command fast path: an anchored grammar over the *whole* normalized
utterance (``^…$`` with word boundaries). It only matches short, unambiguous
imperative commands ("volume up", "lock the screen", "open chrome") and maps them
straight to a tool with arguments — no LLM call, ~1 ms. There is no substring
matching, so "How do I reboot my router?" can never become a reboot.

Stage 2 — everything else goes to the LLM agent with native tool calling. The
set of tools offered depends on the utterance shape: questions and statements
("what is…", "explain…", "should I…", "I want to…") only get read-only tools;
side-effectful tools are offered only for imperative requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

from automation.web_utils import KNOWN_SITES, site_url


@dataclass
class RouteDecision:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    rule: str = ""
    confidence: float = 1.0


_WAKE = re.compile(r"^(?:(?:hey|hi|ok|okay|yo|hello)\s+)?jarvis\b[\s,.:!-]*", re.I)
_POLITE_PREFIX = re.compile(
    r"^(?:please|pls|plz|kindly|just|now|so|um+|uh+|okay|ok|"
    r"can you|could you|would you|will you|can u|could u|would you please|could you please|can you please|"
    r"i want you to|i need you to|i'd like you to|go ahead and)\s+",
    re.I,
)
_POLITE_SUFFIX = re.compile(r"\s+(?:please|pls|for me|right now|now|thanks|thank you|asap)$|(?<!\bin)(?<!\bof)(?<!\bfor)(?<!\bmy)(?<!\bthe)(?<!\bopen)[\s,]+jarvis$", re.I)


def normalize(text: str) -> Tuple[str, bool]:
    """Lower-case, strip wake word / politeness / trailing punctuation.

    Returns (normalized_text, had_polite_prefix).
    """
    t = re.sub(r"\s+", " ", text.strip())
    t = _WAKE.sub("", t)
    t = t.strip(" \t\n.,!?;:\"'")
    polite = False
    for _ in range(4):
        new = _POLITE_PREFIX.sub("", t)
        if new == t:
            break
        polite = True
        t = new.strip()
    for _ in range(3):
        new = _POLITE_SUFFIX.sub("", t)
        if new == t:
            break
        t = new.strip()
    return t.strip(" .,!?"), polite


# ── Utterance shape ──────────────────────────────────────────────────────

IMPERATIVE_VERBS = {
    "open", "launch", "start", "run", "close", "quit", "exit", "kill", "end", "terminate", "turn", "switch",
    "set", "change", "increase", "decrease", "raise", "lower", "reduce", "boost", "mute", "unmute", "play",
    "pause", "resume", "skip", "next", "previous", "stop", "lock", "shutdown", "shut", "restart", "reboot",
    "sleep", "hibernate", "log", "sign", "put", "send", "message", "msg", "text", "email", "mail", "tell",
    "ask", "dm", "whatsapp", "notify", "search", "google", "go", "navigate", "visit", "browse", "take",
    "capture", "grab", "record", "replay", "minimize", "maximize", "restore", "snap", "show", "hide", "type",
    "click", "press", "scroll", "fill", "copy", "paste", "create", "make", "write", "save", "delete", "remove",
    "trash", "move", "rename", "organize", "organise", "sort", "zip", "unzip", "compress", "extract",
    "convert", "download", "install", "uninstall", "enable", "disable", "connect", "disconnect", "empty",
    "clear", "cancel", "undo", "remind", "schedule", "add", "dim", "brighten", "compose", "draft", "reply",
    "use", "perform", "execute", "automate", "repeat", "activate", "deactivate", "flush", "edit", "append",
    "remember", "forget", "note", "focus", "close", "bring", "minimise", "maximise", "toggle", "eject",
    "like", "subscribe", "unsubscribe", "find", "watch", "follow", "share", "duplicate", "back",
    "new", "refresh", "reload",
}
_NOT_IMPERATIVE_AFTER = {"tell": {"me", "us"}, "show": {"me"}, "ask": {"me", "you"}, "make": {"sense"}, "note": {"that"}}


def is_imperative(text: str) -> bool:
    """True when the utterance is a request to *do* something (side-effect tools may be offered)."""
    t, _ = normalize(text)
    words = t.lower().split()
    if not words:
        return False
    first = words[0]
    if first not in IMPERATIVE_VERBS:
        return False
    nxt = words[1] if len(words) > 1 else ""
    return nxt not in _NOT_IMPERATIVE_AFTER.get(first, set())


_AFFIRM = re.compile(
    r"^(?:yes|yeah|yep|yup|ya|sure|ok|okay|confirm|confirmed|do it|go ahead|proceed|yes please|please do|"
    r"affirmative|correct|send it|yes do it|go for it|yes go ahead|yes confirm|do that|yes send it|absolutely|y)"
    r"(?:\s+(?:please|jarvis|thanks))?$",
    re.I,
)
_NEGATE = re.compile(
    r"^(?:no|nope|nah|cancel|cancel it|cancel that|don'?t|do not|don'?t do (?:it|that)|abort|never ?mind|"
    r"no thanks|stop|wait|hold on|not now|negative|n)(?:\s+(?:please|jarvis|thanks))?$",
    re.I,
)


def is_affirmative(text: str) -> bool:
    t = re.sub(r"[.!?,]+", "", _WAKE.sub("", text.strip().lower())).strip()
    return bool(_AFFIRM.match(t))


def is_negative(text: str) -> bool:
    t = re.sub(r"[.!?,]+", "", _WAKE.sub("", text.strip().lower())).strip()
    return bool(_NEGATE.match(t))


# ── Stage 1: exact-command grammar ───────────────────────────────────────

KNOWN_APPS = {
    "chrome", "google chrome", "edge", "microsoft edge", "firefox", "brave", "opera", "notepad", "notepad++",
    "calculator", "calc", "paint", "vs code", "vscode", "visual studio code", "code", "visual studio",
    "pycharm", "intellij", "android studio", "spotify", "vlc", "discord", "slack", "telegram", "whatsapp",
    "zoom", "teams", "microsoft teams", "outlook", "word", "excel", "powerpoint", "onenote", "terminal",
    "windows terminal", "powershell", "cmd", "command prompt", "explorer", "file explorer", "task manager",
    "settings", "control panel", "camera", "photos", "store", "microsoft store", "steam", "obs", "obs studio",
    "snipping tool", "clock", "calendar", "mail", "postman", "git bash", "docker desktop", "figma", "notion",
    "obsidian", "anydesk", "skype", "epic games", "whatsapp desktop", "telegram desktop", "word pad", "wordpad",
}
_SYSTEM_TOOL_NAMES = {
    "task manager": ("task_manager", ""), "settings": ("settings", ""), "windows settings": ("settings", ""),
    "file explorer": ("file_explorer", ""), "explorer": ("file_explorer", ""), "my files": ("file_explorer", ""),
    "bluetooth settings": ("settings", "bluetooth"), "wifi settings": ("settings", "network-wifi"),
    "display settings": ("settings", "display"), "sound settings": ("settings", "sound"),
}
_FORBIDDEN_APP_WORDS = {"learning", "source", "with", "about", "my career", "a", "an", "the", "to", "doing", "working", "over"}


def _is_known_app(name: str) -> bool:
    n = name.strip().lower()
    if n in KNOWN_APPS:
        return True
    try:
        from automation.application_controller import dynamic_app_manager

        dynamic_app_manager._ensure_index()
        return n in dynamic_app_manager._apps_cache
    except Exception:
        return False


def _num(m: re.Match, group: str, default: Optional[int] = None) -> Optional[int]:
    v = m.groupdict().get(group)
    return int(v) if v and v.isdigit() else default


Rule = Tuple[Pattern[str], Callable[[re.Match], Optional[RouteDecision]], str]
_RULES: List[Rule] = []


def rule(pattern: str, name: str = ""):
    rx = re.compile(r"^(?:" + pattern + r")$", re.I)

    def deco(fn: Callable[[re.Match], Optional[RouteDecision]]):
        _RULES.append((rx, fn, name or fn.__name__))
        return fn

    return deco


_DEVICE = r"(?:the |my |this )?(?:pc|computer|laptop|system|machine|desktop)"
_PCT = r"(?:\s*(?:percent|per cent|%))?"

# Safety controls first.
rule(r"(?:emergency stop|stop|stop it|stop everything|stop all(?: automation| actions)?|halt|abort everything|kill switch)", "stop")(
    lambda m: RouteDecision("automation_control", {"action": "stop"}))
rule(r"(?:resume|resume automation|continue automation|clear (?:the )?emergency stop)", "resume")(
    lambda m: RouteDecision("automation_control", {"action": "resume"}))
rule(r"(?:enable|turn on|start|activate|switch on) dry[ -]?run(?: mode)?|dry[ -]?run (?:on|mode on)", "dry_run_on")(
    lambda m: RouteDecision("automation_control", {"action": "dry_run_on"}))
rule(r"(?:disable|turn off|stop|deactivate|switch off|exit) dry[ -]?run(?: mode)?|dry[ -]?run (?:off|mode off)", "dry_run_off")(
    lambda m: RouteDecision("automation_control", {"action": "dry_run_off"}))
rule(r"automation status|(?:is )?dry[ -]?run (?:on|enabled|active)|safety status", "automation_status")(
    lambda m: RouteDecision("automation_status", {}))

# Time & system information (read-only).
rule(r"(?:what(?:'s| is) the )?(?:current |local )?time(?: is it)?(?: now| right now)?|what time is it(?: now)?|tell me the time|time please", "time")(
    lambda m: RouteDecision("get_time_date", {}))
rule(r"what(?:'s| is) (?:the |today'?s )?date(?: today)?|what day is (?:it|today)(?: today)?|today'?s date|what(?:'s| is) today", "date")(
    lambda m: RouteDecision("get_time_date", {}))


@rule(r"(?:what(?:'s| is) (?:my |the )?|check (?:my |the )?|show (?:me )?(?:my |the )?|how much )?(?P<what>battery(?: level| percentage| status| life)?|cpu(?: usage| load)?|ram(?: usage)?|memory usage|disk(?: space| usage)?|storage|system status|system info(?:rmation)?|pc status|computer status|temperature)(?: left| do i have| is left)?", "status")
def _status(m: re.Match) -> RouteDecision:
    w = m.group("what").lower()  # noqa: E501
    focus = "battery" if "battery" in w else "cpu" if "cpu" in w else "ram" if ("ram" in w or "memory" in w) else "disk" if ("disk" in w or "storage" in w) else "temperature" if "temp" in w else ""
    return RouteDecision("system_status", {"focus": focus})


# Volume.
rule(r"(?:increase|raise|turn up|boost|bump up|pump up) (?:the )?(?:volume|sound|audio)(?: by (?P<n>\d{1,3})" + _PCT + r")?|(?:volume|sound) up(?: by (?P<n2>\d{1,3})" + _PCT + r")?|louder|turn it up", "volume_up")(
    lambda m: RouteDecision("set_volume", {"action": "up", "step": _num(m, "n") or _num(m, "n2") or 10}))
rule(r"(?:decrease|lower|turn down|reduce|drop) (?:the )?(?:volume|sound|audio)(?: by (?P<n>\d{1,3})" + _PCT + r")?|(?:volume|sound) down(?: by (?P<n2>\d{1,3})" + _PCT + r")?|quieter|softer|turn it down", "volume_down")(
    lambda m: RouteDecision("set_volume", {"action": "down", "step": _num(m, "n") or _num(m, "n2") or 10}))
rule(r"(?:set|change|put|make) (?:the )?(?:volume|sound) (?:to|at) (?P<n>\d{1,3})" + _PCT + r"|volume (?:to )?(?P<n2>\d{1,3})" + _PCT, "volume_set")(
    lambda m: RouteDecision("set_volume", {"action": "set", "level": _num(m, "n") if _num(m, "n") is not None else _num(m, "n2")}))
rule(r"(?:mute|unmute|toggle mute)(?: (?:the )?(?:volume|sound|audio|speakers?|pc|computer|laptop))?", "mute")(
    lambda m: RouteDecision("set_volume", {"action": "mute"}))
rule(r"what(?:'s| is) the (?:current )?volume(?: level)?|current volume|volume level", "volume_get")(
    lambda m: RouteDecision("set_volume", {"action": "get"}))

# Brightness.
rule(r"(?:increase|raise|turn up|boost) (?:the )?(?:screen )?brightness(?: by (?P<n>\d{1,3})" + _PCT + r")?|brightness up|brighter|make (?:it|the screen) brighter|brighten(?: the)?(?: screen)?", "brightness_up")(
    lambda m: RouteDecision("set_brightness", {"action": "up", "step": _num(m, "n") or 10}))
rule(r"(?:decrease|lower|turn down|reduce) (?:the )?(?:screen )?brightness(?: by (?P<n>\d{1,3})" + _PCT + r")?|brightness down|dimmer|dim(?: the)?(?: screen| display)?|make (?:it|the screen) darker", "brightness_down")(
    lambda m: RouteDecision("set_brightness", {"action": "down", "step": _num(m, "n") or 10}))
rule(r"(?:set|change|put|make) (?:the )?(?:screen )?brightness (?:to|at) (?P<n>\d{1,3})" + _PCT + r"|brightness (?:to )?(?P<n2>\d{1,3})" + _PCT, "brightness_set")(
    lambda m: RouteDecision("set_brightness", {"action": "set", "level": _num(m, "n") if _num(m, "n") is not None else _num(m, "n2")}))

# Media.
rule(r"(?:play|pause|resume|unpause)(?: the)? (?:music|song|track|playback|media|video)|play|pause|play pause", "media_toggle")(
    lambda m: RouteDecision("media_control", {"action": "play_pause"}))
rule(r"(?:next|skip)(?: the)?(?: track| song)?|skip (?:this )?(?:track|song)|play (?:the )?next (?:track|song)", "media_next")(
    lambda m: RouteDecision("media_control", {"action": "next"}))
rule(r"previous(?: track| song)?|(?:play )?(?:the )?(?:previous|last) (?:track|song)|go back a (?:track|song)", "media_prev")(
    lambda m: RouteDecision("media_control", {"action": "previous"}))

# Windows.
rule(r"(?P<a>minimi[sz]e|maximi[sz]e|restore|close) (?:this |the |current |active |that )?window|(?P<a2>minimi[sz]e|maximi[sz]e) (?:it|this|that)", "window_active")(
    lambda m: RouteDecision("window_control", {"action": (m.group("a") or m.group("a2")).lower().replace("ise", "ize")}))
rule(r"(?:show|go to)(?: the)? desktop|(?:minimi[sz]e|hide) (?:all|everything)(?: windows)?", "show_desktop")(
    lambda m: RouteDecision("window_control", {"action": "show_desktop"}))
rule(r"snap (?:this |the )?(?:window )?(?:to )?(?:the )?(?P<d>left|right)(?: side)?", "snap")(
    lambda m: RouteDecision("window_control", {"action": "snap", "direction": m.group("d")}))
rule(r"(?:switch|change) windows?|alt tab|next window", "switch_window")(
    lambda m: RouteDecision("window_control", {"action": "switch"}))
rule(r"(?:list|show)(?: me)?(?: all)?(?: the)?(?: open)? windows|what windows are open", "list_windows")(
    lambda m: RouteDecision("list_windows", {}))

# Lock / power (power always needs confirmation in the gateway).
rule(r"lock(?: the| my)?(?: screen| pc| computer| laptop| system| workstation| it)?", "lock")(lambda m: RouteDecision("lock_screen", {}))
rule(r"(?:shut ?down|power off|turn off|switch off) " + _DEVICE + r"(?: now)?|shut ?down", "shutdown")(
    lambda m: RouteDecision("power_control", {"action": "shutdown"}))
rule(r"(?:restart|reboot) " + _DEVICE + r"(?: now)?|restart|reboot", "restart")(lambda m: RouteDecision("power_control", {"action": "restart"}))
rule(r"(?:put|send) " + _DEVICE + r" to sleep|(?:sleep|suspend) " + _DEVICE, "sleep")(lambda m: RouteDecision("power_control", {"action": "sleep"}))
rule(r"hibernate(?: " + _DEVICE + r")?", "hibernate")(lambda m: RouteDecision("power_control", {"action": "hibernate"}))
rule(r"(?:log ?off|log out|sign out)(?: of (?:windows|my account|the pc))?", "logoff")(lambda m: RouteDecision("power_control", {"action": "logoff"}))
rule(r"(?:cancel|abort|stop) (?:the )?(?:scheduled )?(?:shutdown|shut down|restart|reboot)", "cancel_shutdown")(lambda m: RouteDecision("cancel_shutdown", {}))

# Screenshot.
rule(r"(?:take|capture|grab|save)(?: a| the)? (?:screenshot|screen ?shot|screen capture|snapshot of (?:the|my) screen)|screenshot", "screenshot")(
    lambda m: RouteDecision("take_screenshot", {}))

# Network.
@rule(r"(?:turn|switch) (?P<s>on|off) (?:the )?(?P<d>wi-?fi|wifi|bluetooth)|(?:turn|switch) (?:the )?(?P<d2>wi-?fi|wifi|bluetooth) (?P<s2>on|off)|(?P<v>enable|disable) (?:the )?(?P<d3>wi-?fi|wifi|bluetooth)|(?P<d4>wi-?fi|wifi|bluetooth) (?P<s4>on|off)", "network")
def _network(m: re.Match) -> RouteDecision:
    device = (m.group("d") or m.group("d2") or m.group("d3") or m.group("d4") or "").lower()
    state = (m.group("s") or m.group("s2") or m.group("s4") or ("on" if m.group("v") == "enable" else "off")).lower()
    return RouteDecision("network_toggle", {"device": "bluetooth" if "blue" in device else "wifi", "enable": state == "on"})


rule(r"(?:empty|clear|clean)(?: out)? (?:the |my )?(?:recycle bin|trash|bin)", "recycle_bin")(lambda m: RouteDecision("empty_recycle_bin", {}))

# Kill a process (always confirmed; exact-name match in the gateway).
rule(r"(?:kill|force (?:close|quit|stop)|end task(?: for)?|terminate) (?:the )?(?:process )?(?P<p>[\w.+-]{2,60})(?: process)?", "kill")(
    lambda m: RouteDecision("kill_process", {"target": m.group("p").lower()}))

# Memory.
rule(r"remember(?: that)? (?P<f>.{3,300})", "remember")(lambda m: RouteDecision("memory_remember", {"fact": m.group("f")}))
rule(r"what do you (?:know|remember) about me|what have you (?:saved|stored|remembered) about me|what do you remember|tell me what you know about me|who am i", "recall")(
    lambda m: RouteDecision("memory_recall", {}))
rule(r"forget (?:about )?(?:that )?(?P<t>.{2,120})", "forget")(lambda m: RouteDecision("memory_forget", {"topic": m.group("t")}))

# Model switching (per session).
rule(r"set (?:the )?(?:model|llm) to (?P<m3>[\w./:\- ]{2,60})", "set_model")(
    lambda m: RouteDecision("switch_model", {"model": m.group("m3").strip().lower()}))
rule(r"(?:switch|change|swap)(?: the)?(?: model| llm| brain| ai)? to (?P<m>[\w./:\- ]{2,60}?)(?: model)?|use (?P<m2>gemini|groq|openai|gpt|claude|anthropic|ollama|local|deepseek|openrouter|lm ?studio)(?: model)?", "switch_model")(
    lambda m: RouteDecision("switch_model", {"model": (m.group("m") or m.group("m2")).strip().lower()}))


# Workflows.
rule(r"(?:start )?record(?:ing)?(?: a)?(?: new)? workflow(?: called| named)? (?P<n>.{2,60})|start recording (?P<n2>.{2,60})", "record_start")(
    lambda m: RouteDecision("workflow_record_start", {"name": (m.group("n") or m.group("n2")).strip().lower()}))
rule(r"stop recording(?: (?:the )?workflow)?|(?:done|finish|end) recording", "record_stop")(lambda m: RouteDecision("workflow_record_stop", {}))
rule(r"(?:list|show)(?: me)?(?: my| the| all)?(?: saved)? workflows|what workflows do (?:i|you) have", "workflow_list")(
    lambda m: RouteDecision("workflow_list", {}))


@rule(r"(?:replay|run workflow|play workflow|repeat workflow) (?P<n>[\w -]{2,60}?)(?: with (?P<p>.+))?", "replay")
def _replay(m: re.Match) -> RouteDecision:
    from tools.builtin.workflows import parse_params

    return RouteDecision("workflow_replay", {"name": m.group("n").strip().lower(), "params": parse_params(m.group("p") or "")})


# Reminders, notes, to-dos, briefing, weather, clipboard (Phase 5).
@rule(r"(?:remind me|set (?:a |an )?reminder|set (?:an )?alarm)(?: for| to| at| in| on| every)? (?P<rest>.{3,200})", "reminder")
def _reminder(m: re.Match) -> Optional[RouteDecision]:
    from services.timeparse import parse_when

    rest = m.group("rest").strip()
    if parse_when(rest) is None:
        return None
    return RouteDecision("set_reminder", {"text": rest, "when": rest})


@rule(r"set (?:a |the )?timer (?:for )?(?P<rest>.{2,60})|(?P<rest2>\d+ (?:minutes?|mins?|seconds?|hours?)) timer", "timer")
def _timer(m: re.Match) -> Optional[RouteDecision]:
    from services.timeparse import parse_when

    rest = (m.group("rest") or m.group("rest2")).strip()
    when = rest if rest.lower().startswith("in ") else f"in {rest}"
    return RouteDecision("set_reminder", {"text": "⏲ Timer finished", "when": when}) if parse_when(when) else None


rule(r"(?:what are|show|list|show me|read)(?: me)? my (?:upcoming )?reminders|(?:do i have )?any reminders|my reminders", "reminders_list")(
    lambda m: RouteDecision("list_reminders", {}))
rule(r"(?:cancel|delete|remove|clear)(?: my| the| all)? reminders?(?: (?:about|for|to) )?(?P<q>.{0,80})", "reminder_cancel")(
    lambda m: RouteDecision("cancel_reminder", {"query": m.group("q").strip() or "all"}))
rule(r"(?:take|make|write)(?: a)? note(?: that|:)? (?P<t>.{2,500})|note (?:down|this)[: ]+(?P<t2>.{2,500})", "note_add")(
    lambda m: RouteDecision("note_add", {"text": (m.group("t") or m.group("t2")).strip()}))
rule(r"(?:show|read|list)(?: me)?(?: all)? my notes|what are my notes", "notes_list")(lambda m: RouteDecision("notes_list", {}))
rule(r"add (?P<t>.{2,200}?) to (?:my |the )?(?:to-?do|todo|task)(?: list)?", "todo_add")(
    lambda m: RouteDecision("todo_add", {"text": m.group("t").strip()}))
rule(r"(?:what(?:'s| is) on |show(?: me)? |read )?my (?:to-?do|todo|task)(?: list|s)?|(?:show|list)(?: my)? (?:to-?dos|todos|tasks)", "todo_list")(
    lambda m: RouteDecision("todo_list", {}))
rule(r"(?:mark|tick off|check off|complete) (?P<t>.{2,200}?)(?: as)?(?: done| complete| finished)?", "todo_done")(
    lambda m: RouteDecision("todo_done", {"query": m.group("t").strip()}))
rule(r"remove (?P<t>.{2,200}?) from (?:my |the )?(?:to-?do|todo|task)(?: list)?", "todo_remove")(
    lambda m: RouteDecision("todo_done", {"query": m.group("t").strip(), "remove": True}))
rule(r"(?:give me |read me )?(?:my |the )?(?:daily |morning )?briefing|brief me|what(?:'s| does) my day look like|what(?:'s| is) my day like", "briefing")(
    lambda m: RouteDecision("daily_briefing", {}))
rule(r"(?:what(?:'s| is)|how(?:'s| is)) the weather(?: like)?(?: (?:in|at|for) (?P<l>[\w ,.-]{2,40}?))?(?: today| now| right now)?|weather(?: (?:in|at|for) (?P<l2>[\w ,.-]{2,40}))?", "weather")(
    lambda m: RouteDecision("weather", {"location": (m.group("l") or m.group("l2") or "").strip()}))
rule(r"(?:show|what(?:'s| is) in)(?: me)? my clipboard(?: history)?|what did i (?:just )?copy|clipboard history", "clipboard")(
    lambda m: RouteDecision("clipboard_history", {}))
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
rule(r"paste (?:what|the thing|the text) i copied (?P<n>\d+|one|two|three|four|five|six|seven|eight|nine|ten) (?:items?|things?|times?|copies) (?:ago|back)", "paste_history")(
    lambda m: RouteDecision("paste_from_history", {"index": (int(m.group("n")) if m.group("n").isdigit() else _NUM_WORDS[m.group("n").lower()]) + 1}))

# Routines / modes, undo, file housekeeping, developer helpers.
rule(r"(?:start|activate|enable|turn on|begin|enter|switch to) (?:the |my )?(?P<n>[\w -]{2,40}?) mode", "routine_mode")(
    lambda m: RouteDecision("run_routine", {"name": f"{m.group('n').strip().lower()} mode"}))
rule(r"(?:run|start|do) (?:the |my )?(?P<n>[\w -]{2,40}?) routine", "routine_run")(
    lambda m: RouteDecision("run_routine", {"name": m.group("n").strip().lower()}))
rule(r"(?:list|show)(?: me)?(?: my| all)? routines|what routines do (?:i|you) have", "routine_list")(lambda m: RouteDecision("list_routines", {}))
rule(r"undo(?: that| it| this)?(?: last)?|undo the last (?:thing|action|operation|change|move|file operation)", "undo")(
    lambda m: RouteDecision("undo_last_action", {}))
rule(r"(?:organi[sz]e|tidy(?: up)?|clean up|sort) (?:my |the )?(?P<f>downloads|desktop|documents|pictures)(?: folder)?(?: by (?P<by>type|date|month))?", "organize")(
    lambda m: RouteDecision("organize_folder", {"folder": m.group("f").lower(), "by": "date" if (m.group("by") or "").lower() in ("date", "month") else "type"}))
rule(r"find(?: the)? duplicates?(?: files)? in (?:my |the )?(?P<f>[\w ./\\:~-]{2,80}?)(?: folder)?", "duplicates")(
    lambda m: RouteDecision("find_duplicate_files", {"folder": m.group("f").strip()}))
rule(r"open (?:my |the )?(?P<p>[\w .-]{2,40}?) project(?: in (?:vs ?code|visual studio code|code))?", "open_project")(
    lambda m: RouteDecision("open_project", {"project": m.group("p").strip()}))
rule(r"git status(?: (?:of|for|in) (?:my |the )?(?P<p>[\w .-]{2,40}?)(?: project)?)?", "git_status")(
    lambda m: RouteDecision("git_status", {"project": (m.group("p") or ".").strip()}))
rule(r"(?:summari[sz]e|show|what are) (?:today'?s|my today'?s) commits(?: (?:in|for|of) (?:my |the )?(?P<p>[\w .-]{2,40}?)(?: project)?)?", "git_today")(
    lambda m: RouteDecision("git_today_commits", {"project": (m.group("p") or ".").strip()}))
rule(r"run (?:the )?tests(?: (?:in|for|of) (?:my |the )?(?P<p>[\w .-]{2,40}?)(?: project)?)?", "run_tests")(
    lambda m: RouteDecision("run_tests", {"project": (m.group("p") or ".").strip()}))


# Browser searches.
@rule(r"(?:search|look up|find|play) (?:for )?(?P<q>.+?) on youtube|search youtube for (?P<q2>.+)|youtube (?P<q3>.+)|open youtube and (?:search|look up|find|play) (?:for )?(?P<q4>.+)", "youtube")
def _youtube(m: re.Match) -> RouteDecision:
    q = next(g for g in (m.group("q"), m.group("q2"), m.group("q3"), m.group("q4")) if g)
    return RouteDecision("search_in_browser", {"query": q.lower(), "engine": "youtube"})


@rule(r"google (?P<q>.+)|search google for (?P<q2>.+)|search (?:for )?(?P<q3>.+?) on google|open google and search (?:for )?(?P<q4>.+)", "google")
def _google(m: re.Match) -> RouteDecision:
    q = next(g for g in (m.group("q"), m.group("q2"), m.group("q3"), m.group("q4")) if g)
    return RouteDecision("search_in_browser", {"query": q.lower(), "engine": "google"})


# Open / close apps and websites.
@rule(r"(?:open|launch|start|run|fire up|bring up)(?: up)? (?:the |my )?(?P<x>[\w .+&'-]{1,40}?)(?: app| application| program| website| site| browser)?", "open")
def _open(m: re.Match) -> Optional[RouteDecision]:
    x = m.group("x").strip().lower()
    if not x or x in _FORBIDDEN_APP_WORDS or len(x.split()) > 4:
        return None
    if x in _SYSTEM_TOOL_NAMES:
        tool, page = _SYSTEM_TOOL_NAMES[x]
        return RouteDecision("open_system_tool", {"tool": tool, "page": page})
    url = site_url(x)
    if x in KNOWN_SITES or (url and "." in x):
        return RouteDecision("open_website", {"url": url or x})
    if _is_known_app(x):
        return RouteDecision("open_app", {"name": x})
    return None  # unknown target → let the LLM decide (it may ask)


rule(r"(?:go to|visit|navigate to|browse to|open)(?: the)?(?: website| site| url| link| page)? (?P<x>[\w./:?=&%#+-]{2,120})", "go_to_site")(
    lambda m: RouteDecision("open_website", {"url": site_url(m.group("x"))}) if site_url(m.group("x")) else None)


@rule(r"(?:close|quit|exit) (?:the |my )?(?P<x>[\w .+-]{2,40}?)(?: app| application| program)?", "close_app")
def _close(m: re.Match) -> Optional[RouteDecision]:
    x = m.group("x").strip().lower()
    if x in ("this", "it", "that", "everything", "all"):
        return None
    return RouteDecision("close_app", {"name": x}) if _is_known_app(x) else None


def fast_route(text: str) -> Optional[RouteDecision]:
    """Match the whole utterance against the command grammar. None → use the LLM."""
    t, _ = normalize(text)
    if not t or len(t) > 160:
        return None
    for rx, fn, name in _RULES:
        m = rx.match(t)
        if m:
            decision = fn(m)
            if decision is not None:
                decision.rule = name
                return decision
    return None
