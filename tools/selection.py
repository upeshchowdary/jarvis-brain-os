"""Pick the relevant subset of tools for one request.

Sending 80+ tool schemas on every call costs thousands of prompt tokens (latency, money)
and lowers tool-choice accuracy. We always offer a small core set plus the groups whose
keywords appear in the request (or recent conversation).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from tools.base import Tool

CORE = {
    "clarify", "get_time_date", "web_search", "memory_recall", "system_status", "open_app", "open_website",
    "search_in_browser", "computer_task", "automation_status", "memory_remember",
}

GROUPS: Dict[str, Dict[str, Sequence[str]]] = {
    "system": {
        "tools": ["set_volume", "media_control", "set_brightness", "window_control", "lock_screen", "power_control",
                  "cancel_shutdown", "close_app", "kill_process", "take_screenshot", "network_toggle", "open_system_tool",
                  "empty_recycle_bin", "list_windows", "list_processes", "list_installed_apps", "automation_control"],
        "keywords": ["volume", "sound", "louder", "quieter", "mute", "music", "song", "track", "brightness", "screen", "window",
                     "minimi", "maximi", "snap", "lock", "shut", "restart", "reboot", "sleep", "hibernate", "wifi", "wi-fi",
                     "bluetooth", "process", "kill", "task", "screenshot", "app", "apps", "close", "quit", "recycle", "settings",
                     "battery", "cpu", "ram", "memory usage", "running", "installed", "stop", "dry run", "program"],
    },
    "files": {
        "tools": ["files_list", "files_size", "files_info", "files_read", "file_create", "file_append", "folder_create",
                  "file_rename", "file_move", "file_copy", "file_delete", "organize_folder", "bulk_rename", "zip_files",
                  "unzip_file", "undo_last_action", "find_duplicate_files", "find_files", "search_file_contents"],
        "keywords": ["file", "files", "folder", "directory", "desktop", "download", "documents", "pictures", "rename", "move",
                     "copy", "delete", "trash", "zip", "unzip", "compress", "extract", "organize", "organise", "tidy", "clean",
                     "duplicate", "undo", ".txt", ".pdf", ".py", ".docx", "size", "disk", "storage", "modified"],
    },
    "docs": {
        "tools": ["document_read", "documents_ask", "extract_tables_to_excel", "convert_document", "files_read"],
        "keywords": ["pdf", "docx", "word", "document", "excel", "xlsx", "csv", "table", "slides", "pptx", "powerpoint",
                     "convert", "summar", "notes from", "report", "assignment", "syllabus", "contract", "resume"],
    },
    "productivity": {
        "tools": ["set_reminder", "list_reminders", "cancel_reminder", "note_add", "notes_list", "todo_add", "todo_list",
                  "todo_done", "clipboard_history", "paste_from_history", "daily_briefing", "weather"],
        "keywords": ["remind", "reminder", "alarm", "timer", "schedule", "every", "tomorrow", "tonight", "note", "todo",
                     "to-do", "to do", "task list", "clipboard", "copied", "paste", "briefing", "brief", "weather",
                     "temperature outside", "my day", "agenda"],
    },
    "dev": {
        "tools": ["git_status", "git_today_commits", "run_tests", "open_project", "run_named_command"],
        "keywords": ["git", "commit", "branch", "repo", "test", "tests", "pytest", "project", "vs code", "vscode", "code",
                     "build", "ping", "ipconfig", "dns", "error", "bug"],
    },
    "web": {
        "tools": ["read_webpage", "research", "browser_task"],
        "keywords": ["research", "article", "website", "page", "link", "url", "http", "news", "latest", "compare", "sources",
                     "browser", "online", "buy", "order", "form", "sign up", "book", "search"],
    },
    "messaging": {
        "tools": ["send_message", "compose_email"],
        "keywords": ["message", "msg", "text", "whatsapp", "telegram", "email", "mail", "send", "tell", "dm", "reply",
                     "slack", "discord", "invite", "inform"],
    },
    "vision": {
        "tools": ["screen_look"],
        "keywords": ["screen", "see", "look", "showing", "displayed", "visible", "on my monitor", "this error", "popup",
                     "what's this", "what is this", "image", "picture"],
    },
    "workflow": {
        "tools": ["workflow_record_start", "workflow_record_stop", "workflow_list", "workflow_replay", "run_routine",
                  "list_routines"],
        "keywords": ["record", "replay", "workflow", "routine", "mode", "automate", "repeat", "macro"],
    },
    "memory": {
        "tools": ["memory_forget", "switch_model"],
        "keywords": ["forget", "about me", "remember", "model", "switch", "gemini", "groq", "ollama", "openai", "claude", "gpt"],
    },
}

MAX_TOOLS = 28


def select_tools(query: str, available: Iterable[Tool], context: str = "") -> List[Tool]:
    pool = {t.name: t for t in available}
    text = f" {query.lower()} {context.lower()} "
    wanted: List[str] = [n for n in CORE if n in pool]
    for group in GROUPS.values():
        if any(k in text for k in group["keywords"]):
            wanted += [n for n in group["tools"] if n in pool]
    # External MCP tools: include when their server is mentioned, or all if there are only a few.
    mcp = [n for n in pool if n.startswith("mcp_")]
    if len(mcp) <= 8:
        wanted += mcp
    else:
        wanted += [n for n in mcp if re.search(rf"\b{re.escape(n.split('_')[1])}\b", text)]
    seen, ordered = set(), []
    for name in wanted:
        if name not in seen:
            seen.add(name)
            ordered.append(pool[name])
    # Anything not covered by groups (e.g. future tools) is still reachable when few tools were chosen.
    if len(ordered) < 14:
        grouped = set(CORE) | {n for g in GROUPS.values() for n in g["tools"]}
        ordered += [t for n, t in pool.items() if n not in grouped and n not in seen and not n.startswith("mcp_")]
    return ordered[:MAX_TOOLS]
