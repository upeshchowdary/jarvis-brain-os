"""Utility functions for JSON parsing, string sanitation, and token estimation."""

import re
import json
import socket
from datetime import datetime, timezone
from typing import Dict, Any


def is_internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    """Blocking connectivity check — never call from async code (use core.connectivity.is_online)."""
    for target in (host, "1.1.1.1"):
        try:
            sock = socket.create_connection((target, port), timeout=timeout)
            sock.close()
            return True
        except OSError:
            continue
    return False


def get_current_datetime_utc() -> str:
    """Return ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_current_datetime_local() -> Dict[str, str]:
    """Return local system date and time in human-readable 12-hour, 24-hour, and ISO formats."""
    now = datetime.now().astimezone()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "full_date": now.strftime("%A, %B %d, %Y"),
        "time_12h": now.strftime("%I:%M:%S %p"),
        "time_12h_short": now.strftime("%I:%M %p"),
        "time_24h": now.strftime("%H:%M:%S"),
        "timezone": now.strftime("%Z (%z)"),
        "formatted_full": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
    }


def estimate_token_count(text: str) -> int:
    """Estimate token length of a given text (approx. 4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_json(text: str) -> Any:
    """Extract the first JSON object *or array* from an LLM reply (handles ``` fences and prose)."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(cleaned[i:])
                return value
            except ValueError:
                continue
    return None


def extract_and_clean_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON object from model outputs."""
    text = text.strip()
    if not text:
        return {}

    # Strip markdown code blocks
    if "```" in text:
        text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    # Search for outermost JSON braces
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
