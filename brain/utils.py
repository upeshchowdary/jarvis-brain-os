"""Utility functions for JSON parsing, string sanitation, and token estimation."""

import re
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional


def get_current_datetime_utc() -> str:
    """Return ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def estimate_token_count(text: str) -> int:
    """Estimate token length of a given text (approx. 4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


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
