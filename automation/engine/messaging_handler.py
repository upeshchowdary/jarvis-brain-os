"""Messaging intent parsing and drafting (pure — sending happens via the gateway).

``plan_message`` turns an instruction like "tell Rahul on WhatsApp I'll be late"
into an ``ActionRequest`` for ``message.send`` / ``email.compose``. The gateway then
shows a preview and requires confirmation before anything is sent.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from automation.automation_logger import automation_logger
from automation.gateway import ActionRequest
from brain.utils import extract_and_clean_json

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def _detect_app_from_text(text: str) -> Optional[str]:
    """Detect messaging/email app from text using strict word boundaries."""
    q = text.lower()
    if EMAIL_RE.search(text) or re.search(r"\b(?:gmail|email|e-mail|mail)\b", q):
        return "gmail"
    if re.search(r"\b(?:whatsapp|wa|wapp|whats\s*app)\b", q):
        return "whatsapp"
    if re.search(r"\b(?:telegram|tg|tele)\b", q):
        return "telegram"
    if re.search(r"\bdiscord\b", q):
        return "discord"
    if re.search(r"\bslack\b", q):
        return "slack"
    return None


_STOP_WORDS = {
    "him", "her", "them", "me", "us", "it", "that", "a", "the", "in", "on", "my", "this", "whatsapp", "telegram",
    "discord", "slack", "gmail", "email", "mail", "what", "how", "please", "and", "to", "message", "msg",
}


def _heuristic_contact(goal: str) -> str:
    email_m = EMAIL_RE.search(goal)
    if email_m:
        return email_m.group(0)
    patterns = [
        r"(?:message|msg|text|tell|ask|dm|mail|email)\s+([A-Za-z][a-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\b",
        r"\b(?:to)\s+([A-Za-z][a-z0-9_]+)\b",
        r"\b([A-Za-z][a-z0-9_]+)\s+on\s+(?:whatsapp|telegram|discord|slack)\b",
        r"(?:whatsapp|telegram|discord|slack)\s+(?:message\s+to\s+|to\s+)?([A-Za-z][a-z0-9_]+)\b",
    ]
    for p in patterns:
        m = re.search(p, goal, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in _STOP_WORDS and name.split()[0].lower() not in _STOP_WORDS:
                return name.title()
    return ""


def _heuristic_message(goal: str) -> str:
    m = re.search(r"\b(?:saying|that|say|asking|ask|tell\s+\w+)\s+['\"]?(.+?)['\"]?$", goal, re.IGNORECASE)
    if m:
        return m.group(1).strip(" '\"")
    m = re.search(r"\b(?:msg|message|text|dm)\s+\w+\s+(.+)$", goal, re.IGNORECASE)
    if m:
        return m.group(1).strip(" '\"")
    return goal


async def extract_messaging_intent(goal: str) -> Tuple[str, str, str, str]:
    """Return (contact_or_email, message_body, app, subject) using the LLM, with heuristic fallback."""
    from core.llm import llm

    prompt = (
        f'Instruction: "{goal}"\n\n'
        "Extract the communication request as JSON with keys: contact (person name or email address exactly as "
        "the user said it), subject (email subject, else empty), message (the full message to send; write a polite, "
        "complete body if the user asked you to compose or invite formally; otherwise keep the user's words), "
        "app (gmail|whatsapp|telegram|discord|slack). Reply with JSON only."
    )
    detected_app = _detect_app_from_text(goal)
    try:
        res = await llm.chat([{"role": "user", "content": prompt}], model=None, temperature=0.1, max_tokens=400, purpose="fast")
        data = extract_and_clean_json(res.content)
        if not data:
            raise ValueError("no JSON in reply")
        contact = str(data.get("contact", "")).strip() or _heuristic_contact(goal)
        message = str(data.get("message", "")).strip() or _heuristic_message(goal)
        subject = str(data.get("subject", "")).strip()
        app = detected_app or str(data.get("app", "")).strip().lower() or ("gmail" if "@" in contact else "whatsapp")
        return contact, message, app, subject
    except Exception as exc:
        automation_logger.warning(f"[Messaging] LLM extraction unavailable ({exc}); using heuristics.")
        app = detected_app or ("gmail" if "@" in goal else "whatsapp")
        subject = "Invitation" if "invite" in goal.lower() else ""
        return _heuristic_contact(goal), _heuristic_message(goal), app, subject


async def plan_message(goal: str) -> Tuple[Optional[ActionRequest], str]:
    """Build the gateway request for a messaging instruction. Returns (request, error)."""
    contact, text, app, subject = await extract_messaging_intent(goal)
    if not contact:
        return None, "Who should I send it to?"
    if app == "gmail":
        if not EMAIL_RE.fullmatch(contact):
            return None, f"I need an email address for {contact}."
        return ActionRequest("email.compose", {"to": contact, "subject": subject, "body": text}), ""
    if not text:
        return None, f"What should I say to {contact}?"
    return ActionRequest("message.send", {"app": app, "contact": contact, "text": text}), ""
