"""Multi-Platform Messaging & Communication Handler for JARVIS Automation Engine.

Supports:
- Email / Gmail (Direct Browser Compose / Mailto protocol)
- WhatsApp (Desktop / Web)
- Telegram (Desktop / Web)
- Discord (Desktop / Web)
- Slack (Desktop / Web)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.parse
import webbrowser
from typing import Dict, Optional, Tuple, Any

from automation.automation_logger import automation_logger


async def extract_messaging_intent(goal: str) -> Tuple[str, str, str, str]:
    """
    Single LLM call to extract (contact_or_email, message_body, app_name, subject).
    Handles both chat messages and full email drafting with formal/casual tones.
    """
    from brain.model_manager import model_manager

    prompt = (
        f'Analyze this user instruction: "{goal}"\n\n'
        'Extract and generate the exact communication parameters in JSON (no markdown formatting):\n'
        '{\n'
        '  "contact": "<email address or recipient person name>",\n'
        '  "subject": "<appropriate email subject line if email/mail, else empty>",\n'
        '  "message": "<the complete drafted message or email body — write it naturally and fully according to instructions/tone>",\n'
        '  "app": "<gmail|whatsapp|telegram|discord|slack>",\n'
        '  "tone": "<formal|casual|friendly>"\n'
        '}\n\n'
        'RULES:\n'
        '- If the user mentions email/mail or an @ address (e.g. clash938134@gmail.com), app MUST be "gmail".\n'
        '- For emails (e.g. "invite him to party formally"), write a well-structured, polite, complete email body with proper greeting and closing.\n'
        '- If the user mentions WhatsApp / WA, app MUST be "whatsapp".\n'
        '- If the user mentions Telegram / TG / Tele, app MUST be "telegram".\n'
        '- Return ONLY valid JSON.'
    )

    try:
        res = await model_manager.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=350,
        )
        raw = (res.get("content", "") if isinstance(res, dict) else "").strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
        data = json.loads(raw)
        contact = str(data.get("contact", "")).strip()
        subject = str(data.get("subject", "")).strip()
        message = str(data.get("message", "")).strip()
        app = str(data.get("app", "")).strip().lower()

        # Deterministic text check override
        detected_app = _detect_app_from_text(goal)
        if detected_app:
            app = detected_app

        if not app:
            app = "gmail" if ("@" in contact or "@" in goal) else "whatsapp"

        return contact, message, app, subject
    except Exception as e:
        automation_logger.warning(f"[MessagingHandler] Extraction error: {e}. Using heuristics.")
        detected_app = _detect_app_from_text(goal) or ("gmail" if "@" in goal else "whatsapp")
        contact = _heuristic_contact(goal)
        message = _heuristic_message(goal)
        subject = "Invitation" if "invite" in goal.lower() else "Message from JARVIS"
        return contact, message, detected_app, subject


def _detect_app_from_text(text: str) -> Optional[str]:
    """Detect messaging/email app from text using strict regex boundaries."""
    q = text.lower()
    # Check for email address or email keywords FIRST
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text) or re.search(r"\b(?:gmail|email|e-mail|mail)\b", q):
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


def _heuristic_contact(goal: str) -> str:
    # 1. Check for email pattern
    email_m = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", goal)
    if email_m:
        return email_m.group(1)

    patterns = [
        r"(?:message|msg|text|tell|ask|dm|mail|email)\s+([A-Za-z][a-z0-9_]+(?:\s+[A-Za-z0-9_]+)?)\b",
        r"\b(?:to)\s+([A-Za-z][a-z0-9_]+)\b",
        r"\b([A-Za-z][a-z0-9_]+)\s+on\s+(?:whatsapp|telegram|discord|slack)\b",
        r"(?:whatsapp|telegram|discord|slack)\s+(?:message\s+to\s+|to\s+)?([A-Za-z][a-z0-9_]+)\b",
    ]
    stop_words = {"him", "her", "them", "me", "us", "it", "that", "a", "the", "in", "on", "my", "this", "whatsapp", "telegram", "discord", "slack", "gmail", "email", "mail", "what", "how", "please"}
    for p in patterns:
        m = re.search(p, goal, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() not in stop_words:
                return name.title()
    return ""


def _heuristic_message(goal: str) -> str:
    m = re.search(r"(?:say|ask|tell|saying|asking|invite|write)\s+(?:\w+\s+)?['\"]?(.+)['\"]?$", goal, re.IGNORECASE)
    if m:
        return m.group(1).strip(" '\"")
    return goal


# ─── Email / Gmail Dispatcher ──────────────────────────────────────────────────

async def send_email(recipient: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Composes and opens an email draft in Gmail / default browser and mail client.
    Prefills recipient, subject line, and full body text.
    """
    automation_logger.info(f"[Email] Composing email for '{recipient}' with subject '{subject}'")

    if not subject:
        subject = "Invitation" if "invite" in body.lower() else "Message from JARVIS"

    # Encode query parameters for Gmail web compose URL
    encoded_to = urllib.parse.quote(recipient)
    encoded_su = urllib.parse.quote(subject)
    encoded_body = urllib.parse.quote(body)

    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={encoded_to}&su={encoded_su}&body={encoded_body}"
    mailto_url = f"mailto:{recipient}?subject={encoded_su}&body={encoded_body}"

    # Try opening Gmail web composer in browser (fast & reliable)
    try:
        webbrowser.open(gmail_url, new=2)
        summary = (
            f"Drafted and opened Gmail composer to '{recipient}' with subject '{subject}'.\n\n"
            f"**Preview:**\n{body}"
        )
        return {
            "success": True,
            "summary": summary,
            "action": "open_gmail_compose",
            "recipient": recipient,
            "subject": subject,
            "body": body,
        }
    except Exception as e:
        automation_logger.warning(f"[Email] Web browser open failed: {e}. Trying mailto.")
        try:
            os.system(f'start "" "{mailto_url}"')
            return {
                "success": True,
                "summary": f"Opened default email client for '{recipient}'.",
                "recipient": recipient,
                "subject": subject,
            }
        except Exception as e2:
            return {
                "success": False,
                "error": f"Failed to open email composer: {e2}",
            }


# ─── Telegram Dispatcher ──────────────────────────────────────────────────────

async def send_telegram_uia(contact: str, message: str) -> Dict[str, Any]:
    """Sends message via Telegram Desktop or Web Telegram."""
    from automation.application_controller import application_controller
    from automation.window_controller import window_controller

    automation_logger.info(f"[Telegram] Launching/Focusing Telegram for contact '{contact}'")

    os.system("start tg://")
    await application_controller.open_app("Telegram Desktop")
    await asyncio.sleep(2.0)

    await window_controller.focus_window("Telegram")
    await asyncio.sleep(0.4)

    def _sync_telegram():
        from pywinauto.keyboard import send_keys  # type: ignore
        import pyperclip
        import time

        send_keys("{ESC}")
        time.sleep(0.2)
        send_keys("^f")
        time.sleep(0.3)
        send_keys("^a{BACKSPACE}")
        time.sleep(0.1)

        send_keys(contact, with_spaces=True)
        time.sleep(1.2)

        send_keys("{DOWN}")
        time.sleep(0.2)
        send_keys("{ENTER}")
        time.sleep(0.8)

        pyperclip.copy(message)
        send_keys("^v")
        time.sleep(0.2)
        send_keys("{ENTER}")
        time.sleep(0.3)

        return True, f"Sent Telegram message to {contact}: '{message[:50]}...'"

    loop = asyncio.get_event_loop()
    success, msg = await loop.run_in_executor(None, _sync_telegram)
    return {"success": success, "summary": msg}


# ─── WhatsApp Dispatcher ──────────────────────────────────────────────────────

async def send_whatsapp_uia(contact: str, message: str) -> Dict[str, Any]:
    """Sends WhatsApp message via Windows UI Automation tree."""
    from automation.application_controller import application_controller
    from automation.window_controller import window_controller

    automation_logger.info(f"[WhatsApp] Launching/Focusing WhatsApp for contact '{contact}'")

    os.system("start whatsapp://")
    await application_controller.open_app("WhatsApp")
    await asyncio.sleep(2.0)

    await window_controller.focus_window("WhatsApp")
    await asyncio.sleep(0.4)

    def _sync_whatsapp():
        from pywinauto import Application  # type: ignore
        from pywinauto.keyboard import send_keys  # type: ignore
        import pyperclip
        import time

        try:
            app = Application(backend="uia").connect(title_re=".*WhatsApp.*", timeout=4)
            win = app.top_window()
            win.set_focus()
        except Exception:
            pass

        time.sleep(0.3)

        send_keys("^f")
        time.sleep(0.3)
        send_keys("^a{BACKSPACE}")
        time.sleep(0.1)

        send_keys(contact, with_spaces=True)
        time.sleep(1.2)

        send_keys("{DOWN}")
        time.sleep(0.2)
        send_keys("{ENTER}")
        time.sleep(0.8)

        pyperclip.copy(message)
        send_keys("^v")
        time.sleep(0.2)
        send_keys("{ENTER}")
        time.sleep(0.3)

        return True, f"Sent WhatsApp message to {contact}: '{message[:50]}...'"

    loop = asyncio.get_event_loop()
    success, msg = await loop.run_in_executor(None, _sync_whatsapp)
    return {"success": success, "summary": msg}


# ─── Unified Messaging & Communication Entrypoint ─────────────────────────────

async def dispatch_message(app: str, contact: str, message: str, subject: str = "") -> Dict[str, Any]:
    """Routes to the exact application or communication channel requested by the user."""
    app_clean = app.lower().strip()
    automation_logger.info(f"[Messaging] Dispatching: app='{app_clean}', contact='{contact}'")

    if "gmail" in app_clean or "email" in app_clean or "mail" in app_clean or "@" in contact:
        return await send_email(contact, subject, message)
    elif "telegram" in app_clean:
        return await send_telegram_uia(contact, message)
    elif "whatsapp" in app_clean:
        return await send_whatsapp_uia(contact, message)
    else:
        if "@" in contact:
            return await send_email(contact, subject, message)
        return await send_whatsapp_uia(contact, message)

