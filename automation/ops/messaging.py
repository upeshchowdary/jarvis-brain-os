"""Messaging & email operations — always previewed, confirmed, and recipient-verified.

Flow for chat apps (WhatsApp / Telegram desktop):
  1. the gateway shows a preview (app, recipient, text) and waits for confirmation;
  2. the app is opened via its protocol handler (``os.startfile``, no shell);
  3. the contact is searched; the result list is read through UI Automation.
     Exactly one exact match is required — otherwise the matches are listed and
     nothing is sent;
  4. after opening the chat, the chat *header* must show the intended contact;
  5. only then is the text pasted and Enter pressed. The clipboard is restored.

If any verification step is impossible, the message is left as an unsent draft.
Discord and Slack are reported as "not supported yet" — they never fall back to WhatsApp.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import urllib.parse
import webbrowser
from typing import Any, Dict, List, Optional, Sequence, Tuple

from automation.gateway import OpResult, Preview, Risk, gateway

SUPPORTED_CHAT_APPS = {"whatsapp": ("WhatsApp", "whatsapp:", r".*WhatsApp.*"), "telegram": ("Telegram", "tg:", r".*Telegram.*")}
UNSUPPORTED_APPS = {"discord": "Discord", "slack": "Slack", "instagram": "Instagram", "teams": "Microsoft Teams"}


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", name or "")).strip().lower()


def pick_contact(candidates: Sequence[str], contact: str) -> Tuple[Optional[int], List[str]]:
    """Choose the search result to open. Returns (index or None, names to show the user).

    An exact (normalised) name match wins if it is unique; otherwise a unique
    prefix match; anything else is ambiguous and nothing is chosen.
    """
    want = _norm(contact)
    firsts = [_norm(c.splitlines()[0]) if c else "" for c in candidates]
    exact = [i for i, n in enumerate(firsts) if n == want]
    if len(exact) == 1:
        return exact[0], []
    if len(exact) > 1:
        return None, [candidates[i].splitlines()[0] for i in exact]
    prefix = [i for i, n in enumerate(firsts) if n.startswith(want + " ") or n.startswith(want)]
    if len(prefix) == 1:
        return prefix[0], []
    return None, [candidates[i].splitlines()[0] for i in prefix][:8]


def header_matches(header_texts: Sequence[str], contact: str) -> bool:
    """True if one of the chat-header texts is the intended contact."""
    want = _norm(contact)
    return any(_norm(t) == want or _norm(t).startswith(want + " ") for t in header_texts if t)


def _message_preview(args: Dict[str, Any]) -> Preview:
    app = str(args.get("app", "")).lower()
    contact = str(args.get("contact", "")).strip()
    text = str(args.get("text", "")).strip()
    if app in UNSUPPORTED_APPS:
        return Preview(
            f"Sending messages through {UNSUPPORTED_APPS[app]} isn't supported yet. "
            f"I can open {UNSUPPORTED_APPS[app]} for you instead.",
            ok=False,
        )
    if app not in SUPPORTED_CHAT_APPS:
        return Preview(f"I don't know how to send messages with '{app}'.", ok=False)
    if not contact or not text:
        return Preview("I need both a recipient and the message text.", ok=False)
    return Preview(f"Send a {SUPPORTED_CHAT_APPS[app][0]} message to **{contact}**:\n   “{text}”")


def _escape_keys(text: str) -> str:
    return re.sub(r"([{}()+^%~\[\]])", r"{\1}", text)


def _send_desktop_chat(app: str, contact: str, text: str) -> Tuple[bool, str]:
    """Blocking UI flow. Returns (sent, explanation)."""
    if sys.platform != "win32":
        return False, "Desktop messaging is only available on Windows."
    import pyperclip  # type: ignore
    from pywinauto import Application  # type: ignore
    from pywinauto.keyboard import send_keys  # type: ignore

    label, protocol, title_re = SUPPORTED_CHAT_APPS[app]
    os.startfile(protocol)  # type: ignore[attr-defined]
    win = None
    for _ in range(12):
        try:
            win = Application(backend="uia").connect(title_re=title_re, timeout=0.5).top_window()
            break
        except Exception:
            time.sleep(0.5)
    if win is None:
        return False, f"{label} didn't open."
    win.set_focus()
    saved_clipboard = None
    try:
        saved_clipboard = pyperclip.paste()
    except Exception:
        pass
    try:
        send_keys("{ESC}")
        send_keys("^f")
        time.sleep(0.3)
        send_keys("^a{BACKSPACE}")
        send_keys(_escape_keys(contact), with_spaces=True)
        time.sleep(1.3)

        items = [it for it in win.descendants(control_type="ListItem") if it.window_text().strip()]
        names = [it.window_text() for it in items]
        idx, ambiguous = pick_contact(names, contact)
        if idx is None:
            send_keys("{ESC}")
            if ambiguous:
                return False, f"Several contacts match '{contact}': {', '.join(ambiguous)}. Tell me the exact name."
            return False, f"I couldn't find a {label} contact named '{contact}'. Nothing was sent."
        items[idx].click_input()
        time.sleep(0.9)

        rect = win.rectangle()
        header_texts: List[str] = []
        for ctrl in win.descendants():
            try:
                r = ctrl.rectangle()
                if r.top - rect.top < (rect.height() * 0.18) and r.left - rect.left > rect.width() * 0.25:
                    header_texts.append(ctrl.window_text())
            except Exception:
                continue
        if not header_matches(header_texts, contact):
            return False, (
                f"I opened a chat but couldn't verify it belongs to '{contact}', so I did not send the message."
            )

        pyperclip.copy(text)
        send_keys("^v")
        time.sleep(0.25)
        send_keys("{ENTER}")
        return True, f"Sent {label} message to {contact}."
    finally:
        if saved_clipboard is not None:
            try:
                time.sleep(0.2)
                pyperclip.copy(saved_clipboard)
            except Exception:
                pass


@gateway.operation("message.send", Risk.HIGH, "Send a chat message", preview=_message_preview)
async def send_message(app: str, contact: str, text: str) -> OpResult:
    app = app.lower().strip()
    if app in UNSUPPORTED_APPS:
        return OpResult.fail(f"Sending messages through {UNSUPPORTED_APPS[app]} isn't supported yet.")
    if app not in SUPPORTED_CHAT_APPS:
        return OpResult.fail(f"I don't know how to send messages with '{app}'.")
    sent, why = await asyncio.to_thread(_send_desktop_chat, app, contact, text)
    return OpResult(sent, why, data={"app": app, "contact": contact, "sent": sent})


def _email_preview(args: Dict[str, Any]) -> Preview:
    to = str(args.get("to", "")).strip()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}", to):
        return Preview(f"'{to}' isn't a valid email address.", ok=False)
    body = str(args.get("body", ""))
    return Preview(
        f"Open a Gmail draft (you press Send) to **{to}**\n   Subject: {args.get('subject') or '(none)'}\n   {body[:600]}"
    )


@gateway.operation("email.compose", Risk.MEDIUM, "Open a pre-filled email draft", preview=_email_preview)
async def compose_email(to: str, subject: str = "", body: str = "") -> OpResult:
    q = urllib.parse.quote
    gmail = f"https://mail.google.com/mail/?view=cm&fs=1&to={q(to)}&su={q(subject)}&body={q(body)}"
    if await asyncio.to_thread(webbrowser.open, gmail, 2):
        return OpResult.ok(f"Opened a Gmail draft to {to}. Review it and press Send.", data={"url": gmail})
    mailto = f"mailto:{q(to)}?subject={q(subject)}&body={q(body)}"
    if sys.platform == "win32":
        await asyncio.to_thread(os.startfile, mailto)  # type: ignore[attr-defined]
        return OpResult.ok(f"Opened your mail app with a draft to {to}.")
    return OpResult.fail("Couldn't open a browser or mail client.")
