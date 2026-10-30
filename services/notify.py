"""User notifications: Windows toast + spoken alert (if voice is on) + dashboard event."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from xml.sax.saxutils import escape

from core.events import bus
from core.logging import logger

_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml([Console]::In.ReadToEnd())
$toast = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('JARVIS').Show($toast)
"""


def _toast(title: str, message: str) -> None:
    if sys.platform != "win32":
        return
    try:
        from win11toast import notify as w11  # type: ignore

        w11(title, message, app_id="JARVIS")
        return
    except Exception:
        pass
    xml = (
        "<toast><visual><binding template='ToastGeneric'>"
        f"<text>{escape(title)}</text><text>{escape(message)}</text>"
        "</binding></visual><audio src='ms-winsoundevent:Notification.Reminder'/></toast>"
    )
    try:  # argument list, no shell; the XML arrives on stdin and is never interpolated into the script
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST_PS],
                       input=xml, text=True, capture_output=True, timeout=10, shell=False)
    except Exception as exc:
        logger.debug(f"[notify] toast failed: {exc}")


async def notify(title: str, message: str, *, speak: bool = True, kind: str = "notification") -> None:
    logger.info(f"[notify] {title}: {message}")
    bus.publish(kind, {"title": title, "message": message})
    await asyncio.to_thread(_toast, title, message)
    if speak:
        try:
            from voice.voice_manager import voice_manager
            from voice.tts import tts_engine

            if voice_manager.is_running:
                tts_engine.speak(f"{title}. {message}", concise=False)
        except Exception:
            pass
