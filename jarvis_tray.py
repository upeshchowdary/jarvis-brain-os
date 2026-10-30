"""System-tray icon + global hotkey (Win+J) for JARVIS. Optional: pip install .[desktop]."""

from __future__ import annotations

import webbrowser
from typing import Callable

import httpx

from core.config import ensure_api_token, settings


def _post(path: str) -> None:
    try:
        httpx.post(f"http://{settings.HOST}:{settings.PORT}{path}", headers={"Authorization": f"Bearer {ensure_api_token()}"}, timeout=5)
    except httpx.HTTPError:
        pass


def _icon_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(37, 99, 235, 255))
    d.ellipse((22, 22, 42, 42), fill=(255, 255, 255, 255))
    return img


def register_hotkey(url: str) -> None:
    """Win+J opens the console (uses pynput if installed)."""
    try:
        from pynput import keyboard  # type: ignore
    except ImportError:
        return
    hk = keyboard.GlobalHotKeys({"<cmd>+j": lambda: webbrowser.open(url)})
    hk.daemon = True
    hk.start()


def run_tray(url: str, on_quit: Callable[[], None]) -> None:
    import pystray  # type: ignore

    register_hotkey(url)
    menu = pystray.Menu(
        pystray.MenuItem("Open console", lambda: webbrowser.open(url), default=True),
        pystray.MenuItem("Start voice", lambda: _post("/voice/start")),
        pystray.MenuItem("Stop voice", lambda: _post("/voice/stop")),
        pystray.MenuItem("Emergency stop", lambda: _post("/automation/stop")),
        pystray.MenuItem("Resume automation", lambda: _post("/automation/resume")),
        pystray.MenuItem("Quit", lambda icon, _item: (on_quit(), icon.stop())),
    )
    pystray.Icon("jarvis", _icon_image(), "JARVIS", menu).run()
