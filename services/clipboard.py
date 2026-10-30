"""Clipboard history watcher (local only; stored in data/jarvis_productivity.db)."""

from __future__ import annotations

import asyncio
from typing import Optional

from core.config import settings
from core.logging import logger
from services.store import store


class ClipboardWatcher:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._last = ""

    async def start(self) -> None:
        try:
            import pyperclip  # noqa: F401
        except ImportError:
            logger.info("[Clipboard] pyperclip not installed; clipboard history disabled")
            return
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        self._task = None

    async def poll_once(self) -> bool:
        import pyperclip

        try:
            text = await asyncio.to_thread(pyperclip.paste)
        except Exception:
            return False
        if isinstance(text, str) and text.strip() and text != self._last:
            self._last = text
            await store.add_clip(text, settings.CLIPBOARD_HISTORY_SIZE)
            return True
        return False

    async def _loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            await asyncio.sleep(self.interval)


clipboard_watcher = ClipboardWatcher()
