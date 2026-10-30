"""Tiny in-process event bus (dashboard live feed, notifications, transcript)."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Deque, Dict, List


class EventBus:
    def __init__(self, history: int = 200) -> None:
        self._subscribers: List[asyncio.Queue] = []
        self.recent: Deque[Dict[str, Any]] = deque(maxlen=history)

    def publish(self, kind: str, payload: Dict[str, Any]) -> None:
        event = {"type": kind, "ts": time.time(), **payload}
        self.recent.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)


bus = EventBus()


def _gateway_listener(event: str, payload: Dict[str, Any]) -> None:
    bus.publish(f"gateway.{event}", {"data": payload})


def wire_gateway() -> None:
    from automation.gateway import gateway

    if _gateway_listener not in gateway.listeners:
        gateway.listeners.append(_gateway_listener)
