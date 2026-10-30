"""Non-blocking internet connectivity hint, cached and refreshed in the background.

Used only to *order* LLM providers (local first when offline) — never to refuse a
request, because some networks block raw DNS/TCP probes while HTTPS still works.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

_PROBES = (("1.1.1.1", 53), ("8.8.8.8", 53))
_TTL = 15.0
_state: dict = {"online": None, "checked": 0.0}
_task: Optional[asyncio.Task] = None


async def _probe(timeout: float = 0.8) -> bool:
    for host, port in _PROBES:
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
            writer.close()
            return True
        except (OSError, asyncio.TimeoutError):
            continue
    return False


async def is_online(max_age: float = _TTL) -> bool:
    """Return the cached connectivity state, probing (≤0.8 s per host) only when stale."""
    if _state["online"] is None or time.monotonic() - _state["checked"] > max_age:
        _state["online"] = await _probe()
        _state["checked"] = time.monotonic()
    return bool(_state["online"])


def last_known() -> Optional[bool]:
    return _state["online"]


async def _refresher(interval: float) -> None:
    while True:
        _state["online"] = await _probe()
        _state["checked"] = time.monotonic()
        await asyncio.sleep(interval)


def start_background_refresh(interval: float = 10.0) -> None:
    global _task
    if _task is None or _task.done():
        _task = asyncio.get_running_loop().create_task(_refresher(interval))


def set_state_for_tests(online: Optional[bool]) -> None:
    _state["online"] = online
    _state["checked"] = time.monotonic() if online is not None else 0.0
