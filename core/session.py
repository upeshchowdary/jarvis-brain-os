"""Per-session runtime state with LRU + TTL eviction.

Model choice, the last screen analysis and other per-conversation state live here,
so one session switching models never affects voice / CLI / other API sessions.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.config import settings


@dataclass
class SessionState:
    session_id: str
    model_override: Optional[str] = None
    personality: Optional[str] = None
    vision_context: Optional[str] = None
    vision_at: float = 0.0
    last_tool: Optional[str] = None
    scratch: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def recent_vision(self, max_age: float = 180.0) -> Optional[str]:
        if self.vision_context and time.time() - self.vision_at <= max_age:
            return self.vision_context
        return None

    def set_vision(self, description: str) -> None:
        self.vision_context = description[:4000]
        self.vision_at = time.time()


class SessionStore:
    def __init__(self, max_sessions: Optional[int] = None, ttl: Optional[float] = None) -> None:
        self._data: "OrderedDict[str, SessionState]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions or settings.MAX_SESSIONS
        self.ttl = ttl or settings.SESSION_TTL_SECONDS

    def get(self, session_id: str) -> SessionState:
        now = time.time()
        with self._lock:
            self._evict(now)
            state = self._data.get(session_id)
            if state is None:
                state = SessionState(session_id)
                self._data[session_id] = state
            self._data.move_to_end(session_id)
            state.last_active = now
            return state

    def peek(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._data.get(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)

    def _evict(self, now: float) -> None:
        for sid in [s for s, st in self._data.items() if now - st.last_active > self.ttl]:
            self._data.pop(sid, None)
        while len(self._data) >= self.max_sessions:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


sessions = SessionStore()
