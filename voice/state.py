"""State Machine and Lifecycle Management for JARVIS Voice & Microphone Module.

Defines internal states:
- MIC_OFF
- MIC_INITIALIZING
- MIC_READY
- LISTENING
- WAKE_DETECTED
- TRANSCRIBING
- UNDERSTANDING
- PLANNING
- EXECUTING
- VERIFYING
- SPEAKING
- PAUSED
- INTERRUPTED
- ERROR
"""

from __future__ import annotations

import enum
import time
from typing import Callable, List, Optional
from loguru import logger


class VoiceState(enum.Enum):
    MIC_OFF = "MIC_OFF"
    MIC_INITIALIZING = "MIC_INITIALIZING"
    MIC_READY = "MIC_READY"
    LISTENING = "LISTENING"
    WAKE_DETECTED = "WAKE_DETECTED"
    TRANSCRIBING = "TRANSCRIBING"
    UNDERSTANDING = "UNDERSTANDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    SPEAKING = "SPEAKING"
    PAUSED = "PAUSED"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


class VoiceStateManager:
    """Thread-safe State Machine for Voice interactions with event listeners."""

    def __init__(self, initial_state: VoiceState = VoiceState.MIC_OFF) -> None:
        self._current_state: VoiceState = initial_state
        self._previous_state: VoiceState = initial_state
        self._last_transition_time: float = time.perf_counter()
        self._listeners: List[Callable[[VoiceState, VoiceState, Optional[str]], None]] = []
        self._error_message: Optional[str] = None

    @property
    def current_state(self) -> VoiceState:
        return self._current_state

    @property
    def previous_state(self) -> VoiceState:
        return self._previous_state

    @property
    def state_duration(self) -> float:
        return time.perf_counter() - self._last_transition_time

    @property
    def error_message(self) -> Optional[str]:
        return self._error_message

    def add_listener(self, callback: Callable[[VoiceState, VoiceState, Optional[str]], None]) -> None:
        """Register a callback for state changes: callback(new_state, old_state, message)."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[VoiceState, VoiceState, Optional[str]], None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    def transition_to(self, new_state: VoiceState, message: Optional[str] = None) -> None:
        """Transitions state machine to new state and notifies all registered listeners."""
        if self._current_state == new_state and not message:
            return

        old_state = self._current_state
        self._previous_state = old_state
        self._current_state = new_state
        self._last_transition_time = time.perf_counter()

        if new_state == VoiceState.ERROR:
            self._error_message = message or "Unknown voice error"
        elif old_state == VoiceState.ERROR and new_state != VoiceState.ERROR:
            self._error_message = None

        logger.debug(f"[VoiceState] {old_state.value} -> {new_state.value} {f'({message})' if message else ''}")

        for listener in self._listeners:
            try:
                listener(new_state, old_state, message)
            except Exception as e:
                logger.warning(f"[VoiceState] Listener error: {e}")

    def is_listening(self) -> bool:
        return self._current_state in (VoiceState.LISTENING, VoiceState.WAKE_DETECTED)

    def is_active(self) -> bool:
        return self._current_state not in (VoiceState.MIC_OFF, VoiceState.ERROR, VoiceState.PAUSED)


voice_state_manager = VoiceStateManager()
