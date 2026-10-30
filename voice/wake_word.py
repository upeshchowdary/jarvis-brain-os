"""Wake Word Detection Engine for JARVIS Voice Module.

Supports natural trigger variations:
- "Hey JARVIS"
- "JARVIS"
- "Okay JARVIS" / "OK JARVIS"
- "Hi JARVIS"

Features:
- Streaming acoustic/phonetic trigger matching
- Follow-up conversation window (hands-free subsequent turns without repeated wake words)
- Low-latency, low-CPU standby monitoring
"""

from __future__ import annotations

import re
import time
from typing import Optional, Tuple
from loguru import logger

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager


class WakeWordDetector:
    """Detects wake word triggers in continuous voice input or transcribed text."""

    def __init__(self) -> None:
        self.enabled = voice_config.WAKE_WORD_ENABLED
        self.wake_words = [w.lower() for w in voice_config.WAKE_WORDS]
        self.followup_window_sec = voice_config.FOLLOWUP_WINDOW_SECONDS
        self._last_interaction_time: float = 0.0

        # Regex patterns for fast acoustic/phonetic matching
        self._wake_patterns = [
            re.compile(r"\b(?:hey|hi|ok|okay|hello|yo)?\s*jarvis\b", re.IGNORECASE),
            re.compile(r"\bjarvis\b", re.IGNORECASE),
            re.compile(r"\b(?:hey|hi|okay)\s+jarvis\b", re.IGNORECASE),
        ]

    def record_interaction(self) -> None:
        """Records timestamp of completed interaction for follow-up conversation window."""
        self._last_interaction_time = time.perf_counter()

    def is_in_followup_window(self) -> bool:
        """Returns True if the user is in an active conversational turn within the follow-up window."""
        if not voice_config.REQUIRE_WAKE_WORD_FOR_FOLLOWUPS and self._last_interaction_time > 0:
            elapsed = time.perf_counter() - self._last_interaction_time
            return elapsed <= self.followup_window_sec
        return False

    def detect_in_text(self, text: str) -> Tuple[bool, str]:
        """
        Check if text contains wake word.
        
        Returns:
            (detected: bool, stripped_command: str)
            - detected: True if wake word is present or within follow-up window
            - stripped_command: text with wake word prefix stripped off
        """
        if not self.enabled:
            return True, text.strip()

        cleaned_text = text.strip()

        # 1. Check direct wake word patterns
        for pattern in self._wake_patterns:
            match = pattern.search(cleaned_text)
            if match:
                # Extract command after wake word
                end_pos = match.end()
                command = cleaned_text[end_pos:].strip(" ,.?!:;\n\t")
                logger.info(f"[WakeWord] Wake word detected in text: '{cleaned_text[:match.end()]}'")
                voice_state_manager.transition_to(VoiceState.WAKE_DETECTED, "Wake word matched")
                self.record_interaction()
                return True, command

        # 2. Check active follow-up conversational window
        if self.is_in_followup_window():
            logger.debug("[WakeWord] In active follow-up window (no wake word required).")
            self.record_interaction()
            return True, cleaned_text

        return False, cleaned_text

    def strip_wake_word(self, text: str) -> str:
        """Strips wake word prefix from beginning of text if present."""
        cleaned = text.strip()
        for pattern in self._wake_patterns:
            m = pattern.match(cleaned)
            if m:
                return cleaned[m.end():].strip(" ,.?!:;\n\t")
        return cleaned


wake_word_detector = WakeWordDetector()


class AcousticWakeWord:
    """Optional on-device acoustic wake word ("hey jarvis") using openWakeWord (Apache-2.0 code).

    Detecting the wake word on raw audio means nothing is transcribed until the user
    addresses JARVIS — lower latency, lower CPU and better privacy than transcribing
    everything and searching the text. Note: openWakeWord's bundled pre-trained models
    are released for non-commercial use (CC BY-NC-SA); fine for a personal assistant.
    """

    FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, what openWakeWord expects

    def __init__(self) -> None:
        self._model = None
        self._buffer = b""
        self.error: Optional[str] = None

    def available(self) -> bool:
        if self._model is not None:
            return True
        if self.error:
            return False
        try:
            import openwakeword  # type: ignore
            from openwakeword.model import Model  # type: ignore

            try:
                openwakeword.utils.download_models([voice_config.OPENWAKEWORD_MODEL])
            except Exception:
                pass
            self._model = Model(wakeword_models=[voice_config.OPENWAKEWORD_MODEL], inference_framework="onnx")
            return True
        except Exception as exc:
            self.error = str(exc)
            logger.warning(f"[WakeWord] openWakeWord unavailable ({exc}); falling back to transcript matching.")
            return False

    def process(self, chunk: bytes) -> bool:
        """Feed raw 16-bit mono 16 kHz audio; returns True when the wake word is heard."""
        if not self.available():
            return False
        import numpy as np

        self._buffer += chunk
        frame_bytes = self.FRAME_SAMPLES * 2
        fired = False
        while len(self._buffer) >= frame_bytes:
            frame, self._buffer = self._buffer[:frame_bytes], self._buffer[frame_bytes:]
            scores = self._model.predict(np.frombuffer(frame, dtype="<i2"))
            if max(scores.values(), default=0.0) >= voice_config.OPENWAKEWORD_THRESHOLD:
                fired = True
        if fired:
            self._model.reset()
            self._buffer = b""
            voice_state_manager.transition_to(VoiceState.WAKE_DETECTED, "Acoustic wake word")
        return fired


acoustic_wake_word = AcousticWakeWord()
