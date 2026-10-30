"""Real-Time Voice Interruption (Barge-In) Monitor for JARVIS Voice Module.

Features:
- Continuously listens for user interruption while JARVIS is speaking
- Instant speech termination when user says "Stop", "Cancel", "Abort", "Wait", "Hold on"
- Immediate escalation to halt active automation actions
"""

from __future__ import annotations

import threading
import time
from typing import Optional
from loguru import logger

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager
from voice.tts import tts_engine
from voice.audio_capture import audio_capture
from voice.vad import vad_detector
from automation.gateway import gateway


class BargeInMonitor:
    """Monitors live microphone stream while TTS is active and cancels speech on user interruption."""

    def __init__(self) -> None:
        self.enabled = voice_config.ENABLE_BARGE_IN
        self.interrupt_keywords = [k.lower() for k in voice_config.INTERRUPT_KEYWORDS]
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start_monitoring(self) -> None:
        """Start listening for user barge-in."""
        if self._is_monitoring or not self.enabled:
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._barge_in_worker, daemon=True, name="JARVIS-BargeIn")
        self._monitor_thread.start()
        self._is_monitoring = True

    def stop_monitoring(self) -> None:
        """Stop barge-in monitor."""
        if not self._is_monitoring:
            return

        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.5)
        self._is_monitoring = False

    def is_interrupt_command(self, text: str) -> bool:
        """Checks if transcribed text is an explicit stop / cancel command."""
        t_low = text.lower().strip(" .?!,")
        return any(k == t_low or t_low.startswith(k + " ") or t_low.endswith(" " + k) for k in self.interrupt_keywords)

    def trigger_interruption(self, reason: str = "user interruption") -> None:
        """Immediately interrupts speech output and stops current automation."""
        logger.info(f"[BargeIn] Interruption triggered: {reason}")
        # 1. Stop Text-To-Speech immediately
        tts_engine.stop()

        # 2. Halt any running non-critical automation
        if "stop" in reason.lower() or "cancel" in reason.lower() or "abort" in reason.lower():
            gateway.emergency_stop(reason)

        voice_state_manager.transition_to(VoiceState.INTERRUPTED, reason)

    def _barge_in_worker(self) -> None:
        """Worker thread checking for energy bursts while TTS is speaking."""
        speech_frame_count = 0

        while not self._stop_event.is_set():
            if tts_engine.is_speaking:
                chunk = audio_capture.read_chunk(timeout=0.05)
                if chunk:
                    # Require slightly higher energy threshold during TTS to avoid echo
                    barge_threshold = audio_capture.energy_threshold * 1.8
                    if vad_detector.is_speech_frame(chunk, barge_threshold):
                        speech_frame_count += 1
                        if speech_frame_count >= 2:  # ~120ms of sustained speech
                            self.trigger_interruption("User started speaking during TTS")
                            speech_frame_count = 0
                            time.sleep(0.3)
                    else:
                        speech_frame_count = max(0, speech_frame_count - 1)
            else:
                speech_frame_count = 0
                time.sleep(0.05)


barge_in_monitor = BargeInMonitor()
