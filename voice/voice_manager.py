"""Central Voice Interaction Orchestrator for JARVIS.

Integrates:
- AudioCapture
- VoiceActivityDetector (VAD)
- WakeWordDetector
- SpeechToTextEngine (STT)
- CommandNormalizer
- TextToSpeechEngine (TTS)
- BargeInMonitor
- BrainManager & AutomationOrchestrator
"""

from __future__ import annotations

import asyncio
from typing import Optional, Dict, Any, Callable
from loguru import logger

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager
from voice.audio_capture import audio_capture
from voice.vad import vad_detector
from voice.wake_word import wake_word_detector
from voice.stt import stt_engine
from voice.tts import tts_engine
from voice.barge_in import barge_in_monitor


class VoiceManager:
    """Master controller orchestrating the hands-free conversational voice loop."""

    def __init__(self) -> None:
        self.session_id: str = "voice_session_primary"
        self._is_running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._on_command_callbacks: list[Callable[[str], None]] = []

    @property
    def is_running(self) -> bool:
        return self._is_running

    def register_command_listener(self, callback: Callable[[str], None]) -> None:
        """Register a callback for observed spoken commands."""
        self._on_command_callbacks.append(callback)

    async def start(self) -> bool:
        """Starts the continuous voice listening and execution pipeline."""
        if self._is_running:
            return True

        logger.info("[VoiceManager] Starting JARVIS Voice Interaction System...")

        # 1. Start audio hardware stream
        stream_ok = audio_capture.start_stream()
        if not stream_ok:
            logger.error("[VoiceManager] Failed to start audio capture stream.")
            return False

        # 2. Calibrate ambient noise
        audio_capture.calibrate_ambient_noise(duration_sec=voice_config.NOISE_CALIBRATION_DURATION)

        # 3. Start barge-in monitor
        barge_in_monitor.start_monitoring()

        self._is_running = True
        self._loop_task = asyncio.create_task(self._voice_loop())
        voice_state_manager.transition_to(VoiceState.MIC_READY, "Voice system online & listening")
        logger.info("[VoiceManager] Voice system is online and ready.")
        return True

    async def stop(self) -> None:
        """Stops the voice interaction pipeline."""
        if not self._is_running:
            return

        self._is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        barge_in_monitor.stop_monitoring()
        tts_engine.stop()
        audio_capture.stop_stream()
        voice_state_manager.transition_to(VoiceState.MIC_OFF, "Voice system offline")
        logger.info("[VoiceManager] Voice system stopped.")

    async def process_spoken_command(self, raw_audio: bytes) -> Dict[str, Any]:
        """Processes a single captured speech utterance through STT -> Brain -> Automation -> TTS."""
        # 1. Speech-To-Text Transcription
        stt_result = await stt_engine.transcribe(raw_audio)
        if not stt_result.get("success") or not stt_result.get("text"):
            voice_state_manager.transition_to(VoiceState.MIC_READY, "No speech recognized")
            return {"success": False, "error": "Unrecognized speech"}

        spoken_text = stt_result["text"]
        logger.info(f"[VoiceManager] Heard: '{spoken_text}'")

        # 2. Wake Word Validation (if enabled)
        has_wake_word, command = wake_word_detector.detect_in_text(spoken_text)
        if voice_config.WAKE_WORD_ENABLED and not has_wake_word:
            logger.debug(f"[VoiceManager] Ignored speech without wake word: '{spoken_text}'")
            voice_state_manager.transition_to(VoiceState.MIC_READY, "Awaiting wake word")
            return {"success": False, "ignored": True, "reason": "No wake word"}

        # If user just said "Hey JARVIS" with no trailing command, acknowledge briefly
        if not command.strip():
            logger.info("[VoiceManager] Wake word detected without command. Responding with prompt.")
            tts_engine.speak("Yes? How can I help you?", priority=True)
            wake_word_detector.record_interaction()
            return {"success": True, "action": "wake_acknowledgment"}

        # 3. Check for Emergency Stop / Barge-In Command
        if barge_in_monitor.is_interrupt_command(command):
            logger.info(f"[VoiceManager] Direct stop command detected: '{command}'")
            barge_in_monitor.trigger_interruption("User voice stop command")
            tts_engine.speak("Stopped.", priority=True)
            return {"success": True, "action": "interrupted"}

        # Notify listeners
        for cb in self._on_command_callbacks:
            try:
                cb(command)
            except Exception:
                pass

        # 4. Cognitive Execution via BrainManager
        voice_state_manager.transition_to(VoiceState.UNDERSTANDING, f"Understanding: '{command}'")
        from brain.pipeline import brain

        output = await brain.handle(command, self.session_id, channel="voice")

        wake_word_detector.record_interaction()

        # 5. Spoken Feedback Response
        response_text = output.response or "Done."
        logger.info(f"[VoiceManager] JARVIS Response: '{response_text[:80]}...'")
        tts_engine.speak(response_text)

        return {
            "success": True,
            "command": command,
            "intent": output.intent,
            "response": response_text,
            "metadata": output.metadata,
        }

    async def _voice_loop(self) -> None:
        """Continuous background listening loop."""
        while self._is_running:
            try:
                # Do not accumulate audio if currently speaking (unless barge-in triggers)
                if tts_engine.is_speaking:
                    await asyncio.sleep(0.05)
                    continue

                chunk = audio_capture.read_chunk(timeout=0.1)
                if chunk is None:
                    await asyncio.sleep(0.02)
                    continue

                pre_buf = [audio_capture.get_buffered_audio(duration_sec=0.4)]
                is_speaking, is_complete, audio_data = vad_detector.process_frame(
                    raw_frame=chunk,
                    energy_threshold=audio_capture.energy_threshold,
                    pre_buffer=pre_buf,
                )

                if is_complete and audio_data:
                    # Process completed speech segment
                    await self.process_spoken_command(audio_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[VoiceManager] Voice loop exception: {e}")
                await asyncio.sleep(0.1)


voice_manager = VoiceManager()
